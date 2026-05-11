# vLLM 请求处理

> 以 `vLLM/vllm-upstream` 的 v1 实现为准。本文主线走 `AsyncLLM`，因为它最能体现“多个用户同时发请求”时的真实行为。同步版 `LLMEngine` 走的是同一套调度与 worker 逻辑，只是输出回收方式不同。

## 先看全链路

```text
用户请求
  -> AsyncLLM.generate()
  -> AsyncLLM.add_request()
  -> OutputProcessor.add_request()
  -> EngineCore.add_request_async()
  -> Scheduler.add_request()
  -> waiting / skipped_waiting
  -> Scheduler.schedule()
  -> SchedulerOutput
  -> EngineCore.step()
  -> GPUModelRunner.execute_model()
  -> GPUModelRunner.sample_tokens()
  -> Scheduler.update_from_output()
  -> OutputProcessor.process_outputs()
  -> RequestOutputCollector / API 返回
```

vLLM 的关键不是“每个请求单独跑完再处理下一个”，而是把所有请求放进统一的请求池里，每一轮只看“谁还能再推进多少 token”。所以从调度器视角看，不存在严格的 prefill / decode 两个大阶段，只有 `num_computed_tokens` 跟 `num_tokens_with_spec` 的追赶关系。

## 一组具体例子

下面用一个简化配置说明高并发请求如何被混排：

```text
max_num_seqs = 4
max_num_batched_tokens = 8
scheduling policy = FCFS
```

同时来了 4 个用户请求：

| 请求 | 到达时间 | prompt tokens | max_tokens | 说明 |
|---|---:|---:|---:|---|
| A | T0 | 3 | 5 | 先到 |
| B | T0 | 2 | 4 | 先到 |
| C | T0 | 3 | 2 | 先到 |
| D | T1 | 6 | 3 | 后到 |

这个例子故意把 token 数做成容易看清的数字。真实线上请求当然会更长、更乱，但调度原则一样。

## 第 1 步：用户请求进入 AsyncLLM

调用入口通常是：

```python
AsyncLLM.generate(...)
```

内部先走：

```python
AsyncLLM.add_request(...)
```

这一步做了几件事：

1. `InputProcessor.process_inputs(...)`
   - 把 prompt / chat / multimodal 输入标准化成 `EngineCoreRequest`
   - 完成 tokenization、参数归一化、`request_id` 绑定
2. `OutputProcessor.add_request(...)`
   - 在 API 侧创建 `RequestState`
   - 为每个请求准备一个 `RequestOutputCollector`
3. `engine_core.add_request_async(...)`
   - 把请求送到后台的 `EngineCore`

所以多个用户同时请求时，API 侧并不是直接进模型，而是每个请求先拥有自己的状态容器：

```text
用户 A -> RequestOutputCollector(A)
用户 B -> RequestOutputCollector(B)
用户 C -> RequestOutputCollector(C)
```

这样后面就算 batch 混在一起，返回结果也不会串。

## 第 2 步：EngineCore 把请求放进 waiting

`EngineCore.add_request(...)` 最终会调用：

```python
Scheduler.add_request(request)
```

Scheduler 里不是立刻执行，而是把请求放到队列中：

- `waiting`：普通待调度请求
- `skipped_waiting`：被结构化输出、远端 KV、流式输入等条件暂时卡住的请求

对应逻辑在：

- `Scheduler.add_request(...)`
- `Scheduler._enqueue_waiting_request(...)`
- `Scheduler._select_waiting_queue_for_scheduling(...)`

如果调度策略是 `FCFS`，就先来先服务；如果是 `PRIORITY`, 就按优先级和到达时间排序。这里的队列本质上只是“候选池”，真正执行要等 `schedule()`。

## 第 3 步：Scheduler.schedule() 组成这一轮 batch

这是整条链路里最关键的一步。

```python
Scheduler.schedule()
```

它做的事情可以概括成：

1. 先处理 `running` 里的老请求
2. 再用剩余 token budget 填充 `waiting`
3. 如果 KV cache 不够，可能 preempt 低优先级请求
4. 产出一份 `SchedulerOutput`

### 3.1 先跑 running

`running` 里存的是已经占着 KV cache 的请求。调度器每一轮先看这些老请求还能推进多少 token。

在我们的例子里，T0 时 A/B/C 同时进入 `running`。第一轮 batch 可以直接把三条请求一起塞进去：

```text
batch 1 = A(3) + B(2) + C(3) = 8 tokens
```

因为总和刚好等于 `max_num_batched_tokens`，所以这一轮不会再接纳新的 waiting 请求。

### 3.2 再跑 waiting

到 T1 时，D 到达。A/B/C 已经在 `running` 里了，D 先进入 `waiting`。

下一轮调度时，先给 A/B/C 各推进 1 个输出 token：

```text
running 部分 = A(1) + B(1) + C(1) = 3 tokens
```

此时还剩 5 个 token budget，于是 D 可以顺手吃掉 5 个 prompt token：

```text
waiting 部分 = D(5)
```

于是第二轮 batch 变成：

```text
batch 2 = A(1) + B(1) + C(1) + D(5) = 8 tokens
```

这就是 vLLM 的 continuous batching：**不同阶段、不同长度、不同到达时间的请求，可以在同一轮里混着跑。**

### 3.3 资源不够时怎么办

如果 `kv_cache_manager.allocate_slots(...)` 发现 KV block 不够，`Scheduler` 会尝试抢占一个 running 请求。FCFS 下通常弹出一个尾部请求；PRIORITY 下会挑最低优先级的请求。被抢占的请求会被重新放回 waiting，后面再恢复。

这也是为什么 vLLM 能扛高并发：不是“永不冲突”，而是“冲突时有明确的回退路径”。

### 3.4 SchedulerOutput 是什么

`schedule()` 不直接给 GPU tensor，而是生成协议对象：

- `scheduled_new_reqs`
- `scheduled_cached_reqs`
- `num_scheduled_tokens`
- `scheduled_spec_decode_tokens`
- `scheduled_encoder_inputs`
- `preempted_req_ids`
- `finished_req_ids`

这份输出是 scheduler 和 worker 的边界。scheduler 只负责说“谁、多少 token、哪些 block”；worker 负责把它翻译成真正的模型输入。

## 第 4 步：EngineCore.step() 驱动 worker 执行

`EngineCore.step()` 的主流程是：

```python
scheduler_output = self.scheduler.schedule()
future = self.model_executor.execute_model(scheduler_output, non_block=True)
model_output = future.result()
if model_output is None:
    model_output = self.model_executor.sample_tokens(grammar_output)
engine_core_outputs = self.scheduler.update_from_output(
    scheduler_output, model_output
)
```

它把“调度”与“模型执行”分开了。

## 第 5 步：GPUModelRunner.execute_model() 把 request batch 变成张量 batch

真正把请求转成 GPU 可吃的 tensor 的，是：

```python
GPUModelRunner.execute_model(...)
```

这一步通常包含：

1. `finish_requests(scheduler_output)`
2. `free_states(scheduler_output)`
3. `add_requests(scheduler_output)`
4. `update_requests(scheduler_output)`
5. `block_tables.apply_staged_writes()`
6. `dispatch_cg_and_sync_dp(...)`
7. `prepare_inputs(...)`
8. `prepare_attn(...)`
9. `model(**model_inputs)` 或 `cudagraph_manager.run_fullgraph(...)`

### 5.1 输入准备

`prepare_inputs(...)` 会把请求列表压平为：

- `input_ids`
- `positions`
- `req_ids`
- `num_scheduled_tokens`
- `slot_mapping`

### 5.2 attention 元数据

`prepare_attn(...)` 会准备：

- `block_tables`
- `slot_mappings`
- `attn_metadata`

这一步把“请求级 KV cache block”翻译成“attention kernel 能读懂的地址表”。

### 5.3 模型前向

最后进入模型 forward。这里不会再看“这是哪个用户”，只看一个 batch 里的 tensor。

如果开了 CUDA graph，就走 `run_fullgraph`；否则直接 `model(**model_inputs)`。

## 第 6 步：sample_tokens() 产出下一 token

模型前向得到 hidden states 后，最后一层会进入：

```python
GPUModelRunner.sample_tokens(...)
```

这里完成：

1. `Sampler.sample(...)`
2. `PromptLogprobsWorker.compute_prompt_logprobs(...)`
3. 多卡/流水线场景下的广播或回收
4. 生成 `ModelRunnerOutput`

如果是最后一个 PP rank，就直接采样；如果不是最后一个 PP rank，则先收发中间结果，再由末端统一采样。

对用户来说，这一步的结果就是：

```text
A 生成一个新 token
B 生成一个新 token
C 生成一个新 token
D 可能还在补 prefill
```

## 第 7 步：Scheduler.update_from_output() 回写状态

模型返回后，调度器把结果写回请求状态：

```python
Scheduler.update_from_output(scheduler_output, model_runner_output)
```

核心动作是：

1. 按 `req_id` 找到请求
2. `Request.append_output_token_ids(...)`
3. `check_stop(...)`
4. 更新 `num_computed_tokens`
5. 处理 stop / max_tokens / stop string
6. 释放完成请求的 KV 和 encoder 状态

如果某个请求已经结束，后续会被移出运行集合；如果还没结束，它会保留在 `running` 中，下一轮继续推进。

### 7.1 这里为什么能知道哪个 token 属于谁

因为 `SchedulerOutput.num_scheduled_tokens` 和 `ModelRunnerOutput.req_id_to_index` 都保留了请求顺序映射。这样 batch 里混了很多请求，回写时也能一一对上。

### 7.2 结束条件

请求可能因为这些原因结束：

- 生成到 `max_tokens`
- 遇到 stop string
- 遇到 stop token
- 被外部 abort
- 发生错误

结束后，调度器会标记 finished，并准备释放相关状态。

## 第 8 步：OutputProcessor 把 token 变成用户可读文本

`Scheduler.update_from_output(...)` 之后，`EngineCoreOutputs` 会被送回 API 侧。`AsyncLLM` 的后台任务：

```python
AsyncLLM._run_output_handler()
```

会持续执行：

```python
engine_core.get_output_async()
output_processor.process_outputs(...)
```

`OutputProcessor.process_outputs(...)` 会做三件事：

1. detokenize token ids
2. 更新 logprobs / metrics
3. 生成 `RequestOutput`，放进对应请求的 `RequestOutputCollector`

于是 A/B/C/D 各自的 `generate()` 协程，只会收到自己的结果，不会串台。

## 用时间线把上面的例子串起来

### T0：A/B/C 同时到达

```text
waiting = [A, B, C]
running = []
```

调度后：

```text
batch 1 = A(3) + B(2) + C(3)
running = [A, B, C]
```

### T1：D 到达

```text
waiting = [D]
running = [A, B, C]
```

调度后：

```text
batch 2 = A(1) + B(1) + C(1) + D(5)
running = [A, B, C, D]
```

### T2：继续推进

接下来每一轮都重复同一件事：

1. `schedule()` 先推进 running
2. 余下 token budget 再喂 waiting
3. `execute_model()` 跑前向
4. `sample_tokens()` 采样
5. `update_from_output()` 回写
6. `OutputProcessor` 发回各自的流

这就是 vLLM 为什么能在高并发下维持吞吐：**不是让请求排队等前一个完全结束，而是把每一轮 token 预算尽量填满。**

## 关键原则

1. **请求和 batch 解耦**
   - 每个请求先有自己的 `RequestState`
   - 真正执行时再被 scheduler 混成 batch

2. **调度和执行解耦**
   - `SchedulerOutput` 只描述“要跑什么”
   - worker 才负责真正算 tensor

3. **输出和请求回收解耦**
   - `OutputProcessor` 把结果转成用户可读输出
   - 结束请求会被单独清理，不影响其他请求

4. **连续批处理**
   - 新来的请求不必等旧请求结束
   - running 和 waiting 可以在同一轮里混跑

5. **KV cache 是核心资源**
   - 不是单纯算力问题
   - `allocate_slots()` 能不能成功，直接决定这轮能不能推进

## 推荐继续看的源码

- `vLLM/vllm-upstream/vllm/v1/engine/async_llm.py`
- `vLLM/vllm-upstream/vllm/v1/engine/core.py`
- `vLLM/vllm-upstream/vllm/v1/core/sched/scheduler.py`
- `vLLM/vllm-upstream/vllm/v1/core/sched/request_queue.py`
- `vLLM/vllm-upstream/vllm/v1/worker/gpu/model_runner.py`
- `vLLM/vllm-upstream/vllm/v1/engine/output_processor.py`

如果要把这篇文档和 mini-vLLM 对照起来，最值得看的对应关系是：

- `engine/scheduler.py`
- `engine/block_manager.py`
- `engine/model_runner.py`
- `layers/attention.py`
- `layers/linear.py`

