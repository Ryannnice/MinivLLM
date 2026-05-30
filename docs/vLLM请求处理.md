# vLLM 多请求调度处理

> 分析对象：`vLLM/vllm-upstream` 的 v1 推理链路。本文用一个高并发例子贯穿 `AsyncLLM -> EngineCore -> Scheduler -> GPUModelRunner -> OutputProcessor`，重点解释多个用户请求如何被混成 batch，又如何准确返回给各自用户。

## 0. 先记住三个配置

本文使用一个教学用简化配置：

```text
max_num_seqs = 4
max_num_batched_tokens = 8
scheduling policy = FCFS
```

| 配置 | 管什么 | 在例子中的含义 |
|---|---|---|
| `max_num_seqs` | 同时处于运行态的请求数 | 最多 4 个请求一起占用调度槽 |
| `max_num_batched_tokens` | 一轮调度中实际要推进的 token 总数 | 通常等于这一轮 prefill token + decode token 的总预算；源码里实际使用的是 `max_num_scheduled_tokens`，默认等于它 |
| `max_tokens` | 单个请求最多生成的输出 token 数 | 由用户或 API 参数传入，不包含 prompt token |

注意：`max_num_batched_tokens` 不是只算输入，也不是只算输出，而是**本轮实际送进模型计算的总 token 预算**。如果源码显式设置了 `max_num_scheduled_tokens`，调度器会优先用它。

## 1. 贯穿全文的并发请求例子

同时来了 4 个用户请求：

| 请求 | 到达时间 | prompt tokens | `max_tokens` | 说明 |
|---|---:|---:|---:|---|
| A | T0 | 3 | 5 | 先到 |
| B | T0 | 2 | 4 | 先到 |
| C | T0 | 3 | 2 | 先到 |
| D | T1 | 6 | 3 | 后到 |

可以把它理解成：

```text
A: 输入 3 个 token，最多再生成 5 个 token
B: 输入 2 个 token，最多再生成 4 个 token
C: 输入 3 个 token，最多再生成 2 个 token
D: 输入 6 个 token，最多再生成 3 个 token
```

`max_tokens` 是请求侧参数，例如 OpenAI 风格请求里的 `max_tokens` 或 vLLM 内部的 `SamplingParams(max_tokens=...)`。

## 2. 全链路总览

一条请求进入 vLLM 后，会经过这条链：

```text
用户请求
  -> AsyncLLM.generate()
  -> AsyncLLM.add_request()
  -> InputProcessor.process_inputs()
  -> OutputProcessor.add_request()
  -> EngineCore.add_request_async()
  -> Scheduler.add_request()
  -> Scheduler.schedule()
  -> EngineCore.step()
  -> GPUModelRunner.execute_model()
  -> GPUModelRunner.sample_tokens()
  -> Scheduler.update_from_output()
  -> OutputProcessor.process_outputs()
  -> RequestOutputCollector
  -> 用户收到流式输出
```

分层看更清楚：

| 层 | 主要函数 | 负责什么 |
|---|---|---|
| API 层 | `AsyncLLM.generate()` | 接收用户请求，返回异步生成流 |
| 输入处理 | `InputProcessor.process_inputs()` | tokenize，生成 `EngineCoreRequest` |
| 输出处理 | `OutputProcessor.add_request()` | 在输出处理器里登记请求，并绑定请求队列 |
| 引擎核心 | `EngineCore.step()` | 串起 schedule、execute、sample、update |
| 调度器 | `Scheduler.schedule()` | 决定本轮跑哪些请求、多少 token |
| Worker | `GPUModelRunner.execute_model()` | 把请求 batch 转成 GPU tensor 并前向 |
| 采样 | `GPUModelRunner.sample_tokens()` | 从 logits 采样出新 token |
| 回写 | `Scheduler.update_from_output()` | 把新 token 写回对应 request |

AsyncLLM 是 vLLM 的异步请求入口。

你可以把它理解成：
用户发请求
 -> AsyncLLM 接收
 -> 后台引擎跑
 -> 结果异步流式返回

它的核心作用有三件：
1. 接收请求
    - 处理 generate() / add_request()
    - 把用户输入转成内部 EngineCoreRequest
2. 启动后台输出循环
    - _run_output_handler()
    - 持续从 EngineCore 拉结果
3. 把结果分发回每个请求
      - 每个请求有自己的 RequestOutputCollector
      - 所以多个用户同时请求时，不会串结果

## 3. Step 1：请求进入 AsyncLLM

用户 A/B/C 在 T0 同时进入：

```text
A -> AsyncLLM.generate(request_id="A", prompt=..., SamplingParams(max_tokens=5))
B -> AsyncLLM.generate(request_id="B", prompt=..., SamplingParams(max_tokens=4))
C -> AsyncLLM.generate(request_id="C", prompt=..., SamplingParams(max_tokens=2))
```

每个请求都会走：

```python
AsyncLLM.add_request(...)
InputProcessor.process_inputs(...)
OutputProcessor.add_request(...)
engine_core.add_request_async(...)
```

这一阶段最重要的结果是两个：

```text
EngineCoreRequest:
  request_id
  prompt_token_ids
  sampling_params.max_tokens
  arrival_time
  priority

RequestOutputCollector:
  每个请求一个独立输出队列
```

所以即使后面 A/B/C/D 被混在一个 batch 里执行，API 侧仍然知道：

```text
A 的输出放回 A 的队列
B 的输出放回 B 的队列
C 的输出放回 C 的队列
```

## 4. Step 2：Scheduler 接收请求，但不立刻执行

`EngineCore` 收到请求后，会调用：

```python
Scheduler.add_request(request)
```

调度器内部维护两个主要等待队列：

| 队列 | 放什么 |
|---|---|
| `waiting` | 普通待调度请求 |
| `skipped_waiting` | 暂时不能调度的请求，例如等待结构化输出 grammar、远端 KV、流式输入 |

T0 时 A/B/C 进入后：

```text
waiting = [A, B, C]
running = []
```

这里还没有模型执行，只是把请求放进调度器的候选池。

## 5. Step 3：第一轮 schedule，把 A/B/C 拼成 batch

调度入口：

```python
Scheduler.schedule()
```

vLLM v1 的调度器不是按固定的 prefill 阶段、decode 阶段切开，而是看每个请求：

```text
还差多少 token 没算？
```

内部核心状态可以理解为：

```text
num_computed_tokens      已经算过多少 token
num_tokens_with_spec     当前请求总共希望模型追到哪里
num_new_tokens           本轮还能推进多少 token
```

### 5.1 T0 的调度结果

A/B/C 都是新请求，prompt 还没算：

| 请求 | 本轮要算 | token 数 |
|---|---|---:|
| A | prompt prefill | 3 |
| B | prompt prefill | 2 |
| C | prompt prefill | 3 |

合计：

```text
3 + 2 + 3 = 8
```

刚好等于 `max_num_batched_tokens = 8`，所以第一轮 batch 是：

```text
batch 1 = A(3) + B(2) + C(3)
```

调度后状态：

```text
waiting = []
running = [A, B, C]
```

这一轮虽然调度的是 prompt prefill token，但生成模型通常会在 prefill 前向结束后，用每个请求最后一个 prompt 位置的 logits 采样出首个输出 token。也就是说，`max_num_batched_tokens` 统计的是本轮送入模型计算的 token，不等于本轮最终返回给用户的 token 数。

这里的 `running` 表示请求已经被接纳进运行集合，并且有自己的 KV cache block 生命周期。

## 6. Step 4：T1 时 D 到达，第二轮混合 prefill 和 decode

D 在 T1 到达：

```text
D -> prompt tokens = 6, max_tokens = 3
```

它先进入 waiting：

```text
waiting = [D]
running = [A, B, C]
```

下一轮调度时，vLLM 先看 `running` 里的 A/B/C。**它们 prompt 已经算完，接下来每个请求通常推进 1 个 decode token：**

```text
running 部分 = A(1) + B(1) + C(1) = 3 tokens
```

本轮总预算是 8，还剩：

```text
8 - 3 = 5 tokens
```

于是调度器继续从 `waiting` 里拿 D。D 的 prompt 有 6 个 token，但本轮只剩 5 个 token 预算，所以先吃 5 个 prompt token：

> 这里假设 `enable_chunked_prefill=True`，也就是 v1 scheduler 的默认行为。  
> 如果关闭 chunked prefill，D 的 prompt 长度 6 > 剩余预算 5，调度器会先停下，不会把 D 切成 5 个 token 排进去。

```text
waiting 部分 = D(5)
```

第二轮 batch 变成：

```text
batch 2 = A(1) + B(1) + C(1) + D(5) = 8 tokens
```

这就是 continuous batching：

```text
同一轮 batch 里可以同时有：
  - 老请求的 decode token
  - 新请求的 prefill token
  - 不同长度、不同到达时间的请求
```

## 7. Step 5：batch 里怎么知道 token 属于谁

vLLM 不是靠 token 本身识别归属，而是靠 batch 元数据记录每个 request 的连续区间。

对于第二轮：

```text
A -> 1 token
B -> 1 token
C -> 1 token
D -> 5 token
```

worker 侧可以整理成：

```text
req_ids         = [A, B, C, D]
num_scheduled   = [1, 1, 1, 5]
query_start_loc = [0, 1, 2, 3, 8]
input_ids       = [A1, B1, C1, D1, D2, D3, D4, D5]
```

`query_start_loc` 是前缀和，表示每个请求在扁平 `input_ids` 里的边界：

| 请求 | 区间 | 含义 |
|---|---|---|
| A | `input_ids[0:1]` | A 本轮的 1 个 token |
| B | `input_ids[1:2]` | B 本轮的 1 个 token |
| C | `input_ids[2:3]` | C 本轮的 1 个 token |
| D | `input_ids[3:8]` | D 本轮的 5 个 token |

对应源码对象：

| 字段 | 作用 |
|---|---|
| `SchedulerOutput.num_scheduled_tokens` | `req_id -> 本轮 token 数` |
| `InputBatch.req_ids` | `batch_idx -> req_id` |
| `InputBatch.query_start_loc` | `batch_idx -> input_ids 起止边界` |
| `InputBatch.idx_mapping` | `batch_idx -> worker 内部 request state 下标` |
| `ModelRunnerOutput.req_id_to_index` | 采样结果回写时用的 request 映射 |

一句话：

```text
vLLM 先按 request 分段拼 tensor，再用 query_start_loc 记录每段边界。
```

## 8. Step 6：SchedulerOutput 是调度器和 worker 的协议

`Scheduler.schedule()` 不直接创建 GPU tensor，而是返回 `SchedulerOutput`。

对于第二轮，它可以理解成：

```text
SchedulerOutput:
  num_scheduled_tokens:
    A: 1
    B: 1
    C: 1
    D: 5

  scheduled_cached_reqs:
    A, B, C

  scheduled_new_reqs:
    D

  total_num_scheduled_tokens:
    8
```

含义：

| 字段 | 例子中的含义 |
|---|---|
| `scheduled_new_reqs` | D 第一次被调度，需要把完整请求信息发给 worker |
| `scheduled_cached_reqs` | A/B/C 之前已经在 worker 中缓存过，只发增量 |
| `num_scheduled_tokens` | 每个请求本轮推进多少 token |
| `total_num_scheduled_tokens` | 本轮总 token 数，不能超过 `max_num_batched_tokens` |
| `finished_req_ids` | 告诉 worker 哪些请求已经结束，可以释放缓存状态 |

这个对象是边界：

```text
Scheduler 只管“谁该跑、跑多少、分到哪些 KV block”。
Worker 负责“怎么变成 GPU tensor、怎么执行模型”。
```

## 9. Step 7：EngineCore.step() 串起一轮执行

`EngineCore.step()` 是一轮推理迭代的主循环：

```python
scheduler_output = self.scheduler.schedule()
future = self.model_executor.execute_model(scheduler_output, non_block=True)
grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
model_output = future.result()
if model_output is None:
    model_output = self.model_executor.sample_tokens(grammar_output)
engine_core_outputs = self.scheduler.update_from_output(
    scheduler_output, model_output
)
```

这一轮实际做了四件事：

```text
schedule  决定这轮跑什么
execute   跑模型前向
sample    从 logits 得到新 token
update    把结果写回 request 状态
```

## 10. Step 8：GPUModelRunner 把调度结果变成 GPU 输入

真正把 `SchedulerOutput` 转成模型输入的是：

```python
GPUModelRunner.execute_model(...)
```

核心动作：

| 动作 | 作用 |
|---|---|
| `finish_requests(scheduler_output)` | 清理已经结束的请求 |
| `free_states(scheduler_output)` | 释放 worker 侧状态 |
| `add_requests(scheduler_output)` | 把新请求加入 worker 缓存 |
| `update_requests(scheduler_output)` | 更新老请求的增量 token |
| `prepare_inputs(...)` | 生成 `input_ids`、`positions`、`query_start_loc` |
| `prepare_attn(...)` | 生成 `block_tables`、`slot_mappings`、`attn_metadata` |
| `model(**model_inputs)` | 执行 Transformer forward |

### 10.1 为什么需要 block table 和 slot mapping

Scheduler 只知道请求持有哪些 KV block，例如：

```text
A -> blocks [10]
B -> blocks [11]
C -> blocks [12]
D -> blocks [20]
```

attention kernel 需要的是更底层的地址信息：

```text
这个 token 的 K/V 应该写到哪个 block 的哪个 slot？
这个请求历史上的 K/V 应该从哪些 block 读取？
```

所以 worker 要把请求级 block 信息翻译成：

```text
block_tables   请求 -> 物理 block 列表
slot_mappings  本轮每个 token -> KV cache 写入位置
attn_metadata  attention backend 需要的所有元数据
```

### 10.2 mixed batch 在 Transformer 里为什么不会串

第二轮的 mixed batch 进入 embedding 后，会变成一个矩阵：

```text
input_ids      = [A1, B1, C1, D1, D2, D3, D4, D5]
hidden_states  = embedding(input_ids)
hidden_states.shape = [8, hidden_size]
```

这个矩阵里确实混着多个 request。但大多数 Transformer 层是逐 token 独立计算的，例如 RMSNorm、Linear、MLP：

```text
Y = X @ W

Y[0] = X[0] @ W  # A
Y[1] = X[1] @ W  # B
Y[2] = X[2] @ W  # C
Y[3] = X[3] @ W  # D1
...
```

这些操作只是批量处理多行，不会让 A 的 hidden state 读到 B 或 D。

真正可能跨 token 交互的是 self-attention。逻辑上，如果不加限制，`Q @ K.T` 会让所有 token 互相看见。vLLM 通过 attention metadata 把不同 request 隔开：

```text
query_start_loc = [0, 1, 2, 3, 8]

A: input_ids[0:1]  只能读 A 的 KV blocks
B: input_ids[1:2]  只能读 B 的 KV blocks
C: input_ids[2:3]  只能读 C 的 KV blocks
D: input_ids[3:8]  只能读 D 的 KV blocks，并在 D 内部使用 causal mask
```

逻辑 attention 可见性可以理解成：

```text
        A_hist A1 | B_hist B1 | C_hist C1 | D1 D2 D3 D4 D5
A1        1   1  |   0   0  |   0   0  | 0  0  0  0  0
B1        0   0  |   1   1  |   0   0  | 0  0  0  0  0
C1        0   0  |   0   0  |   1   1  | 0  0  0  0  0
D1        0   0  |   0   0  |   0   0  | 1  0  0  0  0
D2        0   0  |   0   0  |   0   0  | 1  1  0  0  0
D3        0   0  |   0   0  |   0   0  | 1  1  1  0  0
D4        0   0  |   0   0  |   0   0  | 1  1  1  1  0
D5        0   0  |   0   0  |   0   0  | 1  1  1  1  1
```

实际实现通常不会构造这么大的 dense mask，而是把 `query_start_loc`、`seq_lens`、`block_tables`、`slot_mappings`、`positions` 传给高性能 attention kernel。kernel 按这些元数据只访问合法的 K/V：

```text
普通层：逐 token 独立，天然不串。
Attention：按 request 边界和 KV block 表读取，只在同一 request 内做 causal attention。
```

## 11. Step 9：模型前向和采样

模型前向可能走两条路：

```text
普通路径: model(**model_inputs)
CUDA graph 路径: cudagraph_manager.run_fullgraph(...)
```

前向结束后，最后一个 pipeline stage 会执行：

```python
GPUModelRunner.sample_tokens(...)
```

它会：

1. 根据 hidden states 计算 logits
2. 按 sampling 参数采样 token
3. 生成 `ModelRunnerOutput`
4. 更新 worker 内部 request state

第二轮中，采样结果可能是：

```text
A -> token a1
B -> token b1
C -> token c1
D -> 仍在 prefill，暂时没有输出 token
```

注意：prefill 请求不一定每个 chunk 都立刻返回用户可见 token。D 本轮只完成了前 5 个 prompt token，完整 prompt 还有 1 个 token 没算完，因此它通常还不会采样输出。

## 12. Step 10：Scheduler.update_from_output() 回写请求状态

模型输出回到调度器后，调用：

```python
Scheduler.update_from_output(scheduler_output, model_runner_output)
```

调度器会按 `req_id` 回写：

```text
A: append_output_token_ids(a1)
B: append_output_token_ids(b1)
C: append_output_token_ids(c1)
D: num_computed_tokens 增加 5，但还没生成输出 token
```

同时检查停止条件：

| 停止条件 | 例子 |
|---|---|
| 达到 `max_tokens` | C 最多生成 2 个，生成满后结束 |
| 遇到 stop token | 例如 EOS |
| 遇到 stop string | 输出文本命中停止字符串 |
| 用户中断 | 客户端断开，API 调用 abort |
| 错误 | 模型或执行异常 |

如果请求结束，调度器会释放它的 KV cache block，并从 `running` 中移除。

## 13. Step 11：OutputProcessor 把 token 返回给用户

`Scheduler.update_from_output()` 生成 `EngineCoreOutputs` 后，API 侧后台任务会持续拉取：

```python
engine_core.get_output_async()
output_processor.process_outputs(...)
```

`OutputProcessor.process_outputs(...)` 做三件事：

| 动作 | 说明 |
|---|---|
| detokenize | 把 token id 转成文本 |
| 组装 `RequestOutput` | 包含 text、token_ids、logprobs、finish_reason |
| 放入请求自己的队列 | `RequestOutputCollector.put(...)` |

所以这一轮结束后：

```text
RequestOutputCollector(A) 收到 A 的新文本
RequestOutputCollector(B) 收到 B 的新文本
RequestOutputCollector(C) 收到 C 的新文本
RequestOutputCollector(D) 暂时没有可见输出
```

用户侧的 `AsyncLLM.generate()` 是一个异步生成器，会不断从自己的 collector 里取结果并 yield：

```text
用户 A 只看到 A 的流式输出
用户 B 只看到 B 的流式输出
用户 C 只看到 C 的流式输出
用户 D 等 prefill 完成后才开始看到输出
```

## 14. 把时间线压缩成一张表

| 时间 | waiting | running | 本轮 batch | 输出 |
|---|---|---|---|---|
| T0 前 | `[A, B, C]` | `[]` | 还未调度 | 无 |
| T0 调度 | `[]` | `[A, B, C]` | `A(3)+B(2)+C(3)=8` | 通常会顺便采样出 A/B/C 各自的首个输出 token |
| T1 前 | `[D]` | `[A, B, C]` | 还未调度 | 无 |
| T1 调度 | `[]` 或 `[D剩余]` | `[A, B, C, D]` | `A(1)+B(1)+C(1)+D(5)=8` | A/B/C 各可能输出 1 token |
| T2 以后 | 视完成情况变化 | 未完成请求继续保留 | 继续混排 prefill/decode | 完成的请求释放 KV |

这个表就是 vLLM 高并发推理的核心：

```text
请求不断到达
请求不断完成
调度器每轮重新拼 batch
GPU 每轮尽量吃满 token budget
```

## 15. 资源不够时怎么办

如果 `kv_cache_manager.allocate_slots(...)` 发现 KV cache block 不够，scheduler 会尝试抢占请求：

```text
KV block 不够
  -> 找一个可抢占 request
  -> 从 running 移除
  -> 放回 waiting
  -> 释放或延迟释放相关资源
  -> 后面再恢复或重算
```

FCFS 下通常抢占尾部请求；PRIORITY 下会优先抢占低优先级请求。

这说明 vLLM 的高并发不是“无限塞请求”，而是在这些约束中做动态平衡：

```text
max_num_seqs
max_num_batched_tokens
KV cache block 数
LoRA 数量限制
encoder / multimodal budget
priority / FCFS 策略
```

## 16. 关键源码位置

| 主题 | 文件 |
|---|---|
| 异步入口 | `vLLM/vllm-upstream/vllm/v1/engine/async_llm.py` |
| 引擎主循环 | `vLLM/vllm-upstream/vllm/v1/engine/core.py` |
| 请求对象 | `vLLM/vllm-upstream/vllm/v1/request.py` |
| 调度器 | `vLLM/vllm-upstream/vllm/v1/core/sched/scheduler.py` |
| 请求队列 | `vLLM/vllm-upstream/vllm/v1/core/sched/request_queue.py` |
| 调度输出 | `vLLM/vllm-upstream/vllm/v1/core/sched/output.py` |
| GPU runner | `vLLM/vllm-upstream/vllm/v1/worker/gpu/model_runner.py` |
| InputBatch | `vLLM/vllm-upstream/vllm/v1/worker/gpu/input_batch.py` |
| 输出处理 | `vLLM/vllm-upstream/vllm/v1/engine/output_processor.py` |

## 17. 和 mini-vLLM 的对应关系

| vLLM 概念 | mini-vLLM 路径 | 学习重点 |
|---|---|---|
| Scheduler | `mini-vllm/src/myvllm/engine/scheduler.py` | waiting/running、token budget、preemption |
| KV block 管理 | `mini-vllm/src/myvllm/engine/block_manager.py` | block 分配、释放、append |
| ModelRunner | `mini-vllm/src/myvllm/engine/model_runner.py` | 把调度结果变成模型输入 |
| Attention | `mini-vllm/src/myvllm/layers/attention.py` | prefill/decode、KV cache 使用 |
| Linear | `mini-vllm/src/myvllm/layers/linear.py` | TP 行并行、列并行、权重加载 |

一句话总结：

```text
vLLM 的请求处理不是“一个用户一个 batch”，而是“所有用户共享一个动态 token 调度池”。
Scheduler 决定每轮 token 怎么混排，GPUModelRunner 负责把混排结果变成 tensor，
OutputProcessor 再把采样结果按 request_id 分发回各自用户。
```
