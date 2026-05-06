# vLLM.md

## 1. 文档范围

- 分析对象：`vLLM/vllm-upstream`
- 本地源码基线：`92a7c121b62a1484b68c0a27d1ecefd1a84f78fc`
- 关注重点：`vllm/v1` 调度执行主线、KV Cache/PagedAttention、continuous batching、Python 调度层到 C++/CUDA/后端 kernel 的落地链路
- 不展开的话题：OpenAI 兼容 API 表层协议、模型数学原理、单个模型结构细节
- 证据规则：所有关键结论都尽量落到具体源码路径、核心类、关键函数或执行链
- 分析维度：本文按四条硬件友好主线拆解
  1. 请求调度与批处理摊销
  2. KV Cache/显存组织
  3. attention/operator 执行路径
  4. 图执行与并行通信

## 2. 核心结论

1. vLLM 的“continuous batching”本质不是一个单独模块，而是 `Scheduler` 以 `num_computed_tokens` 追赶 `num_tokens_with_spec` 的统一调度模型。
2. PagedAttention 在系统层面的真正价值不是某一个 CUDA kernel，而是“逻辑块管理 + block table + slot mapping + backend-specific KV layout”构成的整套 KV 虚拟内存协议。
3. `Scheduler`、`KVCacheManager`、`Executor`、`GPUModelRunner` 四层分工非常清晰：前者决定“哪些 token 该算”，KV 层决定“KV 放到哪里”，`Executor` 负责跨 worker/进程编排，`GPUModelRunner` 负责“把逻辑调度翻译成设备输入并执行”。
4. 在 v1 主线中，attention backend 已经从“单一 paged attention kernel”演进为“多后端可插拔执行系统”，FlashAttention、Triton、ROCm、自定义 kernel 都可能成为最终落点。
5. prefix cache 命中的最小单位是“完整 block”，而且即使命中全部前缀，最后一个 token 仍可能需要重算以产出 logits；这决定了它不是语义级“完全跳过 prompt”。
6. vLLM 的高吞吐来自三层摊销：调度摊销、元数据准备摊销、kernel/graph 执行摊销，而不是单点算子优化。

## 3. 系统边界与分层地图

| 层级 | 关键路径 | 主要职责 | 不负责 |
|---|---|---|---|
| Engine API 层 | `vllm/v1/engine/llm_engine.py` | 接收请求、输入标准化、输出后处理、驱动 `EngineCoreClient` | 不直接做调度和 kernel 执行 |
| Engine Core 层 | `vllm/v1/engine/core.py` | 初始化 executor、profiling 可用显存、创建 KV 配置、主循环 `schedule -> execute -> update` | 不关心具体 attention backend 细节 |
| 调度层 | `vllm/v1/core/sched/scheduler.py` | 运行队列/等待队列、token budget、preemption、spec decode、encoder budget、KV connector 协调 | 不直接分配底层张量 |
| KV 管理层 | `vllm/v1/core/kv_cache_manager.py` `vllm/v1/core/block_pool.py` `vllm/v1/core/kv_cache_coordinator.py` | block 分配、prefix cache 命中、block 生命周期、逻辑 KV 组协调 | 不执行模型前向 |
| Executor 层 | `vllm/v1/executor/*` | 把 `EngineCore` 的执行请求转成跨 worker/进程的 `collective_rpc`，分离 `execute_model` 与 `sample_tokens` 两阶段 | 不直接决定调度策略或 attention backend |
| Worker/Runner 层 | `vllm/v1/worker/gpu_worker.py` `vllm/v1/worker/gpu/model_runner.py` | 将 `SchedulerOutput` 翻译为设备输入，准备 block table/slot mapping/attention metadata，执行模型 | 不决定高层调度策略 |
| Attention Backend 层 | `vllm/v1/attention/*` `vllm/model_executor/layers/attention/attention.py` | 选择 backend，定义 KV 形状、stride、metadata builder、forward 逻辑 | 不管理请求级 block 生命周期 |
| Native Ops 层 | `vllm/_custom_ops.py` `csrc/attention/*.cu` | C++/CUDA/Triton/ROCm 自定义 kernel | 不理解请求队列语义 |

## 4. 端到端请求生命周期

```mermaid
flowchart TD
    A[LLMEngine.add_request] --> B[InputProcessor.process_inputs]
    B --> C[EngineCore.add_request]
    C --> D[Scheduler.add_request]
    D --> E[Scheduler.schedule]
    E --> F[KVCacheManager.allocate_slots / get_computed_blocks]
    F --> G[SchedulerOutput]
    G --> H[Executor.execute_model]
    H --> I[GPUModelRunner.execute_model]
    I --> J[prepare_inputs / prepare_attn]
    J --> K[build_attn_metadata]
    K --> L[Attention.forward]
    L --> M[backend.do_kv_cache_update + backend.forward]
    M --> N[FlashAttention / Triton / custom ops / paged kernels]
    G --> O[Scheduler.get_grammar_bitmask]
    N --> P[Executor.sample_tokens if needed]
    O --> P
    P --> Q[GPUModelRunner.sample_tokens]
    Q --> R[Scheduler.update_from_output]
    R --> S[EngineCoreOutput -> OutputProcessor]
```

举例子：
vLLM 调度侧在 GPUModelRunner.execute_model() 里调用 self.model(**model_inputs)，模型内部的 Transformer
layers 在具体模型的 forward() 里循环执行。  

在 vLLM v1 里，Transformer 前向是在 GPUModelRunner.execute_model() 里触发的，关键调用是：
vLLM/vllm-upstream/vllm/v1/worker/gpu/model_runner.py:1118
model_output = self.model(**model_inputs)

这行会进入具体模型类的 forward()。以 Qwen3 为例，调用链大致是：
EngineCore.step()
 -> Executor.execute_model()
 -> GPUWorker.execute_model()
 -> GPUModelRunner.execute_model()
 -> self.model(**model_inputs)
 -> Qwen3ForCausalLM.forward()
 -> Qwen3Model / Qwen2Model.forward()
 -> for layer in self.layers: layer(...)

真正逐层执行 Transformer block 的循环在 Qwen2/Qwen3 共享的模型主体里：
vLLM/vllm-upstream/vllm/model_executor/models/qwen2.py:408
for idx, layer in enumerate(islice(self.layers, self.start_layer, self.end_layer)):
    hidden_states, residual = layer(positions, hidden_states, residual)

每个 layer(...) 执行一个 decoder layer，里面再执行 attention 和 MLP。以 Qwen3 的 decoder layer 为例：
vLLM/vllm-upstream/vllm/model_executor/models/qwen3.py:216
hidden_states = self.self_attn(...)
hidden_states = self.mlp(hidden_states)




### 4.1 入口与引擎主循环

- 入口对象是 `vllm/v1/engine/llm_engine.py` 中的 `LLMEngine`。
- `LLMEngine.add_request()` 先把原始 prompt、sampling params、LoRA、多模态输入等交给 `InputProcessor`，生成 `EngineCoreRequest`。
- `LLMEngine` 并不直接调度模型，而是通过 `EngineCoreClient.make_client()` 连接 `EngineCore`。
- 真正的内循环在 `vllm/v1/engine/core.py::EngineCore.step()`：
  1. `scheduler.schedule()`
  2. `model_executor.execute_model(..., non_block=True)`
  3. `scheduler.get_grammar_bitmask(...)`
  4. 等待 `future.result()`；若 worker 仅完成前向则再走 `model_executor.sample_tokens(...)`
  5. `scheduler.update_from_output(...)`

### 4.2 初始化阶段

`EngineCore.__init__()` 的关键动作：

1. 创建 `Executor`
2. 通过 `model_executor.get_kv_cache_specs()` 获取模型需要的 KV 规格
3. 用 `model_executor.determine_available_memory()` profiling 可用显存
4. 调用 `get_kv_cache_configs(...)` 生成 `KVCacheConfig`
5. `model_executor.initialize_from_config(...)` 真正分配 KV cache 并 warmup
6. 创建 `Scheduler`

这一点很关键：vLLM 的 KV 系统不是静态写死的，它是在模型加载后、显存 profiling 后、结合 backend 能力动态决定。

## 5. 核心状态对象与协议

| 对象 | 路径 | 所有者 | 作用 |
|---|---|---|---|
| `Request` | `vllm/v1/request.py` | Scheduler | 请求级状态机，记录 prompt/output/spec tokens、`num_computed_tokens`、`num_output_placeholders`、block hashes |
| `SchedulerOutput` | `vllm/v1/core/sched/output.py` | Scheduler | 一轮调度的设备侧执行描述，包括每个 request 计划计算的 token 数、block 变更、spec decode 元数据等 |
| `KVCacheConfig` | `vllm/v1/kv_cache_interface.py` | EngineCore / Worker | 描述 KV tensor、group、block size、layout 需求 |
| `KVCacheBlocks` | `vllm/v1/core/kv_cache_utils.py` | KVCacheManager | Scheduler 与 KV 管理层之间的抽象接口，隐藏内部 block 结构 |
| `InputBatch` | `vllm/v1/worker/gpu/input_batch.py` | GPUModelRunner | 一轮 batch 的扁平化设备输入描述，包括 `query_start_loc`、`seq_lens`、`input_ids`、`positions` |
| `CommonAttentionMetadata` | `vllm/v1/attention/backend.py` | Attention metadata builder | backend 无关的 attention 输入协议 |
| `slot_mapping` | `vllm/v1/worker/gpu/block_table.py` | Worker | 逻辑 token 位置到物理 KV 槽位的映射 |
| `block_tables` | `vllm/v1/worker/gpu/block_table.py` | Worker | 请求视角的“逻辑序列 -> 物理 block 列表”映射 |

### 5.1 `Request` 为什么是 continuous batching 的核心

`vllm/v1/request.py::Request` 中最关键的三个字段：

- `num_computed_tokens`
- `num_tokens_with_spec`
- `num_output_placeholders`

这三个字段决定了一轮调度还需要为该请求推进多少 token。`Scheduler.schedule()` 的核心不是“这个请求处于 prefill 还是 decode”，而是“这个请求还差多少 token 没被计算”。

## 6. 系统不变量

### 不变量 1：调度推进以 token 差值为中心，而不是以阶段名为中心

在 `vllm/v1/core/sched/scheduler.py::Scheduler.schedule()` 中，作者直接写明：

- 没有严格意义上的 “prefill phase” 和 “decode phase”
- 调度器只关心让 `num_computed_tokens` 追上 `num_tokens_with_spec`

因此，chunked prefill、spec decode、prefix cache、future jump decoding 都能统一进一个框架。

### 不变量 2：KV 的逻辑抽象是 block，物理布局交给 backend

- Scheduler/KV manager 只处理 block id 和 block 数量。
- 物理 KV tensor 的 shape、stride、layout 由 backend 在 `vllm/v1/worker/gpu/attn_utils.py::_reshape_kv_cache()` 中决定。
- 同一个高层调度逻辑可以落在不同 KV 物理布局上。

### 不变量 3：prefix cache 只缓存完整 block，而且最后一个 token 可能必须重算

`vllm/v1/core/kv_cache_manager.py::get_computed_blocks()` 中，命中的 prefix 以完整 block 为单位统计。
即使所有 prompt token 都命中，也会把 `max_cache_hit_length` 设成 `request.num_tokens - 1`，避免最后一个 token 完全不经过前向，从而拿不到 logits。

### 不变量 4：preemption 是真实回退，不是“暂停后原地恢复”

在 `Scheduler._preempt_request()` 中，请求会：

- 释放 KV blocks
- 释放 encoder cache
- 重置局部进度
- 重新回到 waiting 队列

因此 preemption 的代价很高，系统依赖 admission 和 token budget 降低抖动。

### 不变量 5：PagedAttention 是“系统协议”，不是“单 kernel 名称”

`vllm/v1/attention/ops/paged_attn.py::PagedAttention` 在 v1 中主要暴露 `split_kv_cache()` 和 `write_to_paged_cache()` 这类 page 化 KV 操作入口。
真正的 decode/prefill 执行路径可能落到：

- `FlashAttention`
- `Triton`
- `ROCm custom paged attention`
- `_C.paged_attention_v1/v2`

所以 “用了 PagedAttention” 在 v1 里更准确地说是“用了 page-based KV 管理协议”。

## 7. 深入拆解 A：调度器与 Continuous Batching

## 7.1 调度预算

`Scheduler` 初始化时最关键的预算字段：

- `max_num_running_reqs = max_num_seqs`
- `max_num_scheduled_tokens`
- `max_model_len`
- `encoder_compute_budget`
- `num_lookahead_tokens`（spec decode）

这一组预算共同决定“一轮能推进多少请求、多少 token、多少 encoder work”。

## 7.2 运行中请求优先，等待中请求后补

`Scheduler.schedule()` 的基本顺序：

1. 先调度 `running`
2. 再调度 `waiting`
3. 若 block 不够则 preempt 低优先级 request
4. 最终形成 `SchedulerOutput`

这样做的直接效果是：

- decode 请求可以持续推进，避免短请求被长 prompt 完全压住
- chunked prefill 可以与 decode 混排
- request 的“热状态”尽量保留在 worker 常驻内存中

## 7.3 `num_new_tokens` 的统一定义

对于 running request，调度器的核心式子是：

`num_new_tokens = num_tokens_with_spec + num_output_placeholders - num_computed_tokens`

这正是 continuous batching 的本体：请求不是被分成 prefill/decode 两个静态阶段，而是被看作一个不断追赶“应有计算进度”的 token 流。

## 7.4 长 prompt 与 chunked prefill

如果 `long_prefill_token_threshold` 被设置，长 prompt 会被主动切块。
若 `enable_chunked_prefill` 为假，等待队列中的长 request 在 token budget 不足时会直接停住，而不是被部分推进。

这意味着：

- chunked prefill 是吞吐优化
- 但它也带来 admission 复杂度和 KV churn 风险

## 7.5 admission 保护

`Scheduler` 还可以在 `scheduler_reserve_full_isl` 打开时调用 `KVCacheManager.can_fit_full_sequence()`。
这不是普通“当前 chunk 能不能塞下”，而是“整条序列最终能不能放得下”。

它的意义是防止 chunked prefill 只看眼前可行、后续反复 preempt/recompute。

## 7.6 spec decode 与 structured output 不是外挂

在 v1 中，spec decode、grammar bitmask、async scheduling 都直接嵌进主热路径：

- `EngineCore.step_with_batch_queue()`
- `GPUModelRunner.sample_tokens()`
- `Scheduler.update_draft_token_ids(...)`
- `Scheduler.get_grammar_bitmask(...)`

所以它们不是“旁路特性”，而是吞吐模型的一部分。

## 8. 深入拆解 B：KV Cache、Block Pool 与 PagedAttention

## 8.1 Block Pool：全局物理资源池

`vllm/v1/core/block_pool.py::BlockPool` 负责：

- 初始化所有 `KVCacheBlock`
- 维护 free block queue
- 维护 block hash -> cached block 的映射
- 负责 eviction / touch / free

从系统视角看，它相当于“KV 页框分配器”。

## 8.2 `KVCacheManager`：请求视角的逻辑 KV 管理器

`KVCacheManager` 封装了几类关键操作：

- `get_computed_blocks(request)`：查 prefix cache 命中
- `can_fit_full_sequence(...)`：admission 检查
- `allocate_slots(...)`：给本轮新 token 分配 block
- `usage`：汇报 KV 使用率

其中 `allocate_slots()` 的注释本身已经给出了非常清晰的布局语义：

- 已计算的本地 token
- 新命中的 prefix token
- connector 中外部已有 token
- 本轮新算 token
- speculative lookahead token

vLLM 真正难的地方就在这里：它不是简单地“追加新 token”，而是在一个混合缓存状态上做 block 级资源调度。

## 8.3 prefix cache 命中为何是 block 级

block hash 的生成与请求状态绑定在 `Request.update_block_hashes()`。
命中查找由 `KVCacheCoordinator.find_longest_cache_hit(...)` 完成。

这带来三个直接后果：

1. 命中的最小单位是完整 block
2. block 对齐会影响是否能完全复用
3. 某些 backend 或 cache mode 会要求更严格的对齐策略

## 8.4 逻辑 block 与物理 KV tensor 布局严格分离

在 `vllm/v1/worker/gpu/attn_utils.py` 中：

- `_allocate_kv_cache()` 先按字节分配原始张量
- `_reshape_kv_cache()` 再根据 backend 的 `get_kv_cache_shape()` 和 `get_kv_cache_stride_order()` 把原始张量重解释为 backend 需要的布局
- `build_slot_mappings_by_layer()` 再把逻辑 slot mapping 绑定到具体 layer

这意味着 scheduler 只知道 `block_size`，并不知道底层 `kv_cache` 是 `NHD` 还是 `HND`、是 fused K/V 还是 split K/V。

## 8.5 `block_table` 与 `slot_mapping` 是系统协议的桥

Worker 侧 `vllm/v1/worker/gpu/block_table.py` 做两件事：

- `gather_block_tables()`：把请求已拥有的 block id 拉平成 batch 视角表
- `compute_slot_mappings()`：把 token 在逻辑序列中的位置映射到物理 KV 槽位

没有这一步，调度器的“逻辑 block 列表”无法变成 kernel 可消费的物理地址。

## 8.6 native paged attention kernel 在哪里

典型 wrapper 在 `vllm/_custom_ops.py`：

- `paged_attention_v1(...)`
- `paged_attention_v2(...)`

底层 CUDA 实现位于：

- `csrc/attention/paged_attention_v1.cu`
- `csrc/attention/paged_attention_v2.cu`

但在 v1 主线里，很多实际路径已经通过 `FlashAttention` 或 `Triton` backend 实现统一 attention，所以不能把整个系统简化为“直接调用 paged_attention_v2”。

## 9. 深入拆解 C：从调度输出到设备执行

## 9.1 `GPUModelRunner.execute_model()` 是翻译层核心

`vllm/v1/worker/gpu/model_runner.py::GPUModelRunner.execute_model()` 的关键步骤：

1. 清理已结束请求状态
2. 合并/更新 batch 内常驻 request state
3. `prepare_inputs(...)`
4. `prepare_attn(...)`
5. 构建 `attn_metadata`
6. 决定走 CUDA graph replay 还是 eager/model forward
7. 返回 hidden states / `IntermediateTensors` 或缓存本轮执行状态，采样在独立的 `sample_tokens(...)` 阶段完成

它本质上是“把请求级状态机翻译为单轮设备执行描述”。

## 9.2 `InputBatch`：扁平化 batch 协议

`vllm/v1/worker/gpu/input_batch.py::InputBatch` 包含：

- `query_start_loc`
- `seq_lens`
- `seq_lens_cpu_upper_bound`
- `input_ids`
- `positions`
- `logits_indices`
- `cu_num_logits`

这些字段共同描述了一轮混合 prefill/decode batch 的扁平表示。

特别重要的是：

- `prepare_prefill_inputs(...)` 用 Triton kernel 从 request state 中抽取 prompt token
- `prepare_pos_seq_lens(...)` 计算 `positions` 和 `seq_lens`

这两步都是为了避免 Python 逐 token 打包，并把批量元数据准备下沉到 Triton/device 侧。

## 9.3 attention metadata builder 是后端桥梁

`vllm/v1/worker/gpu/attn_utils.py::build_attn_metadata(...)` 会把：

- `query_start_loc`
- `seq_lens`
- `block_tables`
- `slot_mappings`
- `positions`

转换成 backend-specific metadata。

这意味着 backend 的切换点并不在高层调度器，而在 worker 侧 metadata builder。

## 9.4 Layer 级 attention 不是直接读 scheduler

`vllm/model_executor/layers/attention/attention.py::Attention.forward()` 并不理解 request queue。
它通过 forward context 拿到当前 layer 对应的：

- attention metadata
- slot mapping
- KV cache view

然后执行：

- `unified_kv_cache_update`
- `unified_attention_with_output`

这保证了 layer 层只关心“本层本轮该如何更新 KV、如何做 attention”，不关心上层调度。

## 9.5 graph 与 eager 只是执行外壳不同

在 `GPUModelRunner.execute_model()` 中：

- 若 `batch_desc.cg_mode == FULL`，走 CUDA graph replay
- 否则走 eager / piecewise compiled path

但无论哪种模式，输入协议本身都一样：都依赖 `InputBatch + block_tables + slot_mappings + attn_metadata`。

## 10. 性能模型

| 维度 | 关键机制 | 代码锚点 | 主要收益 | 主要代价 |
|---|---|---|---|---|
| 显存/KV 驻留 | BlockPool + prefix cache + page 化 KV | `block_pool.py` `kv_cache_manager.py` | 提升并发、减少重复 prefill | 对齐、hash、eviction 复杂 |
| Host 端摊销 | Triton metadata prep kernels | `gpu/input_batch.py` `gpu/block_table.py` | 降低 Python 逐 token 开销 | 维护 CPU/GPU 双视图复杂 |
| 算子效率 | FlashAttention/Triton/custom ops | `v1/attention/backends/*` | 提升 kernel 吞吐 | backend 分叉、layout 差异 |
| 图执行 | CUDAGraph replay | `gpu/model_runner.py` | 降低 launch overhead | shape 受限、warmup 成本 |
| 并行通信 | DP/PP/DCP + KV connector | `gpu_worker.py` `distributed/kv_transfer/*` | 扩大吞吐/上下文长度 | 协调复杂、回退路径多 |
| Admission 策略 | full-sequence reserve / preemption | `scheduler.py` | 减少抖动与重算 | 保守策略会牺牲利用率 |

## 11. 测试与证据地图

源码外的验证证据主要集中在：

- `tests/v1/core/`：Scheduler、KV、请求状态机
- `tests/v1/`：worker/spec decode/KV connector
- `tests/distributed/`：context parallel、并行路径
- `docs/` 和 upstream design notes：概念性说明

阅读顺序建议是先代码后测试：

1. `vllm/v1/engine/core.py`
2. `vllm/v1/core/sched/scheduler.py`
3. `vllm/v1/core/kv_cache_manager.py`
4. `vllm/v1/worker/gpu/model_runner.py`
5. `vllm/v1/worker/gpu/attn_utils.py`
6. `tests/v1/core/*`

## 12. 容易误解的点

- `continuous batching` 不是“多个请求拼成 batch”这么简单，而是请求进度状态机在多轮迭代中的持续重排。
- `PagedAttention` 不是“一个 kernel 名”，而是整个 page 化 KV 协议。
- prefix cache 不是“命中就完全不算”。
- `prefill`/`decode` 在 attention backend 和 runtime shape 上有意义，但在调度器内部不是两套独立算法。
- 不能假定所有 backend 的 KV layout 一样；这是 backend 自由度的一部分。
- 不能假定所有环境都走同一个 runner；`VLLM_USE_V2_MODEL_RUNNER` 会改变执行主线。

## 13. 对 mini-vllm 的可迁移启示

1. 如果要保留教学友好性，最值得先学的是 `Request` 的状态变量设计，而不是先复刻复杂 kernel。
2. 只要想做真正的 continuous batching，就必须显式建模 `num_computed_tokens`，而不是只维护“当前生成到第几步”。
3. KV 系统最好分成三层：
   - 请求逻辑层
   - block 资源层
   - backend 物理布局层
4. block table 和 slot mapping 必须是显式协议对象；否则调度器和 kernel 层会强耦合。
5. prefix cache 的真实复杂度在“对齐、block 命中、最后一 token 重算”，这些边界比 cache hit rate 本身更重要。

## 14. 模型与 vLLM 的界限

简短答案：新模型来了，你只要写 `model_executor/models/<name>.py` 一个文件（外加可能少量配置/registry 改动）。vLLM 的调度器、KV cache、attention kernel、采样、CUDA graph、量化、TP/PP/EP 都不用碰。

但"一个文件"不等于"工作量小"——下面拆开看。

### 14.1 界限在哪里：vLLM 的 contract

vLLM 给模型作者画了一条很清楚的线，模型类必须遵守这套接口，剩下的事 vLLM 全包：

| vLLM 负责（model 之外的所有东西） | 模型作者负责（模型类内部） |
|---|---|
| Scheduler / continuous batching | 模型层级结构 |
| PagedAttention KV cache 分配 | 各层调用顺序 |
| prefill / decode 区分 + attn metadata | Embedding → N×Block → Norm → LMHead |
| Attention kernel（FlashAttn / FlashInfer / Triton） | 用 vLLM 提供的 Linear / Norm / RoPE / Attention |
| 量化（FP8 / AWQ / GPTQ / MXFP4 ...） | QKV / MLP 用 ColumnParallel + RowParallel 组合 |
| TP / PP / EP 通信 | MoE 用 vLLM 的 FusedMoE |
| CUDA graph / torch.compile | 权重加载映射（HF 名 → 本地切分参数） |
| Sampler / logits processor | 自己模型独有的算子（如 MLA、Indexer） |
| HF 权重加载入口 | 实现 `forward(input_ids, positions, ...)` 签名 |

也就是说：只要你的 `forward` 长得"标准"，所有性能/分布式特性都白送。

### 14.2 一个新模型要做的事（按工作量从小到大）

**A. 注册（很小）**

- 在 `vllm/model_executor/models/registry.py` 里加一行 `"DeepseekV4ForCausalLM": ("deepseek_v4", "DeepseekV4ForCausalLM"),`，把 HF `config.architectures[0]` 映射到你的模型类。
- 如果是新 HF config 类型，可能在 `vllm/transformers_utils/configs/` 加一个适配。

**B. 写模型文件（核心工作）**

看 `qwen3.py`（340 行）vs `deepseek_v4.py`（1568 行）的差距，就能看出工作量分布。

结构上必须有的几样（从已有文件可总结的最小骨架）：

1. `<Model>MLP` — 用 `MergedColumnParallelLinear(gate_up)` + `act_fn` + `RowParallelLinear(down)` 拼出 SwiGLU。
2. `<Model>Attention` — 用 `QKVParallelLinear` / `MergedQKVParallelLinear` + `get_rope(...)` + `Attention(...)` + `RowParallelLinear(o_proj)`。这里只是"调用"`Attention` 类，真正的 PagedAttention 在 vLLM 内部。
3. `<Model>DecoderLayer` — `input_layernorm` → `self_attn` → `post_attention_layernorm` → `mlp`，带 residual。
4. `<Model>Model` — `VocabParallelEmbedding` + `make_layers(num_layers, lambda: DecoderLayer(...))` + 最后一个 `RMSNorm`。
5. `<Model>ForCausalLM` — 包一层，加 `ParallelLMHead` + `LogitsProcessor`，并实现：
   - `forward(input_ids, positions, intermediate_tensors, inputs_embeds)` —— 签名固定，对应 `_model_forward` 那次调用。
   - `compute_logits(hidden_states, sampling_metadata)`。
   - `load_weights(weights: Iterable[tuple[str, Tensor]])` —— HF 权重名 → 本地参数的映射。

**C. 权重加载（中等，但容易踩坑）**

HF checkpoint 的命名跟你内部参数名几乎不会一一对齐：

- HF 里 `q_proj / k_proj / v_proj` 三份 → 你内部一个合并的 `qkv_proj`。
- HF 里 `gate_proj / up_proj` 两份 → 你内部一个 `gate_up_proj`。
- MoE 的 expert 权重命名形态各异。

vLLM 提供 `AutoWeightsLoader` + `WeightsMapper` + `default_weight_loader` 把这事做成"声明式映射"。但每个新模型都要写一遍这张映射表，且要对张量切分维度敏感（前面 ColumnParallel/RowParallel 的讨论就是为这个服务的）。

### 14.3 工程量到底花在哪儿

参考 `deepseek_v4.py` 的 1568 行，可以看出真正费力的是模型本身的"非标准"部分，不是 vLLM 集成：

| 来源 | 占比（粗略） | 例子 |
|---|---|---|
| 标准 transformer 骨架（Embed / Layer / LMHead / forward） | 20–30% | 任何模型都长得差不多 |
| 模型独有算子 | 30–50% | DeepseekV4 的 MLA（Multi-head Latent Attention）、Indexer、Yarn-style RoPE、Mamba/SSM、滑动窗口、interleaved layer 等 |
| MoE 路由 + expert 并行 | 10–30%（仅 MoE 模型） | FusedMoE、router、bias、shared experts、EP 通信、量化 expert |
| 权重加载映射 | 5–15% | HF 命名 → 内部命名、合并 QKV/gate_up、stacked params、量化权重 |
| 量化路径适配 | 0–15% | FP8 / MXFP4 expert、按层跳过量化 (`is_layer_skipped`) |
| 多模态/视觉/编码器（如果有） | 单独再写一倍 | `deepseek_vl2.py`、`llama4.py` 之类 |

为什么 `qwen3.py` 只有 340 行而 `deepseek_v4.py` 接近 1600 行？前者是"标准 GQA + dense MLP"，几乎全程用 vLLM 现成原语就能拼完；后者引入了 MLA、专用 Indexer、复杂的 MoE 路由（带 bias 的 fused topk）、多种量化路径，这些是算法本身的复杂度，不是 vLLM 把事情搞复杂了。

### 14.4 哪些情况会突破"只改一个文件"的边界

绝大多数模型一个文件就够。下面这些会向 vLLM 内部蔓延：

1. **全新的 attention 形态**（MLA、线性 attention、RWKV、Mamba SSM）
   → 需要在 `model_executor/layers/` 下新增一个层（`deepseek_v4_attention.py`、`mamba/...`），并且可能要新增一种 attention backend / metadata。这就是为什么 `deepseek_v4.py` 顶部要 `from vllm.model_executor.layers.deepseek_v4_attention import ...`——这部分是 vLLM 团队跟模型作者一起加进去的，不是用户在 model file 里能搞定的。
2. **新的并行策略**（Expert Parallel 的新变体、Sequence Parallel 新模式）→ 改 `distributed/`、`v1/worker/`。
3. **新的量化格式** → `model_executor/layers/quantization/` 新增方法。
4. **新的 KV cache 结构**（如 MLA 的 latent KV、SSM 的 state cache）→ 改 `v1/core/kv_cache_manager.py`、attention metadata、worker。
5. **新的调度需求**（chunked prefill 的新形式、prefix caching 的新粒度）→ 改 `v1/core/sched/`。

第 1、4 项就是 DeepSeek-V2/V3/V4 真正"贵"的地方——MLA 一上来，vLLM 是要在 KV cache 那一层为它特化的。但这种侵入式改动是模型上游团队 + vLLM core 团队协作的，不是适配普通新模型时要面对的。

### 14.5 对应到 mini-vllm

mini-vllm 把这条边界画得更直白，可以当对照：

- vLLM 提供的能力 ↔ `src/myvllm/layers/`（`linear.py` / `attention.py` / `rotary_embedding.py` / `layernorm.py` / `embedding_head.py` / `sampler.py`）+ `engine/`（scheduler / block manager / model_runner）。
- 每个模型自己写的部分 ↔ `src/myvllm/models/qwen3.py` / `llama.py`，里面就是"用 layers 拼骨架 + 写 load_weights"。
- 模型注册 ↔ `engine/model_runner.py.__init__` 里那个根据目录名 `match` 选模型类的位置——加一个 case 即可。

### 14.6 结论

- 不需要"重新实现一遍"：Transformer 主干、attention kernel、KV cache、调度、采样、量化、TP/PP/EP，vLLM 已经给好了。
- 要做的是"翻译"：把 HF 那份用 `nn.Linear` / `F.scaled_dot_product_attention` 写的 reference 实现，翻译成"用 vLLM 的并行原语 + `Attention` 类拼出来的版本"，再写一份权重名映射。这部分的活儿大约是几百到一千多行 Python，1–3 天到 1–2 周，取决于模型有多怪。
- 真正"贵"的是模型本身的非标准算子：MLA、Indexer、特殊 RoPE、复杂 MoE 路由、新型 KV state——这些既是论文创新点，也是适配的工程量来源。如果一个新模型只是把 Llama 加宽加深、换 RoPE base，那 200 行就够了；如果它发明了一种新 attention，那它需要在 `layers/` 里加新基础设施，这一步通常不是"用户做模型适配"的范畴，而是 vLLM core 接收新机制的过程。

一句话：vLLM 给 Transformer 划了一道"Lego 接口"，模型作者只搭 Lego，引擎部分不动——除非这个模型自带新形状的积木。
