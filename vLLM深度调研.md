# vLLM 深度调研

> 调研范围：本文基于 2026-05-01 可访问的 vLLM 官方文档、PagedAttention 论文与 upstream 官方仓库源码撰写。源码基线固定到 commit `92a7c121b62a1484b68c0a27d1ecefd1a84f78fc`。本文不做功能罗列，重点回答四件事：`vLLM 为什么能把吞吐做高`、`这些结论在源码里如何落地`、`哪些对象/字段是真正的系统骨架`、`哪些不变量最值得 MinivLLM 复用`。

> 证据约定：`源码事实` 指类、函数、字段、调用关系与注释；`工程判断` 指根据源码结构与 serving 常识得到的推断，不把它写成 benchmark 结论；`非目标` 是训练栈、optimizer/backward、以及未进入主干的第三方 patch。

## 1. 结论先行

vLLM 的性能不是某一个 attention kernel 单点优化出来的，而是四层运行时共同作用的结果：

1. `Scheduler` 把请求推进抽象成 token debt，而不是把请求硬拆成 prefill phase / decode phase。
2. `KV cache` 被建模成 block/page 资源，prefix hit、admission、preemption、reuse 都围绕同一套 block 协议收敛。
3. `Worker + Attention backend + BlockTable` 把调度结果翻译成真实设备 batch、slot mapping 和 paged attention 访存。
4. `Compilation + CUDAGraph + Executor + Connector` 继续把 decode 固定开销、分布式拓扑和缓存迁移压到系统层。

一句话概括：

> **vLLM 用 block 化 KV 管理支撑连续调度，再用 backend、graph、executor、connector 把这套调度模型扩展到多硬件、多拓扑和多工作负载。**

## 2. 工程边界与目录地图

理解 vLLM，先不要盯住某个 kernel；先看它把 serving runtime 切成了哪些稳定边界。

| 层次 | 关键文件 | 主要契约 | 为什么关键 |
| --- | --- | --- | --- |
| 用户入口 | [`v1/engine/llm_engine.py`](https://github.com/vllm-project/vllm/blob/92a7c121b62a1484b68c0a27d1ecefd1a84f78fc/vllm/v1/engine/llm_engine.py) | 请求规范化、输出回组装 | 把 API 面和 runtime 面隔开 |
| EngineCore | [`v1/engine/core.py`](https://github.com/vllm-project/vllm/blob/92a7c121b62a1484b68c0a27d1ecefd1a84f78fc/vllm/v1/engine/core.py) | `add_request()` / `step()` 主循环 | 是 V1 runtime 总装点 |
| Scheduler | [`v1/core/sched/scheduler.py`](https://github.com/vllm-project/vllm/blob/92a7c121b62a1484b68c0a27d1ecefd1a84f78fc/vllm/v1/core/sched/scheduler.py) | 本轮谁前进、前进多少、是否抢占 | continuous batching 的真正核心 |
| KV 系统 | [`v1/core/kv_cache_manager.py`](https://github.com/vllm-project/vllm/blob/92a7c121b62a1484b68c0a27d1ecefd1a84f78fc/vllm/v1/core/kv_cache_manager.py)、[`v1/core/block_pool.py`](https://github.com/vllm-project/vllm/blob/92a7c121b62a1484b68c0a27d1ecefd1a84f78fc/vllm/v1/core/block_pool.py) | prefix hit、slot 分配、block 生命周期 | PagedAttention 的系统收益都在这里释放 |
| 协议对象 | [`v1/request.py`](https://github.com/vllm-project/vllm/blob/92a7c121b62a1484b68c0a27d1ecefd1a84f78fc/vllm/v1/request.py)、[`v1/core/sched/output.py`](https://github.com/vllm-project/vllm/blob/92a7c121b62a1484b68c0a27d1ecefd1a84f78fc/vllm/v1/core/sched/output.py) | Request、SchedulerOutput、status 字段 | feature 越多，越要靠协议对象稳住边界 |
| Worker / ModelRunner | [`v1/worker/gpu/model_runner.py`](https://github.com/vllm-project/vllm/blob/92a7c121b62a1484b68c0a27d1ecefd1a84f78fc/vllm/v1/worker/gpu/model_runner.py)、[`v1/worker/gpu/input_batch.py`](https://github.com/vllm-project/vllm/blob/92a7c121b62a1484b68c0a27d1ecefd1a84f78fc/vllm/v1/worker/gpu/input_batch.py) | 把 scheduler output 变成设备输入批次 | 调度和算子之间的翻译层 |
| Attention backend | [`v1/attention/backend.py`](https://github.com/vllm-project/vllm/blob/92a7c121b62a1484b68c0a27d1ecefd1a84f78fc/vllm/v1/attention/backend.py)、[`v1/attention/selector.py`](https://github.com/vllm-project/vllm/blob/92a7c121b62a1484b68c0a27d1ecefd1a84f78fc/vllm/v1/attention/selector.py) | backend 选择、metadata 协议 | attention 不是单函数而是一套派发体系 |
| Paged Attention 执行 | [`v1/attention/ops/paged_attn.py`](https://github.com/vllm-project/vllm/blob/92a7c121b62a1484b68c0a27d1ecefd1a84f78fc/vllm/v1/attention/ops/paged_attn.py)、[`v1/worker/gpu/block_table.py`](https://github.com/vllm-project/vllm/blob/92a7c121b62a1484b68c0a27d1ecefd1a84f78fc/vllm/v1/worker/gpu/block_table.py) | block table、slot mapping、decode 访存路径 | 把 block 化 KV 变成真实执行 |
| 编译与图执行 | [`compilation/cuda_graph.py`](https://github.com/vllm-project/vllm/blob/92a7c121b62a1484b68c0a27d1ecefd1a84f78fc/vllm/compilation/cuda_graph.py)、[`compilation/passes/pass_manager.py`](https://github.com/vllm-project/vllm/blob/92a7c121b62a1484b68c0a27d1ecefd1a84f78fc/vllm/compilation/passes/pass_manager.py) | capture/replay、pass 重写、runtime wrapper | 压低 decode 高频小步固定开销 |
| 执行器与分布式 | [`v1/executor/abstract.py`](https://github.com/vllm-project/vllm/blob/92a7c121b62a1484b68c0a27d1ecefd1a84f78fc/vllm/v1/executor/abstract.py)、[`distributed/parallel_state.py`](https://github.com/vllm-project/vllm/blob/92a7c121b62a1484b68c0a27d1ecefd1a84f78fc/vllm/distributed/parallel_state.py) | 单进程/多进程/Ray、TP/EP/CP 进程组 | 把单卡 runtime 拉成服务系统 |
| Connector / 外部缓存 | [`distributed/kv_transfer/kv_connector/base.py`](https://github.com/vllm-project/vllm/blob/92a7c121b62a1484b68c0a27d1ecefd1a84f78fc/vllm/distributed/kv_transfer/kv_connector/base.py)、[`distributed/ec_transfer/ec_transfer_state.py`](https://github.com/vllm-project/vllm/blob/92a7c121b62a1484b68c0a27d1ecefd1a84f78fc/vllm/distributed/ec_transfer/ec_transfer_state.py) | KV/encoder cache 搬运协议 | disaggregated serving 的关键拼图 |

## 3. 一次请求在 vLLM 里如何被推进

把一条请求主链拉直之后，很多“为什么快”都会落回同一条控制流。

1. 用户请求经 [`LLMEngine`](https://github.com/vllm-project/vllm/blob/92a7c121b62a1484b68c0a27d1ecefd1a84f78fc/vllm/v1/engine/llm_engine.py) 标准化，形成 engine request。
2. [`EngineCore.add_request()`](https://github.com/vllm-project/vllm/blob/92a7c121b62a1484b68c0a27d1ecefd1a84f78fc/vllm/v1/engine/core.py) 把请求交给 runtime，进入 waiting queue。
3. [`EngineCore.step()`](https://github.com/vllm-project/vllm/blob/92a7c121b62a1484b68c0a27d1ecefd1a84f78fc/vllm/v1/engine/core.py) 驱动一轮 scheduler + executor 主循环。
4. [`Scheduler.schedule()`](https://github.com/vllm-project/vllm/blob/92a7c121b62a1484b68c0a27d1ecefd1a84f78fc/vllm/v1/core/sched/scheduler.py) 计算该请求本轮还能前进多少 token。
5. [`KVCacheManager.get_computed_blocks()`](https://github.com/vllm-project/vllm/blob/92a7c121b62a1484b68c0a27d1ecefd1a84f78fc/vllm/v1/core/kv_cache_manager.py) 查 prefix hit，再由 [`allocate_slots()`](https://github.com/vllm-project/vllm/blob/92a7c121b62a1484b68c0a27d1ecefd1a84f78fc/vllm/v1/core/kv_cache_manager.py) 申请 block。
6. Scheduler 产出 [`SchedulerOutput`](https://github.com/vllm-project/vllm/blob/92a7c121b62a1484b68c0a27d1ecefd1a84f78fc/vllm/v1/core/sched/output.py)，executor 把它下发到 worker。
7. [`GPUModelRunner.prepare_inputs()`](https://github.com/vllm-project/vllm/blob/92a7c121b62a1484b68c0a27d1ecefd1a84f78fc/vllm/v1/worker/gpu/model_runner.py) 构造 [`InputBatch`](https://github.com/vllm-project/vllm/blob/92a7c121b62a1484b68c0a27d1ecefd1a84f78fc/vllm/v1/worker/gpu/input_batch.py)，再由 `prepare_attn()` 拼出 block tables 和 slot mappings。
8. `model_state.prepare_attn()` 与 attention backend 生成 metadata，按 full graph / piecewise / eager 路径执行模型。
9. `sample()` 或 rejection sampler 产出 token，`postprocess()` 更新 host/device 两侧状态镜像。
10. output processor 把底层 token 流整理成用户侧可见结果。

关键观察：

- scheduler 只决定“本轮补多少债”。
- KV 系统只决定“哪些债已经被 cache 还掉、哪些债还有空间继续还”。
- worker/backend 只决定“这些债在设备上如何执行掉”。

这个边界足够窄，所以 prefix cache、spec decode、KV transfer、disaggregated serving 才能在同一套框架里叠加。

## 4. 核心状态对象与协议对象

很多人把 vLLM 理解成 feature 组合，其实它更像一组协议对象稳定交互的 runtime。

| 对象 | 关键文件 | 值得盯的字段/方法 | 作用 |
| --- | --- | --- | --- |
| `Request` | [`v1/request.py`](https://github.com/vllm-project/vllm/blob/92a7c121b62a1484b68c0a27d1ecefd1a84f78fc/vllm/v1/request.py) | `num_computed_tokens`、`num_tokens_with_spec`、`num_output_placeholders`、`status` | 把请求推进状态压成少量字段 |
| `SchedulerOutput` | [`v1/core/sched/output.py`](https://github.com/vllm-project/vllm/blob/92a7c121b62a1484b68c0a27d1ecefd1a84f78fc/vllm/v1/core/sched/output.py) | `num_scheduled_tokens`、scheduled reqs、spec decode 信息 | scheduler 和 worker 的协议面 |
| `KVCacheBlocks` | [`v1/core/kv_cache_manager.py`](https://github.com/vllm-project/vllm/blob/92a7c121b62a1484b68c0a27d1ecefd1a84f78fc/vllm/v1/core/kv_cache_manager.py) | cached blocks / new blocks | 告诉 worker 哪些 KV 已存在、哪些需新写 |
| `KVCacheBlock` | [`v1/core/kv_cache_utils.py`](https://github.com/vllm-project/vllm/blob/92a7c121b62a1484b68c0a27d1ecefd1a84f78fc/vllm/v1/core/kv_cache_utils.py) | `block_id`、引用关系、生命周期 | block pool 的最小资源单位 |
| `BlockPool` | [`v1/core/block_pool.py`](https://github.com/vllm-project/vllm/blob/92a7c121b62a1484b68c0a27d1ecefd1a84f78fc/vllm/v1/core/block_pool.py) | `free_block_queue`、`cached_block_hash_to_block`、`touch()` | 统一分配、缓存、回收、淘汰 |
| `InputBatch` | [`v1/worker/gpu/input_batch.py`](https://github.com/vllm-project/vllm/blob/92a7c121b62a1484b68c0a27d1ecefd1a84f78fc/vllm/v1/worker/gpu/input_batch.py) | `num_scheduled_tokens`、`seq_lens_cpu_upper_bound`、`query_start_loc` | scheduler 输出在设备侧的批次表示 |
| `BlockTables` | [`v1/worker/gpu/block_table.py`](https://github.com/vllm-project/vllm/blob/92a7c121b62a1484b68c0a27d1ecefd1a84f78fc/vllm/v1/worker/gpu/block_table.py) | `gather_block_tables()`、`compute_slot_mappings()` | 把 request 级 KV 布局转成 kernel 可读张量 |
| `RequestState` / `req_states` | [`v1/worker/gpu/model_runner.py`](https://github.com/vllm-project/vllm/blob/92a7c121b62a1484b68c0a27d1ecefd1a84f78fc/vllm/v1/worker/gpu/model_runner.py) | `num_computed_tokens.gpu`、`num_computed_tokens_np`、`prefill_len` | worker 侧的 host/device 状态镜像 |

最值得注意的不是对象数量，而是状态如何闭环：

1. `Request` 保存逻辑推进状态。
2. `SchedulerOutput` 把“这一轮要做什么”序列化。
3. `InputBatch` 把逻辑状态压成设备输入。
4. `req_states` 在 worker 内同时维护 GPU 张量和 CPU 镜像。
5. `postprocess()` 再把本轮执行结果写回，等待下一轮调度。

## 5. 三个系统不变量

### 5.1 请求推进不变量：token debt

[`Scheduler.schedule()`](https://github.com/vllm-project/vllm/blob/92a7c121b62a1484b68c0a27d1ecefd1a84f78fc/vllm/v1/core/sched/scheduler.py) 的核心思想非常简单：每个请求只有“已经算了多少 token”和“总共需要算到哪里”两个关键位置。源码里真正被不断比较的就是：

- `request.num_computed_tokens`
- `request.num_tokens_with_spec`
- `request.num_output_placeholders`

因此：

- 长 prompt prefill 只是欠债很多。
- 单步 decode 只是每轮只欠很少债。
- chunked prefill 只是对大债分期偿还。
- speculative decode 只是把草稿 token 也算进待偿还区间。

这个抽象比“prefill 阶段 / decode 阶段”更强，因为新增能力不需要重写调度范式。

### 5.2 缓存不变量：KV 是 block/page 资源

[`KVCacheManager`](https://github.com/vllm-project/vllm/blob/92a7c121b62a1484b68c0a27d1ecefd1a84f78fc/vllm/v1/core/kv_cache_manager.py) 不直接管理“每请求一大段连续显存”，而是管理：

- prefix 命中
- block 分配
- admission gate
- sliding window / common prefix
- block 生命周期

这意味着缓存系统天然面向“共享、抢占、迁移、重用”，而不是只面向“存”。

### 5.3 执行不变量：设备差异被封在 worker/backend/graph

设备差异、backend 差异、graph 差异不应污染顶层 scheduler。vLLM 把它们封在：

- [`v1/worker/gpu/model_runner.py`](https://github.com/vllm-project/vllm/blob/92a7c121b62a1484b68c0a27d1ecefd1a84f78fc/vllm/v1/worker/gpu/model_runner.py)
- [`v1/attention/backend.py`](https://github.com/vllm-project/vllm/blob/92a7c121b62a1484b68c0a27d1ecefd1a84f78fc/vllm/v1/attention/backend.py)
- [`v1/attention/selector.py`](https://github.com/vllm-project/vllm/blob/92a7c121b62a1484b68c0a27d1ecefd1a84f78fc/vllm/v1/attention/selector.py)
- [`compilation/cuda_graph.py`](https://github.com/vllm-project/vllm/blob/92a7c121b62a1484b68c0a27d1ecefd1a84f78fc/vllm/compilation/cuda_graph.py)

正因为这个分层稳定，vLLM-Ascend 才能在不改写顶层调度语义的前提下接管平台实现。

## 6. Scheduler 深入：continuous batching 在源码里到底是什么

continuous batching 在 vLLM 里不是 marketing phrase，而是 `schedule()` 每轮都在做的一组具体决策。

### 6.1 `schedule()` 的核心不是 phase，而是“追平欠债”

源码里 `schedule()` 一开始就建立了三个局部表：

- `req_to_new_blocks`
- `num_scheduled_tokens`
- `scheduled_spec_decode_tokens`

同时初始化两个预算：

- `token_budget = self.max_num_scheduled_tokens`
- `encoder_compute_budget = self.max_num_encoder_input_tokens`

随后对每个 running request 计算：

- 本轮理论上还欠多少 token
- 是否受 `long_prefill_token_threshold` 限制
- 是否受 `max_model_len` 限制
- 是否还要为 encoder inputs 预留预算

也就是说，scheduler 调度的不是“多少个请求”，而是“在多种预算共同约束下，本轮哪些请求可以前进多少 token”。

### 6.2 running queue 优先，waiting queue 再尝试进入

`schedule()` 先处理 `self.running`，这是为了保持已经在系统里的请求持续推进，避免频繁冷热切换。只有 running requests 处理完或者预算还有余量时，waiting requests 才会进入下一轮 admission。

这一步很关键，因为它让 vLLM 的 FCFS/priority 语义和吞吐优化发生在同一个调度器里，而不是拆成两个系统。

### 6.3 `num_output_placeholders` 是异步与 spec decode 能接进主系统的关键

源码里 `num_new_tokens` 的计算不是简单的：

- `num_tokens_with_spec - num_computed_tokens`

而是会把 `num_output_placeholders` 一起带上。这是异步调度与 speculative path 能并入主调度器的关键：placeholder 不是旁路状态，而是正式进入请求推进方程。

这也是为什么 vLLM 能在不重写 scheduler 主循环的情况下叠加 async scheduling、draft token、未来 jump decoding 优化。

### 6.4 `allocate_slots()` 失败时，scheduler 会主动 preempt

真正体现“操作系统味道”的地方是这里：

1. scheduler 先尝试为 request 申请新 block；
2. 如果 `allocate_slots()` 返回 `None`，说明当前 cache pressure 无法容纳本轮前进；
3. 调度器不会直接卡死，而是按策略挑选一个 running request 抢占；
4. 抢占后恢复 token budget、spec decode 信息和 encoder budget；
5. 被抢占请求进入 preempted 列表，等待后续重新推进。

这套机制的代价是局部回退；收益是系统整体不会被少量长尾请求拖住。

### 6.5 waiting queue 的 admission 不是“有空位就进”

对 waiting requests，scheduler 会先做两类判断：

1. 是否有 prefix hit，先通过 `get_computed_blocks()` 找到已有计算结果。
2. 是否能“放得下完整序列”，通过 `can_fit_full_sequence()` 做 admission gate。

这一步很重要。因为如果只有 chunked prefill 没有 full-sequence admission，系统可能只看见“第一小块能进”，却在后面几轮被显存压力拖垮。vLLM 把这个风险提前暴露给 scheduler。

### 6.6 `get_num_common_prefix_blocks()` 说明 scheduler 还感知批内共享

`schedule()` 末尾还会询问 `KVCacheManager.get_num_common_prefix_blocks()`。这不是附带统计，而是让批内共享前缀从“隐含优化”变成“显式信号”。一旦某批 request 有明显公共前缀，共享会反过来影响 prefix cache 收益、KV usage 和吞吐判断。

**源码抓手**

- [`v1/core/sched/scheduler.py`](https://github.com/vllm-project/vllm/blob/92a7c121b62a1484b68c0a27d1ecefd1a84f78fc/vllm/v1/core/sched/scheduler.py)：`class Scheduler`、`schedule()`、`_try_schedule_encoder_inputs()`
- [`v1/core/sched/request_queue.py`](https://github.com/vllm-project/vllm/blob/92a7c121b62a1484b68c0a27d1ecefd1a84f78fc/vllm/v1/core/sched/request_queue.py)：waiting/running 队列组织
- [`v1/core/sched/output.py`](https://github.com/vllm-project/vllm/blob/92a7c121b62a1484b68c0a27d1ecefd1a84f78fc/vllm/v1/core/sched/output.py)：scheduler 输出协议

## 7. KV Cache 深入：PagedAttention 的价值为什么发生在系统层

很多介绍只说“PagedAttention 节省显存”，但源码里更重要的事实是：它把 KV cache 变成了 scheduler 可协同控制的资源系统。

### 7.1 `get_computed_blocks()` 把 prefix cache 接进了主调度路径

`KVCacheManager.get_computed_blocks()` 做了三件值得注意的事：

1. 如果 prefix caching 关闭，或者请求被标记为 `skip_reading_prefix_cache`，直接返回空命中。
2. 即使全 prompt 都命中，也会故意把最后一个 token 留给重算，以便产出 logits。
3. `find_longest_cache_hit()` 返回的是“完整 block”的命中，因此 prefix hit 不只是 token 级优化，而是和 block 对齐策略绑定的。

这说明 prefix cache 在 vLLM 里不是后处理优化，而是改变 scheduler 输入的一等功能。

### 7.2 `allocate_slots()` 不只是分配空间，而是在执行一套多阶段协议

`allocate_slots()` 的源码注释非常重要，它把 token 区间拆成：

- 已本地计算 `comp`
- 新命中 prefix 的 `new_comp`
- 外部 connector 提供的 `ext_comp`
- 本轮真正要新算的 `new`
- speculative lookahead 的 `lookahead`

然后按三个阶段推进：

1. 先清理不再需要的旧 block，判断 free blocks 是否足够。
2. 处理 prefix / external tokens 对应的 block 组织。
3. 为本轮待计算 token 和 lookahead token 申请新 block，并决定哪些 block 需要 cache。

这意味着 `allocate_slots()` 不是“malloc 一段 KV”；它是在把 prefix cache、remote KV、sliding window、spec decode 和 admission 统一到同一个内存协议里。

### 7.3 `BlockPool` 真正持有了“缓存系统”的运行时形态

[`BlockPool`](https://github.com/vllm-project/vllm/blob/92a7c121b62a1484b68c0a27d1ecefd1a84f78fc/vllm/v1/core/block_pool.py) 不是一个薄容器，它直接维护：

- `free_block_queue`：自由块队列，同时承担 eviction 顺序
- `cached_block_hash_to_block`：从 block hash 到 cached block 的查找表
- `null_block`：用于稀疏路径和特殊占位
- `cache_full_blocks()`：把 full blocks 注册到 prefix cache
- `touch()`：更新块的最近使用顺序

因此 block pool 承担的是“缓存分配器 + prefix cache 索引 + 淘汰序控制器”的复合角色。

### 7.4 common prefix 不只是命中率统计，而是批处理共享信号

`get_num_common_prefix_blocks()` 的意义，不只是“统计一下共享了多少块”，而是让 batch 级 prefix 共享成为可量化、可调度的系统信号。只要这个信号存在，scheduler 的 batch 选择、prefix reuse 收益和 block pressure 判断都会变得更稳定。

### 7.5 为什么说 PagedAttention 的价值在系统层

如果只把 paged attention 当成 kernel，就只能得到“非连续 KV 也能算 attention”这个结论；但在 vLLM 里，它真正释放的价值是：

- 允许 scheduler 以 block 为单位抢占和恢复请求
- 允许 prefix cache 直接复用历史 block
- 允许 remote KV / offload / external cache 进入统一协议
- 允许 worker 在 decode 期间只重组 block table，而不需要搬整条连续序列

这才是 vLLM 相比传统“每请求一大块连续 KV”设计的根本差异。

**源码抓手**

- [`v1/core/kv_cache_manager.py`](https://github.com/vllm-project/vllm/blob/92a7c121b62a1484b68c0a27d1ecefd1a84f78fc/vllm/v1/core/kv_cache_manager.py)：`KVCacheManager`、`get_computed_blocks()`、`allocate_slots()`、`can_fit_full_sequence()`
- [`v1/core/block_pool.py`](https://github.com/vllm-project/vllm/blob/92a7c121b62a1484b68c0a27d1ecefd1a84f78fc/vllm/v1/core/block_pool.py)：`BlockPool`、`cache_full_blocks()`、`touch()`
- [`v1/core/kv_cache_coordinator.py`](https://github.com/vllm-project/vllm/blob/92a7c121b62a1484b68c0a27d1ecefd1a84f78fc/vllm/v1/core/kv_cache_coordinator.py)：多 group KV、sliding window、prefix block 组织
- [`config/cache.py`](https://github.com/vllm-project/vllm/blob/92a7c121b62a1484b68c0a27d1ecefd1a84f78fc/vllm/config/cache.py)：block size、dtype、prefix caching、offload 配置

## 8. Worker / Attention Backend：调度结果如何变成设备执行

`SchedulerOutput` 还不是设备能直接吃的东西。真正的执行翻译发生在 worker 与 attention backend。

### 8.1 `prepare_inputs()` 把 request 级状态压成 `InputBatch`

[`GPUModelRunner.prepare_inputs()`](https://github.com/vllm-project/vllm/blob/92a7c121b62a1484b68c0a27d1ecefd1a84f78fc/vllm/v1/worker/gpu/model_runner.py) 做的不是简单 gather，它要同时准备：

- `idx_mapping` / `expanded_idx_mapping`
- `query_start_loc`
- `seq_lens`
- `num_scheduled_tokens`
- `logits_indices`
- `seq_lens_cpu_upper_bound`

其中最值得盯的是 `seq_lens_cpu_upper_bound`。源码直接用：

- `req_states.num_computed_tokens_np + num_scheduled_tokens`

构造一个 CPU 侧上界张量，再塞回 `InputBatch`。这说明 worker 执行路径始终保留了 host 侧对“本轮序列长度”的镜像感知，而不是把所有状态都丢给 device。

### 8.2 `prepare_attn()` 做的是真正的 paged KV 翻译

`prepare_attn()` 紧接着调用 `BlockTables.gather_block_tables()` 和 `compute_slot_mappings()`：

- block tables：每个 request 当前引用哪些 block
- slot mappings：本轮 token 应该写入哪些 slot

这里是调度系统和 paged attention kernel 的真正交界面。scheduler 看到的是 request 和 token；kernel 看到的是 block table 和 slot mapping；中间的翻译层就是 model runner。

### 8.3 `execute_model()` 把 eager、piecewise graph、full graph 放进同一入口

`execute_model()` 的结构非常像一个小 runtime：

1. 先更新 request states、释放已完成请求、应用 block table staged writes。
2. 决定本轮 `BatchDescriptor` 与 `uniform_tok_count`，并通过 `dispatch_cg_and_sync_dp()` 选择运行模式。
3. 真实请求走 `prepare_inputs()` / `prepare_attn()`；dummy run 走 dummy batch。
4. `model_state.prepare_attn()` 构造 backend metadata。
5. full graph 走 `cudagraph_manager.run_fullgraph()`；其他模式直接调用 `model()`。
6. 最后执行 sample、rejection sampling、postprocess。

这说明 worker 层并不是“调用一下模型”；它本身就是一个负责 batch 编排、graph 选择、attention metadata 构造和状态回写的运行时。

### 8.4 sample / rejection sampler / postprocess 共同维护状态闭环

`sample()` 会根据是否存在草稿 token 在两条路径之间分流：

- 普通请求：走 sampler
- speculative decode：走 rejection sampler

随后 `postprocess()` 做三类关键回写：

1. 更新 `req_states.num_computed_tokens.gpu`
2. 更新 `num_computed_prefill_tokens`
3. 乐观推进 `req_states.num_computed_tokens_np`

这一步说明 worker 不是被动执行器，而是 request state 真实推进的一部分。

### 8.5 attention backend selector 的价值是稳定协议，而不是“多几个 kernel”

[`v1/attention/backend.py`](https://github.com/vllm-project/vllm/blob/92a7c121b62a1484b68c0a27d1ecefd1a84f78fc/vllm/v1/attention/backend.py) 与 [`selector.py`](https://github.com/vllm-project/vllm/blob/92a7c121b62a1484b68c0a27d1ecefd1a84f78fc/vllm/v1/attention/selector.py) 的真正价值在于：

- prefill 和 decode 可以走不同 backend
- 不同模型家族可以选择不同 metadata builder
- graph 支持能力可以按 backend 申报
- worker 不需要知道每个 backend 的内核细节，只要遵守 metadata 协议

这才是后端体系可扩展的关键。

**源码抓手**

- [`v1/worker/gpu/model_runner.py`](https://github.com/vllm-project/vllm/blob/92a7c121b62a1484b68c0a27d1ecefd1a84f78fc/vllm/v1/worker/gpu/model_runner.py)：`prepare_inputs()`、`prepare_attn()`、`execute_model()`、`postprocess()`
- [`v1/worker/gpu/input_batch.py`](https://github.com/vllm-project/vllm/blob/92a7c121b62a1484b68c0a27d1ecefd1a84f78fc/vllm/v1/worker/gpu/input_batch.py)：`InputBatch`
- [`v1/worker/gpu/attn_utils.py`](https://github.com/vllm-project/vllm/blob/92a7c121b62a1484b68c0a27d1ecefd1a84f78fc/vllm/v1/worker/gpu/attn_utils.py)：graph 支持判断、metadata 构造辅助逻辑
- [`v1/attention/backend.py`](https://github.com/vllm-project/vllm/blob/92a7c121b62a1484b68c0a27d1ecefd1a84f78fc/vllm/v1/attention/backend.py)
- [`v1/attention/selector.py`](https://github.com/vllm-project/vllm/blob/92a7c121b62a1484b68c0a27d1ecefd1a84f78fc/vllm/v1/attention/selector.py)

## 9. Compile / CUDAGraph：为什么 vLLM 越来越像 runtime system

decode 每步只吐少量 token，最容易暴露 host/runtime 固定开销。vLLM 后期大量工程都在处理这个问题。

### 9.1 `cudagraph_manager` 在 model runner 初始化阶段就是一等公民

GPUModelRunner 初始化时不会把 graph 当成“最后再打开的优化开关”；它会尽早：

- 根据 compilation config 解析 cudagraph mode
- 初始化 `ModelCudaGraphManager`
- 让 speculator 与 graph manager 对接

这说明在 vLLM 的设计里，graph runtime 是 worker 的正式组成部分，而不是实验性旁路。

### 9.2 full / piecewise / eager 三种模式在同一入口分流

`execute_model()` 中的 `dispatch_cg_and_sync_dp()` 会根据：

- 本轮 token 数
- DP 同步结果
- encoder-decoder 特例
- profile run / eager 强制条件

决定本轮走：

- full graph replay
- piecewise graph
- eager

因此图执行不是一个布尔开关，而是一套 runtime mode 选择器。

### 9.3 为什么 `uniform_tok_count` 和 dummy run 很关键

graph capture/replay 想稳定命中，就必须控制 batch 形状。vLLM 为此显式维护：

- `uniform_tok_count`
- `BatchDescriptor`
- dummy run
- dummy attention metadata

这些设计的本质都是在说：decode 高频路径里，shape discipline 和 kernel FLOPs 同等重要。

**源码抓手**

- [`compilation/cuda_graph.py`](https://github.com/vllm-project/vllm/blob/92a7c121b62a1484b68c0a27d1ecefd1a84f78fc/vllm/compilation/cuda_graph.py)
- [`compilation/compiler_interface.py`](https://github.com/vllm-project/vllm/blob/92a7c121b62a1484b68c0a27d1ecefd1a84f78fc/vllm/compilation/compiler_interface.py)
- [`compilation/passes/pass_manager.py`](https://github.com/vllm-project/vllm/blob/92a7c121b62a1484b68c0a27d1ecefd1a84f78fc/vllm/compilation/passes/pass_manager.py)
- [`compilation/passes/vllm_inductor_pass.py`](https://github.com/vllm-project/vllm/blob/92a7c121b62a1484b68c0a27d1ecefd1a84f78fc/vllm/compilation/passes/vllm_inductor_pass.py)

## 10. Serving 拓扑与分布式扩展

vLLM 不只是单机 demo。它已经具备把同一套 runtime 扩展到多进程、多机和缓存分离拓扑的能力。

### 10.1 executor 把同一套 EngineCore 拉到不同部署形态

`abstract.py`、`multiproc_executor.py`、`ray_executor.py` 的作用不是简单换 launcher，而是让同一个 scheduler / worker / output 协议能跑在：

- 单进程
- 多进程
- Ray 集群

这正是“服务系统”和“本地推理 demo”之间的分水岭。

### 10.2 并行与 MoE 不是外围功能，而是主系统延伸

[`distributed/parallel_state.py`](https://github.com/vllm-project/vllm/blob/92a7c121b62a1484b68c0a27d1ecefd1a84f78fc/vllm/distributed/parallel_state.py) 管理 TP/DP/CP/EP 进程组；[`model_executor/layers/linear.py`](https://github.com/vllm-project/vllm/blob/92a7c121b62a1484b68c0a27d1ecefd1a84f78fc/vllm/model_executor/layers/linear.py) 把 row/column/QKV parallel 落到层定义；[`distributed/eplb/eplb_state.py`](https://github.com/vllm-project/vllm/blob/92a7c121b62a1484b68c0a27d1ecefd1a84f78fc/vllm/distributed/eplb/eplb_state.py) 再把 expert parallel load balancing 工程化。

也就是说，分布式不是单独系统，而是从层定义、worker 执行到 executor 装配的一条贯通链。

### 10.3 connector 说明 vLLM 已经超出“本地显存内闭环”

`kv_connector/base.py` 和 `ec_transfer_state.py` 的存在说明：

- 远端 KV
- encoder cache
- disaggregated prefill / decode

都已经被纳入正式系统边界，而不是脚本层 hack。

## 11. 观测性、失败模式与性能判读

### 11.1 可观测性是内建的，不是外置脚本

- [`v1/core/kv_cache_metrics.py`](https://github.com/vllm-project/vllm/blob/92a7c121b62a1484b68c0a27d1ecefd1a84f78fc/vllm/v1/core/kv_cache_metrics.py)：KV usage / hit 指标
- [`v1/metrics/stats.py`](https://github.com/vllm-project/vllm/blob/92a7c121b62a1484b68c0a27d1ecefd1a84f78fc/vllm/v1/metrics/stats.py)：scheduler / prefix cache 等统计
- [`v1/spec_decode/metrics.py`](https://github.com/vllm-project/vllm/blob/92a7c121b62a1484b68c0a27d1ecefd1a84f78fc/vllm/v1/spec_decode/metrics.py)：spec decode 指标
- [`compilation/counter.py`](https://github.com/vllm-project/vllm/blob/92a7c121b62a1484b68c0a27d1ecefd1a84f78fc/vllm/compilation/counter.py)：编译计数
- [`v1/metrics/perf.py`](https://github.com/vllm-project/vllm/blob/92a7c121b62a1484b68c0a27d1ecefd1a84f78fc/vllm/v1/metrics/perf.py)：FLOPs / bytes 估计

### 11.2 三类最常见的性能误判

1. 只盯 kernel，不看 `token_budget` 是否真的打满。
2. 只看显存占用，不看 prefix hit、block reuse 和 preemption 是否在恶化。
3. 只看单次 decode latency，不看 graph capture 命中和 host-side 固定开销。

### 11.3 三类最常见的工程风险

1. cache pressure 过大导致 preemption 频繁，吞吐看似上升但尾延迟恶化。
2. shape 不稳定导致 graph 命中率低，decode 固定成本重新暴露。
3. distributed / connector 路径增加新的状态同步面，没有指标时很难定位瓶颈。

## 12. 对 MinivLLM 的直接启发

如果把这份调研转成 MinivLLM 的实现建议，优先级如下：

1. 先把请求推进状态压成少量字段，至少显式建模 `已算 token`、`目标 token`、`placeholder token`。
2. 先把 KV cache 的 block 协议建稳，再谈 prefix cache、spec decode、offload 或远端 KV。
3. 让 scheduler、KV manager、worker 三层接口尽量窄，避免 feature 跨层互调。
4. 尽早在 runtime 里埋点，而不是等性能问题出现后再补 instrumentation。
5. 如果未来要做 Ascend/NPU 后端，必须守住“顶层调度不携带设备假设”这个约束。

## 13. 参考来源

- [vLLM 仓库](https://github.com/vllm-project/vllm/tree/92a7c121b62a1484b68c0a27d1ecefd1a84f78fc)
- [vLLM 官方文档](https://docs.vllm.ai/)
- [PagedAttention 论文](https://arxiv.org/abs/2309.06180)
