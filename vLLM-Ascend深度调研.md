# vLLM-Ascend 深度调研

> 调研范围：本文基于 2026-05-01 可访问的 vLLM-Ascend 官方文档、官方博客、官方仓库，以及在线源码路径撰写。说明：这些 Ascend 源码并不在当前仓库内，而是通过官方 GitHub 仓库交叉阅读确认。本文锁定的 vLLM-Ascend 仓库快照为 `d1f66849c9ea1459ca6a4c6c213ac3fec2ec1c9c`，对照的 upstream vLLM 快照为 `92a7c121b62a1484b68c0a27d1ecefd1a84f78fc`。重点不是复述“Ascend 也支持哪些 feature”，而是回答：`Ascend 相比 upstream 到底新增了哪些真实优化层`、`这些优化在源码里怎么落地`、`为什么这些改动对 NPU 平台有意义`。

> 证据约定：本文显式区分三类陈述。`源码事实` 指类/函数职责、源码注释、控制流与调用路径；`对照结论` 指在上述两个仓库快照上的横向比较；`工程推断` 指从代码形态、官方文档和平台常识得到的性能/稳定性判断，不把它写成 benchmark 结论。

## 1. 先给结论

vLLM-Ascend 的价值，不是“把 CUDA 版 vLLM 翻译成 NPU 版”，而是：

1. **在 platform 层把 Ascend 接进 vLLM 的硬件抽象**
2. **在 attention backend、graph runtime、compiler pass、scheduler、KV/PD 兼容、host binding 几层补齐 Ascend 专用实现**
3. **让 upstream vLLM 的快路径，在 Ascend 平台上变成一套真正可执行、可调优、可扩展的 serving runtime**

如果要用一句话概括：

> **upstream vLLM 解决“LLM serving 一般怎么快”，vLLM-Ascend 解决“这些快路径在 Ascend 上怎么落地，并继续围绕 NPU 的图执行、访存、通信和 NUMA/IRQ 行为做平台专属优化”。**

---

## 2. 这份文档怎么读

理解 vLLM-Ascend，最容易犯的错是：

- 只看 feature matrix；
- 或只看一两个 torch_npu 调用；
- 或把所有差异都归结成“换了 kernel”。

更好的阅读顺序是：

1. 先看 `platform.py`，确认插件边界；
2. 再看 `attention_v1.py`，确认 attention/KV 运行时是否真的重建了；
3. 再看 `acl_graph.py`，理解 graph mode 不是文档口号，而是完整 runtime；
4. 再看 `graph_fusion_pass_manager.py`，理解 Ascend 的 compile/pass 栈与 upstream 的分岔；
5. 再看 `scheduler_dynamic_batch.py` 与 `recompute_scheduler.py`，理解 Ascend 对调度层加了什么；
6. 最后看 `cpu_binding.py`，理解为什么 Host/NUMA/IRQ 也是性能的一部分。

这份文档也按这个顺序组织。

---

## 3. 代码地图：Ascend 的增量主要集中在哪几层

先把增量层次拉平，会更容易判断哪些是真正的 Ascend 专属优化。

| 层次 | 关键路径 | 作用 | 相比 upstream 的主要增量 |
| --- | --- | --- | --- |
| 平台接入层 | `vllm_ascend/platform.py` | 注册设备、编译后端、pass manager、attention backend | 把 vendor-specific 行为收拢到 platform 抽象内 |
| Attention/KV 执行层 | `vllm_ascend/attention/attention_v1.py` | 定义 KV layout、metadata、paged attention/FIA 运行时 | 不是简单替换 kernel，而是重建 NPU attention runtime |
| Graph runtime 层 | `vllm_ascend/compilation/acl_graph.py` | capture/replay、workspace、graph 参数管理 | 用 Ascend 的 NPUGraph/ACLGraph 语义重建图执行层 |
| Graph fusion 层 | `vllm_ascend/compilation/graph_fusion_pass_manager.py` | NPU 编译期 pass 组合 | 不走 Triton 路径，自己定义 NPU pass 栈 |
| 调度专项层 | `vllm_ascend/core/scheduler_dynamic_batch.py` | 根据 profiling/SLO 动态 refine budget | upstream 没有这套 SLO 查表式动态预算 |
| 复杂场景兼容层 | `vllm_ascend/core/recompute_scheduler.py` | PD/KV transfer/MTP/hybrid model 兼容 | 为 full graph、subprocess、MLA spec 等补大量工程细节 |
| 主机侧优化层 | `vllm_ascend/cpu_binding.py` | CPU affinity、NUMA memory migration、IRQ binding | 把 host 侧抖动控制做成系统级优化，而非附属脚本 |

这张表非常重要，因为它告诉你：

> **Ascend 真正下重手的地方，不是“也支持 prefix caching”，而是“把 platform 到 runtime 的整个性能路径重新接到 NPU 平台上”。**

---

## 4. 平台层：`platform.py` 是总开关，不是普通配置文件

如果只选一个文件看 Ascend 插件边界，应该先看 `vllm_ascend/platform.py`。

### 4.1 它显式声明“这是一块 NPU，不是一块伪装成 CUDA 的设备”

`NPUPlatform` 里直接声明了：

- `device_name = "npu"`
- `device_control_env_var = "ASCEND_RT_VISIBLE_DEVICES"`
- `dispatch_key = "PrivateUse1"`
- `supported_quantization = [ASCEND_QUANTIZATION_METHOD, COMPRESSED_TENSORS_METHOD]`

这意味着 Ascend 插件并不是用一堆条件分支偷偷兼容 upstream，而是在平台层正式接管：

- device identity
- device visibility
- dispatch 语义
- quantization 方法集合

### 4.2 它把 compile/pass manager/backend 入口都改成 Ascend 自己的

这里最关键的不是变量，而是几个入口函数：

- `get_pass_manager_cls()`
  - 返回 `vllm_ascend.compilation.graph_fusion_pass_manager.GraphFusionPassManager`
- `get_compile_backend()`
  - 返回 `vllm_ascend.compilation.compiler_interface.AscendCompiler`
- `get_attn_backend_cls()`
  - 根据配置和设备类型，在
    - `AscendAttentionBackend`
    - `AscendMLABackend`
    - `AscendSFABackend`
    - 某些 310P 专属 backend
    之间分派

### 4.3 upstream-vs-Ascend 的第一个核心 delta

upstream vLLM 也有 platform 抽象，但 Ascend 在这里的增量很明确：

- upstream 的“平台差异”更多停留在设备族；
- Ascend 插件把 **编译器、图融合入口、attention backend dispatch、量化路径** 全都纳入 platform 统一分派；
- 这使得上层框架仍能保持 vLLM 的平台抽象，而不需要把 vendor patch 散落到 executor/worker/model runner 各处。

### 4.4 为什么这通常会更快或更可维护（工程推断）

严格说，`platform.py` 本身不直接让模型更快。  
它的价值是：

- 让所有真正提速的路径可以在统一抽象下被选中；
- 降低维护和主干漂移成本；
- 避免“性能 patch 散落 everywhere”导致后续优化难以叠加。

这是 **性能可持续性** 的基础。

---

## 5. Attention/KV 运行时：`attention_v1.py` 不是换 kernel，而是重建一整套 NPU 路径

如果说 `platform.py` 解决“从哪进入”，那 `attention_v1.py` 解决的是“进入之后真正跑什么”。

这份文件是 Ascend 插件里最值得深读的实现之一。

### 5.1 第一个核心点：它显式定义了 Ascend 的 KV cache 物理布局

`get_kv_cache_shape()` 直接返回：

```python
(2, num_blocks, block_size, num_kv_heads, head_size)
```

也就是：

- 第 0 维：K/V
- 后面按 `num_blocks -> block_size -> num_kv_heads -> head_size` 组织

这意味着 Ascend 插件没有沿用“默认 CUDA 世界的隐式 layout 假设”，而是把 KV cache 的物理契约显式固定下来。

### upstream-vs-Ascend delta

- upstream 更强调通用 paged KV 抽象；
- Ascend 在 backend 里明确把 KV 的物理形状与后续 NPU attention 路径绑定起来。

### 可能的硬件后果（工程推断）

这给后续：

- paged attention kernel
- block copy/swap
- graph capture 时的 workspace/layout

都提供了稳定前提。

### 5.2 第二个核心点：它自己实现 block copy / swap

`swap_blocks()` 和 `copy_blocks()` 直接在 backend 层提供：

- block 级别交换
- block 级别复制

这不是小补丁，而是很关键的信号：

> **Ascend backend 不只是负责 attention 前向，还接手了 paged KV 系统的物理块操作。**

也就是说，Ascend 插件不是“上层调度 + 下层 kernel”两层模型，而是已经把中间这层 block-level runtime 也接了过来。

### 5.3 metadata builder 从一开始就围绕 graph mode 设计

`AscendAttentionMetadataBuilder` 里几个实现细节非常说明问题：

- `get_cudagraph_support()` 直接返回 `AttentionCGSupport.ALWAYS`
- `decode_threshold` 会把 speculative token 数一起算进去
- 还显式限制 `decode_threshold <= 16`
  - 因为 `npu_fused_infer_attention_score` 的 TND layout 有上限
- `build_for_graph_capture()` 目前只覆盖：
  - `DecodeOnly`
  - `ChunkedPrefill`
  - `SpecDecoding`

### upstream-vs-Ascend delta

upstream 也有 graph/cudagraph 支持，但 Ascend 的 attention metadata builder 更强烈地体现了：

- graph capture 是主路径之一；
- metadata 的组织要直接服务于 graph-compatible attention 调用；
- 甚至 speculative token 数、decode threshold 这种细节也已经与底层 NPU 算子限制绑定。

### 可能的硬件后果（工程推断）

这意味着 Ascend 插件不是先“普通跑通”再后补 graph，而是从 metadata 层就为 graph 路径做约束管理。

### 5.4 第三个核心点：它不是一个 attention backend，而是多条 NPU attention 路径的调度器

`update_graph_params()` 暴露出一个关键事实：  
Ascend runtime 并不只有一条 attention 执行路径。

### 路径 A：Paged Attention

在 `using_paged_attention(...)` 为真时：

- 先调 `torch_npu._npu_paged_attention_get_workspace`
- 再调 `torch_npu._npu_paged_attention`

### 路径 B：FIA / fused infer attention

否则走：

- `torch_npu.npu_fused_infer_attention_score.out(...)`
- 带 `block_table`
- 带 `actual_seq_lengths`
- 带 `workspace`
- 某些量化路径还能带 `antiquant_scale` / `offset`

### 为什么这很重要

这说明 `attention_v1.py` 的实际职责不是“实现一个类”，而是：

- 根据 attention 形态分派；
- 根据 graph 模式分派；
- 根据 paged attention / FIA 分派；
- 根据量化模式携带不同参数；
- 根据 speculative / draft model 路径继续兼容。

### upstream-vs-Ascend delta

upstream 当然也有多 backend，但 Ascend 这里的明显增量是：

- backend 里直接持有更强的 runtime orchestration 责任；
- 并且强依赖 `torch_npu` 特定算子与 workspace 协议。

### 可能的硬件后果（工程推断）

这使得 Ascend 插件可以围绕 NPU 自己的：

- workspace 语义
- graph capture 需求
- fused infer attention 算子限制

构建专属快路径，而不是被 upstream 通用路径束缚。

---

## 6. Graph Mode：`acl_graph.py` 是 Ascend 最“像 runtime”的文件之一

如果你想找到“Ascend 真正在系统层做了什么”，`vllm_ascend/compilation/acl_graph.py` 几乎必读。

### 6.1 `ACLGraphWrapper` 不是简单替代 CUDA Graph，而是完整的 graph runtime wrapper

`ACLGraphWrapper` 干的事情可以总结为：

1. 读取 `forward_context.cudagraph_runtime_mode`
2. 根据 `batch_descriptor` 查找或创建 graph entry
3. 首次遇到某 descriptor 时 capture
4. 后续相同 descriptor 走 replay

注意这已经不是“调用一下 NPUGraph API”，而是完整 runtime 行为：

- shape/descriptor 级缓存
- capture/replay 生命周期
- graph entry 管理

### 6.2 capture 阶段做了什么

源码里能直接看到：

- `torch.npu.NPUGraph()` 创建 graph
- `with torch.npu.graph(aclgraph, pool=self.graph_pool):` 做 capture
- `forward_context.capturing = True`
- 某些场景会 patch：
  - `gc.collect`
  - `torch.npu.empty_cache`

### upstream-vs-Ascend delta

upstream 当然也会考虑 graph capture，但 Ascend 这里把“capture 期间避免 GC/empty_cache 干扰”直接做进 wrapper 逻辑里，说明它已经在围绕具体 NPU graph capture 稳定性做工程处理。

### 可能的硬件后果（工程推断）

这类处理不是让单个 matmul 变快，而是：

- 提高 capture 成功率；
- 降低 capture 抖动；
- 避免 piecewise graph capture 时引入非必要干扰。

### 6.3 replay 阶段最关键的不是 replay，而是“顺序保证”

`acl_graph.py` 有一段特别关键的注释：

- 在 async scheduling 或多线程场景里，
- CPU 侧 `record event` 可能比前一轮 graph replay 更早完成；
- 因此 replay 前有时必须 `torch.npu.current_stream().synchronize()`，
- 否则 `update_attn_params` 可能和前一轮 replay 交错。

这段逻辑非常值钱，因为它揭示了一个现实：

> **Graph mode 的瓶颈并不只是 capture/replay API，而是异步调度下的时序正确性。**

### upstream-vs-Ascend delta

这部分是非常 Ascend-specific 的 runtime 约束。  
它说明 Ascend 插件不仅要做 replay，还要把 replay 与上层 async scheduling 的顺序关系重新捋顺。

### 可能的硬件后果（工程推断）

这直接影响：

- graph 参数是否安全更新；
- 某轮 decode 是否会读到错误状态；
- graph 模式在真实 serving 循环中是否稳定。

### 6.4 graph 参数不是一次性固化，而是被做成全局可更新表

`acl_graph.py` 里有一组很关键的全局结构：

- `GraphParams`
- `_graph_params`
- `_draft_graph_params`
- `_draft_graph_prefill_params`

以及：

- `set_graph_params(...)`
- `update_graph_params_workspaces(...)`
- `get_graph_params()`

### 这说明了什么

Ascend graph 路径不是：

- capture 一个固定图；
- 后面盲 replay。

而是：

- 先按 token 数或 capture size 建参数槽位；
- 再随着 step 更新 workspace、attention params、handles、events；
- 让一组 graph entry 服务于动态 token 数场景。

### upstream-vs-Ascend delta

这里的增量不是“也有 graph params”，而是 Ascend 明显把 **workspace 管理 + graph params 管理 + attention runtime 更新** 紧紧绑到了一起。

### 可能的硬件后果（工程推断）

这让 graph mode 真正能在动态 serving workload 中复用，而不是只能在静态 shape demo 里工作。

---

## 7. 编译器图融合：`graph_fusion_pass_manager.py` 说明 Ascend 走的是另一条 compile 路

如果你只盯 runtime，会漏掉 Ascend 的另一个重要增量：**编译期 pass 栈**。

`GraphFusionPassManager` 的注释已经把分歧点讲得很清楚：

- 它对应 upstream 的 `PostGradPassManager`
- 但因为 `torch_npu` 当前不支持 Triton，
- Ascend 必须定义自己的 pass manager。

这句话非常关键。它意味着：

> **Ascend 不能简单照搬 upstream 在 GPU 世界里的 compile/fusion 路线，而是必须在编译器中间层重建一套适合 NPU 的 pass 栈。**

### 7.1 默认启用哪些 pass

从 `configure()` 可见，常见 pass 包括：

- `AddRMSNormQuantFusionPass`
- `QKNormRopeFusionPass`
- `MatmulAllReduceAddRMSNormPass`
- `MulsAddFusionPass`

若 `enable_sp` 打开，还会额外加入：

- `SequenceParallelismPass`
- `SequenceParallelismMoePass`

### 7.2 upstream-vs-Ascend 的核心 delta

upstream 的 compile 路更容易与 GPU/Triton 世界耦合。  
Ascend 这里的显著差异是：

- pass manager 本身就是插件化增量；
- norm-quant、qk-rope、allreduce-rmsnorm、SP 等 pass 被显式组合进 NPU 栈；
- 这些 pass 服务的是 `torch_npu` 可执行的图优化，而不是通用 Triton 生态。

### 7.3 为什么这通常会快（工程推断）

这类 pass 的价值，通常体现在：

- 更少算子边界；
- 更少中间张量；
- 更少 host/runtime launch；
- 通信与计算更容易被融合。

也就是说，它服务的不只是单层算子，而是 **NPU 图执行路径的整体摩擦降低**。

---

## 8. Dynamic Batch：Ascend 不只是有 chunked prefill，而是给它接了 SLO 驱动预算细化

这部分是 Ascend 在调度层最明显的源码级新增之一。

`vllm_ascend/core/scheduler_dynamic_batch.py` 的价值，在于它不是简单再说一遍 decode-first，而是把 **profiling table + SLO** 接到了预算决策里。

### 8.1 `BudgetRefiner` 的核心逻辑很清楚

`BudgetRefiner` 做的事情是：

1. 读取 `profile_table.csv`
2. 按 `(ctx_len, d_num)` 分组
3. 过滤 `cost <= slo_limit`
4. 选出其中 `chunk_size` 最大的一项
5. 存入 `lookup[(ctx_len, d_num)] = chunk_size`

这说明它不是在线学习器，也不是抽象策略层，而是：

> **用离线 profile 表，把“当前 workload 在给定 SLO 下适合的预算上限”查出来。**

### 8.2 `SchedulerDynamicBatch` 如何接入

在 `__init__` 里，`SchedulerDynamicBatch` 直接创建：

```python
self.budget_refiner = BudgetRefiner(
    default_budget=self.scheduler_config.max_num_batched_tokens,
    slo_limit=self.scheduler_config.SLO_limits_for_dynamic_batch,
)
```

而文件注释又明确说：

- 它允许 token budget 动态 refine；
- 仍遵循 decode-first chunked prefill + FCFS；
- 当前主要支持 910B3。

### 8.3 upstream-vs-Ascend 的核心 delta

upstream vLLM 有：

- continuous batching
- chunked prefill
- token budget

Ascend 这里真正新增的是：

- `SLO_limits_for_dynamic_batch`
- `profile_table.csv`
- `BudgetRefiner.lookup`

也就是：

> **把调度预算从静态参数，推进成“由离线 profiling 和时延约束联合决定”的动态量。**

### 8.4 为什么这有实际价值（工程推断）

因为最优 chunk size 和 budget 并不是固定的：

- 长上下文 + 高 decode 负载时，一种预算可能更优；
- 另一种工作负载下，同一预算可能让 TPOT 恶化；
- 如果系统目标是 SLO，而不是单纯最大吞吐，就必须把预算变成 workload-aware。

这正是 Dynamic Batch 的现实意义。

---

## 9. `recompute_scheduler.py`：Ascend 为 PD/KV transfer/full graph 兼容做了很多“不好看但很值钱”的工程

如果只看论文和高层文档，这类代码很容易被忽略。  
但在生产系统里，它往往决定“到底能不能稳定跑”。

### 9.1 它先修 MLA spec manager 的注册时机问题

源码里一开始就有一段很典型的背景说明：

- module-level `spec_manager_map` 的 class key 绑定时机存在问题；
- 在 EngineCoreProc subprocess unpickle 场景下可能出 `KeyError`；
- 所以 Ascend 主动做 `register_ascend_mla_spec_in_manager()`。

### upstream-vs-Ascend delta

这不是“功能新增”，而是 **把特定平台/子进程/序列化路径上的稳定性坑补平**。

### 可能的硬件后果（工程推断）

如果这一步不做，某些 full graph / async / subprocess 路径甚至跑不稳，更谈不上性能。

### 9.2 文件初始化阶段就开始识别多种特殊角色

`RecomputeScheduler` 初始化时会判断：

- `is_mtp_kv_consumer`
- `is_kv_producer`
- `is_hybrid_model`

其中 `is_hybrid_model` 还对某些模型族（如 `qwen3_next` / `qwen3_5`）做专门判断。

这说明它不是一个纯通用 scheduler，而是：

- 为 KV transfer
- 为 PD
- 为 MTP
- 为 hybrid model graph compatibility

做了专门场景分流。

### 9.3 最有意思的一点：为了保住 full graph，会主动给请求补 placeholder token

源码里非常值得记住的一段逻辑是：

- 如果 `is_mtp_kv_consumer` 为真，
- 会用 `PLACEHOLDER_TOKEN_ID` 填充 `request.spec_token_ids`

为什么？

因为 decode node 从 prefill node 拉取 KV 时，full graph 路径仍然需要满足特定 shape/流程约束。

### upstream-vs-Ascend delta

这不是算法层差异，而是 runtime 兼容层差异：

- upstream 更像通用逻辑；
- Ascend 在这里为了 graph-compatible serving，会主动改写中间请求状态。

### 可能的硬件后果（工程推断）

这类“placeholder token”手法本身不会让一个算子更快，  
但它能让 full graph 这条高性能路径继续成立。  
从系统角度看，这是非常值钱的。

---

## 10. `cpu_binding.py`：Ascend 的 host 侧优化不是附录，而是系统性能的一部分

这份文件特别容易被低估。  
很多人一看到 CPU binding，就以为只是 `taskset`。

源码显示远不止如此。

### 10.1 它先构造 per-NPU CPU pool，而不是简单“每进程绑几个核”

`cpu_binding.py` 会：

- 解析 `allowed_cpus`
- 构造 `numa_to_cpu_map`
- 调 `build_global_slice_cpu_pool()`

而 `build_global_slice_cpu_pool()` 的注释明确强调：

- 多进程或多个 DP group 可能共享同一个 cpuset；
- 需要按 **GLOBAL logical NPU ids** 去切 CPU；
- 以避免 CPU 区间重叠。

### upstream-vs-Ascend delta

这已经不是模型框架常见的“小优化脚本”，而是明确针对多 NPU / 多进程环境的资源切片策略。

### 可能的硬件后果（工程推断）

更好的 CPU 切片意味着：

- host 线程争抢更少；
- NUMA 亲和性更稳定；
- tail latency 更可控。

### 10.2 它不仅绑主线程，还识别 `acl_thread` 和 `release_thread`

`get_threads_map(...)` 会专门识别：

- `acl_thread`
- `release_thread`

后续 `bind_threads()` 会：

- 给主进程绑核；
- 给 `acl_thread` 绑到一组 CPU；
- 给 `release_thread` 绑到另一组 CPU。

### upstream-vs-Ascend delta

这很明显是围绕 Ascend runtime 自己的线程模型做的，而不是 generic Linux affinity。

### 可能的硬件后果（工程推断）

把不同职责线程拆开绑定，有助于：

- 减少关键线程抢占；
- 降低 host 侧抖动；
- 改善 runtime 协调效率。

### 10.3 它还做 NUMA-aware memory placement

`bind_memory()` 里直接调用：

- `migratepages`

并按目标 NPU 对应 NUMA node 迁内存。  
源码里的目标也写得很明确：`minimize cross-NUMA traffic`。

### upstream-vs-Ascend delta

这已经不是简单的 CPU affinity，而是：

- CPU placement
- memory placement

两者同时管理。

### 可能的硬件后果（工程推断）

在 ARM + NPU + NUMA 敏感的部署环境下，这类优化对：

- tail latency
- host-device coordination
- 稳定吞吐

往往比一两个微小 kernel 优化还更有价值。

### 10.4 它甚至处理 IRQ 和 irqbalance

`bind_npu_irq()` 会：

- 检查 `/proc/irq` 可写性；
- 只给当前 rank 的 NPU 绑 IRQ，避免多进程互相覆盖；
- 如果发现 `irqbalance` 运行中，还会停掉它并提醒。

### upstream-vs-Ascend delta

这是很典型的“生产系统层优化”，上游通用框架通常不会替你把这层做好。

### 可能的硬件后果（工程推断）

IRQ 漂移与 irqbalance 干扰会直接导致：

- 中断处理不稳定；
- CPU cache locality 被破坏；
- tail latency 抖动。

Ascend 把这层也纳入性能路径，说明它并不是只盯模型内部。

---

## 11. 把这些源码连起来看：Ascend 真正新增的不是 feature，而是 7 条性能通路

把前面几节压缩成一个“工程视角图”，会更容易抓住重点。

### 通路 1：平台接入通路

- `platform.py`
- 作用：统一接管 device / compile / pass manager / attention backend / quantization 入口

### 通路 2：Attention/KV 执行通路

- `attention_v1.py`
- 作用：定义 KV layout、block copy/swap、paged attention / FIA 双路径、graph 参数更新

### 通路 3：Graph runtime 通路

- `acl_graph.py`
- 作用：按 batch descriptor capture/replay，管理 workspace、时序同步、graph params

### 通路 4：编译器图融合通路

- `graph_fusion_pass_manager.py`
- 作用：不依赖 Triton，自己定义 NPU 侧 norm/rope/allreduce/SP 融合 pass

### 通路 5：调度预算通路

- `scheduler_dynamic_batch.py`
- 作用：把离线 profile 表和 SLO 限制接到 token budget 决策

### 通路 6：复杂运行时兼容通路

- `recompute_scheduler.py`
- 作用：让 PD / KV transfer / MTP / hybrid model 仍能维持 full graph 可用

### 通路 7：Host 稳定性通路

- `cpu_binding.py`
- 作用：绑核、迁内存、绑 IRQ、处理 runtime 线程亲和性

这 7 条通路叠起来，才构成“Ascend 版 vLLM 的真实性能画像”。

---

## 12. 与 upstream vLLM 对比，最值得强调的“源码级新增”是什么

如果你要对外讲 vLLM-Ascend，最值得讲的不是“Ascend 也支持 chunked prefill”，而是下面这些真正的源码级新增：

1. **`NPUPlatform` 统一接管 compile/pass/backend/quant 路径**  
   这是插件化和可维护性的根。
2. **`attention_v1.py` 把 Ascend 的 KV 物理布局、block 操作和 attention runtime 都接了过来**  
   这不是单纯“换个 kernel”。
3. **`acl_graph.py` 把 graph capture/replay/workspace/同步顺序做成完整 runtime wrapper**  
   这是 graph mode 真能跑稳的关键。
4. **`GraphFusionPassManager` 明确表明 Ascend 走的是独立 compile/pass 栈，而不是 Triton 路线**
5. **`scheduler_dynamic_batch.py` 把 profiling table + SLO 真正接进预算决策**
6. **`recompute_scheduler.py` 说明 Ascend 为 full graph/PD/KV transfer 做了很多隐藏但关键的兼容工程**
7. **`cpu_binding.py` 表明 Ascend 的性能优化延伸到了 NUMA/IRQ/host runtime 行为**

这七点，才是“Ascend 到底做了什么优化”的最好答案。

---

## 13. 面向学习者：最值得按什么顺序真正读源码

如果你想以后自己也能做这种插件，不建议从 feature guide 开始背。

最有效的阅读顺序是：

1. `platform.py`
   - 先看平台边界如何被接管
2. `attention_v1.py`
   - 看 NPU attention runtime 与 KV 物理布局
3. `acl_graph.py`
   - 看 graph runtime 如何稳定落地
4. `graph_fusion_pass_manager.py`
   - 看 compile/pass 层怎么变成 NPU 专属
5. `scheduler_dynamic_batch.py`
   - 看 Ascend 在调度预算上加了什么
6. `recompute_scheduler.py`
   - 看复杂部署兼容成本到底有多高
7. `cpu_binding.py`
   - 看真正的生产系统为什么不能只盯模型层

按这个顺序读，你会更容易把“适配”和“优化”分开。

---

## 14. 面向分享：三分钟讲明白 vLLM-Ascend 做了什么

如果你要把这份文档讲给别人听，可以压成下面五句话：

1. **Ascend 不是长期重 fork upstream，而是用 `platform.py` 把自己接进 vLLM 的 hardware plugin 抽象。**
2. **它在 `attention_v1.py` 里不只是换 kernel，而是重建了 NPU 的 KV layout、block 操作和多路径 attention runtime。**
3. **它在 `acl_graph.py` 和 `graph_fusion_pass_manager.py` 里重建了一套适合 `torch_npu` 的 graph/compile 快路径。**
4. **它在 `scheduler_dynamic_batch.py` 和 `recompute_scheduler.py` 里继续往调度层和复杂部署兼容层加货。**
5. **它在 `cpu_binding.py` 里把 host 侧 NUMA/IRQ/线程亲和性也纳入性能系统，所以它优化的不只是模型，而是整条 serving runtime。**

这五句话就是骨架。

---

## 15. 最后提醒：哪些地方最容易被写成“看起来很详细，实际上没增量”

写 Ascend 文档时，最容易犯的错有三类：

### 15.1 把 upstream feature 换个名字再讲一遍

例如只说：

- 支持 chunked prefill
- 支持 prefix caching
- 支持 graph mode

这不够，因为这没有回答：

- Ascend 具体在哪里接管了这些能力；
- 相比 upstream，多了哪些 runtime/compile/host 行为。

### 15.2 把 Host/NUMA/IRQ 当成“部署杂项”

从 `cpu_binding.py` 看，这不是杂项，而是性能通路。  
如果忽略这层，会误以为 Ascend 的优化只发生在模型前向。

### 15.3 把“更长”误当成“更深入”

真正高价值的增量应该尽量满足这三个条件：

1. 有明确源码锚点；
2. 有 upstream-vs-Ascend 的行为差异；
3. 有清楚的硬件或 runtime 后果。

只要这三个条件不满足，再长都可能只是填充。

---

## 16. 参考来源

1. vLLM Blog, *Introducing vLLM Hardware Plugin, Best Practice from Ascend NPU*  
   https://blog.vllm.ai/2025/05/12/hardware-plugin.html
2. vLLM-Ascend Docs  
   https://docs.vllm.ai/projects/ascend/en/main/
3. vLLM-Ascend GitHub  
   https://github.com/vllm-project/vllm-ascend
4. vLLM-Ascend 仓库快照：`d1f66849c9ea1459ca6a4c6c213ac3fec2ec1c9c`
5. upstream vLLM 仓库快照：`92a7c121b62a1484b68c0a27d1ecefd1a84f78fc`
6. 源码：`vllm_ascend/platform.py`
7. 源码：`vllm_ascend/attention/attention_v1.py`
8. 源码：`vllm_ascend/compilation/acl_graph.py`
9. 源码：`vllm_ascend/compilation/graph_fusion_pass_manager.py`
10. 源码：`vllm_ascend/core/scheduler_dynamic_batch.py`
11. 源码：`vllm_ascend/core/recompute_scheduler.py`
12. 源码：`vllm_ascend/cpu_binding.py`

---

## 17. 一句话总评

**vLLM-Ascend 的精华不在“把 vLLM 搬到 NPU”，而在它围绕 NPU 的 backend 选择、KV 物理布局、graph capture/replay、compile pass、动态预算、full-graph 兼容以及 NUMA/IRQ 行为，重建了一套真正面向 Ascend 平台的 serving runtime。**
