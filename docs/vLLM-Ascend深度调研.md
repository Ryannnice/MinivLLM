# vLLM-Ascend 深度调研

> 调研范围：本文基于 2026-05-01 可访问的 vLLM-Ascend 官方仓库源码与 upstream vLLM 对照阅读撰写。Ascend 插件源码基线固定到 commit `d1f66849c9ea1459ca6a4c6c213ac3fec2ec1c9c`，对照的 upstream vLLM 固定到 `92a7c121b62a1484b68c0a27d1ecefd1a84f78fc`。本文重点不是重复“Ascend 支持哪些 feature”，而是回答四件事：`哪些上游系统不变量没有变`、`Ascend 真正重写了哪些运行层`、`这些重写为什么对 NPU 平台有意义`、`哪些部分体现出明显的源码级平台债务`。

> 证据约定：`源码事实` 指类、函数、字段、patch 点与控制流；`对照结论` 指 Ascend 插件与 upstream 同时点横向比较；`工程判断` 指根据 NPU 图执行、通信与 host 侧常识做出的合理推断，不把它写成 benchmark 结论。

## 1. 结论先行

vLLM-Ascend 的价值，不在于“把 CUDA 版 vLLM 机械翻译成 NPU 版”，而在于它沿着上游稳定边界，把 NPU 平台真正必须重做的几层都接出来了：

1. `Platform`：重新定义设备身份、dispatch key、compile backend、attention backend 派发与平台配置注入。
2. `Worker / Attention runtime`：围绕 `seq_lens_cpu`、ACLGraph、KV layout、paged attention / FIA 路径重建执行翻译层。
3. `Graph / Compile / Patch`：不是沿用上游 CUDA/Triton 假设，而是单独维护 ACLGraph、torchair、fusion pass 和 meta registration。
4. `Scheduler / Connector / Host`：把 placeholder token、remote KV、dynamic batch、profiling chunk、HCCL 以及 CPU/NUMA/IRQ 绑核纳入正式系统层。

一句话概括：

> **Ascend 没有改写 vLLM 的请求推进与 KV 抽象，而是把 platform、worker、graph、scheduler、communication、host 常数项按 NPU 现实重新做了一遍。**

## 2. 什么没变，什么变了

理解 Ascend 插件最重要的，不是先看某个算子，而是先把“不变量”和“改写项”拆开。

### 2.1 沿用 upstream 的三大不变量

| 不变量 | upstream 落点 | Ascend 是否改写 | 说明 |
| --- | --- | --- | --- |
| 请求推进围绕 token debt | [`v1/core/sched/scheduler.py`](https://github.com/vllm-project/vllm/blob/92a7c121b62a1484b68c0a27d1ecefd1a84f78fc/vllm/v1/core/sched/scheduler.py) | 否 | Ascend 仍然围绕 `num_computed_tokens`、`num_tokens_with_spec`、`num_output_placeholders` 推进请求 |
| KV 仍是 block/page 资源 | [`v1/core/kv_cache_manager.py`](https://github.com/vllm-project/vllm/blob/92a7c121b62a1484b68c0a27d1ecefd1a84f78fc/vllm/v1/core/kv_cache_manager.py) | 否 | Ascend 改的是 layout、copy/swap、offload 和 backend，不是 block 抽象本身 |
| 顶层调度不应携带设备细节 | [`v1/engine/core.py`](https://github.com/vllm-project/vllm/blob/92a7c121b62a1484b68c0a27d1ecefd1a84f78fc/vllm/v1/engine/core.py) | 否 | 设备差异被压到 platform、worker、compilation、patch 层 |

### 2.2 Ascend 真正重写了哪些层

| 层次 | 代表文件 | 相比 upstream 的真实增量 |
| --- | --- | --- |
| Platform | [`platform.py`](https://github.com/vllm-project/vllm-ascend/blob/d1f66849c9ea1459ca6a4c6c213ac3fec2ec1c9c/vllm_ascend/platform.py) | `NPUPlatform` 接管 device identity、compile backend、attention backend dispatch、graph wrapper 选择 |
| Worker | [`worker/model_runner_v1.py`](https://github.com/vllm-project/vllm-ascend/blob/d1f66849c9ea1459ca6a4c6c213ac3fec2ec1c9c/vllm_ascend/worker/model_runner_v1.py)、[`worker/v2/model_runner.py`](https://github.com/vllm-project/vllm-ascend/blob/d1f66849c9ea1459ca6a4c6c213ac3fec2ec1c9c/vllm_ascend/worker/v2/model_runner.py) | `seq_lens_cpu`、ACLGraphWrapper、profile_run、spec/decode 特化 |
| Attention / KV runtime | [`attention/attention_v1.py`](https://github.com/vllm-project/vllm-ascend/blob/d1f66849c9ea1459ca6a4c6c213ac3fec2ec1c9c/vllm_ascend/attention/attention_v1.py) | KV shape、swap/copy、metadata builder、graph params、paged/FIA 分流 |
| Graph / Compile | [`compilation/acl_graph.py`](https://github.com/vllm-project/vllm-ascend/blob/d1f66849c9ea1459ca6a4c6c213ac3fec2ec1c9c/vllm_ascend/compilation/acl_graph.py) | 独立 NPUGraph/ACLGraph runtime，而不是 CUDAGraph shim |
| Scheduler / Serving | [`core/recompute_scheduler.py`](https://github.com/vllm-project/vllm-ascend/blob/d1f66849c9ea1459ca6a4c6c213ac3fec2ec1c9c/vllm_ascend/core/recompute_scheduler.py)、[`core/scheduler_dynamic_batch.py`](https://github.com/vllm-project/vllm-ascend/blob/d1f66849c9ea1459ca6a4c6c213ac3fec2ec1c9c/vllm_ascend/core/scheduler_dynamic_batch.py) | placeholder token、remote KV 状态机、SLO 驱动动态 batch |
| Communication | [`distributed/parallel_state.py`](https://github.com/vllm-project/vllm-ascend/blob/d1f66849c9ea1459ca6a4c6c213ac3fec2ec1c9c/vllm_ascend/distributed/parallel_state.py)、[`distributed/device_communicators/pyhccl.py`](https://github.com/vllm-project/vllm-ascend/blob/d1f66849c9ea1459ca6a4c6c213ac3fec2ec1c9c/vllm_ascend/distributed/device_communicators/pyhccl.py) | HCCL、flashcomm2、O-shard、layer shard |
| Host 优化 | [`cpu_binding.py`](https://github.com/vllm-project/vllm-ascend/blob/d1f66849c9ea1459ca6a4c6c213ac3fec2ec1c9c/vllm_ascend/cpu_binding.py) | CPU、NUMA、IRQ、ACL 线程与 release 线程绑核 |

## 3. 插件边界与代码地图

Ascend 插件最值得学的，是它如何在不破坏上游大框架的前提下，集中接管平台差异。

| 层次 | 关键文件 | 主要职责 | 为什么重要 |
| --- | --- | --- | --- |
| 平台入口 | [`platform.py`](https://github.com/vllm-project/vllm-ascend/blob/d1f66849c9ea1459ca6a4c6c213ac3fec2ec1c9c/vllm_ascend/platform.py) | `NPUPlatform`、compile backend、attention backend、graph wrapper | 所有 NPU 差异的总入口 |
| 附加配置 | [`ascend_config.py`](https://github.com/vllm-project/vllm-ascend/blob/d1f66849c9ea1459ca6a4c6c213ac3fec2ec1c9c/vllm_ascend/ascend_config.py) | PCP/DCP、flashcomm2、sparse C8、dynamic batch、CPU binding | 平台专有配置集中化 |
| Worker | [`worker/model_runner_v1.py`](https://github.com/vllm-project/vllm-ascend/blob/d1f66849c9ea1459ca6a4c6c213ac3fec2ec1c9c/vllm_ascend/worker/model_runner_v1.py)、[`worker/v2/model_runner.py`](https://github.com/vllm-project/vllm-ascend/blob/d1f66849c9ea1459ca6a4c6c213ac3fec2ec1c9c/vllm_ascend/worker/v2/model_runner.py) | NPUModelRunner、`seq_lens_cpu`、ACLGraphWrapper、profile_run | 调度结果到设备执行的翻译层 |
| Attention / MLA / SFA | [`attention/attention_v1.py`](https://github.com/vllm-project/vllm-ascend/blob/d1f66849c9ea1459ca6a4c6c213ac3fec2ec1c9c/vllm_ascend/attention/attention_v1.py)、[`attention/mla_v1.py`](https://github.com/vllm-project/vllm-ascend/blob/d1f66849c9ea1459ca6a4c6c213ac3fec2ec1c9c/vllm_ascend/attention/mla_v1.py) | KV layout、paged/FIA、MLA 路径、graph metadata | Ascend 真正的算子级运行时 |
| 图执行 | [`compilation/acl_graph.py`](https://github.com/vllm-project/vllm-ascend/blob/d1f66849c9ea1459ca6a4c6c213ac3fec2ec1c9c/vllm_ascend/compilation/acl_graph.py) | capture/replay、workspace、graph params | 解决 decode 高频小步固定开销 |
| 图重写 | [`compilation/graph_fusion_pass_manager.py`](https://github.com/vllm-project/vllm-ascend/blob/d1f66849c9ea1459ca6a4c6c213ac3fec2ec1c9c/vllm_ascend/compilation/graph_fusion_pass_manager.py) | pass 顺序、fusion 策略、torchair 接入 | 说明它不是照搬 upstream compile path |
| 调度专项 | [`core/recompute_scheduler.py`](https://github.com/vllm-project/vllm-ascend/blob/d1f66849c9ea1459ca6a4c6c213ac3fec2ec1c9c/vllm_ascend/core/recompute_scheduler.py)、[`core/scheduler_profiling_chunk.py`](https://github.com/vllm-project/vllm-ascend/blob/d1f66849c9ea1459ca6a4c6c213ac3fec2ec1c9c/vllm_ascend/core/scheduler_profiling_chunk.py) | placeholder token、remote KV、profiling chunk | full graph 与服务拓扑兼容的关键 |
| 通信与分片 | [`distributed/parallel_state.py`](https://github.com/vllm-project/vllm-ascend/blob/d1f66849c9ea1459ca6a4c6c213ac3fec2ec1c9c/vllm_ascend/distributed/parallel_state.py)、[`ops/layer_shard_linear.py`](https://github.com/vllm-project/vllm-ascend/blob/d1f66849c9ea1459ca6a4c6c213ac3fec2ec1c9c/vllm_ascend/ops/layer_shard_linear.py) | HCCL、flashcomm2、O-shard、layer shard | 这不是单卡 patch |
| Host 侧 | [`cpu_binding.py`](https://github.com/vllm-project/vllm-ascend/blob/d1f66849c9ea1459ca6a4c6c213ac3fec2ec1c9c/vllm_ascend/cpu_binding.py) | CPU / NUMA / IRQ / thread binding | Ascend 把 host 抖动当成一等性能问题 |

## 4. Ascend 请求主链：哪些段复用上游，哪些段被接管

顶层 `LLMEngine -> EngineCore` 主链仍来自 upstream；真正开始出现 NPU delta 的位置，集中在 platform、scheduler、worker、attention runtime 和 graph wrapper。

1. 上游 `LLMEngine` 与 `EngineCore` 仍负责请求接入、主循环和 executor 装配。
2. [`NPUPlatform`](https://github.com/vllm-project/vllm-ascend/blob/d1f66849c9ea1459ca6a4c6c213ac3fec2ec1c9c/vllm_ascend/platform.py) 在 backend 选择阶段把设备与 graph/runtime 语义切到 `torch.npu`。
3. 若启用 Ascend 专项调度，会进入 [`RecomputeScheduler`](https://github.com/vllm-project/vllm-ascend/blob/d1f66849c9ea1459ca6a4c6c213ac3fec2ec1c9c/vllm_ascend/core/recompute_scheduler.py) 或 [`SchedulerDynamicBatch`](https://github.com/vllm-project/vllm-ascend/blob/d1f66849c9ea1459ca6a4c6c213ac3fec2ec1c9c/vllm_ascend/core/scheduler_dynamic_batch.py)。
4. worker 侧由 [`NPUModelRunner`](https://github.com/vllm-project/vllm-ascend/blob/d1f66849c9ea1459ca6a4c6c213ac3fec2ec1c9c/vllm_ascend/worker/model_runner_v1.py) 或 V2 runner 接管，准备 `seq_lens_cpu`、graph inputs、NPU 版 metadata。
5. attention runtime 通过 [`attention_v1.py`](https://github.com/vllm-project/vllm-ascend/blob/d1f66849c9ea1459ca6a4c6c213ac3fec2ec1c9c/vllm_ascend/attention/attention_v1.py) 在 paged attention 与 FIA 之间分流，并同步 graph params。
6. 若启用 full graph，则 `NPUPlatform` 会返回 `ACLGraphWrapper`，随后由 [`acl_graph.py`](https://github.com/vllm-project/vllm-ascend/blob/d1f66849c9ea1459ca6a4c6c213ac3fec2ec1c9c/vllm_ascend/compilation/acl_graph.py) 承担 capture/replay。
7. 若请求跨节点或采用 PD/disaggregated 拓扑，则 connector、remote KV 和 loader 由 Ascend 插件进一步接管。

关键判断：

- 这不是“上游 scheduler + 下游 kernel”的两层补丁。
- 这是“上游 engine 骨架 + Ascend 运行层重建”的插件体系。

## 5. Platform：`NPUPlatform` 不是配置文件，而是总开关

如果只选一个文件理解插件边界，优先看 [`platform.py`](https://github.com/vllm-project/vllm-ascend/blob/d1f66849c9ea1459ca6a4c6c213ac3fec2ec1c9c/vllm_ascend/platform.py)。

### 5.1 它先显式声明这是一块 NPU，而不是“改名后的 CUDA”

源码里直接定义了：

- `class NPUPlatform(Platform)`
- `device_name = "npu"`
- `dispatch_key = "PrivateUse1"`
- `device_control_env_var = "ASCEND_RT_VISIBLE_DEVICES"`

这一步很关键。后面所有 backend、graph、patch 的选择，都建立在设备身份被显式接管之上。

### 5.2 compile backend、pass manager、attention backend 都在平台层切换

最关键的入口是：

- `get_pass_manager_cls()`
- `get_compile_backend()`
- `get_attn_backend_cls()`

`platform.py` 还会把：

- `SLO_limits_for_dynamic_batch`
- PCP/DCP/flashcomm2 等配置
- graph wrapper 类名

注入到上游配置对象里。也就是说，Ascend 插件不是等进入 worker 后再偷偷换实现，而是在平台层就决定了整条后续路径。

### 5.3 `ACLGraphWrapper` 的选择也发生在平台层

`platform.py` 直接返回：

- `vllm_ascend.compilation.acl_graph.ACLGraphWrapper`

这说明图执行在 Ascend 里不是 worker 层局部 hack，而是平台契约的一部分。对于 decode 高频路径，这一点尤其重要。

### 5.4 可维护性为什么高度依赖 `NPUPlatform`

如果 compile backend、attn backend、dynamic batch、graph wrapper 的分派散落在 executor、model runner、kernel wrapper 各处，后续上游一旦演进，插件就会失控。把这些选择收束到平台层，是 OOT 插件长期可维护的前提。

**源码抓手**

- [`platform.py`](https://github.com/vllm-project/vllm-ascend/blob/d1f66849c9ea1459ca6a4c6c213ac3fec2ec1c9c/vllm_ascend/platform.py)：`NPUPlatform`、`get_pass_manager_cls()`、`get_compile_backend()`、`get_attn_backend_cls()`
- [`ascend_config.py`](https://github.com/vllm-project/vllm-ascend/blob/d1f66849c9ea1459ca6a4c6c213ac3fec2ec1c9c/vllm_ascend/ascend_config.py)
- [`device_allocator/camem.py`](https://github.com/vllm-project/vllm-ascend/blob/d1f66849c9ea1459ca6a4c6c213ac3fec2ec1c9c/vllm_ascend/device_allocator/camem.py)

## 6. Worker / ModelRunner：Ascend 把执行翻译层整套接管

### 6.1 `NPUModelRunner` 不是轻量 wrapper

[`worker/model_runner_v1.py`](https://github.com/vllm-project/vllm-ascend/blob/d1f66849c9ea1459ca6a4c6c213ac3fec2ec1c9c/vllm_ascend/worker/model_runner_v1.py) 和 [`worker/v2/model_runner.py`](https://github.com/vllm-project/vllm-ascend/blob/d1f66849c9ea1459ca6a4c6c213ac3fec2ec1c9c/vllm_ascend/worker/v2/model_runner.py) 都定义了 `class NPUModelRunner(GPUModelRunner)`，但这个继承并不意味着“小改几处就行”。

真正的增量至少包括：

- `ACLGraphWrapper` 接入
- `profile_run()` 的 NPU 路径
- `seq_lens_cpu` 维护
- `optimistic_seq_lens_cpu` 的 NPU 版同步
- async scheduling 下的 event 协调

### 6.2 `_needs_seq_lens_cpu_sync`、`_seq_lens_cpu_event` 暴露了 NPU backend 的额外协议

V1 runner 明确维护：

- `_needs_seq_lens_cpu_sync`
- `_seq_lens_cpu_event`
- `_seq_lens_cpu_event_pending`

而且在 spec decode / async path 下，会在 GPU world 中不再重要的 `_seq_lens_cpu` 镜像上继续做同步。这说明对 Ascend backend 来说，CPU 侧序列长度不是历史包袱，而是当前执行契约的一部分。

### 6.3 `optimistic_seq_lens_cpu` 是 worker 侧的关键桥梁

Ascend runner 会像 upstream 一样先乐观推进 `optimistic_seq_lens_cpu`，但随后又要在某些路径下显式修正、等待事件完成，再把它喂给 attention metadata builder。这说明 Ascend 执行路径比 GPU 路径更依赖 host/device 双侧状态保持一致。

### 6.4 V2 runner 继续为 NPU 保留 `seq_lens_cpu`

[`worker/v2/model_runner.py`](https://github.com/vllm-project/vllm-ascend/blob/d1f66849c9ea1459ca6a4c6c213ac3fec2ec1c9c/vllm_ascend/worker/v2/model_runner.py) 直接写明：

- NPU attention backends 仍需要 `seq_lens_cpu`
- GPUModelRunnerV2 正逐步废弃这一路径
- `_update_seq_lens_cpu()` 需要根据 scheduler output 重新计算每个 request 的长度

这说明 Ascend 并不是简单“复用 V2 worker”，而是在 input buffer 协议层保留了设备特有的数据依赖。

### 6.5 `decode_threshold` 也被带进了 runner

V1 runner 还显式计算 `decode_threshold = 1 + num_speculative_tokens`。这说明 spec decode 在 Ascend world 不是后加优化，而是和 attention 路径选择、graph 兼容性、batch 组织同时建模的。

## 7. Attention / KV Runtime：Ascend 的核心增量几乎都在这里

### 7.1 `attention_v1.py` 重写了 KV cache 的执行契约

[`attention_v1.py`](https://github.com/vllm-project/vllm-ascend/blob/d1f66849c9ea1459ca6a4c6c213ac3fec2ec1c9c/vllm_ascend/attention/attention_v1.py) 最值得盯的函数是：

- `get_kv_cache_shape()`
- `swap_blocks()`
- `copy_blocks()`
- `update_graph_params()`

这意味着 Ascend backend 不只负责“算 attention”，它还直接接管了 block 物理操作与 graph 参数同步。

### 7.2 `AscendAttentionMetadataBuilder` 一开始就是按 graph 路径设计的

`class AscendAttentionMetadataBuilder` 里显式维护：

- `decode_threshold`
- `reorder_batch_threshold`
- `chunked_prefill_enabled`
- `get_cudagraph_support()`

其中最有代表性的细节有两个：

1. `get_cudagraph_support()` 直接返回 `ALWAYS`，说明该 backend 天生按 graph 兼容设计。
2. `decode_threshold` 会把 speculative token 一起算进去，并显式断言不超过 NPU fused infer attention 的限制。

这说明 graph 不是后来补上的开关，而是 attention runtime 的设计前提。

### 7.3 builder 明确优先使用 `_seq_lens_cpu`

`build()` 里优先取：

- `common_attn_metadata._seq_lens_cpu`

其次才是：

- `common_attn_metadata.seq_lens_cpu`

最后才退回 `seq_lens.to("cpu")`

这是一条很强的源码证据：Ascend backend 真的把 CPU 侧序列长度当成执行路径的一等输入，而不是调试遗留字段。

### 7.4 paged attention 与 FIA 是两条真正的执行路径

Ascend runtime 会在 paged attention 与 fused infer attention 之间分流，不是所有请求都走同一 kernel 外壳。这一层差异决定了：

- KV layout
- workspace 组织
- graph replay 形状
- decode/prefill 分界

### 7.5 MLA、KV offload、spec decode 都有 NPU 专门实现

- [`mla_v1.py`](https://github.com/vllm-project/vllm-ascend/blob/d1f66849c9ea1459ca6a4c6c213ac3fec2ec1c9c/vllm_ascend/attention/mla_v1.py)：MLA 路径与量化约束
- [`kv_offload/cpu_npu.py`](https://github.com/vllm-project/vllm-ascend/blob/d1f66849c9ea1459ca6a4c6c213ac3fec2ec1c9c/vllm_ascend/kv_offload/cpu_npu.py)：CPU<->NPU block copy 与 stream/event 协调
- [`spec_decode/dflash_proposer.py`](https://github.com/vllm-project/vllm-ascend/blob/d1f66849c9ea1459ca6a4c6c213ac3fec2ec1c9c/vllm_ascend/spec_decode/dflash_proposer.py)：Ascend 上的 DFlash proposal 路径

这几条支线说明：Ascend 插件不是只保住主路径可运行，而是在高价值扩展路径上都补了设备专门实现。

## 8. Graph Runtime 与 Compile Path：它不是照搬 upstream 的 CUDA/Triton 世界

### 8.1 `acl_graph.py` 是独立 runtime，不是 CUDAGraph shim

[`acl_graph.py`](https://github.com/vllm-project/vllm-ascend/blob/d1f66849c9ea1459ca6a4c6c213ac3fec2ec1c9c/vllm_ascend/compilation/acl_graph.py) 负责 capture/replay、workspace 和 graph 参数维护。它存在的意义不是“给 CUDAGraph 换个名字”，而是把 Ascend 图执行的对象模型独立出来。

### 8.2 compile backend 走的是 `torchair + 自定义 pass`

[`compiler_interface.py`](https://github.com/vllm-project/vllm-ascend/blob/d1f66849c9ea1459ca6a4c6c213ac3fec2ec1c9c/vllm_ascend/compilation/compiler_interface.py) 与 [`graph_fusion_pass_manager.py`](https://github.com/vllm-project/vllm-ascend/blob/d1f66849c9ea1459ca6a4c6c213ac3fec2ec1c9c/vllm_ascend/compilation/graph_fusion_pass_manager.py) 说明它明确选择了另一条编译栈。

典型 pass 包括：

- [`norm_quant_fusion_pass.py`](https://github.com/vllm-project/vllm-ascend/blob/d1f66849c9ea1459ca6a4c6c213ac3fec2ec1c9c/vllm_ascend/compilation/passes/norm_quant_fusion_pass.py)
- [`sequence_parallelism.py`](https://github.com/vllm-project/vllm-ascend/blob/d1f66849c9ea1459ca6a4c6c213ac3fec2ec1c9c/vllm_ascend/compilation/passes/sequence_parallelism.py)
- [`allgather_chunk_noop_pass.py`](https://github.com/vllm-project/vllm-ascend/blob/d1f66849c9ea1459ca6a4c6c213ac3fec2ec1c9c/vllm_ascend/compilation/passes/allgather_chunk_noop_pass.py)
- [`allreduce_rmsnorm_fusion_pass.py`](https://github.com/vllm-project/vllm-ascend/blob/d1f66849c9ea1459ca6a4c6c213ac3fec2ec1c9c/vllm_ascend/compilation/passes/allreduce_rmsnorm_fusion_pass.py)

### 8.3 patch 面说明它必须显式消解上游 GPU 假设

最典型的几处是：

- [`meta_registration.py`](https://github.com/vllm-project/vllm-ascend/blob/d1f66849c9ea1459ca6a4c6c213ac3fec2ec1c9c/vllm_ascend/meta_registration.py)
- [`patch/worker/patch_npugraph_ex_triton.py`](https://github.com/vllm-project/vllm-ascend/blob/d1f66849c9ea1459ca6a4c6c213ac3fec2ec1c9c/vllm_ascend/patch/worker/patch_npugraph_ex_triton.py)
- [`patch/worker/patch_cudagraph.py`](https://github.com/vllm-project/vllm-ascend/blob/d1f66849c9ea1459ca6a4c6c213ac3fec2ec1c9c/vllm_ascend/patch/worker/patch_cudagraph.py)
- [`patch/worker/patch_triton.py`](https://github.com/vllm-project/vllm-ascend/blob/d1f66849c9ea1459ca6a4c6c213ac3fec2ec1c9c/vllm_ascend/patch/worker/patch_triton.py)

这些 patch 覆盖的不是 cosmetic rename，而是：

- FX graph conversion
- graph dispatcher
- Meta 实现
- kernel entry / registry

也就是说，Ascend 能跑通，不是因为平台天然兼容，而是因为插件明确把这些 GPU 假设逐层显式化并修补。

## 9. Scheduler 与 Serving Runtime：Ascend 把服务侧状态机也重写了

### 9.1 `RecomputeScheduler.add_request()` 直接改写了新请求进入系统的方式

[`recompute_scheduler.py`](https://github.com/vllm-project/vllm-ascend/blob/d1f66849c9ea1459ca6a4c6c213ac3fec2ec1c9c/vllm_ascend/core/recompute_scheduler.py) 中的 `add_request()` 会在特定场景下做两类特殊处理：

1. 为 full graph / KV consumer 场景填入 placeholder spec tokens。
2. 对某些 hybrid model producer 场景主动修正 prompt token 列表。

这说明 Ascend 为了维持图匹配与跨节点 KV 消费的一致性，已经把请求入队协议都改了。

### 9.2 `RecomputeSchedulerOutput` 说明 scheduler 输出协议被扩展了

`RecomputeSchedulerOutput(SchedulerOutput)` 额外带有 `recomputed_reqs` 等信息。这意味着 Ascend 不是仅仅沿用上游 SchedulerOutput，而是承认“远端 KV、recompute、placeholder token”需要额外状态。

### 9.3 `_update_waiting_for_remote_kv()` 暴露了 remote KV 状态机

这段代码至少说明三件事：

1. 请求可以处于“等待远端 KV”这一正式状态。
2. KV transfer 完成后，scheduler 需要把 block 缓存回本地 cache manager。
3. full prompt hit 时仍需回退一个 token 以便重新计算 logits。

这说明 Ascend 的 connector/remote KV 路径并不是 worker 层黑盒，而是 scheduler 正式管理的一部分。

### 9.4 `SchedulerDynamicBatch` 不只是 chunked prefill 开关

[`scheduler_dynamic_batch.py`](https://github.com/vllm-project/vllm-ascend/blob/d1f66849c9ea1459ca6a4c6c213ac3fec2ec1c9c/vllm_ascend/core/scheduler_dynamic_batch.py) 直接围绕：

- `SLO limit`
- `chunk_size`
- decode-first chunked prefills

工作。这说明 Ascend 在调度层额外引入了服务级时延约束，而不仅是 token budget。

### 9.5 profiling chunk 把“测一遍再调度”做进了正式路径

- [`scheduler_profiling_chunk.py`](https://github.com/vllm-project/vllm-ascend/blob/d1f66849c9ea1459ca6a4c6c213ac3fec2ec1c9c/vllm_ascend/core/scheduler_profiling_chunk.py)：启动时 profile 多组 chunk 大小
- [`profiling_chunk_predictor.py`](https://github.com/vllm-project/vllm-ascend/blob/d1f66849c9ea1459ca6a4c6c213ac3fec2ec1c9c/vllm_ascend/core/profiling_chunk_predictor.py)：拟合二次 latency 模型并回推 chunk size

这类策略把调度从“静态规则”推进到了“profile-guided runtime policy”。

## 10. Distributed / Shard / Communication：这不是单卡 patch

### 10.1 `parallel_state.py` 真正新增了 Ascend 专项并行组

[`parallel_state.py`](https://github.com/vllm-project/vllm-ascend/blob/d1f66849c9ea1459ca6a4c6c213ac3fec2ec1c9c/vllm_ascend/distributed/parallel_state.py) 中最有代表性的增量是：

- `flashcomm2_otp`
- `flashcomm2_odp`
- `shard weight` 相关 group

这说明 Ascend 不只是复用 global TP group，而是按 O-proj / shard weight 等更细粒度切通信世界。

### 10.2 HCCL 不是薄封装

[`PyHcclCommunicator`](https://github.com/vllm-project/vllm-ascend/blob/d1f66849c9ea1459ca6a4c6c213ac3fec2ec1c9c/vllm_ascend/distributed/device_communicators/pyhccl.py) 直接面向 HCCL library、unique id 和 communicator 初始化。它存在的意义是把通信后端显式切到 NPU 世界，而不是假装 NCCL 语义天然兼容。

### 10.3 layer shard 与 O-shard 说明它在推理图上继续做结构优化

- [`ops/layer_shard_linear.py`](https://github.com/vllm-project/vllm-ascend/blob/d1f66849c9ea1459ca6a4c6c213ac3fec2ec1c9c/vllm_ascend/ops/layer_shard_linear.py)
- [`ops/flashcomm2_oshard_manager.py`](https://github.com/vllm-project/vllm-ascend/blob/d1f66849c9ea1459ca6a4c6c213ac3fec2ec1c9c/vllm_ascend/ops/flashcomm2_oshard_manager.py)

它们继续围绕 attention `o_proj`、权重预取和分片管理做设备专门工程，而不是停留在“能 all-reduce 就行”。

## 11. Host 侧优化：Ascend 把 CPU、NUMA、IRQ 当成一等问题

[`cpu_binding.py`](https://github.com/vllm-project/vllm-ascend/blob/d1f66849c9ea1459ca6a4c6c213ac3fec2ec1c9c/vllm_ascend/cpu_binding.py) 是整套插件里最容易被忽略、但最能说明工程成熟度的文件之一。

### 11.1 绑定策略不是一句 `taskset`

源码里直接写了：

- `MIN_CPUS_PER_NPU = 5`
- 每个 NPU 至少需要 main、ACL、release、IRQ 对应的 CPU 资源
- `allocate()` 会把 CPU pool 切成 `assign_main`、`assign_acl`、`assign_rel`

也就是说，它不是只绑主线程，而是在主动为 runtime 不同角色预留 CPU。

### 11.2 `bind_threads()` 与 `bind_memory()` 说明 NUMA 也是正式性能模型的一部分

`bind_threads()` 会：

- 给主进程绑 `assign_main`
- 给 ACL 线程绑 `assign_acl`
- 给 release 线程绑 `assign_rel`
- 最后通过 `migratepages` 把内存尽量迁到目标 NPU 的 NUMA node

这意味着在 Ascend 设备上，host 内存局部性不是“部署脚本层小优化”，而是 runtime 级因素。

### 11.3 `bind_npu_irq()` 把 IRQ 也纳入运行时管控

`bind_npu_irq()` 会：

- 检查并可能停掉 `irqbalance`
- 从 `/proc/interrupts` 里扫描 `sq_send_trigger_irq`
- 只给当前 rank 的 NPU 绑定 IRQ，避免多进程互相覆盖

这说明在 Ascend world 里，IRQ 抖动足以影响服务稳定性，已经被当成正式的性能面去处理。

## 12. 兼容性边界与工程债务

### 12.1 它不是训练框架

没有 optimizer、backward、checkpoint save、训练 driver。训练并行相关术语出现在源码中，不代表它承担了训练栈职责。

### 12.2 它存在可见的 patch debt

大量 `patch_*` 文件说明上游 CUDA/Triton 假设仍深嵌在生态里。Ascend 能跑通，靠的是把这些假设逐层显式化并修补，而不是平台天然兼容。

### 12.3 full graph 的收益伴随更强 shape discipline

placeholder token、`seq_lens_cpu`、graph params、ACLGraphWrapper 共同说明：为了稳定 graph capture / replay，Ascend 需要比普通 eager path 更严格的形状与状态一致性。

## 13. 对 MinivLLM/Ascend 后端的直接启发

如果 MinivLLM 后续要做 Ascend 或其他 NPU 后端，这份调研里最应该直接复用的是方法，不是 patch 文本。

1. 先稳住 upstream 的顶层不变量，再做平台重写；不要一开始就把设备细节灌进 scheduler。
2. 把平台入口做成明确的 `Platform` 接管点，集中切 compile backend、dispatch key、attention backend、graph wrapper。
3. 提前承认 graph runtime 是独立系统层，而不是“后面再开个优化开关”。
4. 如果设备需要 `seq_lens_cpu`、placeholder token、host binding 这类额外协议，就把它们显式建模，不要藏在临时 patch 里。
5. 对 NPU 平台，communication、loader、connector、CPU/NUMA/IRQ 都应该和 kernel 一样被纳入主文档，而不是附录。

## 14. 参考来源

- [vLLM-Ascend 仓库](https://github.com/vllm-project/vllm-ascend/tree/d1f66849c9ea1459ca6a4c6c213ac3fec2ec1c9c)
- [upstream vLLM 仓库](https://github.com/vllm-project/vllm/tree/92a7c121b62a1484b68c0a27d1ecefd1a84f78fc)
- [vLLM 官方文档](https://docs.vllm.ai/)
- [PagedAttention 论文](https://arxiv.org/abs/2309.06180)
