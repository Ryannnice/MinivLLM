# vLLM-Ascend 如何完成一次 Transformer 推理：从请求调度到 NPU Attention 的全过程拆解

> 本文以一次普通文本生成请求为主线，追踪 vLLM-Ascend 从插件注册、请求调度、输入整理、KV Cache 写入，到 Prefill/Decode Attention、ACL Graph、采样返回的完整路径。重点写清楚“源码里到底发生了什么”，并标出和 MindIE 整图推理栈不同的地方。

---

## 场景设定

```
模型:     任意 vLLM 支持的 Transformer-like Causal LM
          例: Qwen/Qwen3-0.6B, Qwen/Qwen2.5 系列, DeepSeek MLA/MoE 系列

硬件:     Ascend NPU
          官方 quick start 覆盖 Atlas A2 / A3 / 300I Duo 等设备

精度:     FP16 / BF16 / 量化权重, 由模型配置和 quant_config 决定

输入:     "今天天气怎么样"
输出:     逐 token 生成

核心变量:
  L       = Transformer 层数
  H       = hidden_size
  QH      = num_heads
  KVH     = num_kv_heads
  D       = head_dim
  B       = 当前 batch 中请求数
  T       = 当前这次 forward 被调度的 token 总数

KV Cache:
  默认 block_size = 128
  每层 K cache 形状: (num_blocks, block_size, KVH, D)
  每层 V cache 形状: (num_blocks, block_size, KVH, D)
  block table 形状: (max_num_reqs, max_num_blocks_per_req)
  slot_mapping 形状: (T,)
```

先给一个全局结论：

```
vLLM-Ascend 不是把整个 Transformer 改写成 MindIE GE 的全栈图执行。

它的主路径是:
  vLLM V1 Scheduler
    → vLLM-Ascend NPUWorker
    → NPUModelRunner._prepare_inputs
    → vLLM model.forward
    → AscendAttentionBackend
    → torch_npu / 自定义 Ascend 算子
    → AscendSampler

加速重点是:
  1. 用 vLLM 的连续批处理、页式 KV Cache、prefix caching 等调度能力
  2. 用 Ascend NPU attention 算子处理 Prefill / Decode
  3. 用 _npu_reshape_and_cache 写 KV Cache
  4. 用 ACL Graph 减少 host launch 开销
  5. 用 npugraph_ex / FX pass 做局部融合和通信融合
  6. 用 PCP / DCP / KV connector 支撑多卡和长上下文场景
```

和上游 vLLM 的关系可以先压成一句话：

```
vLLM-Ascend 保留 vLLM 的服务入口、V1 Scheduler、Request/KV block 协议、
模型定义和采样闭环；把设备平台、worker、model runner、attention backend、
KV cache 写入算子、图执行 wrapper、部分 scheduler/parallel 扩展和采样 kernel
替换成 Ascend 版本。
```

也就是说，它不是另起一套推理引擎，而是在 vLLM 的硬件抽象边界上做适配：

```
上游 vLLM CUDA 主线:
  CUDAPlatform
    → GPU Worker / GPUModelRunner
    → FlashAttention / FlashInfer / Triton attention
    → reshape_and_cache_flash / CUDA custom ops
    → CUDAGraphWrapper

vLLM-Ascend:
  NPUPlatform
    → NPUWorker / NPUModelRunner / NPUInputBatch
    → AscendAttentionBackend / AscendMLA / AscendSFA
    → _npu_reshape_and_cache / npu_fused_infer_attention_score
    → ACLGraphWrapper / AscendCompiler / npugraph_ex
```

---

## 第一幕：启动阶段 —— vLLM 如何变成 NPU 后端

### Step 0：插件注册

vLLM-Ascend 通过 Python entry point 注册为 vLLM 的硬件插件：

```
setup.py
  entry_points:
    vllm.platform_plugins:
      ascend = vllm_ascend:register

vllm_ascend/__init__.py
  register()
    → "vllm_ascend.platform.NPUPlatform"
```

当用户运行：

```python
from vllm import LLM

llm = LLM(model="Qwen/Qwen3-0.6B")
```

或：

```bash
vllm serve Qwen/Qwen3-0.6B
```

vLLM 发现 Ascend 插件后，会使用 `NPUPlatform` 作为平台实现。

---

### Step 1：NPUPlatform 改写 vLLM 配置

`NPUPlatform.check_and_update_config()` 会把通用 vLLM 配置改成 Ascend 可执行的配置：

```
NPUPlatform.check_and_update_config(vllm_config)
  │
  ├─ init_ascend_config(vllm_config)
  │    读取 additional_config:
  │      ascend_compilation_config
  │      ascend_fusion_config
  │      xlite_graph_config
  │      finegrained_tp_config
  │      profiling_chunk_config
  │      eplb_config
  │      weight_prefetch_config
  │
  ├─ 设置 compilation_config.oot_compiler
  │    → vllm_ascend.compilation.compiler_interface.AscendCompiler
  │
  ├─ 规范化 ACL Graph 模式
  │    FULL_AND_PIECEWISE → PIECEWISE
  │    encoder-decoder    → PIECEWISE
  │    use_inductor       → False
  │
  ├─ 设置 worker class
  │    普通 A2/A3/A5: vllm_ascend.worker.worker.NPUWorker
  │    310P:          vllm_ascend._310p.worker_310p.NPUWorker310
  │    xlite:         vllm_ascend.xlite.xlite_worker.XliteWorker
  │
  ├─ refresh_block_size(vllm_config)
  │    默认 block_size = 128
  │    prefix cache / chunked prefill 开启时强制 128
  │
  └─ 注册 custom_ops = ["all"]  (310P 除外)
```

这一步很关键：上层 vLLM 仍然负责调度、请求队列、KV cache manager，但底层 worker、attention backend、compiler、采样器都切换到了 Ascend 版本。

相比上游 vLLM 的 CUDA 平台，`NPUPlatform` 主要改了这些边界：

```
设备标识:
  CUDAPlatform: device_type="cuda", dispatch_key="CUDA", ray_device_key="GPU"
  NPUPlatform:  device_type="npu",  dispatch_key="PrivateUse1", ray_device_key="NPU"

通信后端:
  CUDA: NCCL
  Ascend: HCCL

worker:
  CUDA: vllm.v1.worker.gpu_worker.Worker
  Ascend: vllm_ascend.worker.worker.NPUWorker
          vllm_ascend._310p.worker_310p.NPUWorker310
          vllm_ascend.xlite.xlite_worker.XliteWorker

attention backend:
  CUDA: 根据设备能力和配置在 FlashAttention / FlashInfer / Triton 等后端中选择
  Ascend: 根据 MLA/SFA/310P 分支固定映射到 AscendAttentionBackend 系列

图执行 wrapper:
  CUDA: vllm.compilation.cuda_graph.CUDAGraphWrapper
  Ascend: vllm_ascend.compilation.acl_graph.ACLGraphWrapper

compile backend:
  CUDA: 上游 torch.compile / inductor / vLLM compile 路径
  Ascend: vllm_ascend.compilation.compiler_interface.AscendCompiler
```

同时 Ascend 会主动清理或覆盖一批 GPU 专属配置，例如 FlashInfer/FlashAttention 相关开关、NVTX tracing、Nsight、NCCL KV transfer buffer 细节、ROCm partial prefill 配置等。这样做的目的不是改变 vLLM 调度语义，而是避免上游 GPU 参数被误带到 NPU 后端后进入无效路径。

---

### Step 2：NPUWorker 初始化设备、通信和自定义算子

```
NPUWorker.__init__()
  │
  ├─ adapt_patch()
  │    给 vLLM 做 Ascend 兼容 patch
  │
  ├─ ops.register_dummy_fusion_op()
  ├─ _register_atb_extensions()        # A5 外的 ATB 扩展
  ├─ register_ascend_customop()
  ├─ init_ascend_config()
  └─ check_ascend_device_type()

NPUWorker.init_device()
  │
  ├─ torch.npu.set_device("npu:{local_rank}")
  ├─ torch.npu.empty_cache()
  ├─ MemorySnapshot()
  ├─ init_distributed_environment(..., backend="hccl")
  ├─ ensure_model_parallel_initialized(TP, PP, PCP, DCP)
  ├─ init_ascend_model_parallel()
  ├─ init_device_properties_triton()
  └─ self.model_runner = NPUModelRunner(...)
```

到这里，NPU 设备、HCCL 通信域、Ascend 自定义算子和 `NPUModelRunner` 都已经准备好。

---

### Step 3：加载模型并准备图模式

```
NPUModelRunner.load_model()
  │
  ├─ get_model(vllm_config)
  │    复用 vLLM 的模型加载框架
  │
  ├─ 如启用 LoRA / EAGLE / EPLB, 初始化对应模块
  │
  └─ 如果 cudagraph_mode.has_full_cudagraphs()
       用 ACLGraphWrapper 包住 model
```

如果启用 full graph，`ACLGraphWrapper` 并不改变模型本身的计算逻辑，而是在 forward 时根据 `BatchDescriptor` 决定：

```
runtime_mode = NONE:
  直接 eager 执行

runtime_mode = FULL / PIECEWISE:
  第一次见到该 batch_descriptor:
    torch.npu.NPUGraph capture
  后续相同 batch_descriptor:
    aclgraph.replay()
```

---

## 第二幕：KV Cache 初始化 —— 先把页式内存池建好

### Step 0：测算可用 KV Cache 显存

vLLM 会先调用 worker 的 profile 流程，估算除 KV Cache 以外的内存占用：

```
NPUWorker.determine_available_memory()
  │
  ├─ 如果用户指定 --kv-cache-memory:
  │    跳过自动估算，但仍 profile_run() 用于编译/预热
  │
  └─ 否则:
       memory_profiling(...)
         └─ NPUModelRunner.profile_run()
              └─ dummy forward

可用于 KV Cache 的内存:
  requested_memory - non_kv_cache_memory
```

`profile_run()` 不只是测内存，也会触发一部分编译、warmup 和算子初始化。Ascend 版本还会在 warmup 后调用一个小的 ATB matmul，避免首个真实请求在 `ReshapeAndCache` 附近吃到冷启动开销。

和上游 GPU worker 相比，这里有两点变化：

```
设备内存统计:
  CUDA worker 用 torch.cuda / NVML 语义看 GPU 内存
  NPUWorker 用 torch.npu 和 MemorySnapshot 统计 NPU 内存

profile 副作用:
  上游 vLLM profile 主要用于估算 non-KV 内存并触发图/算子预热
  Ascend profile 还承担 torch_npu / ATB / custom op 的冷启动摊销
```

因此 Ascend 的 profile_run 更像“内存测算 + 后端预热”的合并阶段，不能只按普通 dry run 理解。

---

### Step 1：分配每层 KV Cache Tensor

KV Cache 的规格由 vLLM 的 `KVCacheConfig` 给出，vLLM-Ascend 在 `initialize_kv_cache()` 里真正分配 NPU tensor：

```
NPUWorker.initialize_from_config(kv_cache_config)
  └─ NPUModelRunner.initialize_kv_cache(kv_cache_config)
       │
       ├─ initialize_attn_backend(kv_cache_config)
       ├─ may_reinitialize_input_batch(kv_cache_config)
       ├─ initialize_kv_cache_tensors(kv_cache_config)
       │    ├─ _allocate_kv_cache_tensors()
       │    └─ _reshape_kv_cache_tensors()
       │
       └─ bind_kv_cache(...)
```

对普通 GQA/MHA attention，每层最终得到：

```
k_cache: (num_blocks, block_size, KVH, D)
v_cache: (num_blocks, block_size, KVH, D)

单层单 block 容量:
  K: block_size × KVH × D × dtype_size
  V: block_size × KVH × D × dtype_size

合计:
  2 × block_size × KVH × D × dtype_size
```

如果 `block_size=128, KVH=8, D=128, dtype=FP16`：

```
单层单 block:
  2 × 128 × 8 × 128 × 2B = 524288B = 512KB

L 层总计:
  512KB × L
```

和上游 vLLM 相比，KV block 的抽象协议没有变：

```
Scheduler / KVCacheManager 仍然只管理:
  request → logical block list → physical block id

attention backend 仍然只消费:
  block_table + slot_mapping + seq_lens
```

变化在后端 tensor layout 和 block_size 偏好：

```
上游 CUDA FlashAttention 常见 KV cache:
  kv_cache: (2, num_blocks, block_size, KVH, D)
  写入: reshape_and_cache_flash(...)
  block_size: 由 backend 选择，常见默认示例是 16

vLLM-Ascend GQA/MHA KV cache:
  k_cache: (num_blocks, block_size, KVH, D)
  v_cache: (num_blocks, block_size, KVH, D)
  写入: torch_npu._npu_reshape_and_cache(...)
  block_size: Ascend 默认 128；prefix cache / chunked prefill 开启时会强制回 128
```

和参考 MindIE 文档中的“一个 Block 包含全部层”不同，vLLM-Ascend 的实际实现仍是每个 attention layer 绑定自己的 K/V cache tensor；vLLM 的 block id 在所有层上语义一致，但物理 tensor 是按 layer 绑定的。

---

### Step 2：NPUInputBatch 持久化请求状态

`NPUModelRunner` 初始化时创建 `NPUInputBatch`：

```
NPUInputBatch
  │
  ├─ token_ids_cpu_tensor: (max_num_reqs, max_model_len)
  │    每个请求一行，保存 prompt token 和后续采样 token
  │
  ├─ num_tokens / num_prompt_tokens / num_computed_tokens
  │    请求级状态
  │
  ├─ MultiGroupBlockTable
  │    每个 KV cache group 一个 BlockTable
  │
  ├─ temperature / top_p / top_k / penalties
  │    采样参数
  │
  └─ sampling_metadata
```

`BlockTable` 里面最重要的是两个 tensor：

```
block_table:  (max_num_reqs, max_num_blocks_per_req)
slot_mapping: (max_num_batched_tokens + padding,)
```

`block_table[row, logical_block_idx] = physical_block_id`

`slot_mapping[token_idx] = physical_block_id * block_size + block_offset`

后面 attention 写 KV Cache 时只看 `slot_mapping`，读历史 KV 时主要看 `block_table + seq_lens`。

相比 `docs/vLLM全过程.md` 里描述的 `GPUModelRunner + InputBatch`，Ascend 版多了一个更明确的 NPU 持久 batch 层：

```
上游 vLLM:
  GPUModelRunner
    → InputBatch
    → CommonAttentionMetadata
    → backend-specific metadata

vLLM-Ascend:
  NPUModelRunner
    → NPUInputBatch
    → AscendCommonAttentionMetadata
    → AscendMetadata / MLA metadata / SFA metadata
```

它的作用不是重新调度请求，而是把 vLLM SchedulerOutput 翻译成 NPU 算子更容易消费的形态：`int32` slot mapping、NPU block table、Ascend attention state、ACL Graph bucket 所需的静态缓冲，以及 PCP/DCP 场景下的额外位置和通信 metadata。

---

## 第三幕：请求进入 —— SchedulerOutput 变成 NPU 输入

假设 tokenizer 得到：

```
原始文本: "今天天气怎么样"
token_ids = [t0, t1, t2, t3]
prompt_len = 4
```

vLLM Scheduler 决定本轮调度这个新请求的全部 4 个 prompt token：

```
SchedulerOutput
  total_num_scheduled_tokens = 4
  num_scheduled_tokens[req_001] = 4
  scheduled_new_reqs = [req_001]
```

模型 forward 前会进入：

```
NPUWorker.execute_model(scheduler_output)
  └─ NPUModelRunner.execute_model(scheduler_output)
       │
       ├─ _update_states(scheduler_output)
       │    更新 NPUInputBatch:
       │      token_ids_cpu[row, 0:4] = [t0,t1,t2,t3]
       │      block_table[row]        = 分配到的物理 block id
       │
       ├─ _prepare_inputs(...)
       ├─ _build_attention_metadata(...)
       ├─ _preprocess(...)
       ├─ set_ascend_forward_context(...)
       ├─ _model_forward(...)
       ├─ compute_logits(...)
       └─ execute_model_state = ...
```

---

### Step 1：提前提交 block table

`_prepare_inputs()` 一开始先做：

```
self.input_batch.block_table.commit_block_table(num_reqs)
```

也就是把 CPU 侧 block table 拷到 NPU。源码里明确写了这是一个优化点：先发起 block table copy，让后续 CPU 侧位置计算和 H2D 拷贝尽量重叠。

这和上游 vLLM 的输入准备思路一致，都是把 SchedulerOutput 展平成“本轮 token batch”。Ascend 的变化是更早把 block table 提交到 NPU，并围绕 NPU kernel 的输入类型组织缓冲，特别是后续 `_compute_slot_mapping_kernel` 和 `_npu_reshape_and_cache` 需要的 `int32` 地址信息。

---

### Step 2：计算 req_indices、positions、query_start_loc

Prefill 阶段只有一个请求，调度 4 个 token：

```
num_scheduled_tokens = [4]
req_indices          = [0, 0, 0, 0]

num_computed_tokens_cpu[0] = 0
query_pos                 = [0, 1, 2, 3]

positions = num_computed_tokens_cpu[req_indices] + query_pos
          = [0, 1, 2, 3]

query_start_loc = [0, 4]
seq_lens        = [4]
```

`positions` 会传给模型，用于 RoPE / M-RoPE / XD-RoPE。

`query_start_loc` 表示每个请求在扁平 token batch 中的起止位置：

```
request 0 的 token 范围:
  [query_start_loc[0], query_start_loc[1]) = [0, 4)
```

---

### Step 3：从 token table 取 input_ids

`NPUInputBatch` 的 token table 是二维：

```
token_ids_cpu_tensor: (max_num_reqs, max_model_len)

row 0:
  [t0, t1, t2, t3, ?, ?, ...]
```

`_prepare_inputs()` 用扁平下标取出本轮要跑的 token：

```
token_indices = positions + req_indices * max_model_len
              = [0, 1, 2, 3]

input_ids = token_ids_cpu_tensor.flatten()[token_indices]
          = [t0, t1, t2, t3]
```

然后把 `input_ids` 拷到 NPU：

```
input_ids.gpu[:4] = [t0, t1, t2, t3]
positions[:4]     = [0, 1, 2, 3]
```

---

### Step 4：计算 slot_mapping

假设 Scheduler 给 `req_001` 分配了物理 block：

```
block_table[row 0] = [Block_42, 0, 0, ...]
block_size = 128
```

每个 token 要写入 KV Cache 的 slot：

```
logical_block_idx = positions // block_size
                  = [0, 0, 0, 0]

block_offset = positions % block_size
             = [0, 1, 2, 3]

physical_block_id = block_table[row, logical_block_idx]
                  = [42, 42, 42, 42]

slot_mapping = physical_block_id * block_size + block_offset
             = [5376, 5377, 5378, 5379]
```

源码里这一步由 `BlockTable.compute_slot_mapping()` 调用 `_compute_slot_mapping_kernel` 在 NPU 上完成。

如果启用 PCP/DCP，公式会变成“虚拟 block + CP rank interleave”，后文单独讲。

---

### Step 5：判定 attention state

`NPUModelRunner._build_attn_state()` 根据每个请求已计算 token 数和本轮调度 token 数，把当前 batch 标记成不同状态：

```
if all(num_computed_tokens_cpu == 0):
    PrefillNoCache

elif all(num_scheduled_tokens == 1):
    DecodeOnly

elif all(num_valid_tokens == 1):
    SpecDecoding 或 ChunkedPrefill

elif enable_chunked_prefill:
    ChunkedPrefill

else:
    PrefillCacheHit
```

当前新请求第一次跑，状态是：

```
AscendAttentionState.PrefillNoCache
```

这决定后续 attention 算子如何解释 K/V：

```
PrefillNoCache:
  当前 Q/K/V 都来自本轮输入，不需要从历史 KV cache 读 K/V

PrefillCacheHit / ChunkedPrefill / DecodeOnly:
  当前新 K/V 要写入 cache，同时 attention 要通过 block_table 读历史 cache
```

这里是 vLLM-Ascend 相比上游 vLLM 较明显的建模变化之一。上游 attention metadata 通常用 prefill/decode 数量、query_start_loc、block_table 等字段表达状态；Ascend 额外显式引入 `AscendAttentionState`：

```
PrefillNoCache
PrefillCacheHit
DecodeOnly
ChunkedPrefill
SpecDecoding
```

这个枚举直接驱动 Ascend attention backend 的分支选择，因为 FIA、PA、MLA/SFA、PCP/DCP、graph capture dummy metadata 对输入形态的要求不同。换句话说，vLLM 的 token debt 调度语义没变，但 NPU backend 需要更早把这一轮 batch 归类成具体算子状态。

---

## 第四幕：Prefill 阶段 —— 一次 forward 处理 prompt token

### Step 0：构建 AttentionMetadata

`_build_attention_metadata()` 会先构建跨层共享的 `AscendCommonAttentionMetadata`：

```
AscendCommonAttentionMetadata
  query_start_loc     = [0, 4]         # NPU tensor
  query_start_loc_cpu = [0, 4]         # CPU tensor
  seq_lens            = [4]
  _seq_lens_cpu       = [4]
  num_reqs            = 1
  num_actual_tokens   = 4
  max_query_len       = 4
  block_table_tensor  = block_table.gpu
  slot_mapping        = [5376, 5377, 5378, 5379]
  positions           = [0, 1, 2, 3]
  attn_state          = PrefillNoCache
```

然后每个 KV cache group 的 builder 生成层级 metadata：

```
AscendAttentionMetadataBuilder.build(...)
  │
  ├─ split_decodes_and_prefills(...)
  ├─ AttentionMaskBuilder.get_attention_mask(...)
  └─ AscendMetadata(...)
       attn_mask
       attn_state
       actual_seq_lengths_q
       actual_seq_lengths_kv
       block_tables
       slot_mapping
       seq_lens
```

同一个 group 内多层共享同一份 metadata 对象，减少重复构建。

对应到上游 vLLM，`AscendCommonAttentionMetadata` 扮演的是 `CommonAttentionMetadata` 的 NPU 版扩展：

```
保留的通用字段:
  query_start_loc
  seq_lens
  block_table
  slot_mapping
  num_reqs / num_actual_tokens / max_query_len

Ascend 额外关心:
  attn_state
  NPU 上的 block table 和 slot_mapping dtype
  FIA/PA 的 actual_seq_lengths_q / actual_seq_lengths_kv
  ACL Graph capture/replay 的固定 buffer
  PCP/DCP/KVComp 所需的附加 metadata
```

所以 attention metadata 的总体结构仍接着 vLLM 的协议往下走，但字段是按 Ascend 算子的参数表重新组织的。

---

### Step 1：Embedding

模型 forward 仍然走 vLLM 的模型定义：

```
input_ids: (4,)
    │
    ▼
embedding
hidden_states: (4, H)
```

这一步和 GPU 版 vLLM 类似，只是 tensor device 是 NPU。

---

### Step 2：进入 Transformer Layer

每层大致仍是标准 decoder layer：

```
hidden_states (T, H)
  │
  ├─ RMSNorm
  ├─ QKV projection
  │    q: (T, QH, D)
  │    k: (T, KVH, D)
  │    v: (T, KVH, D)
  │
  ├─ RoPE / QK norm / model-specific position op
  ├─ Attention
  ├─ O projection
  ├─ residual
  ├─ MLP
  └─ residual
```

这里不要误解成 MindIE 文档里的固定“21 个算子 → 5 个融合 kernel”。vLLM-Ascend 的普通路径仍然沿用 PyTorch/vLLM module graph，只是关键子图会被替换为 NPU 算子，图模式下再做捕获和局部融合。

---

### Step 3：AscendAttentionBackend.forward()

attention 层会调用 Ascend backend：

```
AscendAttentionBackendImpl.forward(
  query:        (T, QH, D)
  key:          (T, KVH, D)
  value:        (T, KVH, D)
  kv_cache:     (k_cache, v_cache)
  attn_metadata
  output
)
```

如果 key/value 非空，先写 KV Cache：

```
reshape_and_cache(...)
  │
  └─ DeviceOperator.reshape_and_cache(...)
       └─ torch_npu._npu_reshape_and_cache(
            key=key[:num_actual_tokens],
            value=value[:num_actual_tokens],
            key_cache=k_cache,
            value_cache=v_cache,
            slot_indices=slot_mapping,
          )
```

对当前例子：

```
key:   (4, KVH, D)
value: (4, KVH, D)

slot_mapping:
  [5376, 5377, 5378, 5379]

写入每层 K/V cache:
  k_cache[Block_42, slots 0..3, :, :] = key[0..3]
  v_cache[Block_42, slots 0..3, :, :] = value[0..3]
```

这一步是 vLLM-Ascend 和普通 PyTorch attention 的关键差异：KV Cache 写入由专门的 NPU reshape/cache 算子完成，而不是 Python 循环或普通 scatter。

如果和上游 CUDA FlashAttention backend 对照，差异更具体：

```
上游 CUDA FlashAttention:
  key_cache, value_cache = kv_cache.unbind(0)
  reshape_and_cache_flash(
    key,
    value,
    key_cache,
    value_cache,
    slot_mapping,
    kv_cache_dtype,
    k_scale,
    v_scale,
  )

Ascend:
  key_cache, value_cache = kv_cache[0], kv_cache[1]
  torch_npu._npu_reshape_and_cache(
    key[:num_actual_tokens],
    value[:num_actual_tokens],
    key_cache,
    value_cache,
    slot_mapping,
  )
```

两者都遵守 `slot_mapping = physical_block_id * block_size + offset` 这个 vLLM 页式 KV 协议；不同的是 CUDA 路径调用 vLLM/FlashAttention 的 cache op，Ascend 路径调用 torch_npu 暴露的 NPU cache op，并把 `slot_mapping` 约束为 NPU 算子需要的类型和形态。

---

### Step 4：PrefillNoCache 的 Fused Infer Attention

然后进入 `forward_impl()`：

```
if DecodeOnly and using_paged_attention(...):
    _npu_paged_attention
else:
    npu_fused_infer_attention_score
```

PrefillNoCache 默认走：

```
torch_npu.npu_fused_infer_attention_score(
  query=query,
  key=key,
  value=value,
  atten_mask=attn_mask,
  block_table=None,
  input_layout="TND",
  block_size=128,
  actual_seq_lengths=[4],
  actual_seq_lengths_kv=[4],
  num_key_value_heads=KVH,
  num_heads=QH,
  scale=1/sqrt(D),
  sparse_mode=3,
)
```

语义上就是：

```
Q: (4, QH, D)
K: (4, KVH, D)
V: (4, KVH, D)

Attention:
  scores = Q × K^T
  causal mask
  softmax
  output = probs × V

输出:
  attn_output: (4, QH, D) → (4, H)
```

注意：即使 `PrefillNoCache` 不需要从历史 KV cache 读 K/V，前面的 `reshape_and_cache` 仍然会把本轮 prompt 的 K/V 写入 cache，给后续 decode 使用。

这里替换的是上游 GPU attention 的核心 kernel：

```
上游 CUDA 常见 prefill:
  flash_attn_varlen_func(...)
    使用 cu_seqlens_q / cu_seqlens_k
    block_table 只在 cache-hit / decode 等路径参与读历史 KV

Ascend PrefillNoCache:
  torch_npu.npu_fused_infer_attention_score(...)
    使用 actual_seq_lengths_q / actual_seq_lengths_kv
    使用 TND layout
    由 attn_state 决定 block_table 是否参与
```

因此不要把 Ascend 的 `npu_fused_infer_attention_score` 简单等同于 CUDA 的 FlashAttention API。它们完成的数学语义相同，都是 causal self-attention，但参数组织、layout 限制、workspace 和 graph update 机制都由各自后端决定。

---

### Step 5：层内剩余计算

attention 输出回到 vLLM 模型层：

```
attn_output: (4, H)
  │
  ├─ output projection
  ├─ residual
  ├─ post-attention norm
  ├─ MLP / MoE
  └─ residual

hidden_states: (4, H)
```

如果启用图编译或 fusion pass，部分子图可能被替换成 Ascend 融合算子：

```
AddRMSNorm + Quant
  → torch.ops.npu.npu_add_rms_norm_quant

QK RMSNorm + RoPE
  → torch.ops.vllm.qkv_rmsnorm_rope

Matmul + AllReduce + AddRMSNorm
  → torch.ops._C_ascend.matmul_allreduce_add_rmsnorm

Muls + Add
  → 对应 fusion pass

Sequence Parallelism / MoE Sequence Parallelism
  → 对 TP/SP 场景插入或替换通信相关子图
```

这些优化来自 `GraphFusionPassManager`，不是一个固定的全层“5 kernel”模板；是否生效取决于模型结构、dtype、compile range、parallel config 和 additional_config。

---

### Step 6：取 logits 并采样第一个输出 token

最后一层输出：

```
hidden_states: (4, H)
```

Prefill 后只需要每个请求最后一个 scheduled token 的 hidden state：

```
logits_indices = query_start_loc[1:] - 1
               = [3]

sample_hidden_states = hidden_states[logits_indices]
                     = hidden_states[[3]]
                     = (1, H)

logits = model.compute_logits(sample_hidden_states)
       = (1, vocab_size)
```

`execute_model()` 此时不直接返回 token，而是保存中间状态：

```
self.execute_model_state = ExecuteModelState(
  scheduler_output,
  logits,
  hidden_states,
  sample_hidden_states,
  attn_metadata,
  positions,
  ...
)

return None
```

随后 vLLM 调用：

```
NPUWorker.sample_tokens(grammar_output)
  └─ NPUModelRunner.sample_tokens(grammar_output)
       ├─ _sample(logits, spec_decode_metadata)
       ├─ _bookkeeping_sync(...)
       └─ ModelRunnerOutput(...)
```

Ascend 采样器做的事情：

```
AscendSampler
  │
  ├─ apply penalties
  │    Triton 可用时使用 Ascend penalties kernel
  │
  ├─ top-k / top-p
  │    A2/A3 上优先 torch.ops._C_ascend.npu_apply_top_k_top_p
  │
  ├─ softmax
  │
  └─ random_sample
       用 exponential 随机数 + argmax
       避免 torch.multinomial 带来的 CPU-NPU 同步
```

相比上游 vLLM 的采样闭环，Ascend 没有改变“logits → sample → update request”的控制流，改的是热点算子：

```
保留:
  penalties / top-k / top-p / softmax / random sample 的语义
  sample_tokens() 和 scheduler.update_from_output() 的闭环位置

替换:
  top-k/top-p 优先走 torch.ops._C_ascend.npu_apply_top_k_top_p
  penalties 可走 Ascend kernel
  random_sample 用 exponential noise + argmax，减少 torch.multinomial 同步
```

所以采样仍是 vLLM 主循环的一部分，不是 NPU attention 之后的离线后处理；Ascend 版本只是尽量把采样中的小而频繁的操作留在设备侧。

假设采样得到：

```
sampled_token_ids = [y0]
```

`_bookkeeping_sync()` 会把 token 写回 `NPUInputBatch`：

```
token_ids_cpu[row, 4] = y0
num_tokens[row]       = 5
output_token_ids     += [y0]
```

Prefill 完成时，每层 KV Cache 状态：

```
Layer 0:
  Block_42 slots 0..3: K/V 已写入

Layer 1:
  Block_42 slots 0..3: K/V 已写入

...

Layer L-1:
  Block_42 slots 0..3: K/V 已写入
```

---

## 第五幕：Decode 阶段 —— 每轮追加 1 个 token

### Decode 与 Prefill 的状态差异

下一轮输入是上一步采样的 token：

```
input token = y0
position    = 4
```

Scheduler 本轮通常只给这个请求调度 1 个 token：

```
num_scheduled_tokens = [1]
num_computed_tokens_cpu[row] > 0

_build_attn_state()
  → AscendAttentionState.DecodeOnly
```

和 Prefill 相比：

```
Prefill:
  Q/K/V 来自多个 prompt token
  attention 看本轮 prompt 内的 causal 前缀
  主要是大矩阵 GEMM + attention

Decode:
  Q/K/V 只来自 1 个新 token
  K/V 需要追加写入 cache
  Q 要看所有历史 K/V
  projection 更接近 GEMV, 很容易受 HBM 带宽限制
```

---

### Decode Step 1：准备 input_ids 和 slot_mapping

`_prepare_inputs()` 对 decode 的计算结果：

```
req_indices = [0]
query_pos   = [0]

num_computed_tokens_cpu[row] = 4
positions = [4]

token_indices = positions + row * max_model_len
              = [4]

input_ids = [y0]

query_start_loc = [0, 1]
seq_lens        = [5]   # 历史 4 + 当前 1
```

KV Cache 写入位置：

```
logical_block_idx = 4 // 128 = 0
block_offset      = 4 % 128  = 4
physical_block_id = 42

slot_mapping = [42 * 128 + 4] = [5380]
```

---

### Decode Step 2：QKV projection

进入模型后，每层仍先做 norm/projection：

```
hidden_states: (1, H)
  │
  └─ QKV projection
       q: (1, QH, D)
       k: (1, KVH, D)
       v: (1, KVH, D)
```

这一步在 decode 中通常是 memory-bound：每次只处理一个 token，矩阵乘从“矩阵 × 矩阵”退化为更小 batch 的矩阵计算，权重读取占主导。

vLLM-Ascend 本身不在这一层手写一个完整的 Transformer decode kernel，而是依靠：

```
torch_npu / ATB / 自定义线性算子
ACL Graph replay
FX fusion pass
weight prefetch 配置
TP/SP/通信融合
```

去降低 launch 和通信开销。

---

### Decode Step 3：追加写 KV Cache

和 Prefill 一样，只要 key/value 非空，就先写 cache：

```
_npu_reshape_and_cache(
  key=(1, KVH, D),
  value=(1, KVH, D),
  slot_indices=[5380],
)
```

写入后，所有层的 Block_42 变成：

```
slots:
  0     1     2     3     4
 [t0]  [t1]  [t2]  [t3]  [y0]
```

这里的 `[t]` 表示该 token 在该层的 K/V cache，而不是 token id 本身。

---

### Decode Step 4：用历史 KV 做 Attention

DecodeOnly 的普通路径仍然是 `npu_fused_infer_attention_score`：

```
_get_fia_params(...)
  key_cache:   (num_blocks, block_size, KVH, D)
  value_cache: (num_blocks, block_size, KVH, D)
  key   = key_cache.view(num_blocks, block_size, -1)
  value = value_cache.view(num_blocks, block_size, -1)

npu_fused_infer_attention_score(
  query=(1, QH, D),
  key=key_cache_view,
  value=value_cache_view,
  block_table=block_table[row],
  actual_seq_lengths=[1],
  actual_seq_lengths_kv=[5],
  input_layout="TND",
  block_size=128,
)
```

语义上：

```
Q_new:   第 5 个 token 的 query
K_all:   slots 0..4 的 key
V_all:   slots 0..4 的 value

output = Attention(Q_new, K_all, V_all)
```

如果满足特定条件，DecodeOnly 也可能走 paged attention：

```
using_paged_attention(runtime_shape, vllm_config) == True
  条件:
    speculative_config is None
    设备不是 A5
    cudagraph_mode == FULL_DECODE_ONLY
    runtime_shape 在 ascend_config.pa_shape_list 中

执行:
  torch_npu._npu_paged_attention(...)
```

源码注释说明，vLLM-Ascend 保留 `_npu_paged_attention` 是因为某些 shape 下它仍比 `npu_fused_infer_attention_score` 更快。

和上游 vLLM 的 decode attention 对照：

```
上游 CUDA:
  常见路径会在 FlashAttention / FlashInfer / Triton / PagedAttention 中选择
  CUDA Graph decode 时也会依赖稳定 batch shape 和对应 backend 的 graph support

vLLM-Ascend:
  DecodeOnly 主路径仍是 npu_fused_infer_attention_score
  只有 FULL_DECODE_ONLY + pa_shape_list 命中 + 非 A5 + 非 spec decode 等条件满足时
  才切到 torch_npu._npu_paged_attention
```

也就是说，Ascend 文档里出现的 “PagedAttention” 更多是 vLLM block table 协议和可选 NPU PA kernel 两层含义：调度侧一直是页式 KV cache；算子侧 decode 主路径未必使用 `_npu_paged_attention`。

---

### Decode Step 5：logits、采样、循环

Decode 每轮的 logits index 很简单：

```
query_start_loc = [0, 1]
logits_indices  = [0]

sample_hidden_states = hidden_states[[0]]
logits = model.compute_logits(sample_hidden_states)
next_token = AscendSampler(logits)
```

然后 `_bookkeeping_sync()` 把新 token 追加回 token table：

```
Step  输入位置  seq_len  slot       采样输出
----  --------  -------  ---------  --------
  1      4        5      Block42:4   y1
  2      5        6      Block42:5   y2
  3      6        7      Block42:6   y3
 ...    ...      ...        ...      ...
```

直到采样到 EOS、达到 max_tokens，或请求被取消。

---

## 第六幕：Chunked Prefill 与 Prefix Cache 命中

真实服务中经常不是“完整 prefill 一次跑完”。当 prompt 很长，或者 batch 中混入 decode 请求时，Scheduler 会切成 chunk：

```
请求已有:
  num_computed_tokens_cpu[row] = 1024

本轮继续 prefill:
  num_scheduled_tokens[row] = 512

attention state:
  ChunkedPrefill 或 PrefillCacheHit
```

此时 `_prepare_inputs()` 计算：

```
positions:
  [1024, 1025, ..., 1535]

seq_lens:
  [1536]

slot_mapping:
  根据 positions 写入新的 cache slots

block_table:
  包含 prefix 已有 block + 新分配 block
```

attention 后端会把 K/V cache 当作历史上下文：

```
current query:
  本轮 chunk 的 tokens

kv:
  block_table 指向的历史 blocks + 当前新写入 blocks

actual_seq_lengths_kv:
  seq_lens_list, 例如 [1536]
```

这就是 vLLM 的 chunked prefill / prefix cache 思路在 Ascend 上的落地方式：调度和 block 管理由 vLLM 负责，NPU backend 接收 `block_table + slot_mapping + seq_lens` 后执行对应 attention。

相比上游 vLLM，这一节最重要的变化不是“有没有 chunked prefill”，而是 chunk 到达后端后的分类和算子选择：

```
上游 vLLM:
  chunked prefill 仍由 Scheduler 的 token budget 截断产生
  attention backend 根据 metadata 处理 prefill/decode 混合 batch

vLLM-Ascend:
  Scheduler 仍然产生同样的 chunked prefill 语义
  _build_attn_state() 显式标成 ChunkedPrefill 或 PrefillCacheHit
  FIA 参数要同时描述当前 query chunk 和历史 KV cache
  PCP/DCP 开启后还要附带 CP slot_mapping、Q/KV all-gather 或 reduce-scatter 信息
```

另外，Ascend 的 `refresh_block_size()` 会在 prefix cache 或 chunked prefill 开启时把普通非 hybrid 模型的 `block_size` 拉回 128。这是和 `docs/vLLM全过程.md` 中常见 16-token block 示例不同的地方：协议相同，但 NPU 后端对 kernel block size 的偏好更强。

---

## 第七幕：ACL Graph —— 减少 Host Launch 开销

### Step 0：为什么需要 ACL Graph

Decode 阶段每轮 token 少，单次 forward 的 host launch、Python 调度、算子间 gap 更容易显得昂贵。ACL Graph 的目标是：

```
第一次:
  capture 一组固定 shape 的 NPU 执行图

后续:
  同 shape / bucket 的 batch 直接 replay
```

vLLM-Ascend 复用 vLLM 的 `CUDAGraphMode` 概念，但底层实现是 Ascend 的 `torch.npu.NPUGraph`。

这也是一个典型的“接口名保留，实现替换”的地方：

```
上游 vLLM:
  CUDAGraphMode
  CUDAGraphWrapper
  torch.cuda.CUDAGraph
  CUDA stream / capture / replay

vLLM-Ascend:
  仍使用 CUDAGraphMode 这个 vLLM 配置枚举
  wrapper 换成 ACLGraphWrapper
  capture/replay 换成 torch.npu.NPUGraph
  attention full graph replay 前还要更新 NPU graph task 参数
```

因此源码里很多字段仍叫 `cudagraph_*`，但在 Ascend 后端语境下实际指的是 ACL Graph 的 bucket、capture size、runtime mode 和 replay 逻辑。

---

### Step 1：选择 graph mode 和 batch bucket

`_determine_batch_execution_and_padding()` 会根据当前 batch 决定：

```
输入:
  num_tokens
  num_reqs
  max_num_scheduled_tokens
  是否 uniform decode
  是否 LoRA
  是否 encoder 输入
  DP padding 需求

输出:
  cudagraph_mode
  batch_descriptor
  num_tokens_padded
```

如果 runtime token 数不在 capture sizes 里，会 pad 到合适 bucket，或退回 eager。

Ascend 平台还会裁剪 capture sizes，因为 ACL Graph 受 stream 资源约束。官方设计文档里提到，piecewise 模式大约会按层数消耗 graph 资源，因此可捕获的 bucket 不能无限多。

相比 CUDA Graph，Ascend 在配置规范化上更保守：

```
FULL_AND_PIECEWISE → PIECEWISE
encoder-decoder    → PIECEWISE
use_inductor       → False
ASCEND_LAUNCH_BLOCKING=1 时禁止 ACL Graph
piecewise capture sizes 会按 ACL stream budget 裁剪
```

上游 vLLM 的 bucket 生成逻辑仍是基础，但 `update_aclgraph_sizes()` 会再根据模型层数、通信形态、HCCL 展开方式和 stream 预算缩小可捕获集合。

---

### Step 2：capture / replay

模型若被包进 `ACLGraphWrapper`，forward 时逻辑是：

```
ACLGraphWrapper.__call__()
  │
  ├─ runtime_mode == NONE:
  │    runnable(*args, **kwargs)
  │
  ├─ batch_descriptor 第一次出现:
  │    aclgraph = torch.npu.NPUGraph()
  │    with torch.npu.graph(aclgraph, pool=graph_pool):
  │        output = runnable(*args, **kwargs)
  │    cache[batch_descriptor] = aclgraph
  │
  └─ batch_descriptor 已存在:
       torch.npu.current_stream().synchronize()
       aclgraph.replay()
       return cached output weak ref
```

这里和普通 PyTorch eager 的主要差异是：后续相同 bucket 不再由 Python 逐算子 launch。

---

### Step 3：Full Graph 下 attention 参数仍要更新

Full graph 的难点是 attention 的 runtime metadata 每轮都变：

```
block_table
seq_lens
actual_seq_lengths_q
actual_seq_lengths_kv
slot_mapping
workspace
```

vLLM-Ascend 的处理方式：

```
capture 时:
  attention backend 记录:
    attn_params
    graph task handle
    external event
    workspace

replay 前:
  update_full_graph_params(...)
    └─ AscendAttentionBackendImpl.update_graph_params(...)
         ├─ torch.npu.graph_task_update_begin(...)
         ├─ 重新下发 npu_fused_infer_attention_score / _npu_paged_attention 参数
         └─ torch.npu.graph_task_update_end(...)
```

所以 Ascend full graph 不是“metadata 固定不变”，而是“图结构固定，attention task 参数在 replay 前被更新”。

---

## 第八幕：npugraph_ex 与 FX 融合 —— 不是整层固定模板，而是模式匹配替换

`AscendCompiler` 是 vLLM-Ascend 的自定义 compiler：

```
NPUPlatform.get_compile_backend()
  → vllm_ascend.compilation.compiler_interface.AscendCompiler

AscendCompiler.compile(...)
  │
  ├─ enable_npugraph_ex=True:
  │    torchair.get_npu_backend(...)
  │    mode = "reduce-overhead"
  │
  └─ enable_npugraph_ex=False:
       GraphFusionPassManager(...)
```

默认 `AscendCompilationConfig.enable_npugraph_ex=True`。

可用 fusion pass 包括：

```
AddRMSNormQuantFusionPass
  AddRMSNorm + Quant → npu_add_rms_norm_quant

QKNormRopeFusionPass
  Q/K split + Q RMSNorm + K RMSNorm + RoPE
    → qkv_rmsnorm_rope

MatmulAllReduceAddRMSNormPass
  matmul + tensor_parallel_all_reduce + add_rmsnorm
    → matmul_allreduce_add_rmsnorm

MulsAddFusionPass
  muls + add 类模式融合

SequenceParallelismPass / SequenceParallelismMoePass
  为 SP 场景改写相关子图
```

这部分最适合和 MindIE 做对比：

```
MindIE:
  倾向服务栈内 GE 编译整图/大粒度融合

vLLM-Ascend:
  以 vLLM runtime 为主
  attention/reshape/cache/采样/通信使用 Ascend 算子
  图模式通过 ACL Graph 捕获 runtime bucket
  FX pass 对匹配到的局部模式做融合
```

---

## 第九幕：Context Parallel —— 长上下文和多卡时 slot_mapping 怎么变

当启用 PCP / DCP 时，KV Cache 会沿序列维度分片。源码里的关键文件是：

```
vllm_ascend/worker/block_table.py
vllm_ascend/worker/pcp_utils.py
vllm_ascend/attention/context_parallel/*
```

CP 的基本变量：

```
pcp_size = prefill_context_parallel_size
dcp_size = decode_context_parallel_size

cp_size = pcp_size * dcp_size
cp_rank = pcp_rank * dcp_size + dcp_rank

cp_kv_cache_interleave_size 默认 = 1
```

普通 slot_mapping：

```
block_idx    = position // block_size
block_offset = position % block_size
slot         = block_table[block_idx] * block_size + block_offset
```

CP slot_mapping：

```
virtual_block_size = block_size * cp_size

logical_block_idx = position // virtual_block_size
virtual_offset    = position % virtual_block_size

target_rank =
  (virtual_offset // cp_kv_cache_interleave_size) % cp_size

local_offset =
  (virtual_offset // (cp_size * cp_kv_cache_interleave_size))
    * cp_kv_cache_interleave_size
  + virtual_offset % cp_kv_cache_interleave_size

如果 target_rank == 当前 rank:
  slot = local_block_id * block_size + local_offset
否则:
  slot = -1
```

也就是说，每张卡只保存自己负责的 token KV，其他 token 的 slot 在本 rank 上标记为无效。

PCP / DCP 对 attention 的影响：

```
PCP:
  prefill 阶段按 sequence 切分输入
  使用 head-tail 分片平衡计算量
  必要时 all-gather KV 或 Q 来完成 attention

DCP:
  复用 TP 通信域
  消除 decode 阶段 KV cache 冗余存储
  decode 时需要在 DCP group 内交换 Q / 输出 / LSE
```

这部分是 vLLM-Ascend 面向长上下文和多卡部署的重要扩展，不影响单卡最小路径的理解。

和上游 vLLM 的普通 TP/DCP 语义相比，Ascend 这里的主要增量是把 CP 做成 NPU 后端的一等路径：

```
PCP:
  目标: 长 prompt prefill 降 TTFT
  做法: 沿 sequence 切 query，使用 head-tail 分片平衡 causal attention 计算
  代价: prefill attention 需要 KV 或 Q 的跨 PCP group 聚合

DCP:
  目标: decode 阶段减少 KV cache 冗余，提高可服务 batch
  做法: 复用 TP 通信域，KV cache 沿 sequence 在 DCP group 内分片
  代价: decode/chunked prefill 需要 Q、输出或 LSE 的 group 内通信

共同变化:
  block_table 里的一个逻辑位置变成 CP virtual block
  slot_mapping 在非本 rank token 上写 -1
  attention backend 必须做 CP-aware 的 FIA / online softmax 更新
```

所以 CP 不是单纯把 `tensor_parallel_size` 改大；它改变的是 KV cache 的存储归属和 attention 的跨卡信息交换方式。

---

## 第十幕：KV Cache 生命周期全貌

用一个请求的生命周期串起来：

```
T0: 请求进入
    Scheduler 为 req_001 建立 request state

T1: Scheduler 分配 KV blocks
    block_table[row] = [Block_42, ...]

T2: Prefill
    _prepare_inputs:
      input_ids  = prompt tokens
      positions  = [0..prompt_len-1]
      slot_mapping 指向 Block_42 slots

    每层 attention:
      _npu_reshape_and_cache 写 prompt K/V
      npu_fused_infer_attention_score 计算 prefill attention

    sampler:
      生成第一个 output token
      写回 token_ids_cpu

T3: Decode loop
    每轮:
      input_ids  = 上轮 output token
      position   = 当前 seq_len
      slot       = 下一个 KV slot
      写入新 K/V
      用 block_table + seq_lens 读取历史 K/V
      采样下一个 token

T4: 请求结束
    vLLM Scheduler / KV cache manager 释放或保留 blocks
      普通请求: block 归还空闲池
      prefix caching: 可复用 prefix block 可能保留
      KV connector / KV pool: 可异步 put/get 到外部 KV 存储
```

KV cache 内存不是每次请求结束都清零；block 被释放后会被后续请求覆盖。正确性依赖 block table 和 slot mapping，而不是物理内存是否清空。

---

## 全过程一图总结

```
用户请求
  │
  ▼
vLLM API / LLM.generate
  │
  ▼
vLLM V1 Scheduler
  │
  ├─ 连续批处理
  ├─ prefix cache 命中
  ├─ chunked prefill
  ├─ block 分配 / 释放
  └─ SchedulerOutput
        │
        ▼
NPUWorker.execute_model
  │
  ├─ 初始化 NPU / HCCL / custom ops
  └─ NPUModelRunner.execute_model
        │
        ├─ _update_states
        │    token_ids_cpu / block_table / request 状态
        │
        ├─ _prepare_inputs
        │    input_ids
        │    positions
        │    query_start_loc
        │    seq_lens
        │    slot_mapping
        │
        ├─ _build_attention_metadata
        │    AscendCommonAttentionMetadata
        │    AscendMetadata per attention group
        │
        ├─ _determine_batch_execution_and_padding
        │    eager / ACL Graph bucket / padding
        │
        ├─ set_ascend_forward_context
        │
        └─ model.forward
              │
              ├─ Embedding / linear / norm / MLP
              │
              ├─ AscendAttentionBackend.forward
              │    │
              │    ├─ _npu_reshape_and_cache
              │    │    写 K/V cache
              │    │
              │    ├─ npu_fused_infer_attention_score
              │    │    Prefill / Chunked / Decode 主路径
              │    │
              │    └─ _npu_paged_attention
              │         特定 Decode full graph shape
              │
              └─ hidden_states
                    │
                    ▼
compute_logits
  │
  ▼
AscendSampler
  │
  ├─ penalties
  ├─ top-k / top-p
  ├─ softmax
  └─ exponential sampling 避免 CPU-NPU 同步
        │
        ▼
ModelRunnerOutput
  │
  ├─ sampled_token_ids
  ├─ logprobs
  ├─ prompt_logprobs
  └─ kv_connector_output
        │
        ▼
Scheduler 更新请求状态
  │
  ├─ 未结束: 下一轮 Decode
  └─ 结束: 释放 / 复用 / 外部 KV pool 保存 blocks
```

如果把这条路径和 `docs/vLLM全过程.md` 的上游 vLLM 路径逐段对齐，可以看到：

```
没有变化或基本复用:
  LLM.generate / vllm serve 入口
  V1 Scheduler 的 token debt 模型
  Continuous batching
  prefix cache / chunked prefill 的调度语义
  KV block table / slot mapping 协议
  vLLM model.forward 的主体 module graph
  logits / sample / scheduler update 闭环

Ascend 替换或新增:
  Platform: CUDAPlatform → NPUPlatform
  Worker: GPU Worker → NPUWorker
  Runner: GPUModelRunner → NPUModelRunner + NPUInputBatch
  Attention: FlashAttention/FlashInfer/Triton → Ascend FIA/PA/MLA/SFA
  KV write: reshape_and_cache_flash → _npu_reshape_and_cache
  Graph: CUDAGraphWrapper → ACLGraphWrapper
  Compiler: 上游 compile backend → AscendCompiler / npugraph_ex / FX fusion pass
  Sampling: 通用采样 kernel → AscendSampler 的 NPU top-k/top-p/random path
  Parallel: 普通 TP/PP/DP → HCCL + PCP/DCP/SP/FlashComm/MoE 通信融合
  Memory/KV transfer: NPU memory profile + Ascend KV connector/KV pool 适配
```

---

## 与 MindIE 文档中那条路径的关键差异

```
维度                    MindIE 文档路径                  vLLM-Ascend 路径
──────────────────────  ─────────────────────────────  ─────────────────────────────
调度层                  MindIE 服务/调度                vLLM V1 Scheduler
KV block 语义            文档示例中 block 覆盖全层        vLLM 每层绑定 K/V cache tensor
默认 block_size          示例 128                        Ascend 默认/常用 128
Prefill attention        GE/FlashAttention 融合描述       npu_fused_infer_attention_score
Decode attention         FlashDecoding 描述               FIA 主路径, PA 特定 shape 可选
KV 写入                  文档描述为 attention 内融合       代码中显式 reshape_and_cache
图执行                   GE 整图编译                      ACL Graph capture/replay bucket
局部融合                 GE pass                          npugraph_ex / FX pattern pass
采样                     服务栈采样                       AscendSampler, 避免 multinomial 同步
多卡长上下文             MindIE 自身并行策略              vLLM TP/PP + Ascend PCP/DCP/SP
```

所以阅读 vLLM-Ascend 时，最重要的不是找一个“固定 5 kernel Transformer layer”，而是沿着这四条数据线看：

```
1. token 数据线:
   token_ids_cpu → input_ids → hidden_states → logits → sampled_token_ids

2. 位置数据线:
   num_computed_tokens → positions → RoPE → query_start_loc

3. KV 数据线:
   block_table → slot_mapping → reshape_and_cache → FIA/PA 读取历史 KV

4. 图执行数据线:
   batch_descriptor → ACL Graph capture/replay → attention graph params update
```

这四条线合起来，就是 vLLM-Ascend 一次 Transformer 推理的全过程。

---

## 参考源码与文档

- `vLLM/vllm-upstream/vllm/platforms/cuda.py`: 上游 CUDA platform、worker class、attention backend 选择、CUDA Graph wrapper。
- `vLLM/vllm-upstream/vllm/platforms/interface.py`: 上游 platform 抽象和 block_size/backend 对齐逻辑。
- `vLLM/vllm-upstream/vllm/v1/attention/backends/flash_attn.py`: 上游 FlashAttention backend、`reshape_and_cache_flash`、prefill/decode attention 对照。
- `vLLM/vllm-ascend/setup.py`: 插件 entry point 与包构建。
- `vLLM/vllm-ascend/vllm_ascend/__init__.py`: `register()` 返回 NPU platform。
- `vLLM/vllm-ascend/vllm_ascend/platform.py`: `NPUPlatform`、配置修正、worker class、attention backend、ACL Graph wrapper。
- `vLLM/vllm-ascend/vllm_ascend/worker/worker.py`: `NPUWorker` 初始化、显存 profile、KV cache 初始化、execute/sample 入口。
- `vLLM/vllm-ascend/vllm_ascend/worker/model_runner_v1.py`: `_prepare_inputs()`、`_build_attention_metadata()`、`execute_model()`、`sample_tokens()`、KV cache tensor 分配。
- `vLLM/vllm-ascend/vllm_ascend/worker/npu_input_batch.py`: `NPUInputBatch` 持久 batch 状态。
- `vLLM/vllm-ascend/vllm_ascend/worker/block_table.py`: block table 和 slot mapping。
- `vLLM/vllm-ascend/vllm_ascend/attention/attention_v1.py`: Ascend GQA/MHA attention backend、FIA、PA、reshape/cache。
- `vLLM/vllm-ascend/vllm_ascend/attention/utils.py`: `AscendCommonAttentionMetadata`、paged attention 判定、CP metadata。
- `vLLM/vllm-ascend/vllm_ascend/device/device_op.py`: `_npu_reshape_and_cache` 等设备算子适配。
- `vLLM/vllm-ascend/vllm_ascend/sample/sampler.py`: Ascend sampler、top-k/top-p、随机采样。
- `vLLM/vllm-ascend/vllm_ascend/compilation/acl_graph.py`: ACL Graph capture/replay 和 graph params。
- `vLLM/vllm-ascend/vllm_ascend/compilation/compiler_interface.py`: `AscendCompiler` 和 npugraph_ex 接入。
- `vLLM/vllm-ascend/vllm_ascend/compilation/graph_fusion_pass_manager.py`: FX fusion pass manager。
- `vLLM/vllm-ascend/docs/source/developer_guide/Design_Documents/ModelRunner_prepare_inputs.md`: 官方输入准备说明。
- `vLLM/vllm-ascend/docs/source/developer_guide/Design_Documents/ACL_Graph.md`: 官方 ACL Graph 设计说明。
- `vLLM/vllm-ascend/docs/source/developer_guide/Design_Documents/npugraph_ex.md`: 官方 npugraph_ex 说明。
- `vLLM/vllm-ascend/docs/source/developer_guide/Design_Documents/context_parallel.md`: 官方 PCP/DCP 说明。
- `vLLM/vllm-ascend/docs/source/developer_guide/Design_Documents/KV_Cache_Pool_Guide.md`: 官方 KV Cache Pool 说明。
- `vLLM/vllm-ascend/docs/source/quick_start.md`: 官方 quick start 和支持设备说明。
