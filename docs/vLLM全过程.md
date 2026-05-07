# vLLM 如何加速一次 Transformer 推理：从 Prefill 到 Decode 的全过程拆解

> 本文参考 `docs/MindIE加速Transformer推理全过程拆解.md` 的写法，用一个具体请求追踪 vLLM v1 在 Prefill 和 Decode 两个阶段到底做了什么。重点不是复述概念，而是把请求状态、调度预算、KV Cache block、slot mapping、attention backend、CUDA graph 和采样闭环串成一条执行链。

---

## 场景设定

```
源码基线: vLLM/vllm-upstream @ 92a7c121b62a1484b68c0a27d1ecefd1a84f78fc
主线版本: vLLM v1
模型示例: Qwen3/Qwen2 风格 decoder-only Transformer
硬件示例: 单张 NVIDIA GPU
精度示例: BF16/FP16 权重 + 激活
输入文本: "今天天气怎么样"
期望输出: "今天北京天气晴朗，气温28度。"

假设 tokenizer 结果:
  prompt token_ids = [104169, 99476, 56278, 104949]
  prompt_len = 4

KV Cache 示例配置:
  block_size = 16 tokens/block
  num_layers = 32
  num_kv_heads = 4
  head_size = 128
  dtype = FP16

单个 KV block 体积:
  page_size_bytes = 2(K+V) x block_size x num_kv_heads x head_size x dtype_size
                  = 2 x 16 x 4 x 128 x 2
                  = 32 KB / layer
  32 层合计约 1 MB
```

这组数字只是为了让流程可视化。vLLM 实际会在 `EngineCore` 初始化时先加载模型、profiling 可用显存，再由 `get_kv_cache_configs(...)` 生成 `KVCacheConfig`，最后把 KV cache 初始化到 worker 上。

关键源码入口：

- `vllm/v1/engine/llm_engine.py::LLMEngine`
- `vllm/v1/engine/core.py::EngineCore`
- `vllm/v1/core/sched/scheduler.py::Scheduler`
- `vllm/v1/core/kv_cache_manager.py::KVCacheManager`
- `vllm/v1/worker/gpu/model_runner.py::GPUModelRunner`
- `vllm/model_executor/layers/attention/attention.py::Attention`
- `vllm/v1/attention/backends/*`

---

## 第一幕：Prefill 阶段 —— prompt token 被第一次算出来

### Step 0：请求进入 Engine，变成可调度的 Request

```
原始文本: "今天天气怎么样"
     │
     ▼
LLMEngine.add_request(...)
     │
     ├─ InputProcessor.process_inputs(...)
     │    - tokenizer / prompt 标准化
     │    - sampling params 标准化
     │    - 多模态 / LoRA / priority 等请求属性归一化
     │
     ▼
EngineCore.add_request(...)
     │
     ▼
Scheduler.add_request(...)
     │
     ▼
Request:
  request_id = "req_001"
  prompt_token_ids = [104169, 99476, 56278, 104949]
  output_token_ids = []
  spec_token_ids = []
  num_prompt_tokens = 4
  num_computed_tokens = 0
  num_tokens_with_spec = 4
  status = WAITING
```

vLLM v1 里最重要的状态不是“这是 prefill 请求”这个标签，而是：

```
还需要计算多少 token =
  num_tokens_with_spec + num_output_placeholders - num_computed_tokens
```

首次进入时：

```
num_new_tokens = 4 + 0 - 0 = 4
```

所以调度器知道：这个请求要推进 4 个 token。

---

### Step 1：EngineCore 主循环触发一轮 schedule -> execute -> sample -> update

`EngineCore.step()` 的主循环非常短，但它串起了整套系统：

```
EngineCore.step()
  │
  ├─ scheduler.schedule()
  │    生成 SchedulerOutput
  │
  ├─ model_executor.execute_model(scheduler_output, non_block=True)
  │    worker / GPUModelRunner 执行模型前向
  │
  ├─ scheduler.get_grammar_bitmask(...)
  │    structured output 约束可插入这里
  │
  ├─ future.result()
  │    等 GPU 前向完成
  │
  ├─ model_executor.sample_tokens(...)
  │    若 execute_model 只完成前向，则单独采样
  │
  └─ scheduler.update_from_output(...)
       把新 token 写回 Request 状态
```

注意这里有两个设计点：

1. 调度、模型前向、采样、状态更新是每轮迭代的闭环。
2. `execute_model()` 和 `sample_tokens()` 可以拆开，给 async scheduling、pipeline parallel、spec decode 留出空间。

---

### Step 2：Scheduler 不是切 phase，而是在追 token debt

`Scheduler.schedule()` 源码注释直接说明：调度器内部没有严格的 decoding phase 或 prefill phase。每个 request 只有 `num_computed_tokens` 和 `num_tokens_with_spec`。

首次请求在 waiting 队列中，调度器会做：

```
WAITING queue:
  req_001:
    num_computed_tokens = 0
    num_tokens_with_spec = 4

调度预算:
  token_budget = max_num_scheduled_tokens
  running slots <= max_num_running_reqs

prefix cache 查询:
  get_computed_blocks(req_001)
    若无命中: num_new_local_computed_tokens = 0
    若有完整 block 命中: 可以跳过对应前缀

本例无命中:
  num_computed_tokens = 0
  num_new_tokens = request.num_tokens - num_computed_tokens
                 = 4 - 0
                 = 4
```

如果 prompt 很长，`long_prefill_token_threshold` 和 `enable_chunked_prefill` 会决定能不能只推进一段 prompt。也就是说，chunked prefill 不是另一套系统，而是同一个 `num_new_tokens` 被 token budget 截断后的结果。

---

### Step 3：KVCacheManager 分配 KV block

调度器不会直接分配 GPU tensor。它调用：

```
KVCacheManager.allocate_slots(
  request=req_001,
  num_new_tokens=4,
  num_new_computed_tokens=0,
  new_computed_blocks=empty,
  num_lookahead_tokens=0,
)
```

`allocate_slots()` 的语义可以理解成一条 token 布局带：

```
----------------------------------------------------------------------
| < comp > | < new_comp > | < ext_comp > | < new > | < lookahead > |
----------------------------------------------------------------------

本例:
  comp      = 0       已经计算过的 token
  new_comp  = 0       prefix cache 新命中的完整 block
  ext_comp  = 0       外部 KV connector 已有的 token
  new       = 4       本轮真正要算的 prompt token
  lookahead = 0       speculative decode 预留 token
```

假设 block_size=16，4 个 token 需要 1 个逻辑 block：

```
BlockPool:
  free:  [Block_42, Block_43, ...]
  used:  []

allocate_slots(req_001, 4):
  取出 Block_42
  req_001 -> [Block_42]

Block_42:
  slot 0..3   将在 Prefill 前向中写入 K/V
  slot 4..15  暂空
```

这一层对应 vLLM 的“PagedAttention 系统协议”：请求只持有 block id 列表，不关心底层 KV cache tensor 的真实 stride 和 layout。

---

### Step 4：SchedulerOutput 描述这一轮要算什么

调度完成后，不是直接调用模型，而是生成一份设备执行描述：

```
SchedulerOutput:
  scheduled_new_reqs:
    req_001

  num_scheduled_tokens:
    req_001 -> 4

  req_to_new_blocks:
    req_001 -> [Block_42]

  scheduled_spec_decode_tokens:
    {}

  total_num_scheduled_tokens = 4
```

这份输出是 scheduler 和 worker 的边界。scheduler 只说“哪些请求、多少 token、分到哪些 block”；worker 负责把它翻译成 GPU 能吃的扁平 tensor 和 attention metadata。

---

### Step 5：GPUModelRunner 把请求批次压平成 InputBatch

`GPUModelRunner.execute_model()` 开始后，先更新 worker 常驻 request state：

```
finish_requests(...)
free_states(...)
add_requests(...)
update_requests(...)
block_tables.apply_staged_writes()
```

然后准备本轮输入：

```
prepare_inputs(scheduler_output, batch_desc)
```

对于本例：

```
num_tokens = 4
num_reqs = 1

InputBatch:
  req_ids = ["req_001"]
  input_ids = [104169, 99476, 56278, 104949]
  positions = [0, 1, 2, 3]
  query_start_loc = [0, 4]
  seq_lens = [4]
  logits_indices = [3]
```

这里的 `logits_indices=[3]` 很关键：prefill 会算 4 个位置的 hidden states，但采样第一个输出 token 只需要最后一个位置的 logits。

vLLM 还会把 decode 请求排在 prefill 请求前面：

```
req_ids = sorted(num_tokens_per_req, key=num_tokens_per_req.get)
```

因为 decode 通常每个请求只算 1 个 token，排在前面更利于常见 decode batch 的图执行和元数据处理。

---

### Step 6：block table 和 slot mapping 把逻辑页翻成物理槽位

接着 runner 调用：

```
prepare_attn(input_batch)
  │
  ├─ block_tables.gather_block_tables(...)
  │    得到每个请求拥有的物理 block id 列表
  │
  └─ block_tables.compute_slot_mappings(...)
       把每个 token position 映射到 KV cache 物理 slot
```

本例的映射可以写成：

```
req_001 block table:
  logical block 0 -> physical Block_42

positions:
  token 0 -> position 0
  token 1 -> position 1
  token 2 -> position 2
  token 3 -> position 3

slot mapping:
  slot_id = block_id * block_size + block_offset

  token 0 -> Block_42 slot 0
  token 1 -> Block_42 slot 1
  token 2 -> Block_42 slot 2
  token 3 -> Block_42 slot 3
```

也就是：

```
Scheduler 视角:
  req_001 -> [Block_42]

Kernel 视角:
  flat token 0 写 KV 到 physical slot 42*16+0
  flat token 1 写 KV 到 physical slot 42*16+1
  flat token 2 写 KV 到 physical slot 42*16+2
  flat token 3 写 KV 到 physical slot 42*16+3
```

没有 `block_table` 和 `slot_mapping`，page 化 KV 只能停留在调度层，attention kernel 不知道该读写哪块显存。

---

### Step 7：构建 attention metadata，选择具体 backend

runner 再把这些字段交给 model state：

```
attn_metadata = model_state.prepare_attn(
  input_batch,
  cg_mode,
  block_tables,
  slot_mappings,
  attn_groups,
  kv_cache_config,
)
```

attention metadata 会包含：

```
query_start_loc = [0, 4]
seq_lens        = [4]
max_query_len   = 4
max_seq_len     = 4
block_table     = [[42, 0, 0, ...]]
slot_mapping    = [Block_42:0, Block_42:1, Block_42:2, Block_42:3]
```

这一步是 backend 分流点。高层调度器不知道最终用 FlashAttention、Triton、FlashInfer 还是 native paged attention；它只维护 block 协议。真正 backend 在 `vllm/v1/attention/backends/*` 中落地。

---

### Step 8：模型前向开始，Embedding 查表

`execute_model()` 最终调用：

```
with set_forward_context(attn_metadata, ..., slot_mapping=slot_mappings_by_layer):
    model_output = self.model(**model_inputs)
```

进入模型后，以 Qwen2/Qwen3 类结构为例：

```
input_ids: (4,)
     │
     ▼
Embedding
     │
     ▼
hidden_states: (4, hidden_size)
     │
     ▼
for layer in layers:
    hidden_states, residual = layer(positions, hidden_states, residual)
```

vLLM 的模型文件只负责把 Transformer 结构用 vLLM 原语拼出来；调度、KV cache、attention backend、采样都在模型文件外部。

---

### Step 9：进入第 0 层 Transformer Block

单层结构可以简化为：

```
X (4, hidden)
  │
  ├─ RMSNorm
  │
  ├─ QKV Parallel Linear
  │    Q: (4, num_heads, head_dim)
  │    K: (4, num_kv_heads, head_dim)
  │    V: (4, num_kv_heads, head_dim)
  │
  ├─ RoPE(Q, K)
  │
  ├─ Attention.forward(Q, K, V)
  │
  ├─ O projection + residual
  │
  ├─ RMSNorm
  │
  ├─ MLP: gate/up/down
  │
  └─ residual
```

这里和 MindIE 的区别很大：

- MindIE 文档重点是 GE 编译器把 21 个操作融合成 5 个大 kernel。
- vLLM 的重点是 runtime 系统把许多请求的 token 混排成一个 batch，并把 KV cache 做成 page/block 协议。

也就是说，vLLM 不只追求“单个请求单层更少 kernel”，还追求：

```
多个请求共享一次调度
多个 token 共享一次 metadata 准备
多个序列共享一个 attention backend 调用
KV 显存以 block 复用和回收
decode 形状稳定时走 CUDA graph
```

---

### Step 10：Attention.forward 更新 KV cache，再做 attention

`vllm/model_executor/layers/attention/attention.py::Attention.forward()` 不读取 scheduler。它从 forward context 拿到当前层的：

```
attn_metadata
slot_mapping
kv_cache view
```

然后执行：

```
query = query.view(-1, num_heads, head_size)
key   = key.view(-1, num_kv_heads, head_size)
value = value.view(-1, num_kv_heads, head_size)

unified_kv_cache_update(key, value, layer_name)
unified_attention_with_output(query, key, value, output, layer_name)
```

如果落到 FlashAttention backend，典型 KV cache 形状是：

```
kv_cache:
  [2, num_blocks, block_size, num_kv_heads, head_size]

key_cache, value_cache = kv_cache.unbind(0)
```

本例第 0 层写入：

```
Layer 0 / Block_42:
  K slot 0 <- token 0 的 K
  K slot 1 <- token 1 的 K
  K slot 2 <- token 2 的 K
  K slot 3 <- token 3 的 K

  V slot 0 <- token 0 的 V
  V slot 1 <- token 1 的 V
  V slot 2 <- token 2 的 V
  V slot 3 <- token 3 的 V
```

attention 计算时，backend 使用：

```
q = query[:num_actual_tokens]
k = key_cache
v = value_cache
cu_seqlens_q = query_start_loc
seqused_k = seq_lens
block_table = block_table
causal = True
```

也就是：Q 是本轮新算出来的 token，K/V 则通过 block table 指向 page 化 KV cache。

---

### Step 11：32 层执行完毕，KV cache 全层填好

Prefill 结束后，每一层都写入了同一批 token 的 K/V：

```
req_001 -> [Block_42]

Block_42:
  Layer 0:  K[slot 0..3]  V[slot 0..3]
  Layer 1:  K[slot 0..3]  V[slot 0..3]
  ...
  Layer 31: K[slot 0..3]  V[slot 0..3]

已用: 4 / 16 slots
空闲: 12 slots
```

从请求视角看，prompt 的 4 个 token 已经完成计算：

```
num_computed_tokens: 0 -> 4
```

但这个状态要等采样和 `scheduler.update_from_output(...)` 后才真正回写到调度器。

---

### Step 12：取最后一个 hidden state，算 logits，采样第一个输出 token

runner 在 `sample_tokens()` 中调用：

```
sample_hidden_states = hidden_states[input_batch.logits_indices]
logits = self.model.compute_logits(sample_hidden_states)
sampler_output = self.sampler(logits, input_batch)
```

本例：

```
hidden_states: (4, hidden_size)
logits_indices = [3]

hidden_states[3]: (1, hidden_size)
     │
     ▼
LM Head / LogitsProcessor
     │
     ▼
logits: (1, vocab_size)
     │
     ├─ logits processors / penalties
     ├─ temperature
     ├─ top-k / top-p
     └─ sample

sampled_token_id = 40001 -> "今"
```

如果有 grammar/structured output，`scheduler.get_grammar_bitmask(...)` 生成的 bitmask 会在 sampling 前应用到 logits 上。

Prefill 这一轮的产物是：

```
输出 token: "今"
KV 副产品: prompt 4 个 token 的全层 K/V 已写入 Block_42
```

---

## 第二幕：Decode 阶段 —— 每轮只追加少量 token

### Decode 与 Prefill 的本质差异

```
                      Prefill                         Decode
输入 token 数          prompt chunk                    通常每请求 1 token
调度状态              WAITING -> RUNNING              RUNNING 继续推进
num_new_tokens         prompt_len - cached_len         1 + draft/lookahead
attention Q            多个 prompt token               新 token
attention K/V          当前 prompt + 已有 cache         全部历史 cache + 新 KV
主要瓶颈              大 prompt 的 attention/GEMM       小 batch GEMV + KV 读取 + launch
vLLM 优化重点          chunked prefill + page KV        continuous batching + CUDA graph
```

vLLM 在 decode 阶段最怕的不是“一个 token 算不出来”，而是：

```
每个请求只来 1 个 token
单请求矩阵很小
Python/metadata/kernel launch 开销占比变大
GPU 容易吃不满
```

所以 decode 的核心策略是：把许多请求的 1 个 token 持续混到同一轮 batch 里。

---

### Decode Step 1：调度第一个输出 token 之后的下一步

Prefill 采样出 `"今"` 后，scheduler 更新请求状态：

```
req_001:
  prompt_token_ids = [104169, 99476, 56278, 104949]
  output_token_ids = [40001]
  num_tokens_with_spec = 5
  num_computed_tokens = 4
  status = RUNNING
```

下一轮 `Scheduler.schedule()` 先处理 running 队列：

```
num_new_tokens =
  num_tokens_with_spec + num_output_placeholders - num_computed_tokens
= 5 + 0 - 4
= 1
```

这就是 decode step：

```
输入 token: 上一步采样出的 "今"
position: 4
本轮只计算 1 个 token
```

---

### Decode Step 1a：KVCacheManager 追加 slot

因为 Block_42 还有空槽：

```
Block_42:
  used: slot 0..3
  next: slot 4
```

`allocate_slots(req_001, num_new_tokens=1)` 不需要新 block，只需要确认当前 block 可追加：

```
req_001 -> [Block_42]

slot mapping:
  decode token position 4 -> Block_42 slot 4
```

如果当前 block 满了，才会从 BlockPool 再取一个新 block：

```
position 16 -> Block_43 slot 0
req_001 -> [Block_42, Block_43]
```

---

### Decode Step 1b：InputBatch 变成 1-token 查询

本轮 `SchedulerOutput`：

```
num_scheduled_tokens:
  req_001 -> 1

total_num_scheduled_tokens = 1
```

`prepare_inputs()` 生成：

```
InputBatch:
  req_ids = ["req_001"]
  input_ids = [40001]
  positions = [4]
  query_start_loc = [0, 1]
  seq_lens = [5]
  logits_indices = [0]
```

注意：

- `query_start_loc=[0,1]` 表示本轮 Q 只有 1 个 token。
- `seq_lens=[5]` 表示 attention 可见的 K/V 历史长度是 5。
- `logits_indices=[0]` 表示这个唯一 hidden state 就要用于采样。

---

### Decode Step 1c：Attention 读全部历史 KV，写 1 个新 KV

进入第 0 层时：

```
X: (1, hidden)
  │
  ├─ RMSNorm + QKV projection
  │
  ├─ RoPE(Q, K_new)
  │
  └─ Attention.forward(Q_new, K_new, V_new)
```

KV cache 操作分两部分：

```
写:
  K_new -> Block_42 slot 4
  V_new -> Block_42 slot 4

读:
  K_all <- Block_42 slot 0..4
  V_all <- Block_42 slot 0..4
```

以 FlashAttention backend 为例，它拿到：

```
q = query[:1]
k = key_cache
v = value_cache
block_table = [[42]]
seqused_k = [5]
max_seqlen_q = 1
max_seqlen_k = 5
```

逻辑计算是：

```
Q_new: (1, num_heads, head_dim)
K_all: (5, num_kv_heads, head_dim)
V_all: (5, num_kv_heads, head_dim)

attention:
  新 token 看 prompt 4 个 token + 自己
```

在 GQA 场景中，多个 Q head 共享一个 KV head；backend 内部会按自己的 kernel 组织读取和计算。vLLM 上层只负责提供正确的 block table、seq_lens 和 slot mapping。

---

### Decode Step 1d：采样下一个 token，并更新状态

第 32 层结束后：

```
hidden_states: (1, hidden_size)
logits_indices = [0]
     │
     ▼
compute_logits + sampler
     │
     ▼
sampled_token_id = token("天")
```

`Scheduler.update_from_output(...)` 后：

```
req_001:
  output_token_ids = ["今", "天"]
  num_tokens_with_spec = 6
  num_computed_tokens = 5
  Block_42 used = 5 / 16
```

下一轮仍然是同一个公式：

```
num_new_tokens = 6 - 5 = 1
```

---

### Decode Step 2~14：循环推进，KV block 逐渐填充

```
Step  输入token   pos   生成token   KV 已用       block table
----  --------   ---   --------   ----------    ----------------
  0   prompt      0-3   "今"       4 tokens      [Block_42]
  1   "今"        4     "天"       5 tokens      [Block_42]
  2   "天"        5     "北"       6 tokens      [Block_42]
  3   "北"        6     "京"       7 tokens      [Block_42]
  4   "京"        7     "天"       8 tokens      [Block_42]
  5   "天"        8     "气"       9 tokens      [Block_42]
  6   "气"        9     "晴"      10 tokens      [Block_42]
  7   "晴"       10     "朗"      11 tokens      [Block_42]
  8   "朗"       11     "，"      12 tokens      [Block_42]
  9   "，"       12     "气"      13 tokens      [Block_42]
 10   "气"       13     "温"      14 tokens      [Block_42]
 11   "温"       14     "28"      15 tokens      [Block_42]
 12   "28"       15     "度"      16 tokens      [Block_42]
 13   "度"       16     "。"      17 tokens      [Block_42, Block_43]
 14   "。"       17     <EOS>     18 tokens      [Block_42, Block_43]
```

当 position=16 时：

```
Block_42 已满:
  slot 0..15

分配 Block_43:
  position 16 -> Block_43 slot 0
```

slot mapping 从此跨 block：

```
position 15 -> Block_42 slot 15
position 16 -> Block_43 slot 0
position 17 -> Block_43 slot 1
```

attention backend 不需要一段连续 KV 内存，它通过 block table 读到完整逻辑序列。

---

## 第三幕：Continuous Batching —— vLLM 真正吃满 GPU 的地方

单个请求的 decode 每轮只有 1 个 token，GPU 利用率很低。vLLM 的关键是把多个请求的不同进度混到同一轮。

假设此时有 4 个请求：

```
req_A: running, decode 1 token, seq_len=512
req_B: running, decode 1 token, seq_len=128
req_C: waiting, prompt 300 tokens
req_D: waiting, prompt 40 tokens
```

一轮 `Scheduler.schedule()` 可能形成：

```
先 running:
  req_A -> 1 token
  req_B -> 1 token

再 waiting:
  req_C -> 128 token chunk   如果 token budget 不够，长 prompt 被切块
  req_D -> 40 tokens

SchedulerOutput:
  req_A: 1
  req_B: 1
  req_C: 128
  req_D: 40

total_num_scheduled_tokens = 170
```

`prepare_inputs()` 会把它压成一个扁平 batch：

```
input_ids:
  [A_decode,
   B_decode,
   C_prompt_0 ... C_prompt_127,
   D_prompt_0 ... D_prompt_39]

query_start_loc:
  [0, 1, 2, 130, 170]

seq_lens:
  [513, 129, 128, 40]
```

vLLM 的“continuous”体现在这里：

```
请求不是等同伴一起结束一个完整阶段
而是每轮都按 token budget 重新拼 batch
running decode 优先保温
waiting prefill 用剩余预算补进来
```

这比“静态 batch”更适合在线服务，因为请求不断到达、不断完成、长度也完全不同。

---

## 第四幕：KV Cache 生命周期全貌

### 1. 初始化：先 profiling，再决定能放多少 KV block

`EngineCore.__init__()` 中 KV 初始化顺序是：

```
model_executor = executor_class(vllm_config)
     │
     ▼
model_executor.get_kv_cache_specs()
     │
     ▼
model_executor.determine_available_memory()
     │
     ▼
get_kv_cache_configs(vllm_config, kv_cache_specs, available_gpu_memory)
     │
     ▼
model_executor.initialize_from_config(kv_cache_configs)
     │
     ▼
Scheduler(..., kv_cache_config=...)
```

这说明 vLLM 的 KV cache 不是写死大小，而是结合模型、backend、显存 profiling、配置动态决定。

---

### 2. 物理布局：backend 决定 KV tensor shape

KV manager 只知道 block；backend 决定 tensor 形状。

FlashAttention backend 的典型形状：

```
[2, num_blocks, block_size, num_kv_heads, head_size]
```

Triton backend 可能使用：

```
[num_blocks, 2, block_size, num_kv_heads, head_size]
```

`attn_utils._reshape_kv_cache()` 会根据 backend 的：

```
get_kv_cache_shape(...)
get_kv_cache_stride_order()
```

把 raw tensor 解释成 backend 需要的 layout。

因此：

```
Scheduler:
  block id / block 数量 / prefix cache

Worker:
  block table / slot mapping

Backend:
  KV tensor shape / stride / kernel
```

三者是分层的。

---

### 3. Prefix Cache：命中单位是完整 block

`KVCacheManager.get_computed_blocks(request)` 会查询 prefix cache：

```
request.block_hashes
     │
     ▼
coordinator.find_longest_cache_hit(...)
     │
     ▼
computed_blocks, num_new_computed_tokens
```

关键限制：

```
命中的 block 必须是 full block
如果全部 prompt 命中，也要重算最后一个 token 来拿 logits
```

所以 prefix cache 不是“语义相同就完全跳过 prompt”，而是：

```
完整 block hash 命中 -> 复用对应 K/V
最后 token 可能重算 -> 产出 logits
未对齐部分 -> 重新计算
```

示例：

```
block_size = 16
prompt_len = 40

可命中的最大完整 block:
  token 0..15
  token 16..31

token 32..39 不构成完整 block，需要计算
如果 40 个都命中，也至少重算最后 token 39
```

---

### 4. Preemption：不是暂停，是释放 KV 后回退

当 running 请求分配 block 失败时，scheduler 会 preempt 低优先级请求：

```
allocate_slots(...) -> None
     │
     ▼
选择 preempted_req
     │
     ▼
_preempt_request(preempted_req)
     │
     ├─ 释放 KV blocks
     ├─ 释放 encoder cache
     ├─ 重置局部进度
     └─ 放回 waiting
```

这不是“暂停后原地恢复”，而是可能引入 recompute。vLLM 因此有 admission gate，例如 `can_fit_full_sequence(...)`，避免 chunked prefill 只看眼前能放下，后续反复抢占。

---

### 5. 请求结束：释放 block，回到 BlockPool

当采样出 EOS 或达到停止条件：

```
OutputProcessor / Scheduler:
  标记 req_001 finished
     │
     ▼
GPUModelRunner.finish_requests(...)
     │
     ▼
free_states(...)
     │
     ▼
KVCacheManager / BlockPool:
  req_001 的 block 引用释放
  Block_42, Block_43 回到 free queue
```

显存中的旧 KV 数据通常不需要清零，后续请求会覆盖对应 slot。

---

## 第五幕：PagedAttention 到底在哪里

很多人把 PagedAttention 理解成某个 CUDA kernel 名，这在 vLLM v1 里太窄了。

更准确的链路是：

```
Scheduler/KVCacheManager:
  request -> logical blocks
     │
     ▼
BlockTables:
  logical blocks -> block_table
  token positions -> slot_mapping
     │
     ▼
Attention metadata:
  query_start_loc / seq_lens / block_table / slot_mapping
     │
     ▼
Attention.forward:
  unified_kv_cache_update
  unified_attention_with_output
     │
     ▼
Backend:
  FlashAttention / Triton / FlashInfer / native paged_attention_v1/v2 / ROCm ...
```

native CUDA paged attention 入口仍然存在：

```
vllm/_custom_ops.py:
  paged_attention_v1(...)
  paged_attention_v2(...)

csrc/attention:
  paged_attention_v1.cu
  paged_attention_v2.cu
  attention_kernels.cuh
```

但 v1 主线常见路径可能走 FlashAttention backend：

```
flash_attn_varlen_func(
  q=query,
  k=key_cache,
  v=value_cache,
  cu_seqlens_q=query_start_loc,
  seqused_k=seq_lens,
  block_table=block_table,
  ...
)
```

所以 PagedAttention 的系统价值是：

```
用 block table 让逻辑连续的序列映射到物理不连续的 KV pages
```

而不是固定等于某一个 kernel。

---

## 第六幕：CUDA Graph 和采样为什么也在主路径里

### CUDA Graph：decode 稳定形状时降低 launch overhead

`GPUModelRunner.execute_model()` 会计算：

```
num_reqs
num_toks
max_query_len
uniform_tok_count
```

再通过：

```
dispatch_cg_and_sync_dp(...)
```

选择：

```
CUDAGraphMode.FULL       完整图 replay
CUDAGraphMode.PIECEWISE  分段图 / 编译片段
CUDAGraphMode.NONE       eager
```

当 batch 描述命中已 capture 的 full graph：

```
cudagraph_manager.run_fullgraph(batch_desc)
```

否则：

```
with set_forward_context(...):
    model_output = self.model(**model_inputs)
```

无论走 graph 还是 eager，核心输入协议都不变：

```
InputBatch + block_tables + slot_mappings + attn_metadata
```

也就是说，CUDA graph 是执行外壳优化，不改变调度/KV 的语义。

---

### Sampling：不是后处理小尾巴，而是状态闭环的一半

vLLM v1 把采样放在 worker 热路径里：

```
sample_hidden_states = hidden_states[input_batch.logits_indices]
logits = model.compute_logits(sample_hidden_states)
sampler_output = sampler(logits, input_batch)
postprocess(input_batch, sampled_tokens, ...)
```

sampler 内部会处理：

```
logits processors
penalties
temperature
top-k
top-p
random / greedy sample
logprobs
```

如果是 spec decode：

```
draft tokens -> 本轮可能产生多个 logits
rejection_sampler -> 接受/拒绝 draft
num_sampled / num_rejected -> 回写请求状态
```

所以采样不只是“拿 logits 选个 token”。它决定下一轮：

```
num_tokens_with_spec
num_output_placeholders
num_computed_tokens
```

这些字段如何变化。

---

## 全过程一图总结

```
时间 →

T0 请求到达
│
├─ LLMEngine.add_request
│   └─ prompt -> Request(req_001)
│
├─ EngineCore.step()
│   └─ Scheduler.schedule()
│       ├─ prefix cache 查询
│       ├─ token_budget 决定 num_new_tokens=4
│       ├─ KVCacheManager.allocate_slots -> Block_42
│       └─ SchedulerOutput(req_001: 4 tokens)
│
├─ GPUModelRunner.execute_model()
│   ├─ prepare_inputs
│   │   input_ids=[4 prompt tokens]
│   │   positions=[0,1,2,3]
│   │   query_start_loc=[0,4]
│   │   seq_lens=[4]
│   │
│   ├─ prepare_attn
│   │   block_table=[[Block_42]]
│   │   slot_mapping=[42:0,42:1,42:2,42:3]
│   │
│   ├─ set_forward_context(attn_metadata, slot_mapping)
│   │
│   └─ model forward
│       ├─ Embedding
│       ├─ Layer 0..31
│       │   ├─ QKV
│       │   ├─ Attention.forward
│       │   │   ├─ 写 K/V 到 Block_42 slot 0..3
│       │   │   └─ FlashAttention/Triton/custom backend
│       │   └─ MLP
│       └─ hidden_states
│
├─ sample_tokens()
│   ├─ hidden_states[last prompt position]
│   ├─ compute_logits
│   ├─ sampler
│   └─ sampled token = "今"
│
├─ Scheduler.update_from_output()
│   └─ num_computed_tokens=4, output=["今"]
│
├─ Decode Step 1
│   ├─ num_new_tokens = 5 - 4 = 1
│   ├─ input_ids=["今"], positions=[4]
│   ├─ slot_mapping=[42:4]
│   ├─ attention 读 Block_42 slot 0..4, 写 slot 4
│   └─ sample -> "天"
│
├─ Decode Step 2..12
│   └─ 每轮追加 1 token，Block_42 从 5/16 填到 16/16
│
├─ Decode Step 13
│   └─ Block_42 满，分配 Block_43
│
└─ EOS
    ├─ 请求完成
    └─ Block_42 / Block_43 释放回 BlockPool


vLLM 的核心优化:

1. 调度:
   continuous batching，每轮按 token debt 和 token budget 重组 batch。

2. KV:
   block/page 化 KV cache，prefix cache 以完整 block 复用，slot mapping 把 token 写入物理槽位。

3. Attention:
   backend 可插拔，FlashAttention/Triton/native paged kernels 共享同一套 block table 协议。

4. 执行:
   decode 形状稳定时用 CUDA graph 降低 launch overhead；prefill/chunked prefill 可走 eager 或 piecewise path。

5. 采样:
   sampler、structured output、spec decode 都进入主状态闭环，不是外挂后处理。
```

---

## 和 MindIE 版本的关键差异

| 维度 | MindIE 文档中的主线 | vLLM 本文主线 |
|---|---|---|
| 优化中心 | GE 图编译，单层算子融合 | runtime 调度，continuous batching，PagedAttention |
| Prefill | 21 个操作融合成少量 kernel | prompt token 被调度成 chunk，写入 page 化 KV |
| Decode | GEMV/FlashDecoding 带宽利用 | 多请求 decode token 混排，CUDA graph 降低小步开销 |
| KV block | 较大 block，贴硬件寻址策略 | page/block 虚拟内存协议，block table + slot mapping |
| 后端边界 | MindIE runtime/GE 管控 | scheduler、KV manager、runner、backend 分层 |
| 核心理解 | 编译器减少 HBM 往返和 launch | 系统持续重排 token，让 GPU 一直有活干 |

一句话：

```
MindIE 更像“把单次 Transformer 前向编译得更紧”；
vLLM 更像“把在线服务中的大量不规则 token 流组织成硬件友好的连续执行”。
```

---

## 对 mini-vllm 的对应关系

本仓库的 `mini-vllm/src/myvllm/` 可以按同一条链理解：

| vLLM upstream 概念 | mini-vllm 对应路径 | 说明 |
|---|---|---|
| `Request` / request state | `engine/sequence.py` | 记录 token、block、完成状态 |
| `Scheduler` | `engine/scheduler.py` | waiting/running 调度、preempt、postprocess |
| `KVCacheManager` / `BlockPool` | `engine/block_manager.py` | block 分配、append、prefix hash |
| `GPUModelRunner` | `engine/model_runner.py` | prepare_prefill、prepare_decode、run_model、sample |
| `Attention` backend | `layers/attention.py` | prefill attention、paged decode、store_kvcache |
| model 文件 | `models/qwen3.py` / `models/llama.py` | 用 layers 拼 Transformer |
| sampler | `layers/sampler.py` | logits 到 token |

mini-vllm 的价值在于把 vLLM 的系统链路压缩成教学版：

```
Sequence -> Scheduler -> BlockManager -> ModelRunner -> Attention -> Sampler
```

要继续贴近 upstream vLLM，最值得优先补的不是复杂 CUDA kernel，而是：

1. 用 `num_computed_tokens` 风格统一 prefill/decode 进度。
2. 显式区分 `block_table` 和 `slot_mapping`。
3. 让 prefix cache 命中以完整 block 为单位，并处理最后 token 重算。
4. 把调度输出和设备输入之间的协议对象写清楚。

---

## 参考资料

- `docs/MindIE加速Transformer推理全过程拆解.md`
- `docs/vLLM.md`
- `docs/vLLM深度调研.md`
- `vLLM/vllm-upstream/vllm/v1/engine/llm_engine.py`
- `vLLM/vllm-upstream/vllm/v1/engine/core.py`
- `vLLM/vllm-upstream/vllm/v1/core/sched/scheduler.py`
- `vLLM/vllm-upstream/vllm/v1/core/kv_cache_manager.py`
- `vLLM/vllm-upstream/vllm/v1/core/block_pool.py`
- `vLLM/vllm-upstream/vllm/v1/worker/gpu/model_runner.py`
- `vLLM/vllm-upstream/vllm/v1/worker/gpu/block_table.py`
- `vLLM/vllm-upstream/vllm/v1/worker/gpu/attn_utils.py`
- `vLLM/vllm-upstream/vllm/model_executor/layers/attention/attention.py`
- `vLLM/vllm-upstream/vllm/v1/attention/backends/flash_attn.py`
- `vLLM/vllm-upstream/vllm/v1/attention/ops/paged_attn.py`
- `vLLM/vllm-upstream/csrc/attention/paged_attention_v1.cu`
- `vLLM/vllm-upstream/csrc/attention/paged_attention_v2.cu`
