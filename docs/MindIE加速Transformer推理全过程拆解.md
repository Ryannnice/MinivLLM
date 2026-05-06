# MindIE 如何加速一次 Transformer 推理：从 Prefill 到 Decode 的全过程拆解

> 本文以一个具体场景为例，逐操作追踪 MindIE 在 Prefill 和 Decode 两个阶段做了哪些加速，每一步的 Tensor 形状、内存布局、硬件调度都写清楚。不讲概念，只讲"到底发生了什么"。

---

## 场景设定

```
模型:     Qwen2-7B (32 层 Transformer, hidden=3584, num_heads=28, num_kv_heads=4, head_dim=128)
硬件:     单卡 Ascend 910B (64GB HBM, AI Core ×30, HBM 带宽 ~1.5 TB/s)
精度:     FP16 (权重 + 激活)
输入:     "今天天气怎么样"
期望输出: "今天北京天气晴朗，气温28度。"

KV Cache 配置:
  cacheBlockSize = 128 tokens/block
  npuMemSize = 34GB (分配给 KV Cache 的显存)
```

---

## 第一幕：Prefill 阶段 —— 一次性处理全部输入 token

### Step 0：Tokenize + KV Cache Block 预分配

```
原始文本: "今天天气怎么样"
              │
              ▼  Qwen2 BPE Tokenizer
token_ids = [104169, 99476, 56278, 104949]     共 4 个 token

在 Prefill 之前, Scheduler 先分配 KV Cache:

  Block 分配:
    需要容纳 4 个 token
    cacheBlockSize = 128 tokens / block
    ⌈4 / 128⌉ = 1 个 Block
    从空闲池取出 Block_42
    页表: req_001 → [Block_42]

  Block_42 的物理内存 (已分配但尚未写入):
    大小 = 128 tokens × 32 layers × 2(K+V) × 4 heads × 128 dim × 2 bytes
         = 128 × 32 × 2 × 4 × 128 × 2
         = 8 MB
    位于 HBM 的 KV Cache 区域
```

---

### Step 1：Embedding —— 查表

```
token_ids: (4,)
     │
     ▼  Embedding 查表
X: shape (4, 3584)    FP16

就是从词表矩阵 (152064, 3584) 中取出第 104169、99476、56278、104949 行。
4 行 × 3584 × 2B = 28 KB，太小了，没有优化空间。
```

---

### Step 2：进入第 0 层 Transformer Block —— 核心优化区

先看没有任何优化时，一个 Transformer Layer 要做什么（21 个操作），再看 MindIE 怎么压缩到 5 个融合 Kernel。

#### 2a. 原始计算图：21 个独立操作

```
X (4, 3584) ─────────────────────────────────────────────────────────────┐
  │                                                                      │
  ├─① RMSNorm → X_norm (4, 3584)                                        │
  │    ├─② X_norm × W_Q (3584,3584) → Q (4, 3584)                       │
  │    ├─③ X_norm × W_K (3584, 512) → K (4, 512)   ← GQA: 只有4个KV头   │
  │    └─④ X_norm × W_V (3584, 512) → V (4, 512)                        │
  │         ├─⑤ Q reshape → (4, 28, 128)                                 │
  │         ├─⑥ K reshape → (4, 4, 128)                                  │
  │         ├─⑦ V reshape → (4, 4, 128)                                  │
  │         ├─⑧ Q 加 RoPE 位置编码                                        │
  │         ├─⑨ K 加 RoPE 位置编码                                        │
  │         ├─⑩ scores = Q × K^T → (28, 4, 4)   注意力分数               │
  │         ├─⑪ scores /= √128                                           │
  │         ├─⑫ causal mask                                               │
  │         ├─⑬ softmax → probs (28, 4, 4)                               │
  │         └─⑭ probs × V → attn (4, 28, 128)                           │
  │              ├─⑮ reshape → (4, 3584)                                  │
  │              └─⑯ × W_O (3584,3584) → O (4, 3584)                    │
  │                                                                      │
  ├─⑰ X + O → X2 (4, 3584)            残差连接                            │
  ├─⑱ RMSNorm(X2) → X2_norm (4, 3584)                                   │
  │    ├─⑲ X2_norm × W_gate (3584,18944) → gate (4,18944)               │
  │    │   X2_norm × W_up   (3584,18944) → up   (4,18944)               │
  │    │   SiLU(gate) ⊙ up → inter (4,18944)                            │
  │    └─⑳ inter × W_down (18944,3584) → ffn_out (4,3584)               │
  │                                                                      │
  └─㉑ X2 + ffn_out → X_next (4, 3584)  残差连接                          │
```

如果在 Eager 模式（如 vLLM-Ascend 当前方式）执行，**每个操作都是独立的 kernel launch**，每次中间结果都写入 HBM 再被下一个操作读出。21 次 kernel launch，约 40 次 HBM 读写。

#### 2b. MindIE GE 编译器的融合：21 步 → 5 个融合 Kernel

**融合 Kernel 1：RMSNorm + QKV MatMul**

```
SRAM：在 GPU / 昇腾芯片里面（片上内存）
HBM：在 GPU / 昇腾芯片外面（板载显存，独立显存颗粒）
原来: ① → HBM → ② → HBM → ③ → HBM → ④
      4 次 kernel launch, 6 次 HBM 读写

融合后: 1 个 Kernel
  ┌─ AI Core SRAM ─────────────────────────────────────────────┐
  │                                                             │
  │  读入 X: (4, 3584) 从 HBM → SRAM                           │
  │                                                             │
  │  第 1 步: RMSNorm                                           │
  │    X_norm = X / RMS(X) * γ                                  │
  │    ↳ 结果留在 SRAM，不写回 HBM                               │
  │                                                             │
  │  第 2 步: QKV MatMul (三合一)                                │
  │    W_Q, W_K, W_V 拼接为 W_QKV: (3584, 4608)                │
  │    [W_Q(3584,3584) | W_K(3584,512) | W_V(3584,512)]        │
  │    QKV = X_norm × W_QKV → (4, 4608)                        │
  │    切分: Q(4,3584), K_new(4,512), V_new(4,512)              │
  │                                                             │
  │  写出 QKV 到 HBM                                            │
  │                                                             │
  └─────────────────────────────────────────────────────────────┘

  节省: RMSNorm 输出 (4,3584)×2B=28KB 的一次 HBM 写 + 三次 HBM 读
        3 个小 MatMul 合并为 1 个大 MatMul (更高的 Cube 利用率)
```

**融合 Kernel 2：RoPE + FlashAttention + KV Cache 写入**

这是最复杂也最关键的融合，将 12 个操作合并为 1 个 Kernel：

```
原来: ⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯
      12 个操作, 中间的 attention matrix (28,4,4) 每次都经过 HBM

融合后: 1 个 FlashAttention Kernel

  ┌─ AI Core 硬件视角 ─────────────────────────────────────────┐
  │                                                             │
  │  昇腾 AI Core 内部有两个计算单元:                              │
  │    Cube 单元: 做矩阵乘 (Q×K^T, P×V)                         │
  │    Vector 单元: 做逐元素操作 (RoPE, Scale, Softmax)           │
  │  两者可以流水线并行!                                           │
  │                                                             │
  │  执行流水线:                                                  │
  │                                                             │
  │  时间 →                                                      │
  │  ┌─────────────┬──────────────┬──────────────┐              │
  │  │ Vector:     │ Vector:      │ Vector:      │              │
  │  │ RoPE(Q_blk) │ Scale+Mask   │ Softmax      │              │
  │  │ RoPE(K_blk) │              │              │              │
  │  ├─────────────┼──────────────┼──────────────┤              │
  │  │             │ Cube:        │ Cube:        │              │
  │  │   (等待)    │ Q×K^T        │ P×V          │              │
  │  └─────────────┴──────────────┴──────────────┘              │
  │  Vector 和 Cube 流水线重叠，不是串行等待！                      │
  │                                                             │
  │  具体步骤:                                                    │
  │                                                             │
  │  1. 从 HBM 读取 Q (4,28,128)                                │
  │     将 Q 按 block 分组 (Bq = 4, 全部在一个 block)             │
  │                                                             │
  │  2. Vector 单元: 对 Q_block 做 RoPE (pos=0,1,2,3)           │
  │     结果留在 SRAM                                             │
  │                                                             │
  │  3. 从 HBM 读取 K_new (4,4,128)                             │
  │     Vector 单元: 对 K_block 做 RoPE                           │
  │                                                             │
  │  4. Cube 单元: scores = Q_roped × K_roped^T → (28, 4, 4)    │
  │     因为 GQA, 实际上每 7 个 Q head 共享 1 个 K head:          │
  │       scores_group0 = Q[0:7] × K[0]^T → (7, 4, 4)          │
  │       scores_group1 = Q[7:14] × K[1]^T → (7, 4, 4)         │
  │       ...                                                    │
  │     这些矩阵很小 (4×4), 全在 SRAM 中                          │
  │                                                             │
  │  5. Vector 单元: scores /= √128, causal_mask, softmax       │
  │     probs: (28, 4, 4) 仍在 SRAM                              │
  │                                                             │
  │  6. Cube 单元: attn_out = probs × V → (4, 28, 128)          │
  │     reshape → (4, 3584)                                      │
  │                                                             │
  │  7. ★ KV Cache 写入 (顺便完成，零额外开销):                    │
  │     K_roped (4, 4, 128) → 写入 Block_42, Layer 0, K 区域     │
  │     V_new   (4, 4, 128) → 写入 Block_42, Layer 0, V 区域     │
  │     这步用 DMA 单元完成, 与 Cube/Vector 并行                   │
  │                                                             │
  │  8. × W_O (3584,3584) → O (4, 3584)                         │
  │     写出到 HBM                                                │
  │                                                             │
  └─────────────────────────────────────────────────────────────┘

  关键: (28, 4, 4) 的 attention matrix 从头到尾在 SRAM 中, 从未到过 HBM
  节省: 11 次 HBM 往返
```

**融合 Kernel 3：残差连接 + RMSNorm**

```
原来: ⑰ X+O → HBM → ⑱ RMSNorm → HBM

融合后:
  ┌─ SRAM ─────────────────────────┐
  │ 读入 X (残差) 和 O (attn 输出)   │
  │ X2 = X + O                      │
  │ X2_norm = RMSNorm(X2)           │
  │ 写出 X2_norm                    │
  │ (X2 也写出, Kernel 5 需要做残差)  │
  └─────────────────────────────────┘
```

**融合 Kernel 4：Gate-Up MatMul + SiLU 激活 + 逐元素乘**

```
原来: ⑲ 两次 MatMul + SiLU + 逐元素乘，多次经过 HBM

融合后:
  ┌─ SRAM ─────────────────────────────────────────────────┐
  │ W_gate 和 W_up 拼接: W_gate_up (3584, 37888)           │
  │                                                         │
  │ gate_up = X2_norm × W_gate_up → (4, 37888)             │
  │ 切分: gate (4, 18944), up (4, 18944)                    │
  │ inter = SiLU(gate) ⊙ up → (4, 18944)                   │
  │                                                         │
  │ SiLU 和逐元素乘在 Vector 单元上完成                       │
  │ MatMul 在 Cube 单元上完成                                │
  │ 两者流水线重叠                                            │
  └─────────────────────────────────────────────────────────┘
```

**融合 Kernel 5：Down MatMul + 残差连接**

```
  inter × W_down (18944, 3584) → ffn_out (4, 3584)
  X_next = X2 + ffn_out   ← 在同一 Kernel 中完成
  写出 X_next → HBM → 传给第 1 层
```

#### 2c. 融合效果量化

```
                          Eager (无优化)      MindIE (图编译)
                          ────────────        ───────────────
Kernel launch 次数 (单层)     21                  5
HBM 读写次数 (单层)           ~40                 ~8
中间张量经过 HBM 的数据量     ~1.8 MB             ~0.4 MB

关键差异不在绝对值,而在scale up后:

当输入 2048 tokens, batch=32 时 (生产场景):
  Eager: 1.8MB × (2048/4) × 32 × 32层 ≈ 944 GB 的 HBM 搬运
  MindIE: 0.4MB × (2048/4) × 32 × 32层 ≈ 211 GB 的 HBM 搬运

  910B HBM 带宽 ~1.5 TB/s:
    Eager: 944GB / 1.5TB/s = 0.63s 光在搬数据
    MindIE: 211GB / 1.5TB/s = 0.14s

  → 4.5x 的 HBM 搬运差距 ≈ 实测 3~4x 的性能差距
    (实际差距略小于理论值, 因为 Eager 也有部分 kernel 并行)
```

---

### Step 3：32 层执行完毕

```
X 依次通过 Layer 0 → Layer 1 → ... → Layer 31
每一层都执行上述 5 个融合 Kernel

同时, 每一层的 KV Cache 都写入了 Block_42:

  Block_42 最终状态:
  ┌──────────────────────────────────────────────────────┐
  │ Layer 0:  K[slot 0..3] ✓  V[slot 0..3] ✓            │
  │ Layer 1:  K[slot 0..3] ✓  V[slot 0..3] ✓            │
  │ ...                                                   │
  │ Layer 31: K[slot 0..3] ✓  V[slot 0..3] ✓            │
  │                                                       │
  │ 已用: 4 / 128 slots (3.1%)                            │
  │ 剩余: 124 slots 空闲                                   │
  └──────────────────────────────────────────────────────┘
```

---

### Step 4：取 logits，采样第一个输出 token

```
X_final: (4, 3584)    ← 4 个位置的隐藏状态

只取最后一个位置 (index 3): X_last: (1, 3584)
  │
  ├─ RMSNorm → (1, 3584)
  │
  └─ LM Head: (1, 3584) × W_vocab^T (3584, 152064) → logits (1, 152064)
     │
     ├─ logits / temperature(0.7)
     ├─ softmax → 概率分布
     ├─ top-p(0.9) 截断: 只保留累积概率 ≤ 0.9 的 token
     └─ 采样 → token_id = 40001 → "今"

⏱ Prefill 完成
  产出: 第 1 个输出 token "今"
  副产品: 4 个输入 token 的 KV Cache 已写入 Block_42
```

---

## 第二幕：Decode 阶段 —— 逐 token 生成，问题完全变了

### Decode 与 Prefill 的本质差异

```
                      Prefill                    Decode
输入 token 数          4 (全部输入)                1 (上一步输出)
MatMul 形状           (4, 3584) × (3584, N)      (1, 3584) × (3584, N)
                      → GEMM (矩阵×矩阵)          → GEMV (矩阵×向量)
计算瓶颈              Compute-bound               Memory-bound
                      (算力不够)                   (带宽不够)
Attention 要看         4 个 token 互相看            1 个新 token 看所有历史
KV Cache              写入 (生成新缓存)             读取 + 追加 1 个

核心矛盾:
  GEMV 的算术强度 = 2N / (N×2B) = 1 FLOP/Byte
  910B 的理论要求 ≈ 算力/带宽 = 320 TFLOPS / 1.5 TB/s ≈ 213 FLOP/Byte
  差 200 多倍! → 算力严重空转, 瓶颈在 HBM 读带宽

  MindIE 在 Decode 阶段的优化重心:
  不是让算力跑得更快, 而是让带宽利用得更满
```

---

### Decode Step 1：生成第 2 个 token

输入: token_id = 40001 ("今"), position = 4

#### Kernel 1：RMSNorm + QKV GEMV

```
X: (1, 3584)
     │
     ▼
┌─ 融合 Kernel 1 ─────────────────────────────────────────┐
│                                                          │
│  RMSNorm: (1, 3584) → (1, 3584)   留在 SRAM              │
│                                                          │
│  QKV GEMV: (1, 3584) × W_QKV (3584, 4608)               │
│                                                          │
│  ★ 这里矩阵乘退化为 GEMV (矩阵×向量):                     │
│    计算量 = 2 × 3584 × 4608 = 33M FLOPs                  │
│    需读取的数据 = W_QKV 权重 3584×4608×2B = 31.5 MB        │
│    算术强度 = 33M / 31.5M ≈ 1 FLOP/Byte                  │
│    → 纯 memory-bound, 时间完全取决于读权重的速度             │
│                                                          │
│  MindIE 的 GEMV 优化:                                     │
│    30 个 AI Core 并行, 每个负责读 W_QKV 的 1/30 列          │
│    AI Core 0: 读取列 [0:154), 计算部分结果                  │
│    AI Core 1: 读取列 [154:307), 计算部分结果                │
│    ...                                                    │
│    AI Core 29: 读取列 [4454:4608), 计算部分结果             │
│    最后 reduce 合并                                        │
│                                                          │
│    等效带宽: 30 × 单 Core 带宽 → 接近 HBM 物理带宽上限      │
│    这是 GE 编译器自动完成的并行拆分                          │
│                                                          │
│  输出: Q(1,3584), K_new(1,512), V_new(1,512)              │
└──────────────────────────────────────────────────────────┘
```

#### Kernel 2：KV Cache 写入 + FlashDecoding

这是 Decode 阶段最关键的操作——新 token 的 KV 写入缓存，并和所有历史 KV 做 Attention。

```
┌─ 融合 Kernel 2 ─────────────────────────────────────────┐
│                                                          │
│  ===== 阶段 A: KV Cache 追加写入 =====                    │
│                                                          │
│  K_new: (1, 4, 128)  已做 RoPE (pos=4)                   │
│  V_new: (1, 4, 128)                                      │
│                                                          │
│  写入 Block_42, Layer 0:                                  │
│    K_new → slot 4 (之前 slot 0~3 已被 Prefill 填充)       │
│    V_new → slot 4                                         │
│                                                          │
│  Block_42 Layer 0 变为:                                   │
│    K: [t0][t1][t2][t3][t4_new][空][空]...[空]             │
│    V: [t0][t1][t2][t3][t4_new][空][空]...[空]             │
│         ↑ 4个Prefill token ↑ 新token                      │
│                                                          │
│  ===== 阶段 B: FlashDecoding 计算 Attention =====         │
│                                                          │
│  需要: Q_new (1, 28, 128) 与 K_all (5, 4, 128) 做 Attn   │
│                                                          │
│  GQA 展开:                                                │
│    28 个 Q head, 4 个 KV head                             │
│    每 7 个 Q head 共享 1 个 KV head                       │
│    Group 0: Q[0:7]  ↔ KV[0]                              │
│    Group 1: Q[7:14] ↔ KV[1]                              │
│    Group 2: Q[14:21] ↔ KV[2]                             │
│    Group 3: Q[21:28] ↔ KV[3]                             │
│                                                          │
│  AI Core 分配 (28 个 head → 30 个 AI Core):                │
│    AI Core 0: head 0    AI Core 1: head 1   ...           │
│    AI Core 27: head 27  AI Core 28-29: 空闲               │
│                                                          │
│  每个 AI Core 的执行 (以 head 0 为例):                     │
│  ┌─ AI Core 0 ──────────────────────────────────────┐    │
│  │                                                   │    │
│  │ 从 HBM 读取:                                      │    │
│  │   Q_head0: (1, 128) = 256 Bytes                   │    │
│  │   K_kv0_all: (5, 128) = 1280 Bytes  从 Block_42 读│    │
│  │   V_kv0_all: (5, 128) = 1280 Bytes  从 Block_42 读│    │
│  │   总读取: 2816 Bytes → 极小, 完全在 SRAM 中         │    │
│  │                                                   │    │
│  │ SRAM 内计算:                                       │    │
│  │   1. RoPE(Q, pos=4)                               │    │
│  │      (K 在 Prefill 时已做过 RoPE, 无需重做)         │    │
│  │                                                   │    │
│  │   2. Cube: scores = Q × K^T → (1, 5)              │    │
│  │      scores = [s0, s1, s2, s3, s4]                │    │
│  │      ← 每个 s_i 表示新 token 对历史第 i 个 token    │    │
│  │        的注意力分数                                 │    │
│  │                                                   │    │
│  │   3. Vector: scores /= √128 = /11.31              │    │
│  │      causal mask: 全可见 (新 token 能看到所有历史)   │    │
│  │      softmax: probs = [p0, p1, p2, p3, p4]        │    │
│  │                                                   │    │
│  │   4. Cube: output = probs × V → (1, 128)          │    │
│  │      = p0*V[0] + p1*V[1] + ... + p4*V[4]          │    │
│  │                                                   │    │
│  │ 写回 output (1, 128) 到 HBM                       │    │
│  └───────────────────────────────────────────────────┘    │
│                                                          │
│  28 个 head 并行完成                                       │
│  concat → (1, 28, 128) → reshape → (1, 3584)              │
│  × W_O → O (1, 3584)                                      │
│                                                          │
│  关键数字:                                                 │
│    每个 AI Core 从 HBM 读取 ~2.8 KB (KV Cache)            │
│    scores (1,5) 始终在 SRAM, 从未到过 HBM                  │
│    这就是 FlashAttention/FlashDecoding 的核心价值           │
└──────────────────────────────────────────────────────────┘
```

#### Kernel 3/4/5：与 Prefill 相同结构，但 shape 不同

```
Kernel 3: Residual(X+O) + RMSNorm
  shape: (1, 3584) → (1, 3584)
  与 Prefill 相同, 只是 batch 维度从 4 变 1

Kernel 4: Gate+Up GEMV + SiLU
  (1, 3584) × W_gate_up (3584, 37888) → (1, 37888)
  这是 Decode 阶段最大的权重读取:
    37888 × 3584 × 2B = 258 MB
    30 个 AI Core 每个读 8.6 MB
    以 1.5 TB/s 带宽: 258MB / 1.5TB/s = 0.17 ms

Kernel 5: Down GEMV + Residual
  (1, 18944) × W_down (18944, 3584) → (1, 3584)
  128 MB 权重读取, 0.09 ms
```

#### Decode Step 1 总耗时拆解

```
Kernel     操作              权重读取     KV Cache读取    耗时(估算)
───────    ──────            ────────     ──────────     ──────────
K1         RMSNorm+QKV GEMV  31.5 MB      -              0.021 ms
K2         FlashDecoding     25.0 MB      ~80 KB         0.017 ms
K3         Res+RMSNorm       ~0 MB        -              0.001 ms
K4         Gate+Up GEMV      258 MB       -              0.172 ms
K5         Down+Res GEMV     128 MB       -              0.085 ms
───────    ──────            ────────     ──────────     ──────────
合计 (单层)                   442.5 MB     ~80 KB         0.296 ms

32 层合计:                    14.2 GB      ~2.6 MB        9.5 ms
LM Head:                     1.04 GB      -              0.7 ms
───────                      ────────     ──────────     ──────────
Decode 单步总计               ~15.2 GB     ~2.6 MB        ~10.2 ms

对比:
  理论下界 = 15.2 GB / 1.5 TB/s = 10.1 ms  (纯带宽瓶颈)
  MindIE 实测 ≈ 10~15 ms  (接近理论下界!)

  → Decode 阶段，MindIE 的优化目标不是减少计算量
    而是把 HBM 带宽利用率逼近 100%
    图编译让 Kernel 之间的 gap 几乎消失 (无 Python 调度开销)
```

---

### Decode Step 2~12：重复循环，KV Cache 逐渐填充

```
Step  输入token   pos   生成token   KV Cache已用   Block_42使用率
────  ────────   ───   ────────   ──────────    ───────────
  1   "今"        4     "天"        5 tokens      3.9%
  2   "天"        5     "北"        6 tokens      4.7%
  3   "北"        6     "京"        7 tokens      5.5%
  4   "京"        7     "天"        8 tokens      6.3%
  5   "天"        8     "气"        9 tokens      7.0%
  6   "气"        9     "晴"       10 tokens      7.8%
  7   "晴"       10     "朗"       11 tokens      8.6%
  8   "朗"       11     "，"       12 tokens      9.4%
  9   "，"       12     "气"       13 tokens     10.2%
 10   "气"       13     "温"       14 tokens     10.9%
 11   "温"       14     "28"       15 tokens     11.7%
 12   "28"       15     "度"       16 tokens     12.5%
 13   "度"       16     "。"       17 tokens     13.3%
 14   "。"       17     <EOS>      18 tokens     14.1%

→ 检测到 EOS, 停止生成
```

每一步的 KV Cache 读取量在增长：

```
Step  KV Cache 读取 (Kernel 2, 每层)    总读取 (32层)
────  ──────────────────────────────    ──────────
  1   5 tokens × 4heads × 128d × 2B × 2(K+V) = 10 KB      320 KB
  7   11 tokens × ... = 22 KB                               704 KB
 14   18 tokens × ... = 36 KB                               1.2 MB

即使到 18 tokens, KV Cache 读取仍然远小于权重读取 (15.2 GB)
KV Cache 读取开始主导性能是在序列 > 2000 tokens 之后:
  2000 tokens: 4 MB/层 × 32 层 = 128 MB → 仍是权重读取的零头
  10000 tokens: 640 MB → 开始与权重读取量可比
  → 长序列场景下, FlashDecoding 的 KV 序列并行切分才真正关键
```

---

## 第三幕：KV Cache 生命周期全貌

```
时间轴:

  T0: 请求到达
      ├─ Scheduler 分配 Block_42 (8 MB)
      └─ Block_42 状态: [空 × 128]

  T1: Prefill 完成
      └─ Block_42 状态: [████____________________...___]
                         4个slot已用   124个空闲

  T2~T14: Decode 循环
      └─ Block_42 状态逐步填充:
         Step 1:  [█████___________________...___]  5/128
         Step 7:  [███████████______________...___] 11/128
         Step 14: [██████████████████_______...___] 18/128

  T15: 生成 EOS, 请求完成
      ├─ 释放 Block_42 → 归还空闲池
      └─ Block_42 内存不清零, 等下个请求覆盖写入

  显存变化:
  ┌─────────────────────────────────────────────────┐
  │ 64 GB 总显存                                      │
  │                                                   │
  │ ┌──────────┐ ┌──────────────┐ ┌───────────────┐  │
  │ │ 模型权重  │ │ KV Cache池   │ │ 激活值+缓冲   │  │
  │ │ ~14 GB   │ │ npuMemSize   │ │ ~2 GB         │  │
  │ │ (固定)   │ │ = 34 GB      │ │ (固定)        │  │
  │ └──────────┘ │              │ └───────────────┘  │
  │              │ Block_42: 8MB│                     │
  │              │ (其中仅      │                     │
  │              │  1.1MB 有效  │                     │
  │              │  数据)       │                     │
  │              │              │                     │
  │              │ 空闲Block:   │                     │
  │              │ 34GB-8MB     │                     │
  │              │ ≈ 4352个Block │                     │
  │              │ 可并发服务    │                     │
  │              │ ~4352个请求   │                     │
  │              │ (每个1Block) │                     │
  │              └──────────────┘                     │
  └─────────────────────────────────────────────────┘

  这就是为什么 cacheBlockSize=128 在短序列场景浪费大:
    本例只用了 18/128 = 14% 的 Block 空间
    但如果 cacheBlockSize=16 (像 vLLM), 需要 ⌈18/16⌉=2 个 Block
    每个 Block 更小, 总浪费更少, 但管理开销更大
    MindIE 选 128 是为了减少寻址开销, 适合昇腾硬件特性
```

---

## 全过程一图总结

```
时间 →
│
│  ┌──── Prefill ─────────────────┐  ┌─── Decode Step 1 ──┐  ┌─ Step 2─┐     ┌─ Step14─┐
│  │                              │  │                     │  │         │     │         │
│  │  输入: 4 tokens              │  │ 输入: 1 token       │  │ 1 token │     │ 1 token │
│  │  ① Embedding                │  │ "今"                │  │ "天"    │ ... │ "。"    │
│  │  ② Layer 0~31:              │  │                     │  │         │     │         │
│  │    5 个融合 Kernel × 32 层   │  │ 5 个融合 Kernel     │  │         │     │ → EOS   │
│  │    (每层 KV Cache 写入)      │  │ × 32 层            │  │         │     │         │
│  │  ③ LM Head + 采样           │  │ K2: FlashDecoding  │  │         │     │         │
│  │  → 第一个token: "今"         │  │   读5个历史KV      │  │ 读6个   │     │ 读18个  │
│  │                              │  │   写1个新KV        │  │ 写1     │     │ 写1     │
│  │  ⏱ ~15ms (compute-bound)    │  │ → "天"             │  │ → "北"  │     │ → EOS   │
│  │                              │  │ ⏱ ~10ms (mem-bound)│  │ ~10ms   │     │ ~11ms   │
│  └──────────────────────────────┘  └─────────────────────┘  └─────────┘     └─────────┘
│
│  KV Cache (Block_42):
│  Prefill后: [████________________________]  4/128
│  Step 1:   [█████_______________________]  5/128
│  Step 2:   [██████______________________]  6/128
│  ...
│  Step 14:  [██████████████████___________] 18/128
│  释放:     [____________________________] → 归还空闲池
│
│  MindIE 在每个阶段的核心优化:
│  Prefill: 图编译融合 21→5 Kernel, 减少 80% HBM 搬运, Cube/Vector 流水线
│  Decode:  GEMV 多 Core 并行, FlashDecoding KV 并行读取, 带宽利用率→100%
│  KV管理:  128-token Block 减少寻址开销, 写入融合进 Attention Kernel
```

---

## 参考资料

- [MindIE Service 开发指南 - 昇腾社区](https://www.hiascend.com/document/detail/zh/mindie/20RC1/mindieservice/servicedev/mindie_service0001.html)
- [MindIE 性能调优流程 - 昇腾官方文档](https://www.hiascend.com/document/detail/zh/mindie/100/mindieservice/servicedev/mindie_service0105.html)
- [MindIE 配置参数说明](https://www.hiascend.com/document/detail/zh/mindie/1.0.RC1/mindieservice/servicedev/mindie_service0005.html)
- [昇腾推理引擎 MindIE - 知乎 (ZOMI)](https://zhuanlan.zhihu.com/p/6878214249)
- [构筑开放基础软件栈，共建昇腾 AI 算力新生态 - 华为](https://www.huawei.com/cn/huaweitech/publication/202503/new-ecology-of-ascend-computing-power)
- [华为昇腾推理对决：开源 vLLM vs 官方 MindIE](https://news.qq.com/rain/a/20250617A01TK100)
- [FlashAttention: Fast and Memory-Efficient Exact Attention (论文)](https://arxiv.org/abs/2205.14135)
