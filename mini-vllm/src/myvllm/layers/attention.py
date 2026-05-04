# 文件内容概览：
# 1. 定义 KV cache 写入用的 Triton kernel。
# 2. 定义 prefill 阶段使用的 FlashAttention 变长 kernel 与 Python 包装函数。
# 3. 定义 decode 阶段使用的 paged attention kernel 与 Python 包装函数。
# 4. 定义统一的 `Attention` 模块，根据当前上下文自动选择 prefill 或 decode 路径。
#
# 在项目中的作用：
# 1. 这是整个项目里最核心的性能文件之一。
# 2. 它把 vLLM 里的两条关键链路落地成代码：prefill 的高效注意力、decode 的分页 KV cache 读取。
# 3. 上层模型虽然长得像普通 Transformer，但真正决定“怎么写 KV、怎么读 KV、怎么加速”的逻辑都在这里。

# 导入 Triton 主库。
import triton
# 导入 Triton 语言子模块，简称 `tl`，用于编写 GPU kernel。
import triton.language as tl
# 导入项目运行时上下文读取函数。
from myvllm.utils import get_context
# 导入 PyTorch 主库。
import torch
# 导入 `torch.nn`，用于定义最终的 `Attention` 模块类。
import torch.nn as nn


# 定义一个 Triton kernel，用于把当前步算出来的 K/V 写入分页 KV cache。
@triton.jit
def store_kvcache_kernel(
    key_ptr,
    value_ptr,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr
):
    # 取当前程序在第 0 维上的 program id，把它解释为“当前处理的是第几个 token”。
    token_idx = tl.program_id(0)
    # 根据 token 对应的 slot mapping，读取它应该写入 KV cache 的线性槽位。
    slot_idx = tl.load(slot_mapping_ptr + token_idx)

    # 如果当前 token 对应的槽位是 -1，说明这个 token 不需要写缓存，直接返回。
    if slot_idx == -1:
        return

    # 根据线性槽位计算该 token 应该落在哪个物理 block。
    block_idx = slot_idx // block_size
    # 根据线性槽位计算该 token 在 block 内部的偏移。
    block_offset = slot_idx % block_size

    # 取当前程序在第 1 维上的 program id，把它解释为“当前处理的是第几个 KV 头”。
    head_idx = tl.program_id(1)

    # 构造 head_dim 范围内的列偏移，用于一次性读取整个向量。
    head_offsets = tl.arange(0, head_dim)
    # 计算当前 token、当前 KV 头在输入 K/V 张量中的线性偏移地址。
    input_offset = (
        token_idx * num_kv_heads * head_dim +
        head_idx * head_dim +
        head_offsets
    )

    # 计算当前 token、当前 KV 头在分页缓存中的线性偏移地址。
    cache_offset = (
        block_idx * block_size * num_kv_heads * head_dim +
        block_offset * num_kv_heads * head_dim +
        head_idx * head_dim +
        head_offsets
    )

    # 从输入 K 张量中读取当前 token、当前头对应的向量。
    key = tl.load(key_ptr + input_offset)
    # 从输入 V 张量中读取当前 token、当前头对应的向量。
    value = tl.load(value_ptr + input_offset)

    # 把 K 向量写入分页缓存的目标位置。
    tl.store(k_cache_ptr + cache_offset, key)
    # 把 V 向量写入分页缓存的目标位置。
    tl.store(v_cache_ptr + cache_offset, value)


# 定义 Python 包装函数，负责准备参数并调用 KV cache 写入 kernel。
def store_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int
):
    # 从 key 张量形状中解析出 token 数、KV 头数和 head_dim。
    num_tokens, num_kv_heads, head_dim = key.shape

    # 如果 key 张量内存不连续，则转成连续内存，方便 Triton kernel 线性寻址。
    if not key.is_contiguous():
        key = key.contiguous()
    # 如果 value 张量内存不连续，则转成连续内存。
    if not value.is_contiguous():
        value = value.contiguous()

    # 断言 K cache 与 V cache 的形状必须完全一致。
    assert k_cache.shape == v_cache.shape, "K and V cache shapes must match"
    # 断言 slot mapping 的长度必须与要写入的 token 数量一致。
    assert slot_mapping.numel() == num_tokens, "Slot mapping size must match number of tokens"

    # 定义 kernel 启动网格：第 0 维遍历 token，第 1 维遍历 KV 头。
    grid = (num_tokens, num_kv_heads)
    # 启动 Triton kernel，把当前批次的 K/V 写入分页缓存。
    store_kvcache_kernel[grid](
        key,
        value,
        k_cache,
        v_cache,
        slot_mapping,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size
    )


# 定义 prefill 阶段的 FlashAttention Triton kernel，支持变长序列。
@triton.jit
def flash_attention_varlen_kernel(
    Q,
    K,
    V,
    O,
    cu_seqlens_q_ptr,
    scale,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # 第 0 维 program id 表示当前处理的是 query 的第几个 block。
    start_m = tl.program_id(0)
    # 第 1 维 program id 表示当前处理的是第几个 query 头。
    off_h = tl.program_id(1)
    # 第 2 维 program id 表示当前处理的是第几个序列。
    seq_idx = tl.program_id(2)

    # 根据 GQA 规则，把 query 头映射到它应使用的 KV 头。
    kv_head_idx = off_h // (num_heads // num_kv_heads)

    # 从累计长度数组中读取当前序列的起始位置。
    seq_start = tl.load(cu_seqlens_q_ptr + seq_idx)
    # 从累计长度数组中读取当前序列的结束位置。
    seq_end = tl.load(cu_seqlens_q_ptr + seq_idx + 1)
    # 计算当前序列长度。
    seq_len = seq_end - seq_start

    # 如果当前 query block 已经越过序列末尾，则无需继续计算。
    if start_m * BLOCK_M >= seq_len:
        return

    # 构造当前 query block 内的 token 偏移。
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # 构造 head_dim 范围的列偏移。
    offs_d = tl.arange(0, head_dim)

    # 计算当前 query block 的内存地址。
    q_ptrs = Q + (seq_start + offs_m[:, None]) * num_heads * head_dim + off_h * head_dim + offs_d[None, :]

    # 标记 query block 内哪些位置落在真实序列长度范围内。
    mask_m = offs_m < seq_len
    # 读取当前 query block。
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)

    # 初始化在线 softmax 的归一化系数 `l_i`。
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    # 初始化在线 softmax 的行最大值 `m_i`。
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - 1e10
    # 初始化输出累加器。
    acc = tl.zeros([BLOCK_M, head_dim], dtype=tl.float32)

    # 计算当前序列需要遍历多少个 K/V block。
    num_blocks = tl.cdiv(seq_len, BLOCK_N)

    # 逐个 K/V block 做在线 softmax 累加。
    for block_n in range(num_blocks):
        # 计算当前 K/V block 的起始 token 位置。
        start_n = block_n * BLOCK_N
        # 构造当前 K/V block 的 token 偏移。
        offs_n = start_n + tl.arange(0, BLOCK_N)

        # 标记当前 K/V block 中哪些位置落在真实序列长度内。
        mask_n = offs_n < seq_len

        # 计算当前 K block 的内存地址。
        k_ptrs = K + (seq_start + offs_n[None, :]) * num_kv_heads * head_dim + kv_head_idx * head_dim + offs_d[:, None]

        # 读取当前 K block。
        k = tl.load(k_ptrs, mask=mask_n[None, :], other=0.0)

        # 计算当前 query block 与当前 K block 的点积注意力分数。
        qk = tl.dot(q, k)
        # 乘上 attention scale。
        qk = qk * scale

        # 构造 causal mask，保证位置只能看见自己及之前的 token。
        mask_causal = (offs_m[:, None] + seq_start) >= (offs_n[None, :] + seq_start)
        # 对非法位置填充极小值。
        qk = tl.where(mask_causal & mask_n[None, :], qk, -1e10)

        # 取当前 block 内每一行的最大值。
        m_ij = tl.max(qk, axis=1)
        # 更新“截至目前”为止的全局行最大值。
        m_i_new = tl.maximum(m_i, m_ij)
        # 计算旧累加器缩放系数。
        alpha = tl.exp(m_i - m_i_new)
        # 计算当前 block 的未归一化 softmax 概率。
        p = tl.exp(qk - m_i_new[:, None])

        # 先把旧累加器根据新的最大值做重缩放。
        acc = acc * alpha[:, None]

        # 计算当前 V block 的内存地址。
        v_ptrs = V + (seq_start + offs_n[:, None]) * num_kv_heads * head_dim + kv_head_idx * head_dim + offs_d[None, :]
        # 读取当前 V block。
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

        # 将当前 block 对输出的贡献累加到结果中。
        acc = acc + tl.dot(p.to(v.dtype), v)

        # 更新归一化分母。
        l_i = l_i * alpha + tl.sum(p, axis=1)
        # 更新每一行的全局最大值。
        m_i = m_i_new

    # 用在线 softmax 的分母对累加器做最终归一化。
    acc = acc / l_i[:, None]

    # 计算输出张量中当前 query block 的写入地址。
    o_ptrs = O + (seq_start + offs_m[:, None]) * num_heads * head_dim + off_h * head_dim + offs_d[None, :]
    # 把结果写回输出张量。
    tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=mask_m[:, None])


# 定义 prefill 阶段 FlashAttention 的 Python 包装函数。
def flash_attention_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale: float,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    # 确保 q 张量内存连续。
    q = q.contiguous()
    # 确保 k 张量内存连续。
    k = k.contiguous()
    # 确保 v 张量内存连续。
    v = v.contiguous()

    # 预先分配与 q 同形状的输出张量。
    output = torch.empty_like(q)

    # 如果 head_dim 不超过 64，则使用较大的 block 配置。
    if head_dim <= 64:
        BLOCK_M = 64
        BLOCK_N = 64
    # 如果 head_dim 不超过 128，则使用中等 block 配置。
    elif head_dim <= 128:
        BLOCK_M = 32
        BLOCK_N = 32
    # 更大的 head_dim 则保守地用更小 block，减少共享内存压力。
    else:
        BLOCK_M = 16
        BLOCK_N = 16

    # 计算当前 batch 中的序列数。
    num_seqs = cu_seqlens.shape[0] - 1

    # 为了计算启动网格，需要先把累计长度搬到 CPU 上取最大序列长度。
    cu_seqlens_cpu = cu_seqlens.cpu()
    # 根据相邻累计长度之差求出最大序列长度。
    max_seq_len = (cu_seqlens_cpu[1:] - cu_seqlens_cpu[:-1]).max().item()

    # 定义三维启动网格：query block 数、头数、序列数。
    grid = (triton.cdiv(max_seq_len, BLOCK_M), num_heads, num_seqs)

    # 启动 FlashAttention kernel。
    flash_attention_varlen_kernel[grid](
        q,
        k,
        v,
        output,
        cu_seqlens,
        scale,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )

    # 返回 prefill 输出。
    return output


# 定义 decode 阶段的 paged attention Triton kernel。
@triton.jit
def paged_attention_decode_kernel(
    output_ptr,
    query_ptr,
    k_cache_ptr,
    v_cache_ptr,
    block_tables_ptr,
    context_lens_ptr,
    scale: tl.constexpr,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    max_num_blocks: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # 第 0 维 program id 对应 batch 中的第几个序列。
    batch_idx = tl.program_id(0)
    # 第 1 维 program id 对应当前处理的是第几个 query 头。
    head_idx = tl.program_id(1)

    # 根据 GQA 规则把 query 头映射到对应的 KV 头。
    kv_head_idx = head_idx // (num_heads // num_kv_heads)

    # 读取当前序列已有的上下文长度。
    context_len = tl.load(context_lens_ptr + batch_idx)

    # 构造 head_dim 范围的向量偏移。
    offs_d = tl.arange(0, head_dim)
    # 计算当前 query 向量在输入张量中的线性地址。
    q_offset = batch_idx * num_heads * head_dim + head_idx * head_dim + offs_d
    # 读取当前 query 向量。
    q = tl.load(query_ptr + q_offset)

    # 初始化输出累加器。
    acc = tl.zeros([head_dim], dtype=tl.float32)
    # 初始化在线 softmax 的分母。
    l_i = 0.0
    # 初始化在线 softmax 的最大值。
    m_i = -1e10

    # 根据 block table 的最大长度和 block_size 估计最多需要遍历多少个 token chunk。
    max_chunks = tl.cdiv(max_num_blocks * block_size, BLOCK_N)

    # 逐个 chunk 遍历历史上下文。
    for chunk_idx in range(max_chunks):
        # 计算当前 chunk 在序列中的起始 token 位置。
        token_start = chunk_idx * BLOCK_N

        # 只有当这个 chunk 还落在有效上下文长度范围内时才继续。
        if token_start < context_len:
            # 构造当前 chunk 内的 token 偏移。
            offs_n = token_start + tl.arange(0, BLOCK_N)
            # 标记当前 chunk 内的哪些 token 属于真实上下文。
            mask_n = offs_n < context_len

            # 初始化当前 chunk 的注意力分数数组。
            qk = tl.zeros([BLOCK_N], dtype=tl.float32) - 1e10

            # 逐 token 读取 K 并计算分数。
            for i in range(BLOCK_N):
                # 计算当前 token 的全局位置。
                token_idx = token_start + i
                # 如果该 token 还在真实上下文长度范围内，则继续。
                if token_idx < context_len:
                    # 计算当前 token 落在逻辑第几个 block。
                    block_num = token_idx // block_size
                    # 计算当前 token 在该 block 内的偏移。
                    block_offset = token_idx % block_size

                    # 如果逻辑 block 编号还在 block table 范围内，则继续。
                    if block_num < max_num_blocks:
                        # 计算 block table 中对应条目的线性地址。
                        block_table_offset = batch_idx * max_num_blocks + block_num
                        # 读取逻辑 block 对应的物理 block id。
                        physical_block_idx = tl.load(block_tables_ptr + block_table_offset)

                        # 如果物理 block id 不为 -1，说明该 block 有效。
                        if physical_block_idx != -1:
                            # 计算当前 token 的 K 向量在分页缓存中的线性地址。
                            k_offset = (
                                physical_block_idx * block_size * num_kv_heads * head_dim +
                                block_offset * num_kv_heads * head_dim +
                                kv_head_idx * head_dim +
                                offs_d
                            )
                            # 读取当前 token 的 K 向量。
                            k_vec = tl.load(k_cache_ptr + k_offset)

                            # 计算当前 query 与该 K 向量的点积分数。
                            score = tl.sum(q * k_vec) * scale

                            # 构造一个 one-hot 位置掩码，用于把当前分数写到 qk 的第 i 个位置。
                            mask_i = tl.arange(0, BLOCK_N) == i
                            # 将当前 token 的分数写入当前 chunk 的注意力分数向量。
                            qk = tl.where(mask_i, score, qk)

            # 对超出真实上下文长度的部分填充极小值。
            qk = tl.where(mask_n, qk, -1e10)

            # 取出当前 chunk 的最大分数。
            m_ij = tl.max(qk)
            # 与历史 chunk 的最大值合并，得到新的全局最大值。
            m_i_new = tl.maximum(m_i, m_ij)
            # 计算旧累加器重缩放因子。
            alpha = tl.exp(m_i - m_i_new)
            # 计算当前 chunk 的未归一化 softmax 权重。
            p = tl.exp(qk - m_i_new)

            # 按新的最大值对旧输出累加器做缩放。
            acc = acc * alpha
            # 按新的最大值对旧分母做缩放。
            l_i = l_i * alpha

            # 再次逐 token 读取 V，并把加权结果累加到输出中。
            for i in range(BLOCK_N):
                # 计算当前 token 的全局位置。
                token_idx = token_start + i
                # 如果该 token 还在有效上下文内，则继续。
                if token_idx < context_len:
                    # 计算逻辑 block 编号。
                    block_num = token_idx // block_size
                    # 计算 block 内偏移。
                    block_offset = token_idx % block_size

                    # 如果逻辑 block 还在 block table 范围内，则继续。
                    if block_num < max_num_blocks:
                        # 计算 block table 条目的线性地址。
                        block_table_offset = batch_idx * max_num_blocks + block_num
                        # 读取物理 block id。
                        physical_block_idx = tl.load(block_tables_ptr + block_table_offset)

                        # 如果物理 block id 有效，则读取对应 V。
                        if physical_block_idx != -1:
                            # 计算 V 向量在线性分页缓存中的地址。
                            v_offset = (
                                physical_block_idx * block_size * num_kv_heads * head_dim +
                                block_offset * num_kv_heads * head_dim +
                                kv_head_idx * head_dim +
                                offs_d
                            )
                            # 读取当前 token 的 V 向量。
                            v_vec = tl.load(v_cache_ptr + v_offset)

                            # 构造 one-hot 掩码，从 `p` 中取出当前 token 的 softmax 权重。
                            mask_i = tl.arange(0, BLOCK_N) == i
                            # 提取当前 token 的权重标量。
                            weight = tl.sum(tl.where(mask_i, p, 0.0))

                            # 把当前 token 的加权 V 累加到输出上。
                            acc = acc + weight * v_vec
                            # 同步更新 softmax 分母。
                            l_i = l_i + weight

            # 更新历史 chunk 最大值。
            m_i = m_i_new

    # 用最终分母归一化累加器，得到 decode 输出。
    output = acc / l_i

    # 计算输出向量在输出张量中的线性地址。
    output_offset = batch_idx * num_heads * head_dim + head_idx * head_dim + offs_d
    # 将输出写回结果张量。
    tl.store(output_ptr + output_offset, output)


# 定义 decode 阶段 paged attention 的 Python 包装函数。
def paged_attention_decode(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int
) -> torch.Tensor:
    # 读取 batch 大小。
    batch_size = query.shape[0]
    # 读取每个序列 block table 的最大逻辑 block 数。
    max_num_blocks = block_tables.shape[1]

    # 保证 query 内存连续，方便 kernel 做线性读取。
    query = query.contiguous()

    # 预先分配输出张量。
    output = torch.empty_like(query)

    # 根据 head_dim 选择每次遍历多少个历史 token。
    BLOCK_N = 64 if head_dim <= 128 else 32

    # 启动网格为“batch 中的序列数 × 头数”。
    grid = (batch_size, num_heads)

    # 启动 decode kernel。
    paged_attention_decode_kernel[grid](
        output,
        query,
        k_cache,
        v_cache,
        block_tables,
        context_lens,
        scale=scale,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
        max_num_blocks=max_num_blocks,
        BLOCK_N=BLOCK_N,
    )

    # 返回 decode 输出。
    return output


# 定义统一的 Attention 模块，对外暴露标准的 `forward(q, k, v)` 接口。
class Attention(nn.Module):
    # 定义初始化函数。
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scale: float = 1.0,
        num_kv_heads: int = None,
        block_size: int = 16,
    ):
        # 调用父类初始化。
        super().__init__()
        # 保存 query 头数。
        self.num_heads = num_heads
        # 保存每个头的维度。
        self.head_dim = head_dim
        # 保存外部传入的 scale。
        self.scale = scale
        # 保存 KV 头数；如果未显式给出，则默认与 query 头数一致。
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        # 保存分页缓存的 block 大小。
        self.block_size = block_size
        # 初始化空的 K/V cache 占位符，后续由 `ModelRunner.allocate_kv_cache()` 真正挂接。
        self.k_cache = self.v_cache = torch.tensor([])

    # 定义统一前向传播接口。
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # 读取当前运行时上下文。
        context = get_context()
        # 取出当前模块挂接到的 K/V cache。
        k_cache, v_cache = self.k_cache, self.v_cache

        # 只要当前模块已经被挂接到真实 cache，且当前上下文提供了 slot mapping，就先把当前步 K/V 写入缓存。
        if k_cache.numel() > 0 and v_cache.numel() > 0 and context.slot_mapping is not None:
            # 如果当前输入是四维张量，说明是 batched 形式，需要先拉平成 `(num_tokens, num_kv_heads, head_dim)`。
            if k.dim() == 4:
                # 解析四维张量的形状。
                B, N, num_kv_heads, head_dim = k.shape
                # 把 K 拉平成连续的二维 token 流。
                k_to_store = k.reshape(B * N, num_kv_heads, head_dim).contiguous()
                # 把 V 拉平成连续的二维 token 流。
                v_to_store = v.reshape(B * N, num_kv_heads, head_dim).contiguous()
            # 如果当前输入本身已经是三维 token 流，则直接保证内存连续即可。
            else:
                # 让 K 变成连续内存。
                k_to_store = k.contiguous()
                # 让 V 变成连续内存。
                v_to_store = v.contiguous()

            # 调用前面定义的辅助函数，把当前步 K/V 写入分页缓存。
            store_kvcache(k_to_store, v_to_store, k_cache, v_cache, context.slot_mapping, self.block_size)

        # 按标准 attention 公式计算最终 scale。
        scale = self.scale / (self.head_dim ** 0.5)

        # 如果当前处于 prefill 阶段，则走 FlashAttention 路径。
        if context.is_prefill:
            # 读取变长序列边界信息。
            cu_seqlens = context.cu_seqlens_q
            # 如果 prefill 路径缺少累计长度数组，则说明调用方准备数据有误。
            if cu_seqlens is None:
                raise ValueError("cu_seqlens_q must be provided for varlen attention")

            # 调用 prefill FlashAttention，得到形状为 `(total_tokens, num_heads, head_dim)` 的输出。
            o = flash_attention_prefill(
                q,
                k,
                v,
                cu_seqlens,
                scale,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim
            )
            # 把三维输出 reshape 成上层线性层需要的 `(total_tokens, num_heads * head_dim)` 形式。
            return o.reshape(o.shape[0], self.num_heads * self.head_dim)
        # 否则说明当前处于 decode 阶段，则走 paged attention 路径。
        else:
            # 调用分页 decode attention，从历史 K/V cache 中读取上下文信息。
            o = paged_attention_decode(
                q,
                k_cache,
                v_cache,
                context.block_tables,
                context.context_lens,
                scale,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                self.block_size
            )
            # 把三维输出 reshape 成上层线性层需要的 `(batch_size, num_heads * head_dim)` 形式。
            return o.reshape(o.shape[0], self.num_heads * self.head_dim)


# 当文件被直接运行时，执行下面的简单示例和性能测试。
if __name__ == "__main__":
    # 构造一个 Attention 层实例并移动到 GPU。
    layer = Attention(num_heads=8, head_dim=64).cuda()
    # 设置一个演示用的 batch、序列长度和总隐藏维度。
    B, N, D = 4, 1024, 512
    # 构造随机 query 张量。
    q = torch.randn(B, N, D).cuda()
    # 构造随机 key 张量。
    k = torch.randn(B, N, D).cuda()
    # 构造随机 value 张量。
    v = torch.randn(B, N, D).cuda()
    # 手动给层挂一个演示用的 K cache。
    layer.k_cache = torch.zeros(B, N, D).cuda()
    # 手动给层挂一个演示用的 V cache。
    layer.v_cache = torch.zeros(B, N, D).cuda()
    # 构造一个简单的 slot mapping。
    slot_mapping = torch.arange(N).cuda()

    # 先做 10 次预热。
    for _ in range(10):
        # 执行一次前向传播。
        _ = layer(q, k, v)

    # 导入 `time`，用于性能测试。
    import time
    # 创建一个列表记录每次测试耗时。
    times = []
    # 正式测试 100 次前向传播。
    for _ in range(100):
        # 在开始计时前同步 CUDA。
        torch.cuda.synchronize()
        # 记录开始时间。
        start_time = time.time()
        # 执行一次前向传播。
        output_tensor = layer(q, k, v)
        # 在结束计时前同步 CUDA。
        torch.cuda.synchronize()
        # 记录结束时间。
        end_time = time.time()
        # 把本次耗时记入列表。
        times.append(end_time - start_time)
    # 计算平均耗时。
    avg_time = sum(times) / len(times)
    # 打印平均耗时。
    print(f"Average inference time over 100 runs: {avg_time * 1000:.4f} ms")
