# 文件内容概览：
# 1. 定义把余弦 / 正弦旋转应用到 query、key 上的辅助函数。
# 2. 定义 `RotaryEmbedding` 模块，负责预计算 RoPE 的 cos/sin 缓存。
# 3. 同时支持普通 RoPE 和 Llama 3.2 风格的长上下文频率修正。
#
# 在项目中的作用：
# 1. 这是 attention 模块获得位置信息的关键组件。
# 2. `Qwen3Attention` 和 `LlamaAttn` 都会在 Q、K 上调用这里的旋转位置编码。
# 3. 它把“token 在序列中的位置”映射为对注意力向量的几何旋转。

# 导入 `torch.nn`，用于定义模块类。
import torch.nn as nn
# 导入 PyTorch 主库，供张量构造和数值运算使用。
import torch


# 定义一个辅助函数，把已经准备好的 cos / sin 应用到输入张量上。
def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # 如果输入是三维张量，说明当前走的是变长扁平化 prefill 路径。
    if x.dim() == 3:
        # 读取输入张量的形状信息：总 token 数、头数、head_dim。
        total_tokens, num_heads, head_dim = x.shape
        # 在 head 维前插入一个维度，让 cos 可以广播到所有注意力头。
        cos = cos.unsqueeze(1) 
        # 在 head 维前插入一个维度，让 sin 可以广播到所有注意力头。
        sin = sin.unsqueeze(1) 

        # 沿最后一维把输入切成两半，分别作为旋转前的两组坐标。
        x1, x2 = x.chunk(2, dim=-1)

        # 计算旋转后的第一半分量。
        out1 = x1 * cos - x2 * sin
        # 计算旋转后的第二半分量。
        out2 = x1 * sin + x2 * cos

        # 在最后一维把旋转后的两半重新拼接起来并返回。
        return torch.cat([out1, out2], dim=-1)
    # 否则输入应为四维张量，说明当前走的是普通 batched 形式。
    else:
        # 读取 batch 维大小。
        B = x.size(0)
        # 读取序列长度。
        seq_len = x.size(1)
        # 读取注意力头数量。
        num_heads = x.size(2)
        # 读取每个头的维度。
        head_dim = x.size(-1)

        # 在 batch 维和 head 维插入长度为 1 的维度，使 cos 可以广播到所有 batch 和所有 head。
        cos = cos.unsqueeze(0).unsqueeze(2)
        # 在 batch 维和 head 维插入长度为 1 的维度，使 sin 可以广播到所有 batch 和所有 head。
        sin = sin.unsqueeze(0).unsqueeze(2)

        # 沿最后一维把输入切成两半。
        x1, x2 = x.chunk(2, dim=-1)

        # 计算旋转后的第一半分量。
        out1 = x1 * cos - x2 * sin
        # 计算旋转后的第二半分量。
        out2 = x1 * sin + x2 * cos

        # 把旋转后的两半沿最后一维拼接起来并返回。
        return torch.cat([out1, out2], dim=-1)


# 定义 RoPE 模块类。
class RotaryEmbedding(nn.Module):
    # 定义初始化函数，包含基础频率、旋转维度、最大位置和 Llama 3 专用参数。
    def __init__(
        self,
        base: int,
        rotary_embedding: int,
        max_position: int = 2048,
        is_llama3: bool = False,
        llama3_rope_factor: float = 32.0,
        llama3_rope_high_freq_factor: float = 4.0,
        llama3_rope_low_freq_factor: float = 1.0,
        llama3_rope_original_max_position_embeddings: int = 8192,
    ):
        # 调用父类初始化。
        super().__init__()
        # 保存 RoPE 的基础频率底数。
        self.base = base
        # 保存要施加旋转的位置编码维度。
        self.rotary_embedding = rotary_embedding
        # 保存允许预计算的最大位置长度。
        self.max_position = max_position
        # 根据标准 RoPE 公式预计算每一对维度的逆频率。
        self.inv_freq = 1 / (base ** (torch.arange(0, self.rotary_embedding, 2) / self.rotary_embedding))
        # 越往后的维度，频率越小，旋转越慢

        # 如果当前选择的是 Llama 3 风格 RoPE，则对逆频率做长上下文修正。
        if is_llama3:
            # 导入数学库，后面要用到圆周率。
            import math
            # 先取出当前逆频率。
            inv_freq = self.inv_freq
            # 把逆频率换算成对应的波长。
            wave_len = 2 * math.pi / inv_freq
            # 如果低频因子和高频因子相同，则走一个更简单的分段缩放策略。
            if llama3_rope_low_freq_factor == llama3_rope_high_freq_factor:
                # 对足够低频的部分做缩放，其余部分保持不变。
                inv_freq = torch.where(
                    wave_len < llama3_rope_original_max_position_embeddings / llama3_rope_high_freq_factor,
                    inv_freq,
                    inv_freq / llama3_rope_factor,
                )
            # 否则走一个平滑过渡版本的缩放策略。
            else:
                # 先计算高低频因子的差值。
                delta = llama3_rope_high_freq_factor - llama3_rope_low_freq_factor
                # 计算每个频率所在的平滑插值系数。
                smooth = (llama3_rope_original_max_position_embeddings / wave_len - llama3_rope_low_freq_factor) / delta
                # 把插值系数裁剪到 0 到 1 之间。
                smooth = torch.clamp(smooth, 0, 1)
                # 根据平滑系数在“原始频率”和“缩放频率”之间插值。
                factor = (1 - smooth) / llama3_rope_factor + smooth
                # 得到修正后的逆频率。
                inv_freq = factor * inv_freq
            # 把修正后的逆频率写回成员变量。
            self.inv_freq = inv_freq

        # 预生成从 0 到 `max_position - 1` 的位置索引。
        positions = torch.arange(self.max_position).float()
        # 用外积的方式计算每个位置、每个频率对应的相位值。
        freqs = torch.einsum("i,j -> ij", positions, self.inv_freq)  

        # 对相位值取 cos，得到余弦缓存。
        cos = torch.cos(freqs)
        # 对相位值取 sin，得到正弦缓存。
        sin = torch.sin(freqs)

        # 把 cos 与 sin 在最后一维拼接起来，方便后续一次索引再拆开。
        cos_sin_cache = torch.cat([cos, sin], dim=-1)
        # 将 cos/sin 缓存注册成 buffer，使其随模块迁移设备但不参与训练。
        self.register_buffer("cos_sin_cache", cos_sin_cache)

    # 使用 `torch.compile` 编译 RoPE 前向过程。
    @torch.compile
    # 定义前向传播，输入是位置索引、query 张量、key 张量。
    def forward(self, positions, query, key):
        # 根据当前位置索引从缓存中取出需要的 cos/sin。
        cos_sin = self.cos_sin_cache[positions]
        # 把拼接在一起的缓存拆回 cos 与 sin 两部分。
        cos, sin = cos_sin.chunk(2, dim=-1)
        # 分别把同一组 cos/sin 旋转到 query 和 key 上，并一起返回。
        return (
            apply_rotary_pos_emb(query, cos, sin),
            apply_rotary_pos_emb(key, cos, sin)
        )


# 当文件被直接执行时，运行下面的简单演示代码。
if __name__ == "__main__":
    # 设置一个示例 base。
    base = 5
    # 设置旋转维度。
    rotary_dim = 16
    # 设置最大位置长度。
    max_position = 100
    # 打印偶数维的索引位置。
    print(torch.arange(0, rotary_dim, 2))
    # 打印 base 的幂次展开结果。
    print(base ** (torch.arange(0, rotary_dim, 2) / rotary_dim))
    # 手动计算一次逆频率，方便观察公式结果。
    inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2) / rotary_dim))
    # 打印逆频率张量。
    print(inv_freq)

    # 构造位置索引张量。
    t = torch.arange(max_position).float()

    # 计算位置与逆频率的外积。
    freqs = torch.einsum("i,j -> ij", t, inv_freq)

    # 打印频率矩阵的形状。
    print(freqs.size())

    # 打印第 3 个位置对应的频率向量。
    print(freqs[2])
