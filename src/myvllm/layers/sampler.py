# 文件内容概览：
# 1. 定义项目中的最终采样层 `SamplerLayer`。
# 2. 它接收 logits 和每个序列各自的 temperature。
# 3. 内部先做温度缩放和 softmax，再通过 Gumbel-Max 风格方式采样 token。
#
# 在项目中的作用：
# 1. 这是模型前向完成之后、把概率分布变成离散 token 的最后一步。
# 2. `ModelRunner.run()` 会在 rank 0 上调用这里的采样逻辑。
# 3. 它把“输出 hidden states -> logits -> 下一个 token”这条链路闭合起来。

# 导入 PyTorch 主库。
import torch
# 导入 `torch.nn`，用于定义模块类。
import torch.nn as nn


# 定义采样层，继承自 `nn.Module`。
class SamplerLayer(nn.Module):
    # 使用中文文档字符串说明该模块的目标。
    """
    根据输入 logits 计算概率分布。
    再结合 temperature 做随机采样。
    输出每个序列采样得到的 token id。
    """

    # 定义初始化函数。
    def __init__(self):
        # 调用父类初始化逻辑。
        super().__init__()

    # 使用 `torch.compile` 编译采样路径。
    @torch.compile
    # 定义前向传播函数，输入为 logits 张量与 temperature 张量。
    def forward(self, logits: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
        # 使用每个序列自己的 temperature 对 logits 做缩放。
        logits /= temperature.unsqueeze(-1)
        # 对缩放后的 logits 做 softmax，得到概率分布。
        probs = torch.softmax(logits, dim=-1)
        # 使用指数噪声实现近似的 Gumbel-Max 采样，并返回最大值对应的 token 下标。
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        # 返回采样后的 token id。
        return sample_tokens
