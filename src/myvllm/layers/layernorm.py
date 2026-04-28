# 文件内容概览：
# 1. 定义项目里使用的 RMSNorm 层实现。
# 2. 同时支持“仅归一化”和“先加残差再归一化”两种前向路径。
# 3. 文件底部附带了简单的 CUDA 性能测试代码。
#
# 在项目中的作用：
# 1. Qwen3 和 Llama 的每个 decoder layer 都会用到这里的归一化层。
# 2. 它承担了 attention 前、attention 后以及最终输出前的稳定化工作。
# 3. 由于项目采用手写模型结构，这里需要自己实现一个可加载权重的 RMSNorm。

# 导入 PyTorch 主库，供张量运算和模块定义使用。
import torch
# 导入 `time`，用于文件底部的简单基准测试。
import time


# 定义一个自定义归一化层，继承自 `torch.nn.Module`。
class LayerNorm(torch.nn.Module):
    # 定义初始化函数，参数 `gamma` 是 RMSNorm 的缩放权重，`eps` 是数值稳定项。
    def __init__(self, gamma: torch.Tensor, eps: float = 1e-5):
        # 调用父类初始化函数。
        super().__init__()
        # 把传入的 `gamma` 复制成新的 `nn.Parameter`，使其成为可训练、可加载的参数。
        self.weight = torch.nn.Parameter(gamma.detach().clone())
        # 保存数值稳定项 `eps`。
        self.eps = eps

    # 定义一个只读属性 `gamma`，作为旧代码兼容别名。
    @property
    # 返回当前层的缩放权重。
    def gamma(self):
        # 返回真实存储参数的 `self.weight`。
        return self.weight

    # 使用 `torch.compile` 编译纯 RMSNorm 路径。
    @torch.compile
    # 定义不带残差的 RMSNorm 前向传播。
    def rms_forward(self, x: torch.Tensor) -> torch.Tensor:
        # 先对最后一维做平方、求均值，再加上 `eps`，得到均方值。
        variance = x.pow(2).mean(dim=-1, keepdim=True) + self.eps
        # 对均方值开平方，得到 RMS 分母。
        sqrt_variance = variance.sqrt()
        # 用输入除以 RMS 分母并乘上可学习权重，得到归一化结果。
        x_norm = x / sqrt_variance * self.weight
        # 返回归一化后的张量。
        return x_norm

    # 定义带残差的 RMSNorm 路径。
    def residual_rms_forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        # 先把主分支输入与残差相加。
        x = x + residual
        # 返回归一化后的结果，以及更新后的残差张量本身。
        return self.rms_forward(x), x

    # 统一的前向传播入口，既支持有残差也支持无残差。
    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None) -> torch.Tensor:
        # 如果调用者传入了残差张量，则走“残差相加后再 RMSNorm”的路径。
        if residual is not None:
            # 返回带残差路径的结果。
            return self.residual_rms_forward(x, residual)
        # 否则直接对输入做 RMSNorm。
        else:
            # 返回无残差路径的结果。
            return self.rms_forward(x)


# 当该文件被直接运行时，执行下面的简单示例和性能测试。
if __name__ == "__main__":
    # 构造一个大尺寸随机输入张量并放到 GPU 上。
    x = torch.randn(8, 4000, 8000).cuda()
    # 构造一个长度为 hidden_size 的缩放向量，数值为 0.5，仅用于演示。
    gamma = torch.full((8000,), 0.5, device="cuda", dtype=x.dtype)
    # 创建 LayerNorm 层实例并移动到 GPU。
    layer = LayerNorm(gamma=gamma).cuda()
    # 构造一个与输入同形状的残差张量。
    residual = torch.full_like(x, fill_value=1)

    # 做若干次预热，避免首次编译和首次 kernel 启动影响统计结果。
    for _ in range(10):
        # 执行一次无残差前向传播。
        _ = layer(x)
    
    # 创建列表记录“无残差”路径的耗时。
    times = []
    # 测试 100 次无残差路径的前向传播性能。
    for _ in range(100):
        # 在开始计时前同步 CUDA。
        torch.cuda.synchronize()
        # 记录起始时间。
        start_time = time.time()
        # 执行一次无残差前向传播。
        _ = layer(x)
        # 在结束计时前同步 CUDA。
        torch.cuda.synchronize()
        # 记录结束时间。
        end_time = time.time()
        # 将本次耗时加入列表。
        times.append(end_time - start_time)
    # 计算无残差路径的平均耗时。
    avg_time = sum(times) / len(times)
    # 打印无残差路径的平均耗时。
    print(f"[Without residuals] Average inference time over 100 runs: {avg_time * 1000:.4f} ms")

    # 清空耗时列表，准备统计带残差路径。
    times.clear()
    # 测试 100 次带残差路径的前向传播性能。
    for _ in range(100):
        # 在开始计时前同步 CUDA。
        torch.cuda.synchronize()
        # 记录起始时间。
        start_time = time.time()
        # 执行一次带残差的前向传播。
        _ = layer(x, residual)
        # 在结束计时前同步 CUDA。
        torch.cuda.synchronize()
        # 记录结束时间。
        end_time = time.time()
        # 将本次耗时加入列表。
        times.append(end_time - start_time)
    # 计算带残差路径的平均耗时。
    avg_time = sum(times) / len(times)
    # 打印带残差路径的平均耗时。
    print(f"[With residuals] Average inference time over 100 runs: {avg_time * 1000:.4f} ms")
