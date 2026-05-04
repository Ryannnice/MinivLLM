# 文件内容概览：
# 1. 定义 `SiluAndMul` 激活层。
# 2. 该层会把输入按最后一维一分为二，对前半部分做 SiLU，再与后半部分逐元素相乘。
# 3. 文件末尾带有一个简单的 GPU 基准测试示例。
#
# 在项目中的作用：
# 1. 这是项目里 MLP 模块使用的核心激活函数。
# 2. `Qwen3MLP` 和 `LlamaMLP` 都会调用这里的实现。
# 3. 它对应现代大模型里常见的 SwiGLU / SiLU-gated 风格前馈激活。

# 导入 PyTorch 主库，并使用简称 `torch` 方便后续调用张量与 CUDA 接口。
import torch
# 导入 `torch.nn` 并命名为 `nn`，用于定义自定义神经网络层。
import torch.nn as nn
# 导入函数式接口 `torch.nn.functional`，这里主要为了调用 `silu`。
import torch.nn.functional as F
# 导入 `time`，供文件底部的简单性能测试使用。
import time


# 定义一个自定义激活层，继承自 `nn.Module`。
class SiluAndMul(nn.Module):
    # 用中文文档字符串说明该模块的语义。
    """
    将输入沿最后一维切成两半。
    前半部分经过 SiLU 激活。
    激活后的结果再与后半部分逐元素相乘。
    """

    # 定义初始化函数。
    def __init__(self):
        # 调用父类 `nn.Module` 的初始化逻辑。
        super().__init__()

    # 使用 `torch.compile` 尝试让这个小算子在大张量场景下获得更好的执行效率。
    @torch.compile
    # 定义前向传播函数，输入是一个张量 `x`。
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 按最后一维把输入均分成两块，前半部分赋给 `x`，后半部分赋给 `y`。
        """
        内存布局：在前面的 Linear 层中，我们通常直接输出双倍维度的结果，然后在这个模块里“一分为二”。
        这样在显存中数据是连续的，读取效率最高。
        """
        x, y = x.chunk(2, -1)
        # 对前半部分做 SiLU，再与后半部分逐元素相乘，并返回结果。
        return F.silu(x) * y


# 当文件被直接执行而不是被导入时，运行下面这段示例代码。
if __name__ == "__main__":
    # 构造一个激活层实例并移动到 CUDA 设备上。
    layer = SiluAndMul().cuda()
    # 构造一个随机输入张量，形状仅用于演示和压测。
    input_tensor = torch.randn(8, 4000, 8000).cuda()
    
    # 先执行若干次预热，避免把首次编译和缓存建立时间计入正式统计。
    for _ in range(10):
        # 调用激活层做一次前向传播。
        _ = layer(input_tensor)

    # 创建一个空列表，用于记录每次测试的耗时。
    times = []
    # 正式执行 100 次前向传播并统计平均时间。
    for _ in range(100):
        # 在计时前同步 CUDA，避免异步执行导致统计失真。
        torch.cuda.synchronize()
        # 记录开始时间。
        start_time = time.time()
        # 执行一次前向传播。
        output_tensor = layer(input_tensor)
        # 在计时结束前再次同步 CUDA，确保 kernel 真正执行完成。
        torch.cuda.synchronize()
        # 记录结束时间。
        end_time = time.time()
        # 将本次耗时追加到列表中。
        times.append(end_time - start_time)
    # 计算 100 次测试的平均耗时。
    avg_time = sum(times) / len(times)
    # 打印平均推理时间，单位换算成毫秒更直观。
    print(f"Average inference time over 100 runs: {avg_time * 1000:.4f} ms")
