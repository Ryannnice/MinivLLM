# 文件内容概览：
# 1. 定义项目里所有[并行线性层]的基础类与具体实现。
# 2. 包括 普通复制线性层、列并行线性层、行并行线性层、合并列并行层和专用的 QKV 列并行层。
# 3. 同时实现了“如何从完整 checkpoint 权重切出当前 rank 分片”的权重加载逻辑。
#
# 在项目中的作用：
# 1. 这是模型做张量并行的核心基础设施。
# 2. attention 的 QKV 投影、MLP 的 gate/up/down 投影、输出投影都建立在这些类上。
# 3. 它负责把“完整模型权重”映射成“当前 GPU 只保留自己负责的那一片参数”。

# 导入 `torch.nn` 并命名为 `nn`，用于定义模块和参数。
import torch.nn as nn
# 导入 PyTorch 主库，供张量与分布式结果张量运算使用。
import torch
# 导入分布式接口，供张量并行时获取 rank / world size 与执行 all_reduce。
import torch.distributed as dist


# 定义所有线性层的公共基类。
class LinearBase(nn.Module):
    # 使用中文文档字符串说明这个基类的职责。
    """
    所有线性层的共同父类。
    这里统一保存张量并行相关信息，并提供权重参数与 bias 参数的初始化框架。
    子类需要各自实现真正的权重加载和前向传播逻辑。
    """

    # 定义初始化函数，参数为输入维度、输出维度、是否带 bias，以及张量并行切分维度。
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        tp_dim: int | None = None
    ):
        # 调用父类初始化逻辑。
        super().__init__()
        # 保存张量并行切分发生在哪个维度上。
        self.tp_dim = tp_dim
        # 获取当前进程在张量并行组中的 rank。
        self.tp_rank = dist.get_rank()
        # 获取当前张量并行组中的总卡数。
        self.tp_size = dist.get_world_size()

        # 创建线性层权重参数，形状约定为 `[output_size, input_size]`。
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        # 给权重参数挂上自定义 loader，便于后续从完整 checkpoint 中切分装载。
        self.weight.weight_loader = self.weight_loader

        # 如果当前线性层需要 bias，则创建 bias 参数。
        if bias:
            # bias 的长度等于当前线性层的输出维度。
            self.bias = nn.Parameter(torch.zeros(output_size))
            # 同样给 bias 挂上自定义 loader。
            self.bias.weight_loader = self.weight_loader
        # 如果不需要 bias，则显式注册一个名为 `bias` 的空参数。
        else:
            self.register_parameter('bias', None)

    # 定义一个抽象的权重加载接口，要求子类实现。
    def weight_loader(self, param: nn.Parameter, loaded_weights: torch.Tensor):
        # 基类不提供实现，直接抛出异常提醒子类覆盖。
        raise NotImplementedError("Subclasses should implement this method.")

    # 定义一个抽象的前向传播接口，要求子类实现。
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 基类不提供实现，直接抛出异常提醒子类覆盖。
        raise NotImplementedError("Subclasses should implement this method.")


# 用多行注释解释自定义 weight_loader 的使用背景。
"""
这些自定义的 `weight_loader` 用来解决一个核心问题：
模型部署时，当前 GPU 上保存的是 “被张量并行切分后的参数”
但从 checkpoint 读进来的通常是 “完整模型参数”

因此在加载权重时，不能简单做 `param.data.copy_(loaded_weight)`
而是需要根据当前 rank 和切分方式，先把完整权重切到属于本卡的那一片，再写入本地参数。
"""


# 定义最简单的[复制式]线性层，不做张量并行切分
class ReplicatedLinear(LinearBase):
    # 定义初始化函数。
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True
    ):
        # 直接调用基类初始化，完整保留输入输出维度。
        super().__init__(input_size, output_size, bias)

    # 定义权重加载逻辑。
    def weight_loader(self, param: nn.Parameter, loaded_weights: torch.Tensor):
        # 复制式线性层不做切分，直接完整拷贝权重即可。
        param.data.copy_(loaded_weights)

    # 定义前向传播逻辑。
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 直接调用标准线性层计算。
        return nn.functional.linear(x, self.weight, self.bias)


# 定义[列并行]线性层：沿输出维度切分。
class ColumnParallelLinear(LinearBase):
    # 定义初始化函数。
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
    ):
        # 获取当前张量并行组的总卡数。
        tp_size = dist.get_world_size()

        # 列并行要求输出维度能被卡数整除。
        assert output_size % tp_size == 0, "Output size must be divisible by tensor parallel size."

        # 传给基类的输出维度是“当前 rank 持有的那一片输出维度”。
        super().__init__(input_size, output_size // tp_size, bias, tp_dim=0)

    # 定义[列并行]的权重加载逻辑。
    def weight_loader(self, param: nn.Parameter, loaded_weights: torch.Tensor):
        # 取出目标参数数据。
        param_data = param.data
        # 完整权重的第 0 维就是总输出维度：output = linear(x, weight, bias)，其中 weight 的形状是：[out_features, in_features]
        full_data_output_size = loaded_weights.size(0)

        # 计算每个 rank 理论应拿到多少输出行。
        shard_size = full_data_output_size // self.tp_size
        # 断言切分后大小与本地参数形状一致。
        assert shard_size == param_data.size(0), "Shard size does not match parameter size."
        
        # 计算当前 rank 在完整输出维上的起始位置。
        start_index = self.tp_rank * shard_size
        # 从完整权重中沿第 0 维切出本 rank 负责的输出分片。tensor.narrow(dim, start, length)：含义分别是：- dim：沿哪一维切  - start：从哪里开始  - length：取多长
        slided_weight = loaded_weights.narrow(0, start_index, shard_size)
        # 把切出来的分片写入本地参数。
        param_data.copy_(slided_weight)

    # 定义前向传播逻辑。
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 当前 rank 只会计算自己负责的那部分输出特征。
        return nn.functional.linear(x, self.weight, self.bias)


# 定义一个“合并多个列并行矩阵”的线性层。
class MergedColumnParallelLinear(ColumnParallelLinear):
    # 定义初始化函数，`output_sizes` 表示被合并的各个输出矩阵大小。
    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = True,
    ):
        # 保存各个子矩阵原始输出维大小。
        self.output_sizes = output_sizes
        # 调用父类初始化，把多个输出维求和后当成一个大矩阵处理。
        super().__init__(input_size, sum(output_sizes), bias)

    # 定义合并列并行层的权重加载逻辑。
    def weight_loader(self, param: nn.Parameter, loaded_weights: torch.Tensor, loaded_weight_id: int):
        # 取出目标参数数据。
        param_data = param.data
        # 计算当前要装载的是第几个子矩阵在本地大矩阵中的偏移位置。
        offset = sum(self.output_sizes[:loaded_weight_id]) // self.tp_size
        
        # 计算当前子矩阵在本 rank 上对应的分片大小。
        shard_size = self.output_sizes[loaded_weight_id] // self.tp_size
        # 在大矩阵参数里先 narrow 到当前子矩阵对应的那一段。
        param_data = param_data.narrow(0, offset, shard_size)
        # 计算完整权重在本 rank 上应切出的起始位置。
        loaded_weights_start_index = self.tp_rank * shard_size
        # 从完整子矩阵里切出当前 rank 负责的输出分片。
        shard_weights = loaded_weights.narrow(0, loaded_weights_start_index, shard_size)
        # 将分片权重写入本地参数的对应区间。
        param_data.copy_(shard_weights)


# 定义专门为 attention QKV 投影服务的列并行线性层。
class QKVColumnParallelLinear(ColumnParallelLinear):
    # 定义初始化函数。
    def __init__(
        self,
        input_size: int,
        head_size: int,
        num_heads: int,
        num_kv_heads: int | None = None,
        bias: bool = False,
    ):
        # 读取当前张量并行组的卡数。
        self.tp_size = dist.get_world_size()
        # 如果未显式指定 KV 头数，则默认与 Q 头数相同。
        num_kv_heads = num_kv_heads or num_heads
        # 保存单个注意力头的维度。
        self.head_size = head_size
        # 计算当前 rank 本地拥有多少个 query 头。
        self.num_heads = num_heads // self.tp_size
        # 计算当前 rank 本地拥有多少个 kv 头。
        self.num_kv_heads = num_kv_heads // self.tp_size
        # 计算当前 rank 本地输出总维度：Q + K + V。
        self.output_size = head_size * (self.num_heads + 2 * self.num_kv_heads)
        # 计算完整大矩阵的总输出维度。
        total_output_size = head_size * (num_heads + 2 * num_kv_heads)
        # 调用父类初始化，让父类继续按列并行方式切分。
        super().__init__(input_size, total_output_size, bias=bias)

    # 定义 QKV 专用的权重加载逻辑。
    def weight_loader(self, param: nn.Parameter, loaded_weights: torch.Tensor, load_weight_id: str):
        # 取出目标参数数据。
        param_data = param.data
        # 检查当前要加载的子权重类型只能是 q / k / v。
        assert load_weight_id in ['q', 'k', 'v'], "load_weight_id must be one of 'q', 'k', 'v'"
        # 如果当前加载的是 query 权重。
        if load_weight_id == 'q':
            # query 分片在本地大矩阵中的起始偏移为 0。
            offset = 0
            # query 分片大小等于本地 query 头数乘以 head_size。
            shard_size = self.head_size * self.num_heads
        # 如果当前加载的是 key 权重。
        elif load_weight_id == 'k':
            # key 分片的起点紧接在 query 分片之后。
            offset = self.head_size * self.num_heads
            # key 分片大小等于本地 kv 头数乘以 head_size。
            shard_size = self.head_size * self.num_kv_heads
        # 如果当前加载的是 value 权重。
        elif load_weight_id == 'v':
            # value 分片的起点在 query 与 key 两段之后。
            offset = self.head_size * self.num_heads + self.head_size * self.num_kv_heads
            # value 分片大小等于本地 kv 头数乘以 head_size。
            shard_size = self.head_size * self.num_kv_heads
        # 理论上走不到这里，但为了防御式编程仍保留异常分支。
        else:
            # 如果传入非法类型，则抛出错误。
            raise ValueError(f"Unknown load_weight_id: {load_weight_id}")

        # 在本地大矩阵中 narrow 出当前 q / k / v 对应的目标区间。
        param_data = param_data.narrow(0, offset, shard_size)
        # 计算完整权重中当前 rank 的起始位置。
        loaded_weights_start_index = self.tp_rank * shard_size
        # 从完整 q / k / v 权重中切出本 rank 对应的输出分片。
        shard_weights = loaded_weights.narrow(0, loaded_weights_start_index, shard_size)

        # 把分片后的权重写入本地目标参数区间。
        param_data.copy_(shard_weights)


# 定义行并行线性层：沿输入维度切分。
class RowParallelLinear(LinearBase):
    # 定义初始化函数。
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
    ):
        # 获取张量并行组中的总卡数。
        tp_size = dist.get_world_size()
        # 行并行要求输入维度可以被卡数整除。
        assert input_size % tp_size == 0, "Input size must be divisible by tensor parallel size."
        # 传给基类的输入维度是“本 rank 实际持有的输入分片大小”。
        super().__init__(input_size // tp_size, output_size, bias, tp_dim=1)

    # 定义行并行的权重加载逻辑。
    def weight_loader(self, param: nn.Parameter, loaded_weights: torch.Tensor):
        # 取出目标参数数据。
        param_data = param.data
        # 完整权重的第 1 维是完整输入维度。
        full_data_input_size = loaded_weights.size(1)
        # 计算每个 rank 在输入维上持有的分片大小。
        shard_size = full_data_input_size // self.tp_size
        # 断言切分后的大小与本地参数形状一致。
        assert shard_size == param_data.size(1), "Shard size does not match parameter size."
        # 计算当前 rank 在完整输入维上的起始偏移。
        start_index = self.tp_rank * shard_size
        # 从完整权重的第 1 维切出本 rank 负责的输入列分片。
        slided_weight = loaded_weights.narrow(1, start_index, shard_size)
        # 把分片写入本地参数。
        param_data.copy_(slided_weight)

    # 定义前向传播逻辑。
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 先用本地输入分片和本地权重计算部分输出。
        result = nn.functional.linear(x, self.weight, self.bias)
        # 如果是多卡张量并行，则把所有 rank 的部分输出求和，还原完整输出。
        if self.tp_size > 1:
            # 对所有 rank 的局部结果做 all_reduce 求和。
            dist.all_reduce(result, op=dist.ReduceOp.SUM)
        # 返回最终输出结果。
        return result


# 当文件被直接执行时，运行下面的简单示例代码。
if __name__ == "__main__":
    # 如果当前分布式环境可用但尚未初始化，则先初始化一个单进程示例环境。
    if dist.is_available() and not dist.is_initialized():
        # 初始化一个基于 gloo 的最小分布式进程组，仅用于本地演示。
        dist.init_process_group(
            backend="gloo",
            init_method="tcp://127.0.0.1:29500",
            rank=0,
            world_size=1,
        )
    # 构造一个基类实例，仅用于演示对象创建。
    layer = LinearBase(input_size=10, output_size=5)
    # 打印创建结果。
    print("LinearBase layer initialized:", layer)
