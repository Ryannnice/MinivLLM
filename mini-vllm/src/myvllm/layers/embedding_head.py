# 文件内容概览：
# 1. 定义 `VocabParallelEmbedding`，用于按词表维度切分 embedding 权重。
# 2. 定义 `ParallelLMHead`，用于在张量并行场景下按分片词表计算 logits。
# 3. 该文件同时处理权重装载、跨卡聚合与 prefill 阶段只取最后一个 token 的逻辑。
#
# 在项目中的作用：
# 1. 这是模型输入端和输出端都要用到的“词表并行”基础设施。
# 2. `Qwen3Model` / `LlamaModel` 的词嵌入来自这里，`Qwen3ForCausalLM` / `LlamaForCausalLM` 的输出头也来自这里。
# 3. 它把“大词表参数很大”这个问题压缩成“每张卡只保存自己负责的那部分词表”。

# 导入 PyTorch 主库。
import torch
# 从 `torch` 中导入 `nn` 命名空间，供模块定义使用。
from torch import nn
# 导入函数式接口 `F`，这里主要用来调用 `embedding`。
import torch.nn.functional as F
# 导入分布式模块，处理多卡场景下的 all_reduce / gather。
import torch.distributed as dist

# 从项目上下文工具中导入 `get_context`，用于 LM Head 判断当前是 prefill 还是 decode。
from myvllm.utils import get_context


# 定义按词表维度切分的并行嵌入层。
class VocabParallelEmbedding(nn.Module):
    # 定义初始化函数，参数分别是总词表大小和 embedding 维度。
    def __init__(self, num_embeddings: int, embedding_dim: int):
        # 调用父类初始化函数。
        super().__init__()
        # 读取当前张量并行的总卡数。
        self.tp_size = dist.get_world_size()
        # 读取当前进程在张量并行组中的 rank。
        self.tp_rank = dist.get_rank()

        # 保存原始词表大小。
        self.num_embeddings = num_embeddings
        # 把词表大小向上补齐到能被并行卡数整除，方便做均匀切分。
        self.padded_num_embeddings = (num_embeddings + self.tp_size - 1) // self.tp_size * self.tp_size
        # 计算当前每张卡要负责的词表分片大小。
        self.num_embeddings_per_partition = self.padded_num_embeddings // self.tp_size
        # 保存每个 token 对应的向量维度。
        self.embedding_dim = embedding_dim

        # 为当前 rank 分配本地词表分片的权重参数。
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
        # 给参数挂上自定义权重加载器，便于从完整 checkpoint 中切出本卡分片。
        self.weight.weight_loader = self.weight_loader

    # 定义权重加载函数，把完整 embedding 权重切到本卡负责的范围内。
    def weight_loader(self, param: nn.Parameter, loaded_weights: torch.Tensor):
        # 取出目标参数的底层数据引用。
        param_data = param.data

        # 计算本 rank 对应分片在完整词表中的起始偏移。
        offset = self.tp_rank * self.num_embeddings_per_partition
        # 当前分片理论上应拥有的词表行数。
        shard_size = self.num_embeddings_per_partition

        # 计算当前分片在真实未补齐词表中的起始位置。
        actual_start = min(offset, self.num_embeddings)
        # 计算当前分片在真实未补齐词表中的结束位置。
        actual_end = min(offset + shard_size, self.num_embeddings)
        # 计算当前分片实际能装载到多少行有效词表权重。
        actual_size = max(0, actual_end - actual_start)

        # 如果当前分片覆盖到了真实词表范围，则拷贝这部分有效权重。
        if actual_size > 0:
            # 从完整权重中切出当前分片对应的有效范围。
            sharded_weights = loaded_weights.narrow(0, actual_start, actual_size)
            # 把切出来的有效范围拷贝到参数前部。
            param_data[:actual_size].copy_(sharded_weights)

        # 如果由于补齐导致本地分片比真实数据大，则把剩余部分清零。
        if actual_size < shard_size:
            # 将填充出来的那部分权重置零，避免脏数据影响结果。
            param_data[actual_size:].zero_()

    # 定义前向传播函数，根据当前 rank 的词表分片返回本地 embedding 结果。
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 构造掩码，只让当前 rank 处理落在自己词表分片范围内的 token。
        mask = (x >= self.tp_rank * self.num_embeddings_per_partition) & \
               (x < (self.tp_rank + 1) * self.num_embeddings_per_partition) & \
               (x < self.num_embeddings)
        # 把全局 token id 平移到当前本地分片的局部索引空间。
        x = mask * (x - self.tp_rank * self.num_embeddings_per_partition)
        # 用局部索引在本地 embedding 权重中查表。
        output = F.embedding(x, self.weight)

        # 如果是多卡张量并行，则需要把各卡的局部结果汇总起来。
        if dist.get_world_size() > 1:
            # 对不属于本卡词表范围的位置再次置零，避免误把本地 id 0 的 embedding 当成有效输出。
            output = mask.unsqueeze(1) * output
            # 对所有 rank 的局部 embedding 结果做求和，得到完整 embedding。
            dist.all_reduce(output, op=dist.ReduceOp.SUM)
        # 返回最终的 embedding 结果。
        return output


# 定义并行 LM Head，它直接继承并复用并行 embedding 的权重切分逻辑。
class ParallelLMHead(VocabParallelEmbedding):
    # 定义初始化函数，参数仍然是总词表大小和隐藏维度。
    def __init__(self, num_embeddings: int, embedding_dim: int):
        # 直接复用父类的初始化过程。
        super().__init__(num_embeddings, embedding_dim)

    # 定义输出头的前向传播，输入是 hidden states。
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 读取当前运行时上下文。
        context = get_context()
        # 如果当前处于 prefill 阶段，则只保留每个序列最后一个 token 的 hidden state。
        if context.is_prefill:
            # 根据 `cu_seqlens_q` 找到每个序列最后一个 token 在扁平化张量中的位置。
            last_token = context.cu_seqlens_q[1:] - 1
            # 只取出每个序列最后一个 token 的 hidden states，并转成连续内存。
            x = x[last_token].contiguous()

        # 对本地词表分片做线性映射，得到局部 logits。
        logits = torch.nn.functional.linear(x, self.weight)
        # 如果是多卡张量并行，还需要把各卡负责的词表分片拼回完整词表。
        if self.tp_size > 1:
            # 仅在 rank 0 上预先分配 gather 列表，用于接收所有 rank 的局部 logits。
            all_logits = [torch.empty(logits.size(), device=logits.device) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
            # 把所有 rank 的局部 logits 收集到目标 rank 0 上。
            dist.gather(logits, gather_list=all_logits, dst=0)
            # 只有 rank 0 才能继续把局部结果拼起来。
            if self.tp_rank == 0:
                # 沿最后一维拼接所有局部词表分片，形成补齐后的完整词表 logits。
                logits = torch.cat(all_logits, dim=-1)
                # 去掉词表补齐带来的多余尾部维度，只保留真实词表大小。
                logits = logits[..., :self.num_embeddings]

        # 返回最终 logits；在多卡模式下只有 rank 0 会得到完整词表 logits。
        return logits
