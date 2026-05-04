# 文件内容概览：
# 1. 统一导出 layers 子目录中最常用的层、模块和工具类。
# 2. 让上层代码可以通过 `from myvllm.layers import *` 直接拿到需要的名字。
# 3. 同时保留 `linear.py` 中通过通配符导出的并行线性层相关符号。
#
# 在项目中的作用：
# 1. 这是 `src/myvllm/layers` 的包入口。
# 2. `models/qwen3.py`、`models/llama.py` 等文件都会从这里集中导入构建模型所需的层。
# 3. 它把底层算子组织成一个统一接口，减少上层模块的导入复杂度。

# 从激活函数文件中导入 `SiluAndMul`，供上层模型直接使用。
from .activation import SiluAndMul
# 从注意力文件中导入 `Attention`，供模型中的自注意力模块复用。
from .attention import Attention
# 从词嵌入与 LM Head 文件中导入并行词嵌入和并行输出头。
from .embedding_head import ParallelLMHead, VocabParallelEmbedding
# 从归一化文件中导入自定义的 RMSNorm 实现。
from .layernorm import LayerNorm
# 通过通配符导入线性层文件中定义的并行线性层和相关公共符号。
from .linear import *
# 从 RoPE 文件中导入旋转位置编码模块。
from .rotary_embedding import RotaryEmbedding
# 从采样文件中导入最终的采样层。
from .sampler import SamplerLayer
