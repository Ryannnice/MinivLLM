# vLLM 深度调研

> 调研范围：本文基于 2026-05-01 可访问的 vLLM 官方文档、官方博客、PagedAttention 论文，以及 upstream 在线源码路径撰写。说明：这些 upstream 源码并不在当前仓库内，而是通过官方 GitHub 仓库交叉阅读确认。本文锁定的 upstream 仓库快照为 `92a7c121b62a1484b68c0a27d1ecefd1a84f78fc`。目标不是做 feature 清单，而是回答两个问题：`vLLM 为什么快`、`这些“快”在源码里是怎么落地的`。

> 证据约定：本文显式区分三类陈述。`源码事实` 指源码注释、类/函数职责、控制流与数据结构；`对照结论` 指在相同时间点对 upstream 代码路径做的横向比较；`性能推断` 指根据代码形态、官方文档和常见 serving 约束做出的工程判断，不把它写成 benchmark 结论。

## 1. 先给结论

vLLM 快，不是因为它把某个 attention kernel 单点做到极致，而是因为它把 LLM serving 里最昂贵的几个资源错配问题同时打掉了：

1. **KV cache 不再按“每请求一大块连续内存”来管理**  
   而是按 block/page 管理，显著降低碎片与过度预留。
2. **调度不再按“整批请求一起走完整阶段”来思考**  
   而是按“每个请求还欠多少 token 没算”来推进。
3. **prefill、decode、prefix cache、spec decode 被统一到了同一套 token 调度抽象里**
4. **高频 decode 小步的 Python/runtime 开销继续被 graph/compile/clean execution loop 压下去**
5. **后期能力继续往系统层扩展**  
   包括 disaggregated serving、KV connector、MoE/DBO、多类 attention backend、量化、KV 量化等。

一句话概括：

> **vLLM 的快，本质上是“用 block 化 KV 管理支撑持续调度，再用 prefix/spec/graph/系统拓扑继续把吞吐、延迟和资源利用率往上推”。**

---

## 2. 这份文档怎么读

如果你是第一次系统理解 vLLM，建议按下面顺序看：

1. 先看“代码地图”，搞清楚主要模块边界。
2. 再看“请求主链”，理解一个请求在 vLLM 里如何被推进。
3. 再看 scheduler，理解 continuous batching 真正是如何成立的。
4. 再看 KV cache manager，理解 PagedAttention 在系统层的真实价值。
5. 最后看 prefix cache、spec decode、graph/compile、分离式 serving 等增强层。

你会发现：  
**vLLM 的第一性原理不是 kernel，而是“调度 + KV 内存系统 + 高速执行路径”三件事绑在一起。**

---

## 3. 代码地图：先知道该看哪里

下面这张“阅读地图”比死记 feature 名字更重要。

| 代码区域 | 代表路径 | 主要职责 | 为什么和“快”直接相关 |
| --- | --- | --- | --- |
| 调度核心 | `vllm/v1/core/sched/scheduler.py` | 决定这一轮给谁算多少 token | continuous batching、chunked prefill、spec decode 都在这里统一 |
| KV 系统 | `vllm/v1/core/kv_cache_manager.py` | block 分配、prefix hit、usage、admission gate | PagedAttention 的系统收益都通过它释放 |
| KV 协调层 | `vllm/v1/core/kv_cache_coordinator.py` 等 | sliding window、多 group KV 管理、prefix block 组织 | 把底层 block pool 组织成 scheduler 可用接口 |
| attention / cache 执行 | 各 attention backend、paged attention 路径 | 真正读写 KV、执行 attention | 把 block table 变成可执行的 kernel 路径 |
| 编译/图执行 | `vllm/compilation/` | torch.compile、graph capture/replay、pass 管理 | 压低 decode 高频小步的固定开销 |
| Serving 增强层 | spec decode、KV connector、disaggregated serving、MoE/DBO | 把“快”从单机扩到更多工作负载与拓扑 | 决定大规模生产环境中的收益上限 |

如果只看 attention kernel，很容易误解 vLLM。  
真正决定吞吐的常常是：

- token budget 有没有被打满；
- request 能不能持续流动；
- prefix 命中后能不能直接减少调度成本；
- decode 每一步的 host/runtime 固定开销够不够低。

---

## 4. 先讲本质：为什么传统 LLM serving 慢

在理解 vLLM 之前，先看它到底在对抗什么。

### 4.1 KV cache 太大，而且增长方式高度不规则

自回归生成时，每个 token 的 K/V 都要长期保留。  
问题不在“算一次 attention”，而在“要一直把过去所有 token 的 K/V 放在显存里”。

传统做法如果按请求连续分配，通常会遇到：

- 过度预留；
- 释放/增长导致碎片；
- 请求长度不规则时，显存利用率很差。

PagedAttention 论文和最早的 vLLM 博客都强调了这一点：  
显存浪费可能高到 60%-80%，而 vLLM 要做的是把浪费压到只剩接近“最后一个 block 的尾部浪费”。

### 4.2 decode 阶段常常不是纯算力问题，而是 memory-bound + runtime-bound

decode 一次只多吐少量 token，单步 GPU 时间可能并不长。  
这会暴露出很多非算子层开销：

- 调度
- 输入准备
- block 分配
- runtime launch
- detokenize
- streaming

如果这些环节做得不好，GPU 即使单核算得快，也会被系统空洞吃掉。

### 4.3 在线请求天然不规则

线上工作负载往往同时具备：

- 到达时间随机；
- prompt 长短不一；
- 输出长度不可预测；
- 部分请求共享前缀；
- 流式返回要求低延迟。

如果还沿用“固定 batch、整批推进”的离线思路，GPU 利用率和用户体验都会很差。

所以 vLLM 真正要解的问题不是“怎么写个快 attention”，而是：

> **如何把不规则在线请求，重写成一套高利用率的运行时系统。**

---

## 5. 请求主链：一个请求在 vLLM 里如何被推进

理解“请求主链”很关键，因为它直接告诉你各个优化点在什么位置生效。

一个简化后的主链可以写成：

1. 请求进入 engine / serving 层
2. 请求进入 waiting 队列
3. scheduler 决定本轮给它补多少 token
4. KV cache manager 判断：
   - 哪些 prefix blocks 已命中
   - 哪些 blocks 还要新分配
   - 当前 block pool 是否足够
5. 若可调度：
   - 形成本轮 scheduled tokens
   - 进入 model executor / attention backend
6. attention backend 依据 block table 读写 KV
7. 生成结果回到请求状态
8. scheduler 下一轮继续推进，直到完成

这条链上最关键的观察是：

- **scheduler 决定“算多少”**
- **kv_cache_manager 决定“能不能放得下、哪些不用重算”**
- **attention backend 决定“如何按 block 真正执行”**

vLLM 的快来自这三者的协同，而不是某一层孤立优化。

---

## 6. 调度器的第一性原理：不是 prefill/decode，而是 token debt

这是整个 vLLM V1 里最值得反复读的源码思想。

在 `vllm/v1/core/sched/scheduler.py` 的 `schedule()` 中，源码注释明确写到：

- scheduler 不区分传统意义上的“prefill phase”或“decoding phase”；
- 每个 request 只有：
  - `num_computed_tokens`
  - `num_tokens_with_spec`
- 调度器要做的事，是让前者追上后者。

这句话的含义非常大：

- chunked prefill 只是“还欠很多 prompt tokens”
- normal decode 只是“还欠 1 个或少量 token”
- speculative decoding 只是“欠的 token 里包含 spec tokens”
- prefix caching 只是“有一部分 token debt 已经被 cache 命中，不需要再补”

也就是说，vLLM 并不是给每种 feature 造一套独立调度器，而是把它们统一成：

> **每个请求当前还欠多少 token 没算。**

这就是 continuous batching 真正能扩展的原因。

### 6.1 源码里具体怎么体现

在 `schedule()` 里，关键流程大致是：

1. 遍历 `self.running`
2. 对每个 request 计算 `num_new_tokens`
3. 用：
   - `token_budget`
   - `max_model_len`
   - encoder budget
   - long prefill threshold
   - spec decode / placeholder 逻辑
   - Mamba block 对齐
   去修正这轮最多能补多少 token
4. 调 `kv_cache_manager.allocate_slots(...)`
5. 如果 block 不够，就 preempt 低优先级请求
6. 如果能分到 block，就把这个 request 加入 scheduled 集合

这意味着 scheduler 真正的资源不是“请求数量”，而是：

- token budget
- encoder budget
- free KV blocks

这是和很多传统 serving 系统非常不一样的地方。

### 6.2 为什么这通常会快（性能推断）

因为这种抽象天然适合处理不规则请求：

- 有的请求只欠 1 个 decode token
- 有的请求欠一大段 prompt
- 有的请求前缀命中后只欠尾巴
- 有的请求带 speculative tokens

统一按 token debt 处理，调度器就能在每个 step 里自由拼出更高利用率的 batch。

---

## 7. continuous batching 真正成立，靠的是“每步重调度 + 可跳过 + 可抢占”

很多文章把 vLLM 简化成“支持 continuous batching”。  
但如果只记这个词，几乎学不到任何实现价值。

从 `scheduler.py` 看，continuous batching 真正成立靠三件事。

### 7.1 每一步都会重调度

这意味着 batch 不是一个静态容器，而是一个持续流动的集合：

- 已在跑的请求继续
- 新请求可以进
- 完成的请求立刻出
- 暂时不能跑的请求可以跳过

### 7.2 某个请求这轮跑不了，不一定阻塞后面请求

源码里有一个很关键的风格：

- 某些情况下用 `continue`
- 而不是直接 `break`

注释明确说，这样的 `continue` 分支意味着它在某些情况下并不严格坚持 FCFS，而会允许后续请求先被调度。

这说明 vLLM 的目标不是“排队最纯粹”，而是“设备利用率尽量高”。

### 7.3 block 不够时会 preempt

`allocate_slots(...)` 如果失败，scheduler 不会简单认输，而是会进入 preemption 路径；在不同 policy 下，被换出的请求选择逻辑并不完全相同：

- 选择低优先级请求；
- 做 preemption；
- 回收资源后继续尝试调度当前请求。

这说明 vLLM 的调度器不是一个被动的 admission queue，而是一个会主动重排资源的运行时。

### 7.4 为什么这三件事一起才有效

如果只有“新请求随时进来”，但：

- 没有 per-step 预算重算，
- 没有前缀命中接入，
- 没有 KV block 可回收分配，
- 没有 preemption，

那 continuous batching 只是一个 marketing 词。

vLLM 强在它把这些机制做成一套可执行的闭环。

---

## 8. PagedAttention 在源码层的真实价值：把 KV cache 变成 scheduler 能操控的资源

很多人一看到 PagedAttention，就只想到“attention kernel 很高级”。  
这其实只看到了半件事。

在系统层，PagedAttention 的真正意义是：

- KV 被切成 block；
- block 可按需分配；
- block 可共享；
- block 可回收；
- prefix hit 可以直接表示成“已有哪些 full blocks 不用重算”。

这件事在 `vllm/v1/core/kv_cache_manager.py` 里体现得很清楚。

### 8.1 `KVCacheBlocks` 不是小工具，而是调度器和 KV 系统之间的协议

`KVCacheBlocks` 的注释已经说明：

- 它是 allocation result；
- 用来隐藏 KV manager 的内部结构；
- 让 scheduler 不需要知道底层 block pool 的细节。

这是非常好的系统设计信号：  
block 管理不是“底层实现细节”，而是主调度循环的协议对象。

### 8.2 `get_computed_blocks(request)` 把 prefix cache 命中直接接到调度器上

这个函数做了几件很关键的事：

1. 如果 prefix caching 关闭，直接返回空命中；
2. 如果开启，就去找已命中的 full blocks；
3. 即使“全部命中”，也仍然保留最后 token 的重算需要，以拿到 logits；
4. 返回：
   - 命中的 blocks
   - 对应的 computed tokens 数

这说明 prefix cache 在 vLLM 里不是后处理，而是 scheduler 的输入之一。

### 8.3 `allocate_slots(...)` 是最该认真读的 KV 入口之一

它的注释非常有价值，因为它直接把 block 生命周期拆成几段：

- `comp`：已计算 tokens
- `new_comp`：本轮因 prefix cache 命中的新已计算 tokens
- `ext_comp`：来自 connector 的外部已计算 tokens
- `new`：本轮要计算的新 tokens
- `lookahead`：为 speculative decode 等预留的 tokens

这张布局图其实揭示了 vLLM 的几个核心能力如何叠加：

- prefix caching
- external KV connector
- spec decode
- normal decode
- chunked prefill

它们并不是几套不同内存模型，而是共享同一套 block allocation 语义。

### 8.4 为什么这通常会快（性能推断）

因为 once you have block-level control：

- prefix hit 直接变成更少的 compute debt
- 更少的 compute debt 直接变成更小的 token budget 占用
- token budget 占用更小，意味着这一轮能塞更多请求
- 塞更多请求，意味着吞吐更高

这就是 PagedAttention 从“省显存”变成“提吞吐”的完整因果链。

---

## 9. chunked prefill 为什么在 vLLM 里不是补丁功能

很多系统把 chunked prefill 当额外模式。  
在 vLLM 里，它更像 scheduler 的自然产物。

从 `scheduler.py` 的 waiting 请求路径看，新请求在进入 running 前会经历：

1. prefix hit 计算
2. 剩余 `num_new_tokens` 计算
3. long prefill threshold 限制
4. token budget 限制
5. encoder budget 限制
6. `allocate_slots(...)` 判定

如果 `enable_chunked_prefill=True`，而剩余 prompt 太长，就会自动被切成当前 budget 能容纳的 chunk。

所以 chunked prefill 不是“又加一套调度逻辑”，而是：

> **waiting request 在统一 token-budget 流程下被自然裁成一个个可运行 chunk。**

### 为什么这重要（源码事实 + 调度推断）

因为它让长 prompt 不再独占几个完整 step：

- decode 请求可以持续推进；
- 短请求不容易被长 prompt 拖死；
- prefix hit 越多，第一轮 chunk 越小；
- prefill 和 decode 真正进入统一调度。

---

## 10. prefix caching 在 vLLM 里不是外挂，而是主调度输入

这一点值得单独强调。

很多系统的前缀缓存像个外挂模块：  
命中后少算一点，仅此而已。

而在 vLLM 里，从调度主链角度看，它更像：

- waiting request 的“起始 debt reduction”
- running request 的“历史 block 复用基础”
- connector / disaggregated serving 的局部形式

更具体地说：

- 本地 prefix hit 通过 `get_computed_blocks(request)` 进入调度；
- 外部 connector 命中通过 `external matched tokens` 进入调度；
- 最终 `num_computed_tokens` 直接改变 `num_new_tokens`。

这意味着 prefix cache 不仅减少 FLOPs，还改变：

- admission 成本；
- token budget 消耗；
- chunked prefill 的 chunk size；
- batch 内 request 混合效率。

从系统视角讲：

> **vLLM 把 prefix caching 做成了 scheduler 的一部分，而不是 scheduler 旁边的缓存插件。**

---

## 11. speculative decoding 为什么必须和 scheduler 紧耦合

spec decode 如果只是一个“外面先猜几个 token，再回来验证”的外挂，很难和真实 serving 系统融合得好。

在 `scheduler.py` 里，vLLM 明确把它纳入了调度状态：

- 初始化阶段就读取 `speculative_config`
- 记录 `num_spec_tokens` 与 `num_lookahead_tokens`
- `num_tokens_with_spec` 直接进入 debt 计算
- 已调度的 speculative tokens 会被记录到 `scheduled_spec_decode_tokens`

这说明 spec decode 在 vLLM 里不是孤立模块，而是：

- 预算层可见；
- block 分配层可见；
- request 状态机可见。

这很重要，因为只有这样它才更容易与：

- chunked prefill
- async scheduling
- KV cache
- graph/compile path

共存在同一 serving 主链中。这里“协同工作良好”属于工程推断，不是本文直接给出的 benchmark 结论。

---

## 12. graph / compile：后期 vLLM 越来越像 runtime system

早期很多人记住 vLLM，是因为 PagedAttention。  
但从 V1 到后续 graph/compile、KV connector、spec decode、DBO 等方向来看，vLLM 已经越来越像一个 runtime system。

### 12.1 为什么图执行很关键

decode 阶段的痛点往往是：

- 单步很短；
- 高频重复；
- host/runtime 固定开销占比高。

所以 graph/compile 的目标不是只做大算子加速，而是压：

- launch overhead
- Python overhead
- 形状稳定场景下的 runtime bookkeeping

### 12.2 为什么这和 scheduler/KV 一样重要

如果你已经通过：

- block 化 KV 提高了并发，
- continuous batching 把 batch 转起来，

那接下来暴露出来的瓶颈往往就是“每一步系统层固定成本”。

因此 vLLM 的快是逐层传导的：

1. 先解决能不能放下更多请求；
2. 再解决能不能把这些请求高效调度；
3. 最后解决每一步本身还有多少 runtime 空洞。

这也是为什么后期 vLLM 的优化越来越偏系统工程而不只是算子工程。

---

## 13. 为什么说 vLLM 的性能是“多层叠加”，不是单点 magic

为了便于记忆，可以把 vLLM 的快分成五层：

### 第 1 层：KV 内存系统

- block/page 化
- prefix hit block 复用
- sliding window / eviction / allocation

### 第 2 层：调度系统

- token debt 抽象
- per-step scheduling
- preemption
- decode-first / chunked prefill

### 第 3 层：重复工作消除

- prefix caching
- external KV connector
- speculative decoding

### 第 4 层：执行路径优化

- graph capture/replay
- torch.compile
- backend-specific kernels
- 更低的 host/runtime 固定成本

### 第 5 层：系统拓扑优化

- disaggregated serving
- MoE/DBO
- 多并行策略组合
- 多节点 KV 传输

vLLM 的优势恰恰在于：  
**这五层不是孤立存在，而是层层放大彼此收益。**

---

## 14. 最容易误解的三件事

### 14.1 “vLLM 快 = PagedAttention 快”

不对。  
PagedAttention 重要，但更多是 **内存系统底座**。  
没有 scheduler、prefix hit 接入和图执行，PagedAttention 的收益释放不出来。

### 14.2 “continuous batching = 新请求随时进 batch”

也不对。  
真正成立的前提包括：

- token debt 调度抽象；
- block allocation 可快速响应；
- 请求可跳过；
- 请求可 preempt；
- budget 是按 token 而不是按请求数来算。

### 14.3 “prefix caching 命中就是单纯少算一点”

还是不够准确。  
在 vLLM 里，它会真正改变：

- 请求进入 running 的成本；
- token budget 占用；
- chunk size；
- batch 混合效率。

所以它是 **调度输入**，不只是“cache 命中后加速一下”。

---

## 15. 面向学习者：最值得抄到自己脑子里的源码主线

如果你是为了自己实现一个 mini-vLLM，最重要的是抓住这条主线：

1. Request/Sequence 状态如何表示
2. Scheduler 如何按 token debt 决定本轮算谁
3. KV cache manager 如何表达 prefix hit / block allocation / admission gate
4. attention backend 如何把逻辑 block 映射成真实 KV 读写
5. compile/graph 如何继续压系统开销

如果这五件事没有连起来，即使你把某个 attention kernel 写出来了，也很难真的复现 vLLM 的“快”。

---

## 16. 面向分享：三分钟讲明白 vLLM 为什么快

如果你要拿这份文档去讲给别人听，可以只讲下面四句话：

1. **vLLM 先把 KV cache 做成 block/page，所以显存碎片低、并发更高。**
2. **它的调度器不是按 prefill/decode 阶段调度，而是按每个请求还欠多少 token 调度，所以 chunked prefill、prefix cache、spec decode 能统一。**
3. **prefix caching 在 vLLM 里不是外挂，而是直接进入主调度循环，所以命中前缀会真实改变 token budget 的利用率。**
4. **后期它继续用 graph/compile/clean runtime path 压低每一步固定开销，因此小步 decode 场景也能持续提速。**

这四句话，基本就是整个系统的骨架。

---

## 17. 对 MinivLLM 的直接启发

结合这个仓库自己的 `MinivLLM` 学习代码，最值得对照的并不是“我是不是也有 attention 层”，而是：

- scheduler 有没有统一 prefill/decode 抽象；
- block manager 是否只是 allocator，还是已经成为调度协议的一部分；
- prefix cache 是否只停留在想法，还是已经真实改变调度输入；
- decode 路径有没有开始暴露系统层固定开销；
- 你的实现是否能自然扩展到 chunked prefill / external KV / spec decode。

这也是为什么学习 vLLM，不能只盯模型层代码。

---

## 18. 参考来源

1. vLLM Blog, *vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention*  
   https://blog.vllm.ai/2023/06/20/vllm.html
2. Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention*  
   https://arxiv.org/abs/2309.06180
3. vLLM Docs, *Paged Attention*  
   https://docs.vllm.ai/en/latest/design/paged_attention/
4. vLLM Docs, *Automatic Prefix Caching*  
   https://docs.vllm.ai/en/latest/design/prefix_caching/
5. vLLM Docs, *Optimization and Tuning*  
   https://docs.vllm.ai/en/stable/configuration/optimization.html
6. vLLM Docs, *Speculative Decoding*  
   https://docs.vllm.ai/en/latest/features/spec_decode.html
7. vLLM Blog, *vLLM V1: A Major Upgrade to vLLM’s Core Architecture*  
   https://blog.vllm.ai/2025/01/27/v1-alpha-release.html
8. upstream 仓库快照：`92a7c121b62a1484b68c0a27d1ecefd1a84f78fc`
9. upstream 源码：`vllm/v1/core/sched/scheduler.py`
10. upstream 源码：`vllm/v1/core/kv_cache_manager.py`
11. upstream 源码：`vllm/compilation/`

---

## 19. 一句话总评

**vLLM 的快，不是“一个快 attention kernel”，而是“把不规则在线请求重写成 token-debt 调度问题，再用 block 化 KV、prefix/spec 复用和 graph/compile 把这套调度真正跑满硬件”。**
