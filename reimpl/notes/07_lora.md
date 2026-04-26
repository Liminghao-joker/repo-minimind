# LoRA：四两拨千斤的艺术  
  
## 1. 今日目标  
  
1. 讲清楚 LoRA 的核心数学原理；  
2. 在 MiniMind 中定位 LoRA 最自然的注入位置；  
3. 能说明冻结哪些参数、训练哪些参数，以及如何做最小验证。  
  
**LoRA 不是替换原始权重，而是在冻结原始权重的前提下，学习一个低秩增量分支。**  
  
---  
  
## 2. 从全参数微调到 LoRA  
  
### 2.1 全参数微调是什么  
  
对于一个线性层，若原始映射写作：  
$y = Wx$  
  
其中 $W$ 是该层的参数矩阵。  
  
在全参数微调（Full Fine-Tuning）中，我们直接更新整个 $W$。这意味着：  
  
- 所有参数都参与反向传播；  
- 梯度、优化器状态、参数副本都会占用显存；  
- 当模型规模较大时，训练成本和存储成本都很高。  
  
这也是 SFT 在大模型场景下经常遇到的现实问题：**算力贵、显存吃紧、存储多个任务版本代价高。**  
  
### 2.2 LoRA 的基本改法  
  
LoRA（Low-Rank Adaptation）的思路是：  
  
- **冻结**原始预训练权重 $W$；  
- 不再直接更新 $W$；  
- 只额外学习一个增量矩阵 $\Delta W$。  
  
于是前向写成：  
  
$$  
y = (W + \Delta W)x  
$$  
  
LoRA 进一步假设这个增量矩阵是**低秩的**，因此把它分解为两个更小的矩阵乘积：  
  
$$  
\Delta W = BA  
$$  
  
于是：  
  
$$  
y = Wx + BAx  
$$  
  
在实际实现中，还会乘上一个缩放系数：  
  
$$  
y = Wx + \frac{\alpha}{r}BAx  
$$  
  
其中：  
  
- $W \in \mathbb{R}^{d_{out} \times d_{in}}$  
- $A \in \mathbb{R}^{r \times d_{in}}$  
- $B \in \mathbb{R}^{d_{out} \times r}$  
- $r \ll \min(d_{out}, d_{in})$  
  
这里的 $r$ 就是 LoRA 的 rank。  
  
---  
  
## 3. 为什么 LoRA 能减少训练参数量  
  
假设原始线性层权重为：  
  
$$  
W \in \mathbb{R}^{d \times d}  
$$  
  
那么：  
  
- 全参数微调需要训练 $d^2$ 个参数；  
- LoRA 只训练：  
  
$$  
A \in \mathbb{R}^{r \times d}, \quad B \in \mathbb{R}^{d \times r}  
$$  
  
总参数量为：  
  
$$  
rd + dr = 2dr  
$$  
  
因此，参数量从 $d^2$ 下降到 $2dr$。当 $r \ll d$ 时，节省非常明显。  
  
例如当 $d=512, r=8$ 时：  
  
- 全参微调：$512^2 = 262144$  
- LoRA：$2 \times 512 \times 8 = 8192$  
  
也就是说，一个 512×512 的投影层，如果只加 rank=8 的 LoRA，训练参数量大约只剩原来的 3.125%。  
  
---  
  
## 4. 低秩约束到底意味着什么  
  
LoRA 的关键不是“少训一点参数”这么简单，而是它对增量矩阵 $\Delta W$ 加了一个**低秩结构约束**。  
  
从线性代数上看：  
  
$$  
\mathrm{rank}(BA) \leq \min(\mathrm{rank}(B), \mathrm{rank}(A)) \leq r  
$$  
  
因此，$\Delta W = BA$ 的秩不会超过 $r$。  
  
这意味着：LoRA 不是允许模型对原权重做任意复杂的改动，而是要求这种改动只能落在一个较低维的子空间里。  
  
直觉上可以这样理解：  
  
- 预训练模型已经有了通用能力；  
- 下游任务微调时，往往不需要“重写整个权重空间”；  
- 许多有效的更新，其实只需要沿着少数几个关键方向做调整。  
  
因此，LoRA 的核心假设是：**适配新任务所需的权重变化，往往近似是低秩的。**  
  
---  
  
## 5. 初始化为什么不会破坏原模型  
  
LoRA 中一个非常关键的工程细节是初始化。  
  
常见做法是：  
  
- $A$ 随机初始化（例如高斯分布或 Kaiming 初始化）；  
- $B$ 初始化为全 0。  
  
这样一来，在训练刚开始时：  
  
$$  
\Delta W = BA = 0  
$$  
  
因此：  
  
$$  
y = Wx + \frac{\alpha}{r}BAx = Wx  
$$  
  
也就是说，**LoRA 分支在初始时不会改变原始模型行为**。  
  
这一点非常重要，因为它意味着训练不是从“破坏原模型”开始，而是从“保持原模型输出不变”开始，再逐渐学习增量修正。  
  
你可以把这件事类比为残差分支：  
  
- 主干是原始预训练模型；  
- LoRA 分支像一个从 0 开始逐步长出来的补丁；  
- 一开始补丁不生效，训练后才逐渐产生作用。  
  
---  
  
## 6. 缩放因子 $\alpha/r$ 的作用  
  
LoRA 前向里常出现：  
  
$$  
\frac{\alpha}{r}  
$$  
  
这里的 $\alpha$ 是一个超参数，用来控制 LoRA 分支的整体强度。  
  
为什么还要除以 $r$？  
  
因为当 rank 变大时，矩阵乘积 $BA$ 的数值尺度可能发生变化。如果不做归一化，rank 变大后 LoRA 分支可能会过强，导致不同 rank 设置下输出尺度不稳定。  
  
所以 $\alpha/r$ 的作用可以概括为：  
  
1. 控制 LoRA 分支的贡献强度；  
2. 让不同 rank 下的输出量级更稳定；  
3. 把“rank 的变化”和“分支强度”部分解耦。  
  
要注意：  
  
- 它**不是学习率**；  
- 它属于前向计算中的缩放系数；  
- 本质上是在调控增量分支对原分支的相对影响。  
  
补充： Rank-Stabilized LoRA 选择的缩放策略是 $\alpha / \sqrt{r}$。
  
---  
  
## 7. LoRA 常加在哪些层  
  
LoRA 常见的注入位置是 **Attention projection layers**，也就是注意力模块中的线性投影层。  
  
原因主要有三点：  
  
1. 这些层参数量大，是模型表达能力的重要位置；  
2. 它们直接影响 query、key、value 或输出映射，对任务适配较敏感；  
3. 工程上改动自然，只需替换对应的 `nn.Linear`。  
  
常见优先级通常是：  
  
- 首先考虑 `q_proj`、`v_proj`  
- 其次可考虑 `k_proj`、`o_proj`  
- 再往后才考虑 FFN 中的线性层  
  
> **LoRA 最典型、最自然的接入点，就是 Transformer block 中 attention 的若干投影层。**  

---  
  
## 8.代码实现  
  
```python  
class LoRALinear(nn.Module):  
def __init__(self, in_features, out_features, r=8, alpha=16):  
super().__init__()  
self.linear = nn.Linear(in_features, out_features, bias=False) # 原始分支
self.linear.weight.requires_grad = False # 冻结全部参数

# 新挂上的 lora_A,lora_B 可训练
self.lora_A = nn.Parameter(torch.randn(r, in_features) * 0.01)  
self.lora_B = nn.Parameter(torch.zeros(out_features, r))  
self.r = r  
self.scaling = alpha / r  
  
def forward(self, x):  
base_out = self.linear(x)  
lora_out = x @ self.lora_A.t() @ self.lora_B.t() * self.scaling # LoRA 分支 
return base_out + lora_out # 最终输出
```

统计参数量进行最小验证：

```python
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"LoRA 可训练参数: {trainable:,}")
```
## 9. 天然正则化？

LoRA 并不是让模型在完整参数空间里自由更新，而是强制它只能通过一个低秩增量去适配任务。这等于人为缩小了可学习更新的自由度。

这种结构限制带来的结果是：

- 更新空间更小；
    
- 不容易对小数据集过度记忆；
    
- 更倾向于学到“最必要的修正方向”。
    

所以它看起来会带一点“结构性正则化”的味道。这源于它的参数化方式本身带有较强的约束。


## 10. Takeways


LoRA 是一种参数高效微调方法。它不是直接更新预训练模型的全部参数，而是在冻结原始权重 $W$ 的前提下，额外学习一个低秩增量 $\Delta W = BA$。因此前向可以写成 $y = Wx + \frac{\alpha}{r}BAx$。

它的核心优势在于参数量从原本的 $d^2$ 下降到 $2dr$，当 $r \ll d$ 时能显著降低训练显存和存储成本。工程上，LoRA 常插在 Transformer 的 attention projection layers，比如 `q_proj`、`v_proj` 等线性层。训练时原始权重冻结，只训练新增的 A 和 B。初始化时通常令 B 为 0，这样训练开始时不会破坏原模型行为。