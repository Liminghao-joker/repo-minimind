# LoRA：从数学原理到 MiniMind 中的参数注入位置

> Day 9 学习笔记 | 2026-04-07

---

## 1. 今日目标

- 会讲 LoRA 的基本数学原理
- 会解释为什么它能减少训练参数量
- 会说明缩放因子 alpha/r 的作用
- 会说明初始化为什么不破坏原模型
- 会定位 MiniMind 中哪些层最适合插 LoRA
- 会区分哪些参数冻结、哪些参数可训练

---

## 2. 核心结论（一句话版）

**LoRA 不是替换 W，而是在冻结 W 的前提下，学习一个低秩增量 ΔW = BA。**

训练时只更新 A 和 B，参数量从 d² 降到 2dr（r << d），大幅减少显存和计算开销。

---

## 3. 数学原理

### 3.1 全参数微调 vs LoRA

| | 全参数微调 | LoRA |
|---|---|---|
| 做法 | W ← W + ΔW（直接更新整个 W） | y = Wx + (BA)x · (α/r) |
| 训练对象 | W 的全部参数 | 只有 A（r×d）和 B（d×r） |
| 参数量级 | d² | 2dr |
| 预训练权重 | 被覆盖 | 完全保留 |

### 3.2 核心公式

```
y = Wx + (B · A) x · (α / r)
```

- **W**：(out_dim, in_dim)，冻结的预训练权重
- **A**：(r, in_dim)，可训练，kaiming 初始化
- **B**：(out_dim, r)，可训练，**初始化为全零**
- **α/r**：缩放因子，控制 LoRA 分支信号强度

### 3.3 为什么低秩近似合理？

**直觉**：预训练模型已经学到了通用知识，微调只是在原有权重上做小的增量调整。研究表明，这个增量矩阵 ΔW 在实际微调任务中是**低秩的**——用 r=8 或 16 就能很好地近似。

**类比**：一张高清照片的"修图"只是局部微调，不需要重画整张图。

### 3.4 alpha/r 缩放因子

- α（alpha）通常固定为一个常数（如 16）
- r 增大时，α/r 变小，自动衰减 LoRA 分支的贡献
- 常用设置：**α = 2r**（即 scaling = 2）
- 作用：防止 r 变大后 LoRA 分支信号过强，淹没原始输出

### 3.5 初始化为什么安全

B 初始化为全零 → 训练开始时 ΔW = 0·A = **零矩阵**

→ 模型初始输出 = Wx + 0 = Wx，与原始模型完全一致

→ 训练过程从"不改变原始模型"开始，逐步学习增量

---

## 4. MiniMind 中的注入位置

### 4.1 源码定位

| 文件 | 类 | 行号 |
|---|---|---|
| `model/model_minimind.py` | `Attention` | 150-221 |
| `model/model_minimind.py` | `FeedForward` | 224-237 |
| `model/model_minimind.py` | `MiniMindBlock` | 360-382 |

### 4.2 Attention 投影层详情

| 投影层 | 定义 | 形状 | 行号 |
|---|---|---|---|
| `q_proj` | `nn.Linear(512, 512, bias=False)` | (512, 512) | 155 |
| `k_proj` | `nn.Linear(512, 128, bias=False)` | (128, 512) | 158 |
| `v_proj` | `nn.Linear(512, 128, bias=False)` | (128, 512) | 161 |
| `o_proj` | `nn.Linear(512, 512, bias=False)` | (512, 512) | 165 |

> 注意：k_proj 和 v_proj 的输出维度是 128（= num_key_value_heads × head_dim = 2 × 64），因为 MiniMind 使用了 GQA。

### 4.3 FFN 投影层详情

| 投影层 | 定义 | 形状 |
|---|---|---|
| `gate_proj` | `nn.Linear(512, 1408, bias=False)` | (1408, 512) |
| `up_proj` | `nn.Linear(512, 1408, bias=False)` | (1408, 512) |
| `down_proj` | `nn.Linear(1408, 512, bias=False)` | (512, 1408) |

### 4.4 注入优先级

**首选**（效果最好，论文推荐）：
1. `q_proj`、`v_proj` —— 所有 Transformer Block 的 Attention 层
2. 追加 `o_proj` 可进一步提升效果

**可选**：
3. `k_proj`
4. FFN 的 `gate_proj`、`up_proj`、`down_proj`（收益递增不如 Attention）

**最自然做法**：遍历所有 `model.model.layers`，对每个 Block 的 Attention 替换目标投影层为 LoRALinear。

---

## 5. 参数冻结与训练对象

### 5.1 三层区分（最重要）

| 层次 | 对象 | 状态 |
|---|---|---|
| 数学对象 | ΔW = BA | 一个低秩矩阵分解 |
| 工程对象 | LoRALinear(original_linear, r, alpha) | 给线性层挂一个旁路分支 |
| 训练对象 | A 和 B | requires_grad=True，其他全部 False |

### 5.2 冻结代码模式

```python
# Step 1: 冻结全部
for param in model.parameters():
    param.requires_grad = False

# Step 2: 替换目标层为 LoRALinear（A 和 B 自动 requires_grad=True）
for layer in model.model.layers:
    layer.attention.q_proj = LoRALinear(layer.attention.q_proj, r=8, alpha=16.0)
    layer.attention.v_proj = LoRALinear(layer.attention.v_proj, r=8, alpha=16.0)

# Step 3: 验证
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"可训练: {trainable:,} / 总计: {total:,} = {trainable/total*100:.2f}%")
```

### 5.3 MiniMind 中的参数量估算

假设对 8 层 Transformer 的 q_proj 和 v_proj 插 LoRA（r=8）：

- 每层 q_proj LoRA 参数：2 × 8 × 512 = 8,192
- 每层 v_proj LoRA 参数：2 × 8 × 512 = 8,192（v_proj 输出 128，但 in_dim=512）
  - 实际：A(8, 512) + B(128, 8) = 4096 + 1024 = 5,120
- 8 层合计 ≈ (8192 + 5120) × 8 ≈ **106,496**
- 模型总参数 ≈ 26M
- 训练比例 ≈ **0.4%**

---

## 6. 最小实现思路

```python
class LoRALinear(nn.Module):
    def __init__(self, original_linear: nn.Linear, r: int = 8, alpha: float = 16.0):
        super().__init__()
        self.original_linear = original_linear
        self.original_linear.weight.requires_grad = False

        in_dim = original_linear.in_features
        out_dim = original_linear.out_features

        self.lora_A = nn.Parameter(torch.empty(r, in_dim))
        self.lora_B = nn.Parameter(torch.zeros(out_dim, r))
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)

        self.scaling = alpha / r

    def forward(self, x):
        return self.original_linear(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
```

关键设计决策：
- B 初始化为零 → ΔW 初始为零 → 不破坏原模型
- A 用 kaiming 初始化 → 训练开始时有合理梯度
- `alpha / r` 控制缩放 → r 变大时自动衰减

---

## 7. 今日易错点 / 真正需要记住的

1. **LoRA 不是替换 W，是在冻结 W 的前提下加一个低秩增量**
2. **B 初始化为零是关键**——保证训练开始时模型行为不变
3. **alpha/r 的作用是缩放**——不是学习率，是控制 LoRA 分支的信号强度
4. **参数量从 d² 降到 2dr**，不是降到 r²
5. **通常只改 Attention 层就够了**，不需要改所有线性层
6. **LoRA 可以随时合并回原权重**：W_merged = W + BA · (α/r)，合并后零推理开销
7. **区分三层**：数学对象(ΔW=BA)、工程对象(旁路分支)、训练对象(只训A和B)

---

## 8. 面试表达模板

> **Q：LoRA 是什么？为什么用 LoRA？**

LoRA（Low-Rank Adaptation）是一种参数高效微调方法。它的核心思想是：不直接微调预训练模型的全部参数，而是冻结原始权重 W，额外学习一个低秩增量矩阵 ΔW = BA，其中 B 是 d×r 的矩阵，A 是 r×d 的矩阵，r 远小于 d。

这样做的好处是，参数量从 d² 量级降到 2dr，比如 MiniMind 的一个 512×512 的投影层，用 r=8 的 LoRA 只需要 8192 个参数，是原来的 3%。而且 B 初始化为零，所以训练开始时模型行为和原始模型完全一致。

在实际应用中，通常只对 Attention 的 Q 和 V 投影层插入 LoRA 就能达到接近全参微调的效果。

> **Q：LoRA 怎么做到不破坏预训练权重？**

通过初始化策略。B 矩阵初始化为全零，所以训练开始时 ΔW = B·A = 0，模型的输出就是原始预训练模型的输出。然后训练过程从零开始逐步学习增量，类似于在预训练权重的基础上做"残差学习"。

> **Q：alpha/r 是什么？**

它是 LoRA 分支的缩放因子。alpha 通常固定（比如 16），r 是秩。当 r 增大时，缩放因子自动变小，防止 LoRA 分支信号过强。常用设置是 alpha = 2r，这样 scaling = 2，是一个经验上效果较好的值。
