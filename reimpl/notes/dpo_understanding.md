# MiniMind 中 DPO 的位置、数据流与基本损失直觉

## Today Goal

理解 MiniMind 中 DPO（Direct Preference Optimization）的完整训练流程：
- 定位 DPO 关键源码位置与调用链
- 梳理从数据加载到 loss 计算的完整数据流
- 建立 DPO loss 的直觉理解（初始值、训练方向、beta 的作用）
- 通过最小实验验证 loss 行为

## Why It Matters

DPO 是 Pretrain → SFT → DPO 链路中的偏好对齐阶段。SFT 教模型"怎么对话"，DPO 教模型"在都能对话的前提下，选择更好的回答"。

## Source Code Path

| 文件 | 位置 | 作用 |
|------|------|------|
| `trainer/train_dpo.py:33` | `dpo_loss()` | DPO 核心公式实现（19 行） |
| `trainer/train_dpo.py:24` | `logits_to_log_probs()` | 从 logits 提取实际 token 的 log 概率 |
| `trainer/train_dpo.py:55` | `train_epoch()` | 训练主循环，ref/policy 双模型前向 + 反向 |
| `trainer/train_dpo.py:175-184` | `__main__` 初始化段 | policy 和 ref 从同一 SFT checkpoint 加载，ref 冻结 |
| `dataset/lm_dataset.py:126-196` | `DPODataset` | 偏好数据加载，返回 x/y/mask 的 chosen/rejected 六元组 |

阅读顺序：`DPODataset` → `train_epoch` → `dpo_loss` → `logits_to_log_probs`

## Data Flow

### 1. 数据构造（DPODataset）

原始数据来自 `dpo.jsonl`，每条包含 `chosen` 和 `rejected`（chat 格式 list）：

```json
{"chosen": [{role, content}, ...], "rejected": [{role, content}, ...]}
```

Dataset 的 `__getitem__` 对每个样本：
- 用 `apply_chat_template` 渲染为模型输入文本
- tokenize 后构造 x（input_ids[:-1]）、y（input_ids[1:]）
- 用 `generate_loss_mask` 只在 assistant 回复区间标记 mask=1

返回 6 个张量（各 shape `(seq_len,)`）：`x_chosen, y_chosen, mask_chosen, x_rejected, y_rejected, mask_rejected`

### 2. Batch 拼接（train_epoch:64-67）

```
x_chosen(4,1024) + x_rejected(4,1024) → x(8,1024)
y_chosen(4,1024) + y_rejected(4,1024) → y(8,1024)
mask_chosen(4,1024) + mask_rejected(4,1024) → mask(8,1024)
```

前半 batch 是 chosen，后半是 rejected，拼成一个大 batch 一次性过模型。

### 3. 双模型前向（train_epoch:73-81）

```
x(8,1024) → ref_model(no_grad) → ref_logits(8,1024,vocab)
                                 → logits_to_log_probs + y
                                 → ref_log_probs(8,1024)

x(8,1024) → model(有梯度)      → policy_logits(8,1024,vocab)
                                 → logits_to_log_probs + y
                                 → policy_log_probs(8,1024)
```

ref 用 `torch.no_grad()` 冻结，policy 正常前向。

### 4. Loss 计算（dpo_loss:37-51）

```
(8,1024) ── mask×sum/length ──→ (8,) 归一化标量
                                  │
                    [0:4] chosen       [4:8] rejected
                        │                  │
              pi_logratios = policy_chosen - policy_rejected
              ref_logratios = ref_chosen - ref_rejected
              logits = pi_logratios - ref_logratios
              loss = -logsigmoid(beta × logits).mean()
```

### 5. 反向传播（train_epoch:87-94）

只更新 policy model 参数，ref model 始终不变。

## Loss Intuition

### 公式对应

$$\mathcal{L} = -\mathbb{E}\left[\log\sigma\left(\beta\left[\log\frac{\pi_\theta(y_w|x)}{\pi_\theta(y_l|x)} - \log\frac{\pi_{ref}(y_w|x)}{\pi_{ref}(y_l|x)}\right]\right)\right]$$

| 代码变量 | 公式对应 | 含义 |
|----------|---------|------|
| `pi_logratios` | $\log\frac{\pi_\theta(y_w)}{\pi_\theta(y_l)}$ | policy 认为 chosen 比 rejected 好多少 |
| `ref_logratios` | $\log\frac{\pi_{ref}(y_w)}{\pi_{ref}(y_l)}$ | ref 认为 chosen 比 rejected 好多少 |
| `logits` | 两者之差 | policy 相对 ref 的**额外**偏好 |
| `beta` | $\beta$ | 控制偏好惩罚强度 |
| `-logsigmoid(...)` | $-\log\sigma(\cdot)$ | 越偏好 chosen → loss 越低 |

### 三个关键状态

| 状态 | logits | loss | 含义 |
|------|--------|------|------|
| policy = ref（初始） | = 0 | **log(2) ≈ 0.693** | 训练起点，policy 没有额外偏好 |
| policy 偏好 chosen | > 0 | → 0 | 训练目标达成 |
| policy 偏好 rejected | < 0 | >> 0.693 | 方向错误，梯度会纠正 |

### beta 的作用

beta 控制损失对偏好差异的敏感度。同样的 log 概率差，beta=1.0 的 loss 响应远大于 beta=0.1。MiniMind 默认 beta=0.1，是一种保守策略。

### 为什么需要 ref model

没有 ref 的约束，policy 会无限制地把 chosen 概率推高、rejected 概率推低，导致生成多样性丧失。ref 充当"锚点"：loss 不是要求 policy 绝对偏好 chosen，而是要求 policy 相对 ref **额外**偏好 chosen。这保持了模型的生成能力。

### 训练超参特征

- 学习率 `4e-8`（极小）：DPO 在已训练好的 SFT 模型上微调，步子太大会遗忘
- 初始权重：policy 和 ref 都从 `full_sft` checkpoint 加载

## Verification

### 实验 1：DPO loss 直觉验证

脚本：`reimpl/experiment/sanity_dpo_loss.py`

用构造的 tensor 验证三个场景：

| 场景 | loss | 结论 |
|------|------|------|
| policy = ref | 0.693147 = log(2) | 精确匹配，确认初始状态 |
| policy 偏好 chosen (beta=0.1) | 0.6118 < 0.693 | loss 下降 |
| policy 偏好 chosen (beta=1.0) | 0.1678 | beta 大 10 倍，降幅显著更大 |
| policy 偏好 rejected | 0.8952 > 0.693 | loss 上升 |

### 实验 2：DPO 单步更新验证

脚本：`reimpl/experiment/exp_dpo_one_step.py`

用 TinyLM（vocab=16, hidden=8）构造 1 条 chosen + 1 条 rejected，跑 1 次前向+反向：

| 指标 | 更新前 | 更新后 | 变化 |
|------|--------|--------|------|
| chosen_policy_score | -3.064 | -2.993 | +0.071 (上升) |
| rejected_policy_score | -3.367 | -3.471 | -0.104 (下降) |
| loss | 0.693 | 0.650 | -0.043 (下降) |

验证了：DPO 一步更新后，policy 对 chosen 的评分上升、对 rejected 的评分下降，loss 下降。

## Key Takeaways

1. **DPO 在训练链路中的位置**：Pretrain → SFT → DPO，站在 SFT 模型上做偏好对齐，不教新知识，教"品味"
2. **双模型设计**：policy 和 ref 从同一 checkpoint 加载，ref 冻结作为锚点，防止 policy 走极端
3. **Batch 拼接策略**：chosen 和 rejected 拼成一个大 batch 过模型，减少 forward 次数
4. **Loss 初始值**：log(2) ≈ 0.693，不是从 0 开始。
5. **Beta 控制训练强度**：beta 越大，同样的偏好差异产生更大的 loss 响应
6. **DPO vs PPO**：DPO 是 off-policy 的，用静态偏好数据集训练，不需要 Reward Model；PPO 是 on-policy 的，需要实时采样 + Reward Model 做教练。DPO 更简单但能力提升有限
7. **loss_mask 的作用**：只在 assistant 回复区间计算 loss，prompt 部分不参与
