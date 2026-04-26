---
title: DPO
tags:
  - minimind
  - llm
  - dpo
---

# DPO

## 1. Core Problem

理解 MiniMind 中 DPO（Direct Preference Optimization）的完整训练流程：

- 定位 DPO 关键源码位置与调用链
    
- 梳理从数据加载到 loss 计算的完整数据流
    
- 建立 DPO loss 的直觉理解（初始值、训练方向、beta 的作用）
    
- 通过最小实验验证 loss 行为
## 2. Conclusions

- DPO 是偏好对齐阶段，站在 SFT 模型之上继续训练。
- 它不是直接对整句做黑盒评分，而是把 chosen / rejected 都转成 token-level log-prob 再聚合。
- policy 和 ref 从同一个 SFT checkpoint 出发，但 ref 冻结不更新。
- DPO 的核心比较对象不是绝对概率，而是 policy 相对 ref 的“额外偏好”。
- 当 `policy = ref` 时，loss 初值是 `log(2) ≈ 0.693`，不是 0。
- 当前这部分我已经有两类真实验证：一类是 sanity loss，另一类是单步参数更新。

## 3. Intuition and Principle

SFT 解决的是“模型要学会按指令回答”。  
但现实里很多回答都不止一个可行版本，真正的问题会变成：

- 哪个回答更好？
- 哪个回答更符合偏好？
- 模型应该把概率更多分给哪个版本？

DPO 的直觉可以先这样记：

- 对同一个 prompt，给模型一个 chosen 和一个 rejected；
- 希望 policy 相比 ref，更偏向 chosen；
- 但又不能完全脱离原模型分布跑飞，所以 ref 要留下来当锚点。

这一点我现在会用一句话概括：

> SFT 教模型“会回答”，DPO 教模型“在多个可回答版本里更偏向好的那个”。

## 4. MiniMind Implementation View

### 4.1 Source Code Path

```text
DPODataset
-> train_epoch
-> dpo_loss
-> logits_to_log_probs
```


| 文件                              | 位置                      | 作用                                        |
| ------------------------------- | ----------------------- | ----------------------------------------- |
| `trainer/train_dpo.py:33`       | `dpo_loss()`            | DPO 核心公式实现                                |
| `trainer/train_dpo.py:24`       | `logits_to_log_probs()` | 从 logits 提取实际 token 的 log 概率              |
| `trainer/train_dpo.py:55`       | `train_epoch()`         | 训练主循环，ref/policy 双模型前向 + 反向               |
| `trainer/train_dpo.py:175-184`  | `__main__` 初始化段         | policy 和 ref 从同一 SFT checkpoint 加载，ref 冻结 |
| `dataset/lm_dataset.py:126-196` | `DPODataset`            | 偏好数据加载，返回 x/y/mask 的 chosen/rejected 六元组  |

阅读顺序：`DPODataset` → `train_epoch` → `dpo_loss` → `logits_to_log_probs`

### 4.2 Data Flow

### 1. 数据构造（DPODataset）

原始数据来自 `dpo.jsonl`，每条包含 `chosen` 和 `rejected`（chat 格式 list）：

`{"chosen": [{role, content}, ...], "rejected": [{role, content}, ...]}` 

Dataset 的 `__getitem__` 对每个样本：

- 用 `apply_chat_template` 渲染为模型输入文本
    
- tokenize 后构造 ` x(input_ids[:-1])、y(input_ids[1:])`
    
- 用 `generate_loss_mask` 只在 assistant 回复区间标记 `mask=1`
    

返回 6 个张量（各 shape `(seq_len,)`）：`x_chosen, y_chosen, mask_chosen, x_rejected, y_rejected, mask_rejected`

### 2. Batch 拼接（train_epoch:64-67）
```text
x_chosen(4,1024) + x_rejected(4,1024) → x(8,1024)  
y_chosen(4,1024) + y_rejected(4,1024) → y(8,1024)  
mask_chosen(4,1024) + mask_rejected(4,1024) → mask(8,1024)
```

前半 batch 是 chosen，后半是 rejected，**拼成一个大 batch 一次性过模型**。

### 3. 双模型前向（train_epoch:73-81）
```text
x(8,1024) → ref_model(no_grad) → ref_logits(8,1024,vocab)  
                                 → logits_to_log_probs + y  
                                 → ref_log_probs(8,1024)  
​  
x(8,1024) → model(有梯度)      → policy_logits(8,1024,vocab)  
                                 → logits_to_log_probs + y  
                                 → policy_log_probs(8,1024)
```



ref 用 `torch.no_grad()` 冻结，policy 正常前向。

### 4. Loss 计算（dpo_loss:37-51）
```text
(8,1024) ── mask×sum/length ──→ (8,) 归一化标量  
                                  │  
                    [0:4] chosen       [4:8] rejected  
                        │                  │  
              pi_logratios = policy_chosen - policy_rejected  
              ref_logratios = ref_chosen - ref_rejected  
              logits = pi_logratios - ref_logratios  
              loss = -logsigmoid(beta × logits).mean()
```



### 5. 反向传播（train_epoch:87-94）

只更新 `policy model` 参数，`ref model` 始终不变。



### 4.4 DPO loss 

$$
\mathcal{L} = -\mathbb{E}\left[\log\sigma\left(\beta\left[\log\frac{\pi_\theta(y_w|x)}{\pi_\theta(y_l|x)} - \log\frac{\pi_{ref}(y_w|x)}{\pi_{ref}(y_l|x)}\right]\right)\right]
$$


| 代码变量               | 公式对应                                          | 含义                                  |
| ------------------ | --------------------------------------------- | ----------------------------------- |
| `pi_logratios`     | $\log\frac{\pi_\theta(y_w)}{\pi_\theta(y_l)}$ | policy 认为 chosen 比 rejected 好多少     |
| `ref_logratios`    | $\log\frac{\pi_{ref}(y_w)}{\pi_{ref}(y_l)}$   | ref 认为 chosen 比 rejected 好多少        |
| `logits`           | 两者之差                                          | policy 相对 ref 的**额外**偏好             |
| `beta`             | $\beta$                                       | 控制偏好惩罚强度，MiniMind 中默认 $\beta = 0.1$ |
| `-logsigmoid(...)` | $-\log\sigma(\cdot)$                          | 越偏好 chosen → loss 越低                |

三个关键状态

| 状态                 | logits | loss               | 含义                 |
| ------------------ | ------ | ------------------ | ------------------ |
| policy = ref（初始）   | = 0    | **log(2) ≈ 0.693** | 训练起点，policy 没有额外偏好 |
| policy 偏好 chosen   | > 0    | → 0                | 训练目标达成             |
| policy 偏好 rejected | < 0    | >> 0.693           | 方向错误，梯度会纠正         |

## 5. Minimal Verification

### 5.1 DPO loss 直觉验证
构造 tensor 验证如下三种情景：

| 场景                          | loss              | 结论                 |
| --------------------------- | ----------------- | ------------------ |
| policy = ref                | 0.693147 = log(2) | 精确匹配，确认初始状态        |
| policy 偏好 chosen (beta=0.1) | 0.6118 < 0.693    | loss 下降            |
| policy 偏好 chosen (beta=1.0) | 0.1678            | beta 大 10 倍，降幅显著更大 |
| policy 偏好 rejected          | 0.8952 > 0.693    | loss 上升            |

### 5.2 DPO 单步更新验证
脚本：`reimpl/experiment/exp_dpo_one_step.py`

用 TinyLM（vocab=16, hidden=8）构造 1 条 chosen + 1 条 rejected，跑 1 次前向+反向：

|指标|更新前|更新后|变化|
|---|---|---|---|
|chosen_policy_score|-3.064|-2.993|+0.071 (上升)|
|rejected_policy_score|-3.367|-3.471|-0.104 (下降)|
|loss|0.693|0.650|-0.043 (下降)|

验证了：DPO 一步更新后，policy 对 chosen 的评分上升、对 rejected 的评分下降，loss 下降。


## 6. Questions

1. DPO 在整个训练链路里的位置是什么？它解决的是“会回答”之后的哪一类问题？
2. 为什么不能把 DPO 简单理解成“对 chosen / rejected 做二分类”？
3. `DPODataset` 为什么会同时返回 `x/y/mask` 的 chosen 和 rejected 六元组？这说明训练仍然建立在什么机制上？
>  token 对齐和 loss mask
4. chosen 和 rejected 为什么会在 `train_epoch` 里拼成一个大 batch 一次性过模型？
>  - 减少 forward 次数；
> - 让 chosen / rejected 两半保持一一对应；
> - 后面在 loss 里可以直接按前半 / 后半切分。
5. `policy` 和 `ref` 的关系是什么？为什么需要 `ref model` ? 为什么 ` ref ` 必须冻结？
>  如果没有 `ref model`，policy 会尽可能把 chosen 概率抬高，把 rejected 概率压低，会影响模型的生成能力。
6. 为什么 `policy = ref` 时 DPO loss 的起点是 `log(2) ≈ 0.693`？
7. `beta` 在 DPO 里控制的是什么？为什么它不是学习率？

## 7. Review

- DPO 在整条训练链路里对应的是偏好对齐阶段，它站在 SFT 之后继续训练，但解决的问题已经从“会不会答”变成了“更偏向哪种回答”。
- DPO 不是直接把两条回答拿来做黑盒比较，它仍然会把 chosen / rejected 放回 causal LM 框架里，计算 token-level log-prob 再做聚合。
- `DPODataset` 返回 chosen / rejected 各自的 `x / y / mask`，这说明它的训练粒度依然落在 token 对齐和 loss mask 上。
- `policy` 和 `ref` 都从同一个 SFT checkpoint 出发，但 ref 保持冻结，作用是给 policy 提供一个稳定的比较锚点。
- DPO 真正优化的不是 policy 对 chosen 的绝对概率，而是 policy 相比 ref 的“额外偏好”。
- `policy = ref` 时，loss 从 `log(2) ≈ 0.693` 起步，这个基准值对调试和理解训练起点很重要。
- `beta` 控制的是偏好差异进入损失后的敏感度，值越大，loss 对同样的 log-prob 差异反应越强。

### Appendix 
#### 一个需要纠正的误区
对于 DPO 数据表面上是 chosen/rejected 两个候选回答，但在训练时，模型不是直接对整句做黑盒评分，而是转成 causal LM 的 next-token prediction 形式，通过 teacher forcing 计算这两条回答各自的 token-level log-prob，再聚合成序列级分数，最后进行偏好比较。
这也是 Dataloader 会同时返回 `x_chosen, x_rejected, y_chosen, y_rejected, mask_chosen, mask_rejected` 的原因。
#### 和 PPO 的比较
- DPO 是 off-policy 的，用静态偏好数据集训练，不需要 Reward Model
- PPO 是 on-policy 的，需要实时采样 + Reward Model 做教练。
- DPO 更简单但能力提升有限

## Related Notes

- [SFT](./05_sft.md)
- [Train Step](./04_train_step.md)
- [LoRA](./07_lora.md)
