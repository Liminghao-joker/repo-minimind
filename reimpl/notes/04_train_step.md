# MiniMind Train Step 的 dataflow、Loss and parameter

## Today Goal

**验证并读懂一次最小 pretrain train step**

`fake batch -> model forward -> logits/loss -> backward -> optimizer.step`

---

## Why It Matters

对于 decoder-only LLM 来说，一次 train step 是训练流程的最小闭环。

它回答的是 4 个最基础的问题：

1. 输入数据以什么形式送进模型？
2. logits 和 labels 是怎样对齐计算 loss 的？
3. 梯度是如何从 loss 回传到 embedding 和各层参数的？
4. 参数到底是在什么时候真正发生更新的？

---

## Intuition

从直觉出发，去认识一次 train_step：

1. 先给模型一段 token 序列
2. 让模型在每个位置预测“下一个 token 应该是什么”
3. 用真实答案和预测结果算出误差（loss）
4. 让误差沿网络反向传播，告诉每一层“哪里预测错了”
5. 最后由优化器根据梯度微调参数
---
## Code Path
一个典型的 train step 流程图归纳如下：
```text
DataLoader → (input_ids, labels)                                                                                                   
                ↓                                                                                                                  
model(input_ids, labels=labels)      ← MiniMindForCausalLM.forward                                                                 
  ├── embed_tokens(input_ids)        ← [B, T] → [B, T, H]                                                                          
  ├── for each MiniMindBlock:                                                                                                      
  │     ├── RMSNorm → Attention (Q/K/V + RoPE + causal mask) → residual                                                            
  │     └── RMSNorm → FFN/SwiGLU → residual                                                                                        
  ├── final RMSNorm                                                                                                                
  ├── lm_head(hidden_states)         ← [B, T, H] → [B, T, V]                                                                       
  ├── shift_logits = logits[:, :-1]                                                                                                
  ├── shift_labels  = labels[:, 1:]                                                                                                
  └── cross_entropy(shift_logits, shift_labels, ignore_index=-100)                                                                 
                ↓                                                                                                                  
loss.backward()                       ← 计算梯度                                                                                   
optimizer.step()                      ← 更新参数                                                                                   
optimizer.zero_grad()                 ← 清零梯度  
```
通过实验，验证最小的 train step。
## 1. 构造最小 batch


```python
B, T = 2, 16
input_ids = torch.randint(3, 6400, (B, T)) # shape:[B, T]
labels = input_ids.clone() # 初始化为与 input_ids 一致
labels[:, -4:] = -100 # 不参与 cross-entropy 的 loss 计算
```

- **Pretrain**：大部分位置都参与 next-token prediction
- **SFT**：通常只让 assistant 回复部分参与 loss，其余 prompt 部分会被 mask 成 `-100`

> 对于 causal language modeling：`-100` 则表示该位置不参与交叉熵计算。

---

## 2. 模型 forward 

```python
res = model(input_ids=input_ids, labels=labels)
logits = res.logits
loss = res.loss
```

模型内部经历了完整的 decoder-only 前向路径：

`input_ids`
`-> embed_tokens`
`-> N × Transformer Block`
`-> final norm`
`-> lm_head`
`-> logits`
`-> shift + cross entropy loss`

这里的 `logits` 的最后 vocab_size 维度衡量了对整个词表的预测分布。


---

## 3. Loss 

外部传入的 `labels` shape 仍然是 `[B, T]`，并没有手动写：

```python
shift_logits = logits[:, :-1, :]
shift_labels = labels[:, 1:]
```
 **shift 对齐是在 `MiniMindForCausalLM.forward` 内部完成的**。

其本质是：

- 位置 `i` 的输出
- 去预测位置 `i+1` 的真实 token

也就是 autoregressive LM 的标准 next-token prediction。

如果写成更直观的形式，就是：

```python
shift_logits = logits[..., :-1, :]
shift_labels = labels[..., 1:]
loss = CE(shift_logits, shift_labels, ignore_index=-100)
```

这里需要特别记住两个点：

### 1）为什么 logits 去尾、labels 去头？
因为当前位置的输出只能预测“下一个 token”，不能预测自己。

### 2）为什么 `-100` 很关键？
因为 `CrossEntropyLoss(ignore_index=-100)` 会跳过这些位置，不让它们贡献 loss 和梯度。

---

## 4. Backward 

有了 loss 之后，最小反向传播只需要：

```python
loss.backward()
```

该实验检查了两类参数：

```python
grad_lm_head = model.lm_head.weight.grad
grad_embed = model.model.embed_tokens.weight.grad
```

---

## 5. 参数更新
`backward` 仅完成了反向梯度的传递，但此时参数还没有进行更新。

- `forward`：建立计算图，得到 `logits/loss`
- `backward`：把梯度写入各参数的 `.grad`
- `optimizer.step()`：真正按照梯度更新参数

所以这个 demo 在 step 前先保存了参数副本：

```python
param_lm_head = model.lm_head.weight.data.clone()
param_embed = model.model.embed_tokens.weight.data.clone()
```

随后执行：

```python
optimizer.step()
optimizer.zero_grad(set_to_none=True)
```

再比较更新前后的参数差值：

```python
lm_head_diff = (model.lm_head.weight.data - param_lm_head).norm().item()
embed_diff = (model.model.embed_tokens.weight.data - param_embed).norm().item()
```

验证参数差值大于 0，说明：参数确实进行了更新。

---

## Shape Flow

结合 MiniMind 默认配置，本次 train step 的主要 shape 流整理如下：

| Stage | Tensor | Shape | Explanation |
|---|---|---:|---|
| Input | `input_ids` | `[2, 16]` | batch 输入 token id |
| Input | `labels` | `[2, 16]` | 与 input_ids 同 shape，部分位置为 `-100` |
| Embedding | hidden states | `[2, 16, 512]` | 词向量表示 |
| Attention | `q` | `[2, 16, 8, 64]` | 8 个 query heads |
| Attention | `k/v` | `[2, 16, 2, 64]` | 2 个 kv heads（GQA） |
| Repeat KV | repeated `k/v` | `[2, 16, 8, 64]` | 匹配 query 头数 |
| Output | `logits` | `[2, 16, 6400]` | 对词表每个 token 的预测 |
| Shift | `shift_logits` | `[2, 15, 6400]` | 去掉最后一个位置 |
| Shift | `shift_labels` | `[2, 15]` | 去掉第一个位置 |
| Loss | `loss` | `scalar` | 交叉熵标量 |


---

## Questions

### Q1：为什么 pretrain 里通常 `labels = input_ids`？
因为 pretrain 的目标是学习一般文本的 next-token distribution，而要预测的下一个 token 本身就包含在原序列里，因此序列中的大多数 token 都可以作为监督信号。但在计算 loss 时要进行 shift 操作，即：
```python
shift_logits = logits[:, :-1, :] # [B, T, vocab_size]
shift_labels = labels[:, 1:] # [B, T]
```

### Q2：为什么 `logits` 是 `[B, T, V]`？
因为 batch 中每个样本的每个位置，都要输出一个对整个词表的预测分布。

### Q3：为什么 loss 计算要做 shift？
因为 decoder-only LM 在位置 `i` 的输出，用来预测位置 `i+1` 的 token，而不是当前位置自己。

### Q4：`backward()` 和 `optimizer.step()` 的区别是什么？
`backward()` 只负责计算并写入梯度；`optimizer.step()` 才真正修改参数值。

---
## Takeways

- `shift_logits` 去尾、`shift_labels` 去头，反映的正是“位置 i 的输出去预测位置 i+1 的 token”。
- `-100` 不是普通 token id，而是交叉熵里的 `ignore_index`；它直接决定哪些位置不参与 loss 和梯度计算。
- `optimizer.step()` 才是真正更新参数的时刻，`backward()` 只是把梯度写到 `.grad` 上。