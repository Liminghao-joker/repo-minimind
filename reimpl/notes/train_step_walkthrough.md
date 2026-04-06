# MiniMind 一次 Pretrain Train Step 的数据流、Loss 计算与参数更新

## Today Goal

打通 MiniMind 一次完整 pretrain train step：从 batch 数据构造到参数更新，验证 forward → loss → backward → optimizer.step 链路连通。

## Why It Matters

一次 train step 是训练全流程的最小闭环。不理解它，后续的梯度累积、混合精度、学习率调度、Pretrain/SFT/LoRA/DPO 的区别都无从谈起。

这是理解所有训练变体的基座。

---

## Code Path（按执行顺序）

### 1. 数据构造：`dataset/lm_dataset.py:45-58`

```python
tokens = [bos_id] + tokenize(text) + [eos_id]
input_ids = tokens + [pad_id] * (max_length - len(tokens))
labels = input_ids.clone()
labels[input_ids == pad_id] = -100
```

**Pretrain vs SFT 的本质区别**：
- Pretrain：labels = input_ids，**每个 token 都学** next-token prediction
- SFT：只有 assistant 回复部分的 labels 非 -100，prompt 部分被 mask

### 2. Forward：`model_minimind.py:457-492`

```
input_ids [B, T]
  → embed_tokens → [B, T, H]
  → ×8 MiniMindBlock:
      RMSNorm → Attention(Q/K/V + RoPE + GQA repeat + causal) + 残差
      RMSNorm → FFN(SwiGLU) + 残差
  → final RMSNorm → lm_head → logits [B, T, V]
  → shift CE loss
```

### 3. Loss 计算：`model_minimind.py:484-488`

```python
shift_logits = logits[..., :-1, :]   # 位置 0~T-2 的预测
shift_labels = labels[..., 1:]       # 位置 1~T-1 的真实 token
loss = CE(shift_logits, shift_labels, ignore_index=-100)
```

**shift 的含义**：位置 i 的输出预测位置 i+1 的 token = next-token prediction。

### 4. Backward + Optimizer Step：`train_pretrain.py:37-46`

```python
scaler.scale(loss).backward()
scaler.unscale_(optimizer)
clip_grad_norm_(model.parameters(), 1.0)
scaler.step(optimizer)
optimizer.zero_grad(set_to_none=True)
```

---

## Shape Flow

默认配置 (B=2, T=16, H=512, V=6400, heads=8, kv_heads=2, head_dim=64)：

| 阶段 | 变量 | Shape | 说明 |
|------|------|-------|------|
| 输入 | input_ids | [2, 16] | |
| 输入 | labels | [2, 16] | PAD 位 = -100 |
| Embed | hidden_states | [2, 16, 512] | embed_tokens 查表 |
| Attention | xq | [2, 16, 8, 64] | Q: 8 个头 |
| Attention | xk, xv | [2, 16, 2, 64] | KV: 2 个头 (GQA) |
| Attention | repeat_kv 后 | [2, 16, 8, 64] | 复制 4 份匹配 Q |
| LM Head | logits | [2, 16, 6400] | |
| Shift | shift_logits | [2, 15, 6400] | 去尾 |
| Shift | shift_labels | [2, 15] | 去头 |
| Loss | loss | scalar | |

---

## Verification

通过 `reimpl/experiment/train_step_demo.py` 用 fake 数据验证：

| 检查项 | 结果 |
|--------|------|
| logits shape = [B, T, V] | PASS |
| loss 为有限正数 | PASS |
| backward 后参数有梯度且非零 | PASS |
| optimizer.step 后参数确实变化 | PASS |

---

## Key Takeaways

### 今天搞懂了什么

1. **Pretrain 的 labels 就是 input_ids 本身**——模型从第一个 token 开始就在学 next-token prediction，和 SFT（只学 assistant 部分）形成对比。

2. **shift 对齐是 autoregressive LM 训练的核心**：logits 去 tail、labels 去 head，实现"用位置 i 的输出预测位置 i+1"。

3. **GQA 的训练侧体现**：KV 头数少（2），通过 `repeat_kv` 复制匹配 Q 头数（8），节省显存但计算量不变。

4. **一次 train step 的真正参数更新点只有 `optimizer.step()`**——forward 只产生计算图，backward 只填梯度，参数到 step 才变。

5. **weight tying**：`embed_tokens.weight` 和 `lm_head.weight` 共享，梯度会累加到同一个张量上。

### 可被追问的点

- **Q: 为什么 Pretrain 不 mask 掉非回复部分？**
  Pretrain 目标是学语言的 next-token distribution，所有文本都是有效监督。SFT 才需要只对齐回复格式。

- **Q: 梯度累积在代码里怎么体现？**
  `loss /= accumulation_steps`，多步 backward 梯度累加，每 N 步才 step + zero_grad 一次。

- **Q: `ignore_index=-100` 的作用？**
  CrossEntropy 遇到 label=-100 时跳过该位置，不贡献 loss 也不贡献梯度，用于 mask padding。

- **Q: `aux_loss` 什么时候非零？**
  使用 MoE 架构时（`use_moe=True`），MoEGate 计算专家负载均衡辅助损失，鼓励专家被均匀使用。
