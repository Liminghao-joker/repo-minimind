"""
MiniMind Pretrain Train Step Demo
=============================================
目标：跑通 forward → loss → backward → optimizer.step 完整链路。

验证 3 件事：
1. forward 输出 logits shape = [B, T, V]，loss 是有限正数
2. backward 后参数有梯度且非零
3. optimizer.step 后参数确实发生变化

不需要：真实数据集、tokenizer、DDP、混合精度、checkpoint
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
from torch import optim
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

# ==================== 板块 1：构造 fake 数据 ====================
# 对应源码：dataset/lm_dataset.py PretrainDataset.__getitem__
# Pretrain 阶段：labels = input_ids（所有有效位都参与 loss），PAD 位 = -100

B, T = 2, 16  # batch_size, seq_len（最小可验证规模）

# 随机 input_ids，范围 [3, 6400)，避开 BOS=1, EOS=2, PAD=0
torch.manual_seed(42)
input_ids = torch.randint(3, 6400, (B, T))

# labels = input_ids，尾部 4 位设 -100 模拟 padding
labels = input_ids.clone()
labels[:, -4:] = -100

print(f"[输入] input_ids shape: {input_ids.shape}")   # 期望 [2, 16]
print(f"[输入] labels shape:    {labels.shape}")       # 期望 [2, 16]
print(f"[输入] labels 中 -100 数量: {(labels == -100).sum().item()}")  # 期望 8

# ==================== 板块 2：初始化模型 + 优化器 ====================
# 对应源码：trainer/train_pretrain.py L106-132 + trainer_utils.init_model

config = MiniMindConfig()  # 默认: hidden=512, layers=8, heads=8, kv_heads=2, vocab=6400
model = MiniMindForCausalLM(config)
model.train()  # 训练模式

optimizer = optim.AdamW(model.parameters(), lr=1e-3)

print(f"\n[模型] hidden_size={config.hidden_size}, num_layers={config.num_hidden_layers}, "
      f"vocab_size={config.vocab_size}")
print(f"[模型] 可训练参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

# 记录 step 前 2 个关键参数的值（用于后续对比）
param_lm_head = model.lm_head.weight.data.clone()
param_embed = model.model.embed_tokens.weight.data.clone()

# ==================== 板块 3：Forward ====================
# 对应源码：model_minimind.py MiniMindForCausalLM.forward L457-492
# 调用链：embed_tokens → ×8 Block(RMSNorm → Attention+残差 → RMSNorm → FFN+残差) → final_norm → lm_head → shift CE loss

res = model(input_ids=input_ids, labels=labels)
logits = res.logits # [B, T, vocab_size]
loss = res.loss # scalar 

print(f"\n[Forward] logits shape: {logits.shape}")     # 期望 [2, 16, 6400]
print(f"[Forward] loss: {loss.item():.4f}")             # 期望有限正数
print(f"[Forward] aux_loss: {res.aux_loss.item():.4f}") # 非 MoE，期望 0.0

# ==================== 板块 4：Backward ====================
# 对应源码：trainer/train_pretrain.py L37
# scaler.scale(loss).backward() 的简化版（无混合精度）

loss.backward()

# 检查梯度
grad_lm_head = model.lm_head.weight.grad
grad_embed = model.model.embed_tokens.weight.grad

print(f"\n[Backward] lm_head.weight.grad norm: {grad_lm_head.norm().item():.4f}")
print(f"[Backward] embed_tokens.weight.grad norm: {grad_embed.norm().item():.4f}")
print(f"[Backward] 梯度非零参数数量: {sum(1 for p in model.parameters() if p.grad is not None)}")

# ==================== 板块 5：Optimizer Step ====================
# 对应源码：trainer/train_pretrain.py L39-46
# 简化版（无梯度累积、无混合精度 scaler、无梯度裁剪）

optimizer.step()
optimizer.zero_grad(set_to_none=True)

# 验证参数确实被更新
lm_head_diff = (model.lm_head.weight.data - param_lm_head).norm().item()
embed_diff = (model.model.embed_tokens.weight.data - param_embed).norm().item()

print(f"\n[Step] lm_head.weight 变化量: {lm_head_diff:.6f}")
print(f"[Step] embed_tokens.weight 变化量: {embed_diff:.6f}")
print(f"[Step] 梯度是否已清零: {model.lm_head.weight.grad is None}")

# ==================== 板块 6：结论 ====================
checks = {
    "logits shape 正确": logits.shape == torch.Size([B, T, config.vocab_size]),
    "loss 为有限正数": torch.isfinite(loss) and loss.item() > 0,
    "lm_head 有梯度": grad_lm_head is not None and grad_lm_head.norm() > 0,
    "embed_tokens 有梯度": grad_embed is not None and grad_embed.norm() > 0,
    "lm_head 参数已更新": lm_head_diff > 0,
    "embed_tokens 参数已更新": embed_diff > 0,
}

print("\n" + "=" * 50)
print("链路验证结果：")
for name, passed in checks.items():
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name}")
all_passed = all(checks.values())
print(f"\n结论: {'一次 train step 跑通!' if all_passed else '存在问题，需要排查'}")
