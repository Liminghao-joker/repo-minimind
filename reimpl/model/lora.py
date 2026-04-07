"""手动实现最小 lora 层"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LoRALinear(nn.Module):
    """最小 LoRA 线性层，y = Wx + (alpha / r) * BAx"""
    def __init__(self, in_features, out_features, r=8, alpha=16):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.linear.weight.requires_grad = False # 冻结原始权重

        self.lora_A = nn.Linear(in_features, r, bias=False)
        # A 高斯初始化
        self.lora_A.weight.data.normal_(mean=0.0, std=0.01)
        # # 可选：也可采用 kaiming_normal_ 初始化
        # self.lora_A = nn.Parameter(torch.empty(r, in_features))
        # nn.init.kaiming_normal_(self.lora_A, a=math.sqrt(5))

        self.lora_B = nn.Linear(r, out_features, bias=False)
        # B 全 0 初始化
        self.lora_B.weight.data.zero_()
        self.r = r
        self.scaling = alpha / r
    
    def forward(self, x):
        """x: [B, in_features]"""
        return self.linear(x) + self.scaling * self.lora_B(self.lora_A(x))
    
from model.model_minimind import MiniMindForCausalLM, MiniMindConfig

# 1. 加载模型，打印原始可训练参数量
config = MiniMindConfig()
model = MiniMindForCausalLM(config)

total_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"原始可训练参数: {total_before:,}") # 25829888
# 2. 冻结全部
for param in model.parameters():
    param.requires_grad = False
total_frozen = sum(p.numel() for p in model.parameters() if p.requires_grad)
assert total_frozen == 0, "冻结失败！"

# 3. 替换 attention 层（以第 0 层 q_proj 为例）
attn = model.model.layers[0].self_attn
attn.q_proj = LoRALinear(config.hidden_size, config.hidden_size, r=8, alpha=16)

# 4. 检查可训练参数只剩 A 和 B
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"LoRA 可训练参数: {trainable:,}")  # 应该 = 8*512 + 512*8 = 8192
# 验证减少了多少参数
reduction = (total_before - trainable) / total_before * 100
print(f"参数减少: {reduction:.2f}%")

# 5. dummy forward 验证输出形状
x = torch.randint(0, config.vocab_size, (1, 16))  # (batch=1, seq_len=16)
out = model(x)
print(f"输出 logits 形状: {out.logits.shape}")  # 应为 (1, 16, vocab_size)