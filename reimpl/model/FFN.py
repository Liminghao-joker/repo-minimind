"""FFN & SwiGLU"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

"""test1: compare relu and silu"""

def compare_relu_and_silu():
    x = torch.linspace(-5, 5, 200)

    relu_y = F.relu(x)
    silu_y = F.silu(x)

    plt.figure(figsize=(10, 5))
    plt.plot(x.numpy(), relu_y.numpy(), label='ReLU', linewidth=2)
    plt.plot(x.numpy(), silu_y.numpy(), label='SiLU (Swish)', linewidth=2)
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    plt.legend(fontsize=14)
    plt.title('ReLU vs SiLU')
    plt.grid(True, alpha=0.3)
    plt.show()

"""test2: implement simple FFN with SwiGLU"""
class SimpleSwiGLU_FFN(nn.Module):
    def __init__(self, d_model, hidden_dim):
        super().__init__()
        #* 定义三个线性层：gate_proj, up_proj, down_proj
        self.gate_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.up_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x):
        # 1. gate_proj(x): d_model -> hidden_dim
        # 2. F.silu(...) -> silu 激活产生门控信号
        # 3. up_proj(x): d_model -> hidden_dim 高维空间
        # 4. gate * up_proj(x) 表示被筛选的信息
        # 5. down_proj(...) 压缩到低维度
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

def compare_relu_gelu_silu():
    x = torch.linspace(-5, 5, 500)

    relu = F.relu(x)
    gelu = F.gelu(x)
    silu = F.silu(x)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(x.numpy(), relu.numpy(), label='ReLU', linewidth=2)
    axes[0].plot(x.numpy(), gelu.numpy(), label='GELU', linewidth=2)
    axes[0].plot(x.numpy(), silu.numpy(), label='SiLU (Swish)', linewidth=2)
    axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.4)
    axes[0].axvline(x=0, color='gray', linestyle='--', alpha=0.4)
    axes[0].set_title('Comparison of ReLU, GELU and SiLU')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    relu_grad = torch.where(x > 0, torch.ones_like(x), torch.zeros_like(x))
    x_grad = x.clone().requires_grad_(True)
    silu_out = F.silu(x_grad)
    silu_out.sum().backward()
    silu_grad = x_grad.grad.clone()

    axes[1].plot(x.numpy(), relu_grad.numpy(), label='ReLU Gradient', linewidth=2)
    axes[1].plot(x.detach().numpy(), silu_grad.numpy(), label='SiLU Gradient', linewidth=2)
    axes[1].set_title('Gradient Comparison')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # compare_relu_and_silu()
    # ffn = SimpleSwiGLU_FFN(d_model=768, hidden_dim=2048)
    # x = torch.randn(2, 10, 768)  # [batch_size, seq_len, d_model]
    # output = ffn(x)
    # print("Input shape:", x.shape)        # 应为 [2, 10, 768]
    # print("FFN output shape:", output.shape)  # 应为 [2, 10, 768]   
    # # 统计参数量
    # total_params = sum(p.numel() for p in ffn.parameters())
    # print("Total parameters in SimpleSwiGLU_FFN:", total_params)  # 3*768*2048 = 4,718,592 params
    compare_relu_gelu_silu()
