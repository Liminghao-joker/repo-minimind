
import torch

def build_rope_cache_pairwise(seq_len: int, head_dim: int, base: float = 10000.0):
    """
    数学直写版：
    生成每对维度 (2i, 2i+1) 对应的 cos/sin，shape = [seq_len, head_dim/2]
    """
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    positions = torch.arange(seq_len).float()
    freqs = torch.outer(positions, inv_freq)  # [seq_len, head_dim/2]
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    return cos, sin


def apply_rope_pairwise(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """
    数学直写版：按相邻 pair 旋转
    x:   [seq_len, head_dim]
    cos: [seq_len, head_dim/2]
    sin: [seq_len, head_dim/2]
    """
    x_even = x[:, 0::2]
    x_odd = x[:, 1::2]

    x_rot_even = x_even * cos - x_odd * sin
    x_rot_odd = x_even * sin + x_odd * cos

    out = torch.empty_like(x)
    out[:, 0::2] = x_rot_even
    out[:, 1::2] = x_rot_odd
    return out


# =========================
# MiniMind 官方风格写法
# =========================

def precompute_freqs_cis_minimind(head_dim: int, end: int, rope_base: float = 10000.0, attn_factor: float = 1.0):
    """
    MiniMind 官方风格：
    1. 先算半维 freqs: [seq_len, head_dim/2]
    2. 再 cat 成整维 cos/sin: [seq_len, head_dim]
    """
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"
    freqs = 1.0 / (rope_base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(end).float()
    freqs = torch.outer(t, freqs).float()  # [seq_len, head_dim/2]

    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    return freqs_cos, freqs_sin


def rotate_half_minimind(x: torch.Tensor):
    """
    MiniMind / LLaMA-NeoX 风格：
    把后半段搬到前面并取负，再拼接前半段
    例：
    x = [a, b, c, d, e, f, g, h]
    -> [-e, -f, -g, -h, a, b, c, d]
    """
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rotary_pos_emb_minimind(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int = 1):
    """
    MiniMind 官方风格：
    q, k : [bsz, seq_len, n_heads, head_dim] 或其他兼容布局
    cos,sin: [seq_len, head_dim]
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    q_embed = (q * cos + rotate_half_minimind(q) * sin).to(q.dtype)
    k_embed = (k * cos + rotate_half_minimind(k) * sin).to(k.dtype)
    return q_embed, k_embed


# =========================
# 为了公平比较：把 pairwise 输入重排成 MiniMind 期望布局
# =========================

def interleaved_to_halfsplit(x: torch.Tensor):
    """
    把数学直写版的相邻配对布局:
    [x0, x1, x2, x3, x4, x5, x6, x7]
    变成 MiniMind rotate_half 适配的半区布局:
    [x0, x2, x4, x6, x1, x3, x5, x7]

    这样 rotate_half 才等价于“每个二维对子旋转 90°”。
    """
    return torch.cat([x[..., 0::2], x[..., 1::2]], dim=-1)


def halfsplit_to_interleaved(x: torch.Tensor):
    """
    把 MiniMind 的半区布局再还原回相邻配对布局，便于和数学版逐元素比较。
    """
    half = x.shape[-1] // 2
    first = x[..., :half]
    second = x[..., half:]

    out = torch.empty_like(x)
    out[..., 0::2] = first
    out[..., 1::2] = second
    return out


def compare_demo():
    torch.set_printoptions(precision=6, sci_mode=False)

    seq_len = 3
    head_dim = 8

    # baseline: 数学直写版输入，默认是“相邻两维一组”
    q_pairwise = torch.tensor([
        [1., 2., 3., 4., 5., 6., 7., 8.],
        [1., 2., 3., 4., 5., 6., 7., 8.],
        [1., 2., 3., 4., 5., 6., 7., 8.],
    ])
    k_pairwise = q_pairwise.clone()

    print("=== baseline: 数学直写版输入（interleaved pair layout） ===")
    print(q_pairwise)

    # ---- 数学直写版 ----
    cos_pair, sin_pair = build_rope_cache_pairwise(seq_len=seq_len, head_dim=head_dim, base=10000.0)
    q_rot_pairwise = apply_rope_pairwise(q_pairwise, cos_pair, sin_pair)

    print("\n=== 数学直写版输出 q_rot_pairwise ===")
    print(q_rot_pairwise)

    # ---- MiniMind 官方风格 ----
    # 先把输入重排到 half-split 布局
    q_half = interleaved_to_halfsplit(q_pairwise)
    k_half = interleaved_to_halfsplit(k_pairwise)

    # 增加 batch / head 维，模拟 attention 内部的真实 shape
    # [bsz, seq_len, n_heads, head_dim]
    q_half_4d = q_half.unsqueeze(0).unsqueeze(2)
    k_half_4d = k_half.unsqueeze(0).unsqueeze(2)

    cos_full, sin_full = precompute_freqs_cis_minimind(head_dim=head_dim, end=seq_len, rope_base=10000.0)

    q_rot_half_4d, _ = apply_rotary_pos_emb_minimind(
        q_half_4d, k_half_4d, cos_full, sin_full, unsqueeze_dim=1
    )

    q_rot_half = q_rot_half_4d.squeeze(0).squeeze(1)  # [seq_len, head_dim]
    q_rot_minimind_back = halfsplit_to_interleaved(q_rot_half)

    print("\n=== MiniMind 风格输入（half-split layout） ===")
    print(q_half)

    print("\n=== MiniMind 风格旋转后（仍是 half-split layout） ===")
    print(q_rot_half)

    print("\n=== MiniMind 风格旋转后，再还原回 interleaved layout ===")
    print(q_rot_minimind_back)

    # ---- 对照比较 ----
    print("\n=== cache shape 对照 ===")
    print("pairwise cos/sin:", cos_pair.shape, sin_pair.shape)   # [seq_len, head_dim/2]
    print("minimind cos/sin:", cos_full.shape, sin_full.shape)  # [seq_len, head_dim]

    print("\n=== 数值是否一致（数学版 vs MiniMind版还原后） ===")
    print(torch.allclose(q_rot_pairwise, q_rot_minimind_back, atol=1e-6))

    print("\n=== 最大绝对误差 ===")
    print((q_rot_pairwise - q_rot_minimind_back).abs().max())

    print("\n=== 检查 position=0 是否不变 ===")
    print("pairwise :", torch.allclose(q_pairwise[0], q_rot_pairwise[0], atol=1e-6))
    print("minimind :", torch.allclose(q_pairwise[0], q_rot_minimind_back[0], atol=1e-6))

    print("\n=== 第 1 对维度 (0,1) 对照 ===")
    print("pairwise pos0/1/2:")
    print(q_rot_pairwise[:, 0:2])
    print("minimind pos0/1/2:")
    print(q_rot_minimind_back[:, 0:2])

    print("\n=== 第 2 对维度 (2,3) 对照 ===")
    print("pairwise pos0/1/2:")
    print(q_rot_pairwise[:, 2:4])
    print("minimind pos0/1/2:")
    print(q_rot_minimind_back[:, 2:4])

    print("\n=== rotate_half 直观测试 ===")
    x = torch.tensor([1., 3., 5., 7., 2., 4., 6., 8.])  # half-split layout
    print("x                =", x)
    print("rotate_half(x)   =", rotate_half_minimind(x))
    print("解释：前半段是偶数维，后半段是奇数维；rotate_half 后相当于对子向量执行 [-y, x]")


if __name__ == "__main__":
    compare_demo()
