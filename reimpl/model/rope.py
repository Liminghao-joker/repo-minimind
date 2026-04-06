import torch

def build_rope_cache(seq_len: int, head_dim: int, base: float = 10000.0):
    """
    生成每对维度应该旋转的角度，也就是 $\cos\theta_{m,i}, \quad \sin\theta_{m,i}$
    """
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))   # w_i.shape == [head_dim/2]
    positions = torch.arange(seq_len).float() # [seq_len]
    # theta[p, i] = p * w_i
    freqs = torch.outer(positions, inv_freq) # m.shape ==[seq_len, head_dim/2]

    """
    positions = [0, 1, 2]
    inv_freq = [w_0, w_1, w_2, w_3]  # head_dim=8 -> 4对维度
    freqs = [[0*w_0, 0*w_1, 0*w_2, 0*w_3],   # position=0
             [1*w_0, 1*w_1, 1*w_2, 1*w_3],   # position=1
             [2*w_0, 2*w_1, 2*w_2, 2*w_3]]   # position=2
    """

    # 把频率转换成 cos 和 sin，后续在 q/k 上进行旋转
    cos = torch.cos(freqs)   # [seq_len, head_dim/2]
    sin = torch.sin(freqs)   # [seq_len, head_dim/2]
    return cos, sin

def apply_rope_pairwise(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """
    对输入的 x 进行 RoPE 旋转
    x:   [seq_len, head_dim]
    cos: [seq_len, head_dim/2]
    sin: [seq_len, head_dim/2]
    """
    # 每一对 (2i, 2i+1) 维度进行旋转
    # x = [1, 2, 3, 4, 5, 6, 7, 8]
    # x_even = [1, 3, 5, 7]
    # x_odd  = [2, 4, 6, 8]
    x_even = x[:, 0::2]   # [seq_len, head_dim/2]，偶数维度
    x_odd  = x[:, 1::2]   # [seq_len, head_dim/2]，奇数维度

    x_rot_even = x_even * cos - x_odd * sin
    x_rot_odd  = x_even * sin + x_odd * cos

    # 把旋转后的结果放到原维度
    out = torch.empty_like(x)
    out[:, 0::2] = x_rot_even
    out[:, 1::2] = x_rot_odd
    # [x1', x2', x3', x4', x5', x6', x7', x8']
    return out

def precompute_freqs_cis(seq_len: int, head_dim: int, rope_base: float = 10000.0, attn_factor: float = 1.0):
    """
    更贴切 Minimind 官方的写法
    保留 `attn_factor` 参数，默认设为 1
    """
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"
    # freqs, attn_factor = 1 / (rope_base ** (torch.arange(0, head_dim, 2)[: (head_dim // 2)].float() / head_dim)), 1.0
    freqs = 1 / (rope_base ** (torch.arange(0, head_dim, 2).float() / head_dim)) # [head_dim/2]
    attn_factor = 1.0
    t = torch.arange(seq_len).float() # [seq_len]
    freqs = torch.outer(t, freqs).float() # [seq_len, head_dim/2]

    #* 不同于手写版 build_rope_cache 的实现，这里采用 cat 使返回的 cos/sin 维度与输入的 x 的 head_dim 保持一致，利于后续广播
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor  # [seq_len, head_dim]
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor  # [seq_len, head_dim]

    return freqs_cos, freqs_sin

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int = 0):
    """
    旋转位置编码的核心实现函数。
    q: [batch_size, num_heads, seq_len, head_dim]
    k: [batch_size, num_kv_heads, seq_len, head_dim] #! 此时还未进行 repeat_kv
    cos, sin: [seq_len, head_dim]
    unsqueeze_dim: 需要在哪个维度上扩展 cos/sin 以匹配 q/k 的维度
    """
    cos = cos.unsqueeze(unsqueeze_dim) # [1, seq_len, head_dim]
    sin = sin.unsqueeze(unsqueeze_dim)

    def rotate_half(x: torch.Tensor):
        """
        不再对 q 采用奇偶来分对，而是直接把后半段拼接到前半段并取反。
        eg: x = [a, b, c, d, e, f, g, h]
            -> [-e, -f, -g, -h, a, b, c, d]
        """
        half = x.shape[-1] // 2
        return torch.cat([-x[..., half:], x[..., :half]], dim=-1)

    q_embed = (q * cos + rotate_half(q) * sin).to(q.dtype)
    k_embed = (k * cos + rotate_half(k) * sin).to(k.dtype)
    return q_embed, k_embed


# GPT test
def demo_v1():
    """
    演示数学直写版的 RoPE 实现
    """
    seq_len = 3
    head_dim = 8

    # 每一行代表一个位置的 q 向量
    q = torch.tensor([
        [1., 2., 3., 4., 5., 6., 7., 8.],
        [1., 2., 3., 4., 5., 6., 7., 8.],
        [1., 2., 3., 4., 5., 6., 7., 8.],
    ])

    cos, sin = build_rope_cache(seq_len=seq_len, head_dim=head_dim, base=10000.0)
    q_rot = apply_rope_pairwise(q, cos, sin)

    print("=== 原始 q ===")
    print(q)
    print("\n=== cos.shape / sin.shape ===")
    print(cos.shape, sin.shape)   # 应为 [3, 4], [3, 4]

    # 看到每个位置的 cos/sin
    print("\n=== cos ===")
    print("cos.shape:", cos.shape)
    print(cos)
    print("\n=== sin ===")
    print("sin.shape:", sin.shape)
    print(sin)

    print("\n=== 旋转后 q_rot ===")
    print(q_rot)

    print("\n=== 检查 position=0 是否不变 ===")
    print(torch.allclose(q[0], q_rot[0], atol=1e-6))  # True，因为位置0角度为0

    print("\n=== 检查不同位置是否发生变化 ===")
    print(torch.allclose(q[1], q_rot[1], atol=1e-6))  # False
    print(torch.allclose(q[2], q_rot[2], atol=1e-6))  # False

    print("\n=== 查看第1对维度(0,1)在不同位置的变化 ===")
    print("pos0:", q_rot[0, 0:2])
    print("pos1:", q_rot[1, 0:2])
    print("pos2:", q_rot[2, 0:2])

    print("\n=== 查看第2对维度(2,3)在不同位置的变化 ===")
    print("pos0:", q_rot[0, 2:4])
    print("pos1:", q_rot[1, 2:4])
    print("pos2:", q_rot[2, 2:4])

def demo_v2():
    """
    演示 MiniMind 风格的 RoPE 实现，并与数学直写版进行对比
    """
    seq_len = 3
    head_dim = 8

    # baseline 输入：每一行代表一个位置的 q 向量
    q = torch.tensor([
        [1., 2., 3., 4., 5., 6., 7., 8.],
        [1., 2., 3., 4., 5., 6., 7., 8.],
        [1., 2., 3., 4., 5., 6., 7., 8.],
    ])

    print("=== 原始 q (interleaved 排布) ===")
    print(q)

    # -------------------------------------------------
    # 1) 数学直写版：pairwise RoPE
    # -------------------------------------------------
    cos_pair, sin_pair = build_rope_cache(seq_len=seq_len, head_dim=head_dim, base=10000.0)
    q_rot_pair = apply_rope_pairwise(q, cos_pair, sin_pair)

    print("\n" + "=" * 60)
    print("1) 数学直写版 apply_rope_pairwise")
    print("=" * 60)
    print("cos_pair.shape:", cos_pair.shape)   # [seq_len, head_dim/2]
    print("sin_pair.shape:", sin_pair.shape)
    print("q_rot_pair:")
    print(q_rot_pair)

    # -------------------------------------------------
    # 2) MiniMind 风格版：apply_rotary_pos_emb
    #    注意：它默认更适配 half-split 排布
    # -------------------------------------------------
    freqs_cos, freqs_sin = precompute_freqs_cis(seq_len=seq_len, head_dim=head_dim)

    # apply_rotary_pos_emb 期望 q/k 形状:
    # q: [batch_size, num_heads, seq_len, head_dim]
    q_for_mm = q.unsqueeze(0).unsqueeze(0)   # [1, 1, seq_len, head_dim]
    k_for_mm = q.unsqueeze(0).unsqueeze(0)

    q_rot_mm, _ = apply_rotary_pos_emb(
        q_for_mm, k_for_mm,
        freqs_cos, freqs_sin,
        unsqueeze_dim=0
    )
    q_rot_mm = q_rot_mm[0, 0]   # [seq_len, head_dim]

    print("\n" + "=" * 60)
    print("2) MiniMind 风格（直接喂 interleaved 输入）")
    print("=" * 60)
    print("freqs_cos.shape:", freqs_cos.shape)   # [seq_len, head_dim]
    print("freqs_sin.shape:", freqs_sin.shape)
    print("q_rot_mm:")
    print(q_rot_mm)

    print("\n=== 直接比较：pairwise vs minimind-direct ===")
    print("allclose:", torch.allclose(q_rot_pair, q_rot_mm, atol=1e-6))
    print("max abs diff:", (q_rot_pair - q_rot_mm).abs().max())

    # -------------------------------------------------
    # 3) 做维度重排：interleaved <-> half-split
    #    让 MiniMind 写法和数学版真正对齐
    # -------------------------------------------------
    def interleaved_to_halfsplit(x: torch.Tensor):
        # [x0,x1,x2,x3,x4,x5,x6,x7]
        # -> [x0,x2,x4,x6,x1,x3,x5,x7]
        return torch.cat([x[..., 0::2], x[..., 1::2]], dim=-1)

    def halfsplit_to_interleaved(x: torch.Tensor):
        half = x.shape[-1] // 2
        even = x[..., :half]
        odd = x[..., half:]
        out = torch.empty_like(x)
        out[..., 0::2] = even
        out[..., 1::2] = odd
        return out

    q_halfsplit = interleaved_to_halfsplit(q)               # [seq_len, head_dim]
    q_halfsplit_4d = q_halfsplit.unsqueeze(0).unsqueeze(0)  # [1,1,seq_len,head_dim]
    k_halfsplit_4d = q_halfsplit.unsqueeze(0).unsqueeze(0)

    q_rot_mm_aligned, _ = apply_rotary_pos_emb(
        q_halfsplit_4d, k_halfsplit_4d,
        freqs_cos, freqs_sin,
        unsqueeze_dim=0
    )
    q_rot_mm_aligned = q_rot_mm_aligned[0, 0]               # [seq_len, head_dim]
    q_rot_mm_back = halfsplit_to_interleaved(q_rot_mm_aligned)

    print("\n" + "=" * 60)
    print("3) MiniMind 风格（先重排到 half-split，再旋转，再转回）")
    print("=" * 60)
    print("q_halfsplit:")
    print(q_halfsplit)
    print("\nq_rot_mm_aligned (half-split 空间下):")
    print(q_rot_mm_aligned)
    print("\nq_rot_mm_back (转回 interleaved 后):")
    print(q_rot_mm_back)

    print("\n=== 最终严格比较：pairwise vs minimind-aligned ===")
    print("allclose:", torch.allclose(q_rot_pair, q_rot_mm_back, atol=1e-6))
    print("max abs diff:", (q_rot_pair - q_rot_mm_back).abs().max())

    print("\n=== 检查 position=0 是否不变 ===")
    print("pairwise:", torch.allclose(q[0], q_rot_pair[0], atol=1e-6))
    print("minimind-aligned:", torch.allclose(q[0], q_rot_mm_back[0], atol=1e-6))

    print("\n=== 查看第1对维度(0,1)在不同位置的变化 ===")
    print("pairwise pos0:", q_rot_pair[0, 0:2])
    print("pairwise pos1:", q_rot_pair[1, 0:2])
    print("pairwise pos2:", q_rot_pair[2, 0:2])

    print("mm-align pos0:", q_rot_mm_back[0, 0:2])
    print("mm-align pos1:", q_rot_mm_back[1, 0:2])
    print("mm-align pos2:", q_rot_mm_back[2, 0:2])

if __name__ == "__main__":
    demo_v2()