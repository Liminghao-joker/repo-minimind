import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# 复用 rope.py 手写实现的 RoPE 函数
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

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """
    旋转位置编码的核心实现函数。
    q: [batch_size, num_heads, seq_len, head_dim]
    k: [batch_size, num_kv_heads, seq_len, head_dim] #! 此时还未进行 repeat_kv
    cos, sin: [seq_len, head_dim]
    unsqueeze_dim: 需要在哪个维度上扩展 cos/sin 以匹配 q/k 的维度
    """
    #! boardcast cos/sin 到 q/k 的维度
    cos = cos.unsqueeze(1) # [B, 1, seq_len, head_dim]
    sin = sin.unsqueeze(1)

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

class MiniAttentionDemo(nn.Module):
    def __init__(
        self,
        vocab_size=100,
        hidden_size=64,
        num_heads=4,
        num_kv_heads=2,
        max_seq_len=128,
    ):
        super().__init__()
        assert hidden_size % num_heads == 0 # head_dim 分头
        assert num_heads % num_kv_heads == 0 # num_kv_groups 分组

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.num_kv_groups = num_heads // num_kv_heads
        self.max_seq_len = max_seq_len

        # embedding
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)

        # q/k/v projection
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=False) 

        # output projection
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)

        # lm_head
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        self.precompute_freqs_cis = precompute_freqs_cis
        self.apply_rotary_pos_emb = apply_rotary_pos_emb

    def repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, n_kv_heads, T, head_dim]
        -> [B, n_heads, T, head_dim]
        """
        if self.num_kv_groups == 1:
            return x
        B, n_kv_heads, T, D = x.shape
        x = x[:, :, None, :, :].expand(B, n_kv_heads, self.num_kv_groups, T, D)
        return x.reshape(B, n_kv_heads * self.num_kv_groups, T, D)
    

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values=None,
        use_cache=False,
    ):
        """
        input_ids: [B, T]
        past_key_values:
            None
            or tuple(past_k, past_v)
            past_k: [B, n_kv_heads, T_past, D]
            past_v: [B, n_kv_heads, T_past, D]
        """
        B, T = input_ids.shape

        # ===== 1. embedding =====
        x = self.embed_tokens(input_ids)   # [B, T, hidden_size]

        # ===== 2. q/k/v projection =====
        q = self.q_proj(x)  # [B, T, n_heads * D]
        k = self.k_proj(x)  # [B, T, n_kv_heads * D]
        v = self.v_proj(x)  # [B, T, n_kv_heads * D]

        # ===== 3. reshape to heads =====
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2) # [B, num_heads, T, head_dim]
        k = k.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2) # [B, num_kv_heads, T, head_dim]
        v = v.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # ===== 4. position ids =====
        past_len = 0 if past_key_values is None else past_key_values[0].shape[2]
        position_ids = torch.arange(past_len, past_len + T, device=input_ids.device) #! position_ids is a tensor rather than a slice
        position_ids = position_ids.unsqueeze(0).expand(B, T)  # [B, T]

        # ===== 5. RoPE on q/k =====
        full_cos, full_sin = self.precompute_freqs_cis(self.max_seq_len, self.head_dim) # [max_seq_len, head_dim]
        full_cos = full_cos.to(input_ids.device)
        full_sin = full_sin.to(input_ids.device)
        cos = full_cos[position_ids]
        sin = full_sin[position_ids]
        q, k = self.apply_rotary_pos_emb(q, k, cos, sin)

        # ===== 6. append kv-cache =====
        if past_key_values is not None:
            past_k, past_v = past_key_values
            k = torch.cat([past_k, k], dim=2)   # [B, n_kv_heads, T, D]
            v = torch.cat([past_v, v], dim=2)

        new_past_key_values = (k, v) if use_cache else None

        # ===== 7. GQA: repeat kv heads =====
        # 在计算 attention 之前将 k/v 恢复成正常维度
        k_for_attn = self.repeat_kv(k)   # [B, n_heads, T_total, D]
        v_for_attn = self.repeat_kv(v)   # [B, n_heads, T_total, D]

        # ===== 8. attention score =====
        attn_scores = torch.matmul(q, k_for_attn.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # shape: [B, n_heads, T, T_total]

        # ===== 9. causal mask =====
        T_total = k_for_attn.shape[2]
        causal_mask = torch.full((T, T_total), float("-inf"), device=input_ids.device)
        causal_mask = torch.triu(causal_mask, diagonal=1 + past_len)
        attn_scores = attn_scores + causal_mask.unsqueeze(0).unsqueeze(0)

        # ===== 10. softmax =====
        attn_weights = F.softmax(attn_scores, dim=-1)

        # ===== 11. weighted sum =====
        attn_output = torch.matmul(attn_weights, v_for_attn)   # [B, n_heads, T, D]

        # ===== 12. merge heads =====
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, self.hidden_size)

        # ===== 13. output projection =====
        hidden_states = self.o_proj(attn_output)   # [B, T, hidden_size]

        # ===== 14. logits =====
        logits = self.lm_head(hidden_states)       # [B, T, vocab_size]

        return {
            "logits": logits,
            "hidden_states": hidden_states,
            "past_key_values": new_past_key_values,
        }