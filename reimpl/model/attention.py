# Day2: Attention mechanism impplementation
# 1. Single head attention
# 2. standard multi-head attention
# 3. compare shape according to MiniMind source code

# some test I can do：

# print shape
# assert
# check softmax
# casual mask
# flash-attention

# TODO: 结构解耦，分别实现 single_head_attention 和 multi_head attention，保证能够跑通；同时继续推进 GQA

import  torch
import torch.nn as nn
import torch.nn.functional as F
import math

def precompute_freqs_cis(dim: int, end: int, rope_base: float = 10000.0):
    """
    dim: head_dim
    end: max_seq_len
    return:
        freqs_cos: [end, dim]
        freqs_sin: [end, dim]
    """

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, 
                         cos: torch.Tensor, sin: torch.Tensor,
                         unsqueeze_dim: int = 1):
    """
    q: [B, num_heads, T, head_dim]
    k: [B, num_kv_heads, T, head_dim]
    cos: [T, head_dim]
    sin: [T, head_dim]
    """
    # Implementation for applying rotary position embedding
    def rotate_half(x: torch.Tensor):
        half = x.shape[-1] // 2
        return torch.cat([-x[..., half:], x[..., :half]], dim=-1)
    
    q_embed = q * cos.unsqueeze(unsqueeze_dim) + rotate_half(q) * sin.unsqueeze(unsqueeze_dim)
    k_embed = k * cos.unsqueeze(unsqueeze_dim) + rotate_half(k) * sin.unsqueeze(unsqueeze_dim)
    return q_embed, k_embed

class SingleHeadSelfAttention(nn.Module):
    """
    x -> q/k/v -> scores -> causal mask -> softmax -> output
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

        # q/k/v projection layers
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        # output projection layer
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor):
        """
        x: [B, T, D]
        return: [B, T, D]

        B: batch_size
        T: seq_len
        D： hidden_dim
        H: num_attention_heads
        """

        # 1. input embedding vector
        B, T, H = x.shape # [B, T, H]
        assert H == self.hidden_size

        print("Input x:", x.shape)


        # 2. q,k,v projection layers
        q = self.q_proj(x) # [B, T, H]
        k = self.k_proj(x) # [B, T, H]
        v = self.v_proj(x) # [B, T, H]
        assert q.shape == (B, T, H)
        assert k.shape == (B, T, H)
        assert v.shape == (B, T, H)
        print("q/k/v shape:", q.shape, k.shape, v.shape)

        # 3. attention
        # falsh-attention and manual implementation
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.hidden_size)  # [B, T, T]
        print("scores shape:", scores.shape)

        # 4. casual mask
        # 右上三角掩码，防止推理时模型看到未来的信息
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1) # [T, T]
        scores = scores.masked_fill(mask, float("-inf")) # [B, T, T]
        assert scores.shape == (B, T, T)

        # 4. softmax
        attn = F.softmax(scores, dim=-1) # [B, T, T]
        print("attn shape:", attn.shape)
        print("attn row sum:", attn.sum(dim=-1)) # should equal to 1

        # 5. output projection layer
        output = attn @ v # [B, T, H]
        print("output shape:", output.shape) # [B, T, D]
        output = self.o_proj(output) # [B, T, H]
        assert output.shape == (B, T, H)
        print("final output shape:", output.shape)

        return output


class MultiHeadSelfAttention(nn.Module):
    """
    x -> q/k/v -> split heads -> scores -> causal mask -> softmax -> output
    """
    def __init__(self, hidden_size: int, num_heads: int = 8):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        assert hidden_size % num_heads == 0

        self.head_dim = hidden_size // num_heads

        # q/k/v projection layers
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        # output projection layer
        self.o_proj = nn.Linear(self.head_dim * num_heads, hidden_size, bias=False)

    def forward(self, x: torch.Tensor):
        """
        x: [B, T, D]
        return: [B, T, D]

        B: batch_size
        T: seq_len
        D： hidden_dim
        H: num_attention_heads
        """

        # 1. input embedding vector
        B, T, H = x.shape # [B, T, H]
        assert H == self.hidden_size

        print("Input x:", x.shape)


        # 2. q,k,v projection layers
        q = self.q_proj(x) # [B, T, H]
        k = self.k_proj(x) # [B, T, H]
        v = self.v_proj(x) # [B, T, H]
        assert q.shape == (B, T, H)
        assert k.shape == (B, T, H)
        assert v.shape == (B, T, H)
        print("q/k/v shape:", q.shape, k.shape, v.shape)

        # multi-head
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2) # [B, num_heads, T, head_dim]
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2) # [B, num_heads, T, head_dim]
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2) # [B, num_heads, T, head_dim]

        # 3. attention
        # falsh-attention and manual implementation
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B, num_heads, T, T]
        print("scores shape:", scores.shape)

        # 4. casual mask
        # 右上三角掩码，防止推理时模型看到未来的信息
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1) # [T, T]
        scores = scores.masked_fill(mask, float("-inf")) # [B, num_heads, T, T]
        assert scores.shape == (B, self.num_heads, T, T)

        # 4. softmax
        attn = F.softmax(scores, dim=-1) # [B, num_heads, T, T]
        print("attn shape:", attn.shape)
        print("attn row sum:", attn.sum(dim=-1)) # should equal to 1

        # 5. output projection layer
        output = attn @ v # [B, num_heads, T, head_dim]
        output = output.transpose(1, 2).contiguous().view(B, T, self.hidden_size)
        print("output shape:", output.shape) # [B, T, D]
        output = self.o_proj(output) # [B, T, H]
        assert output.shape == (B, T, H)
        print("final output shape:", output.shape)

        return output


class GroupedQueryAttention(nn.Module):
    """
    GQA + RoPE
    x -> q/k/v -> RoPE on q/k -> repeat kv -> scores -> causal mask -> softmax -> output
    """
    def __init__(self, hidden_size: int, num_heads: int = 8, num_key_value_heads: int = None):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads if num_key_value_heads is not None else num_heads

        assert hidden_size % num_heads == 0
        assert num_heads % self.num_key_value_heads == 0

        self.head_dim = hidden_size // num_heads
        self.n_rep = self.num_heads // self.num_key_value_heads

        # q/k/v projection layers
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False) # [B, T, num_heads * head_dim]
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=False) # [B, T, num_key_value_heads * head_dim]
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=False) # [B, T, num_key_value_heads * head_dim]
        # output projection layer
        self.o_proj = nn.Linear(self.head_dim * num_heads, hidden_size, bias=False)

    def forward(self, x: torch.Tensor):
        """
        x: [B, T, D]
        return: [B, T, D]

        B: batch_size
        T: seq_len
        D： hidden_dim
        H: num_attention_heads
        """

        # 1. input embedding vector
        B, T, H = x.shape # [B, T, H]
        assert H == self.hidden_size

        print("Input x:", x.shape)


        # 2. q,k,v projection layers
        q = self.q_proj(x) # [B, T, H]
        k = self.k_proj(x) # [B, T, H]
        v = self.v_proj(x) # [B, T, H]
        print("q/k/v shape:", q.shape, k.shape, v.shape)
        assert q.shape == (B, T, self.num_heads * self.head_dim)
        assert k.shape == (B, T, self.num_key_value_heads * self.head_dim)
        assert v.shape == (B, T, self.num_key_value_heads * self.head_dim)

        # GQA
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2) # [B, num_heads, T, head_dim]
        k = k.view(B, T, self.num_key_value_heads, self.head_dim).transpose(1, 2) # [B, num_key_value_heads, T, head_dim]
        v = v.view(B, T, self.num_key_value_heads, self.head_dim).transpose(1, 2) # [B, num_key_value_heads, T, head_dim]
        assert q.shape == (B, self.num_heads, T, self.head_dim)
        assert k.shape == (B, self.num_key_value_heads, T, self.head_dim)
        assert v.shape == (B, self.num_key_value_heads, T, self.head_dim)

        # TODO: RoPE position embedding
        # q/k: [B, num_heads, T, head_dim]

        #* repeat kv
        def repeat_kv(x, n_rep: int): 
            """
            x: [B, Nkv, T, head_dim]
            return: [B, Nkv * n_rep, T, head_dim]
            """
            B, Nkv, T, head_dim = x.shape
            if n_rep == 1: 
                return x
            x = x[:, :, :, None, :].expand(B, Nkv, T, n_rep, head_dim) # [B, Nkv, T, n_rep, head_dim]
            return x.reshape(B, Nkv * n_rep, T, head_dim) # [B, Nkv * n_rep, T, head_dim]
    
        k = repeat_kv(k, self.n_rep) # [B, T, num_heads, head_dim]
        v = repeat_kv(v, self.n_rep) # [B, T, num_heads, head_dim]
        assert k.shape == (B, self.num_heads, T, self.head_dim)
        assert v.shape == (B, self.num_heads, T, self.head_dim)
        print("after repeat k/v shape:", k.shape, v.shape)

        # 3. attention
        # falsh-attention and manual implementation
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B, num_heads, T, T]
        assert scores.shape == (B, self.num_heads, T, T) # [B, Nh, T, T]
        print("scores shape:", scores.shape)

        # 4. casual mask
        # 右上三角掩码，防止推理时模型看到未来的信息
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1) # [T, T]
        scores = scores.masked_fill(mask, float("-inf")) # [B, num_heads, T, T]

        # 4. softmax
        attn = F.softmax(scores, dim=-1) # [B, num_heads, T, T]
        assert attn.shape == (B, self.num_heads, T, T)
        print("attn shape:", attn.shape)
        print("attn row sum:", attn.sum(dim=-1)) # should equal to 1

        # 5. output projection layer
        # [B, Nh, T, Dh] -> [B, T, Nh, Dh] -> [B, T, D]
        output = attn @ v # [B, Nh, T, head_dim]
        output = output.transpose(1, 2).contiguous().view(B, T, self.hidden_size)
        print("output shape:", output.shape) # [B, T, D]
        output = self.o_proj(output) # [B, T, H]
        assert output.shape == (B, T, H)
        print("final output shape:", output.shape)

        return output


if __name__ == "__main__":
    torch.manual_seed(42)
    batch_size = 2
    seq_len = 4
    hidden_size = 8
    num_heads = 2
    x = torch.randn(batch_size, seq_len, hidden_size)

    # # test single head attention
    # single_attn = SingleHeadSelfAttention(hidden_size)
    # single_output = single_attn(x)
    # assert single_output.shape == (batch_size, seq_len, hidden_size)

    # # test multi head attention
    # multi_attn = MultiHeadSelfAttention(hidden_size, num_heads)
    # multi_output = multi_attn(x)
    # assert multi_output.shape == (batch_size, seq_len, hidden_size)

    # test GQA
    num_key_value_heads = 1
    gqa_attn = GroupedQueryAttention(hidden_size, num_heads, num_key_value_heads)
    gqa_output = gqa_attn(x)
    assert gqa_output.shape == (batch_size, seq_len, hidden_size)

    # print("x.shape = ", x.shape)
    # print("out.shape = ", output.shape) # [B, T, H]
