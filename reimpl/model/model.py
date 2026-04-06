from transformers import PretrainedConfig


class MokioMindConfig(PretrainedConfig):
    model_type = "mokiomind"

    def __init__(
        self,
        dropout: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_act: str = "silu",
        hidden_size: int = 512,
        intermediate_size: int = None,
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 8,
        num_key_value_heads: int = 2,
        vocab_size: int = 6400,
        rms_norm_eps: float = 1e-05,
        rope_theta: int = 1000000,
        inference_rope_scaling: bool = False,
        flash_attention: bool = True,
        ############ MoE ############
        use_moe: bool = False,
        num_experts_per_tok: int = 2,
        n_routed_experts: int = 4,
        n_shared_experts: int = 1,
        scoring_func: str = "softmax",
        aux_loss_alpha: float = 0.01,
        seq_aux: bool = True,
        norm_topk_prob: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        self.flash_attention = flash_attention
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob
        self.aux_loss_alpha = aux_loss_alpha
        self.scoring_func = scoring_func

        self.rope_scaling = (
            {
                "beta_fast": 32,
                "beta_slow": 1,
                "factor": 16,
                "original_max_position_embeddings": 2048,
                "attention_factor": 1.0,
                "type": "yarn",
            }
            if self.inference_rope_scaling
            else None
        )

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

#* RMSNorm
# 继承 nn.Module 类
class RMSNorm(nn.Module):
    # init
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    # _norm
    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    # forward
    def forward(self, x):
        return self._norm(x.float()) * self.weight.type_as(x)
    
#* RoPE

#* GQA layer    
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat the key and value tensors for multi-query attention."""
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    # 增添一个维度，并在该维度上复制，最后还原形状
    return x[:, :, :, None, :].expand(bs, slen, num_key_value_heads, n_rep, head_dim).reshape(bs, slen, num_key_value_heads * n_rep, head_dim)

class Attention(nn.Module):
    def __init__(self, args: MokioMindConfig):
        super().__init__()
        # Group heads
        self.num_key_value_heads = args.num_attention_heads if args.num_key_value_heads is None else args.num_key_value_heads
        self.head_dim = args.hidden_size // args.num_attention_heads
        self.n_local_heads = args.num_attention_heads
        self.n_local_kv_heads = args.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        assert args.num_attention_heads % args.num_key_value_heads == 0
        
        #* projection layers
        self.q_proj = nn.Linear(args.hidden_size, args.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(args.hidden_size, args.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(args.hidden_size, args.num_key_value_heads * self.head_dim, bias=False)
        # return to the original size
        self.o_proj = nn.Linear(args.num_attention_heads * self.head_dim, args.hidden_size, bias=False)

        #* dropout layer
        self.attn_dropout = nn.Dropout(args.dropout) # 注意力权重 dropout
        self.resid_dropout = nn.Dropout(args.dropout) # 残差连接 dropout
        self.dropout = args.dropout # 保存 dropout 率

        #? Flash attention
        self.flash = (
            hasattr(torch.nn.functional, "scaled_dot_product_attention") 
            and args.flash_attention
        )

    def forward(self,
                x: torch.Tensor,
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, # kv-cache history
                ):
        # q,k,v
        bsz, seq_len, _ = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x) # [bsz, seq_len, num_attention_heads * head_dim]

        # 把输入拆成多个头
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        # TODO：RoPE Implementation
        

        # TODO: GQA
        #? transpose 到形状 [bsz, n_heads, seq_len, head_dim] 以便矩阵乘法
        xq = xq.transpose(1, 2)
        #
        # n_local_heads = n_local_kv_heads * n_rep
        xk = repeat_kv(xk, self.n_rep).transpose(1, 2)
        xv = repeat_kv(xv, self.n_rep).transpose(1, 2)
    
        # attention mechanism
        # 优先使用 Pytorch 2.0+ 的 scaled_dot_product_attention，如果不可用则回退手动实现
        #? flash attention 
        if self.flash and seq_len > 1 and (past_key_value is None or torch.all(attention_mask == 1)):
            # TODO: is_causal=True 本质上就是 decoder causal mask
            output = F.scaled_dot_product_attention(xq, xk, xv, attn_mask=None, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        # 手动实现
        else:
            scores = xq @ xk.transpose(-2, -1) / math.sqrt(self.head_dim) #! 没有传入 head_dim
            assert scores.shape == (bsz, self.n_local_heads, seq_len, seq_len)
            # mask
            scores[:, :, :, -seq_len:] += torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=scores.device), diagonal=1)

            if attention_mask is not None:
                extended_attention_mask += attention_mask.unsqueeze(1).unsqueeze(2) # [bsz, 1, 1, seq_len]
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
                scores += extended_attention_mask
        
            # softmax
            scores = F.softmax(scores, dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = scores @ xv 

        # dropout + output
        # input: [bsz, n_heads, seq_len, head_dim]
        # 拼接头，输出投影后返回
        output = self.resid_dropout(self.o_proj(output.transpose(1, 2).reshape(bsz, seq_len, -1)))
        assert output.shape == (bsz, seq_len, self.hidden_size)
        return output, past_kv

                                                


class FeedForward():
    pass

