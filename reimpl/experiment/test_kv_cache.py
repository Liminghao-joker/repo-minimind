import torch
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

def print_past_kv_shapes(past_key_values, tag=""):
    print(f"\n===== {tag} =====")
    for layer_idx, (k, v) in enumerate(past_key_values):
        print(f"layer {layer_idx}:")
        print(f"  K shape = {tuple(k.shape)}")
        print(f"  V shape = {tuple(v.shape)}")

def main():
    torch.manual_seed(42)

    # 1. 构造一个小模型，便于快速验证
    config = MiniMindConfig(
        vocab_size=100,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        dropout=0.0,
        flash_attn=False,   # 先关掉，便于走清晰的普通 attention 路径
    )
    model = MiniMindForCausalLM(config)
    model.eval()

    # 2. 构造一个 prompt，长度为 5
    input_ids = torch.tensor([[1, 5, 7, 9, 3]])   # [B=1, T=5]
    attention_mask = torch.ones_like(input_ids)

    # =========================
    # Step 1: prefill
    # =========================
    with torch.no_grad():
        outputs_prefill = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
        )

    past_key_values = outputs_prefill.past_key_values

    print("prefill input_ids shape:", tuple(input_ids.shape)) # [1, 5]
    print_past_kv_shapes(past_key_values, tag="After Prefill") # [1. 5, num_heads, head_dim]

    # 验证 cache 长度 == prompt 长度
    expected_len = input_ids.shape[1]
    for layer_idx, (k, v) in enumerate(past_key_values):
        assert k.shape[1] == expected_len, f"Layer {layer_idx} K cache len mismatch"
        assert v.shape[1] == expected_len, f"Layer {layer_idx} V cache len mismatch"

    print("\n[OK] prefill 后每层 cache 长度都等于 prompt 长度")

    # =========================
    # Step 2: decode 一步
    # 只输入一个新 token
    # =========================
    next_input_ids = torch.tensor([[11]])   # [B=1, T=1]
    next_attention_mask = torch.ones_like(next_input_ids)

    with torch.no_grad():
        outputs_decode = model(
            input_ids=next_input_ids,
            attention_mask=next_attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )

    new_past_key_values = outputs_decode.past_key_values

    print("\ndecode input_ids shape:", tuple(next_input_ids.shape))
    print_past_kv_shapes(new_past_key_values, tag="After One Decode Step")

    # 验证 cache 长度 +1
    for layer_idx, ((old_k, old_v), (new_k, new_v)) in enumerate(zip(past_key_values, new_past_key_values)):
        assert new_k.shape[1] == old_k.shape[1] + 1, f"Layer {layer_idx} K cache did not grow by 1"
        assert new_v.shape[1] == old_v.shape[1] + 1, f"Layer {layer_idx} V cache did not grow by 1"

    print("\n[OK] decode 一步后，每层 cache 长度都增长了 1")

if __name__ == "__main__":
    main()