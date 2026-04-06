"""Tutorial-style sanity test for reimpl/model/forward_demo.py.

This script is intentionally small and readable. It shows:
1. what shape goes into the model,
2. what shape comes out of the first forward pass,
3. how KV cache grows after one decode step,
4. whether cached decoding matches a full forward pass.

Run from repo root:
    python -m reimpl.experiment.test_forward_demo_tutorial
"""

import torch

from reimpl.model.forward_demo import MiniAttentionDemo


def main():
    torch.manual_seed(42)

    model = MiniAttentionDemo(
        vocab_size=100,
        hidden_size=64,
        num_heads=4,
        num_kv_heads=2,
        max_seq_len=128,
    )
    model.eval()

    # 1) Prefill: give the model a short prompt.
    input_ids = torch.tensor([[1, 5, 7, 9]], dtype=torch.long)
    print("=== Prefill ===")
    print("input_ids shape:", tuple(input_ids.shape))

    with torch.no_grad():
        prefill_out = model(input_ids, use_cache=True)

    logits = prefill_out["logits"]
    past_key_values = prefill_out["past_key_values"]

    print("logits shape:", tuple(logits.shape))
    print("hidden_states shape:", tuple(prefill_out["hidden_states"].shape))
    print("cache k shape:", tuple(past_key_values[0].shape))
    print("cache v shape:", tuple(past_key_values[1].shape))

    # 2) Decode: feed one new token with the cache.
    next_ids = torch.tensor([[11]], dtype=torch.long)
    print("\n=== Decode with cache ===")
    print("next_ids shape:", tuple(next_ids.shape))

    with torch.no_grad():
        decode_out = model(next_ids, past_key_values=past_key_values, use_cache=True)

    print("decode logits shape:", tuple(decode_out["logits"].shape))
    print("new cache k shape:", tuple(decode_out["past_key_values"][0].shape))
    print("new cache v shape:", tuple(decode_out["past_key_values"][1].shape))

    # 3) Compare cached decoding with a full forward pass.
    #    For a correct causal cache implementation, the last-token logits should match.
    full_input = torch.tensor([[1, 5, 7, 9, 11]], dtype=torch.long)
    with torch.no_grad():
        full_out = model(full_input, use_cache=False)

    cached_last_logits = decode_out["logits"][:, -1, :]
    full_last_logits = full_out["logits"][:, -1, :]
    max_abs_diff = (cached_last_logits - full_last_logits).abs().max().item()

    print("\n=== Cache consistency check ===")
    print("full last logits shape:", tuple(full_last_logits.shape))
    print("cached last logits shape:", tuple(cached_last_logits.shape))
    print("max abs diff:", max_abs_diff)
    print("match:", torch.allclose(cached_last_logits, full_last_logits, atol=1e-6))

    # 4) Minimal teaching note.
    print("\n=== What to observe ===")
    print("1. prefill cache length equals prompt length")
    print("2. decode cache length grows by 1")
    print("3. cached decode and full forward should be numerically very close")


if __name__ == "__main__":
    main()