import time
import torch
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

def benchmark_generate(model, input_ids, attention_mask, use_cache, max_new_tokens=32, device="cpu"):
    """使用 model.generate 来生成文本，并测量时间"""
    model.eval()
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    # warmup
    with torch.no_grad():
        _ = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=4,
            use_cache=use_cache,
        )

    if device.startswith("cuda"):
        torch.cuda.synchronize()
    start = time.time()

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            use_cache=use_cache,
        )

    if device.startswith("cuda"):
        torch.cuda.synchronize()
    end = time.time()

    return end - start, outputs

def main():
    """对比使用 cache / 不使用 cache，从 32 个 prompt token 生成 64 个新 token 的时间差异"""
    torch.manual_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = MiniMindConfig(
        vocab_size=100,
        hidden_size=256,
        num_hidden_layers=6,
        num_attention_heads=8,
        num_key_value_heads=2,
        max_position_embeddings=256,
        dropout=0.0,
        flash_attn=False,
    )
    model = MiniMindForCausalLM(config).to(device)

    # prompt 长一点，才能更明显体现 cache 的优势
    input_ids = torch.randint(low=0, high=100, size=(1, 128)) # [1, 128]
    attention_mask = torch.ones_like(input_ids)

    t_no_cache, out_no_cache = benchmark_generate(
        model, input_ids, attention_mask,
        use_cache=False,
        max_new_tokens=128,
        device=device
    )

    t_cache, out_cache = benchmark_generate(
        model, input_ids, attention_mask,
        use_cache=True,
        max_new_tokens=128,
        device=device
    )

    print(f"use_cache=False: {t_no_cache:.4f}s, output shape={tuple(out_no_cache.shape)}")
    print(f"use_cache=True : {t_cache:.4f}s, output shape={tuple(out_cache.shape)}")

    speedup = t_no_cache / t_cache if t_cache > 0 else float("inf")
    print(f"speedup = {speedup:.2f}x")

if __name__ == "__main__":
    main()