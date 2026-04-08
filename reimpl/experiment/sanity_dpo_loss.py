"""DPO loss 直觉验证：用构造的 tensor 验证三个关键场景

验证目标：
1. 初始状态 (policy = ref) → loss ≈ log(2) ≈ 0.693
2. 已学好 (policy 偏好 chosen) → loss → 0
3. 学反了 (policy 偏好 rejected) → loss >> 0.693
"""

import math

import torch
import torch.nn.functional as F


def dpo_loss(ref_log_probs, policy_log_probs, mask, beta):
    """复制自 trainer/train_dpo.py:33"""
    seq_lengths = mask.sum(dim=1, keepdim=True).clamp_min(1e-8) # (batch_size, 1)
    ref_log_probs = (ref_log_probs * mask).sum(dim=1) / seq_lengths.squeeze() # (batch_size * 2,)
    policy_log_probs = (policy_log_probs * mask).sum(dim=1) / seq_lengths.squeeze() # (batch_size * 2,)

    batch_size = ref_log_probs.shape[0]
    chosen_ref_log_probs = ref_log_probs[:batch_size // 2]
    reject_ref_log_probs = ref_log_probs[batch_size // 2:]
    chosen_policy_log_probs = policy_log_probs[:batch_size // 2]
    reject_policy_log_probs = policy_log_probs[batch_size // 2:]

    pi_logratios = chosen_policy_log_probs - reject_policy_log_probs
    ref_logratios = chosen_ref_log_probs - reject_ref_log_probs
    logits = pi_logratios - ref_logratios
    loss = -F.logsigmoid(beta * logits)
    return loss.mean()


def make_batch(chosen_val, rejected_val, batch_size=4, seq_len=8):
    """构造一个 batch 的 log_probs 和 mask

    Args:
        chosen_val: chosen 位置每个 token 的 log_prob 标量值
        rejected_val: rejected 位置每个 token 的 log_prob 标量值
        batch_size: chosen + rejected 各 batch_size/2
        seq_len: 序列长度

    Returns:
        log_probs: (batch_size, seq_len) 全部填充为对应值
        mask: (batch_size, seq_len) 只在后半段（模拟 assistant 回复区）为 1
    """
    half = batch_size // 2
    # log_probs: 前 half 行填 chosen_val, 后 half 行填 rejected_val
    log_probs = torch.cat([
        torch.full((half, seq_len), chosen_val),
        torch.full((half, seq_len), rejected_val),
    ])
    # mask: 假设前 3 个 token 是 prompt(0), 后 5 个是 assistant 回复(1)
    mask = torch.cat([
        torch.zeros(batch_size, 3),
        torch.ones(batch_size, 5),
    ], dim=1)
    return log_probs, mask


def main():
    beta = 0.1
    batch_size = 4
    seq_len = 8
    log2 = math.log(2)

    print("=" * 60)
    print(f"DPO Loss 直觉验证 (beta={beta}, batch_size={batch_size})")
    print("=" * 60)

    # ── 场景 1: 初始状态，policy = ref ──
    # chosen_val=2.0, rejected_val=1.0 对 policy 和 ref 都相同
    # → pi_logratios = ref_logratios → logits=0 → loss=-log(sigmoid(0))=log(2)
    ref_lp, mask = make_batch(chosen_val=-1.0, rejected_val=-2.0, batch_size=batch_size, seq_len=seq_len)
    policy_lp, _ = make_batch(chosen_val=-1.0, rejected_val=-2.0, batch_size=batch_size, seq_len=seq_len)
    loss = dpo_loss(ref_lp, policy_lp, mask, beta=beta)
    print(f"\n[场景 1] 初始状态 (policy = ref)")
    print(f"  loss = {loss.item():.6f}")
    print(f"  log(2) = {log2:.6f}")
    print(f"  差异 = {abs(loss.item() - log2):.8f}")
    assert abs(loss.item() - log2) < 1e-5, "初始状态 loss 应接近 log(2)"
    print("  ✓ 验证通过: loss ≈ log(2)")

    # ── 场景 2: 已学好，policy 更偏好 chosen ──
    # ref 不变，policy 的 chosen 提高、rejected 降低
    ref_lp, mask = make_batch(chosen_val=-1.0, rejected_val=-2.0, batch_size=batch_size, seq_len=seq_len)
    policy_lp, _ = make_batch(chosen_val=-0.3, rejected_val=-3.0, batch_size=batch_size, seq_len=seq_len)
    loss = dpo_loss(ref_lp, policy_lp, mask, beta=beta)
    print(f"\n[场景 2] 已学好 (policy 偏好 chosen)")
    print(f"  loss = {loss.item():.6f}")
    assert loss.item() < log2, "学好后 loss 应小于 log(2)"
    print(f"  ✓ 验证通过: loss({loss.item():.4f}) < log(2)({log2:.4f})")

    # ── 场景 2 补充: beta 越大，学好的 loss 下降越快 ──
    loss_high_beta = dpo_loss(ref_lp, policy_lp, mask, beta=1.0)
    print(f"  同数据 beta=1.0 → loss = {loss_high_beta.item():.6f} (更激进)")

    # ── 场景 3: 学反了，policy 偏好 rejected ──
    # ref 不变，policy 的 chosen 降低、rejected 提高
    ref_lp, mask = make_batch(chosen_val=-1.0, rejected_val=-2.0, batch_size=batch_size, seq_len=seq_len)
    policy_lp, _ = make_batch(chosen_val=-3.0, rejected_val=-0.3, batch_size=batch_size, seq_len=seq_len)
    loss = dpo_loss(ref_lp, policy_lp, mask, beta=beta)
    print(f"\n[场景 3] 学反了 (policy 偏好 rejected)")
    print(f"  loss = {loss.item():.6f}")
    assert loss.item() > log2, "学反时 loss 应大于 log(2)"
    print(f"  ✓ 验证通过: loss({loss.item():.4f}) > log(2)({log2:.4f})")

    print("\n" + "=" * 60)
    print("全部验证通过")
    print("=" * 60)
    print("\n总结:")
    print("  初始状态 loss ≈ log(2) ≈ 0.693  → 训练起点")
    print("  学好后 loss → 0                  → 训练目标")
    print("  学反了 loss >> 0.693             → 需要继续纠正")
    print("  beta 控制偏好惩罚强度            → beta 越大训练信号越强")


if __name__ == "__main__":
    main()
