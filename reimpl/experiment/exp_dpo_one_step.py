"""DPO 单步实验：构造 1 条 chosen/rejected 样本，跑一次前向+反向，观察分数变化

目标：
  1. 逐步打印 shape 和 score，看清数据从输入到 loss 的完整路径
  2. 做 1 次 optimizer.step 后，验证 chosen_score 上升、rejected_score 下降
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── 极小模拟模型 ──
class TinyLM(nn.Module):
    """vocab=16, hidden=8 的单层 LM，仅用于观察 DPO 数据流"""

    def __init__(self, vocab_size=16, hidden_size=8):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        # x: (batch, seq_len) → logits: (batch, seq_len, vocab_size)
        return self.head(self.embed(x))


def logits_to_log_probs(logits, labels):
    """复制自 trainer/train_dpo.py:24"""
    log_probs = F.log_softmax(logits, dim=2)
    return torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(-1)


def compute_dpo_loss(ref_log_probs, policy_log_probs, mask, beta):
    """拆分版 dpo_loss，逐步返回中间量"""
    seq_lengths = mask.sum(dim=1, keepdim=True).clamp_min(1e-8)
    ref_lp = (ref_log_probs * mask).sum(dim=1) / seq_lengths.squeeze()
    policy_lp = (policy_log_probs * mask).sum(dim=1) / seq_lengths.squeeze()

    # 前 1 条是 chosen，后 1 条是 rejected
    chosen_ref = ref_lp[0]
    rejected_ref = ref_lp[1]
    chosen_policy = policy_lp[0]
    rejected_policy = policy_lp[1]

    pi_logratios = chosen_policy - rejected_policy
    ref_logratios = chosen_ref - rejected_ref
    logits = pi_logratios - ref_logratios
    loss = -F.logsigmoid(beta * logits)
    return loss.mean(), {
        "chosen_ref": chosen_ref,
        "rejected_ref": rejected_ref,
        "chosen_policy": chosen_policy,
        "rejected_policy": rejected_policy,
        "pi_logratios": pi_logratios,
        "ref_logratios": ref_logratios,
        "logits": logits,
    }


def print_step(title, x, y, mask, ref_log_probs, policy_log_probs, details, loss):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(f"  x (输入):               {tuple(x.shape)}")
    print(f"  y (目标):               {tuple(y.shape)}")
    print(f"  mask:                   {tuple(mask.shape)}")
    print(f"  ref_log_probs (token):  {tuple(ref_log_probs.shape)}")
    print(f"  policy_log_probs (tok): {tuple(policy_log_probs.shape)}")
    print(f"  ── 归一化后的标量 score ──")
    print(f"  chosen_ref_score:       {details['chosen_ref']:.6f}")
    print(f"  rejected_ref_score:     {details['rejected_ref']:.6f}")
    print(f"  chosen_policy_score:    {details['chosen_policy']:.6f}")
    print(f"  rejected_policy_score:  {details['rejected_policy']:.6f}")
    print(f"  ── DPO 中间量 ──")
    print(f"  pi_logratios (C-R):     {details['pi_logratios']:.6f}")
    print(f"  ref_logratios (C-R):    {details['ref_logratios']:.6f}")
    print(f"  logits (pi - ref):      {details['logits']:.6f}")
    print(f"  loss:                   {loss.item():.6f}")


def main():
    torch.manual_seed(42)
    vocab_size = 16
    seq_len = 10
    beta = 0.5

    # ── 1. 构造 1 条 chosen + 1 条 rejected ──
    # chosen: token id 偏向 [4,5,6,7]
    # rejected: token id 偏向 [0,1,2,3]
    x_chosen = torch.tensor([[3, 5, 6, 7, 8, 5, 4, 7, 6, 5]])
    x_rejected = torch.tensor([[3, 1, 0, 2, 8, 1, 0, 2, 1, 0]])
    # y 左移一位
    y_chosen = torch.tensor([[5, 6, 7, 8, 5, 4, 7, 6, 5, 9]])
    y_rejected = torch.tensor([[1, 0, 2, 8, 1, 0, 2, 1, 0, 9]])
    # mask: 前 3 个 token 是 prompt(0), 后 7 个是 assistant 回复(1)
    mask_chosen = torch.tensor([[0, 0, 0, 1, 1, 1, 1, 1, 1, 1]], dtype=torch.float)
    mask_rejected = torch.tensor([[0, 0, 0, 1, 1, 1, 1, 1, 1, 1]], dtype=torch.float)

    # 拼成一个大 batch
    x = torch.cat([x_chosen, x_rejected], dim=0)          # (2, 10)
    y = torch.cat([y_chosen, y_rejected], dim=0)           # (2, 10)
    mask = torch.cat([mask_chosen, mask_rejected], dim=0)  # (2, 10)

    print(f"输入构造完成")
    print(f"  x_chosen:  {x_chosen.tolist()}")
    print(f"  x_reject:  {x_rejected.tolist()}")
    print(f"  mask:      prompt=0, assistant=1 → {mask[0].tolist()}")

    # ── 2. 初始化 policy 和 ref（同一权重） ──
    policy_model = TinyLM(vocab_size=vocab_size)
    ref_model = TinyLM(vocab_size=vocab_size)
    ref_model.load_state_dict(policy_model.state_dict())  # 完全复制
    ref_model.eval()
    ref_model.requires_grad_(False)

    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=1e-2)

    # ── 3. Step 0: 初始前向（policy = ref）──
    with torch.no_grad():
        ref_logits = ref_model(x)
    ref_log_probs = logits_to_log_probs(ref_logits, y)

    policy_logits = policy_model(x)
    policy_log_probs = logits_to_log_probs(policy_logits, y)

    loss, details = compute_dpo_loss(ref_log_probs, policy_log_probs, mask, beta)
    print_step("Step 0: 初始前向 (policy = ref)", x, y, mask,
               ref_log_probs, policy_log_probs, details, loss)

    # ── 4. 反向传播 + 更新 ──
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(f"\n{'─'*60}")
    print(f"  backward + optimizer.step() 完成")
    print(f"{'─'*60}")

    # ── 5. Step 1: 更新后再前向 ──
    with torch.no_grad():
        ref_logits = ref_model(x)
    ref_log_probs_new = logits_to_log_probs(ref_logits, y)

    policy_logits_new = policy_model(x)
    policy_log_probs_new = logits_to_log_probs(policy_logits_new, y)

    loss_new, details_new = compute_dpo_loss(
        ref_log_probs_new, policy_log_probs_new, mask, beta
    )
    print_step("Step 1: 更新后前向", x, y, mask,
               ref_log_probs_new, policy_log_probs_new, details_new, loss_new)

    # ── 6. 对比变化 ──
    print(f"\n{'='*60}")
    print(f"  变化对比")
    print(f"{'='*60}")
    dc = details_new["chosen_policy"] - details["chosen_policy"]
    dr = details_new["rejected_policy"] - details["rejected_policy"]
    dl = loss_new.item() - loss.item()
    print(f"  chosen_policy_score:  {details['chosen_policy']:.6f} → {details_new['chosen_policy']:.6f}  (Δ={dc:+.6f})")
    print(f"  rejected_policy_score:{details['rejected_policy']:.6f} → {details_new['rejected_policy']:.6f}  (Δ={dr:+.6f})")
    print(f"  loss:                 {loss.item():.6f} → {loss_new.item():.6f}  (Δ={dl:+.6f})")

    # 验证
    chosen_up = dc.item() > 0
    rejected_down = dr.item() < 0
    loss_down = dl < 0
    print(f"\n  ✓ chosen_score 上升:   {'是' if chosen_up else '否'}")
    print(f"  ✓ rejected_score 下降: {'是' if rejected_down else '否'}")
    print(f"  ✓ loss 下降:          {'是' if loss_down else '否'}")

    if chosen_up and rejected_down and loss_down:
        print(f"\n  验证通过: DPO 一步更新后，policy 确实更偏好 chosen、更排斥 rejected")
    else:
        print(f"\n  注: 单步更新可能不足以同时满足三个条件（lr 不够大 / beta 不够大）")
        print(f"  但方向应该是正确的，可增大 lr 或 beta 再试")


if __name__ == "__main__":
    main()
