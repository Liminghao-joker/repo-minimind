"""以伪代码形式呈现 DPO 训练流程"""

# 1. 加载 sft 后的模型作为 ref model
policy_model = MiniMindLM.from_pretrained(sft_checkpoint)
ref_model = MiniMindLM.from_pretrained(sft_checkpoint)
ref_model.eval() # 冻结 ref model 参数

# 2. 对每个 batch 偏好对
for batch in dataloader:
    prompt, chosen, rejected = batch

    policy_chosen_logps = get_log_probs(policy_model, prompt, chosen)
    policy_rejected_logps = get_log_probs(policy_model, prompt, rejected)
    ref_chosen_logps = get_log_probs(ref_model, prompt, chosen)
    ref_rejected_logps = get_log_probs(ref_model, prompt, rejected)

    # 计算 DPO loss
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)
    loss = -F.logisigmoid(chosen_rewards - rejected_rewards).mean()

    # 反向传播并更新参数
    loss.backward()
    optimizer.step()