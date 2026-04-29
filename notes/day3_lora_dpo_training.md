# Day 3 - LoRA and DPO Training

## 1. 今日目标

基于昨日得到的 `full_sft_official_pretrain_512.pth`，完成两个后训练分支：

- LoRA identity 微调：尝试用轻量 adapter 注入身份类回答。
- DPO 偏好优化：基于 chosen/rejected 数据，让模型相对 SFT reference 更偏向 chosen。

今天同时补充了 LoRA 和 DPO 的评测脚本，产出了 LoRA 生成对比报告和 DPO preference 离线评测报告。

## 2. Base Model

| item | value |
|---|---|
| base weight | `out/full_sft_official_pretrain_512.pth` |
| hidden_size | 512 |
| num_hidden_layers | 8 |
| params | 25.83M |
| source | official pretrain + mini SFT |
| role today | LoRA 和 DPO 都以它作为起点；DPO 里也作为 reference model |

## 3. LoRA Run

### Command

```bash
cd /workspace/liminghao/Projects/repo-minimind/trainer

python train_lora.py \
  --data_path ../dataset/lora_identity.jsonl \
  --from_weight full_sft_official_pretrain \
  --lora_name lora_identity_official_sft \
  --save_dir ../out/lora \
  --hidden_size 512 \
  --num_hidden_layers 8 \
  --use_wandb \
  --wandb_project MiniMind-Pretrain \
  2>&1 | tee ../logs/day3_lora_identity_official_sft.log
```

注意：`train_lora.py` 没有 `--save_weight` 参数，LoRA 权重名应使用 `--lora_name`。

### Config

| item | value |
|---|---|
| data_path | `dataset/lora_identity.jsonl` |
| samples | 91 |
| from_weight | `full_sft_official_pretrain` |
| lora_name | `lora_identity_official_sft` |
| save_dir | `out/lora/` |
| epochs | 50 |
| batch_size | 32 |
| max_seq_len | 340 |
| lr | 1e-4 |
| dtype | bfloat16 |
| trainable params | LoRA params 0.131M, about 0.50% |
| peak GPU memory | 日志未记录 |
| W&B run | `MiniMind-LoRA-lora_identity_official_sft-Epoch-50-BatchSize-32-LR-0.0001` |

### Logs

日志文件：`logs/day3_lora_identity_official_sft.log`

| step | loss | lr | gpu_mem | note |
|---|---:|---:|---|---|
| epoch1 | 2.7842 | 0.00009991 | 未记录 | 初始 identity loss |
| epoch10 | 2.4582 | 0.00009141 | 未记录 | 仍有波动 |
| epoch20 | 1.9541 | 0.00006891 | 未记录 | 明显低于初始 |
| epoch30 | 1.7731 | 0.00004109 | 未记录 | 继续下降 |
| epoch40 | 1.7421 | 0.00001859 | 未记录 | 进入低学习率阶段 |
| epoch50 | 1.5444 | 0.00001000 | 未记录 | 最终记录点 |

### Output

- LoRA weight: `out/lora/lora_identity_official_sft_512.pth`
- 文件大小约 536KB，符合轻量 adapter 的预期。

## 4. DPO Run

今天实际跑了两次 DPO：

- `dpo_official_sft`: 默认学习率 `4e-8`，loss 基本在 0.693 附近，训练信号很弱。
- `dpo_official_sft_lr1e6`: 调高到 `1e-6` 后重新训练，loss 有更多低于 0.693 的点，离线 preference 评测也有正向提升。

### Command

第一次，默认 `4e-8`：

```bash
cd /workspace/liminghao/Projects/repo-minimind/trainer

python train_dpo.py \
  --data_path ../dataset/dpo.jsonl \
  --from_weight full_sft_official_pretrain \
  --save_weight dpo_official_sft \
  --save_dir ../out \
  --hidden_size 512 \
  --num_hidden_layers 8 \
  --use_wandb \
  --wandb_project MiniMind-Pretrain \
  2>&1 | tee ../logs/day3_dpo_official_sft.log
```

第二次，`1e-6`：

```bash
cd /workspace/liminghao/Projects/repo-minimind/trainer

python train_dpo.py \
  --data_path ../dataset/dpo.jsonl \
  --from_weight full_sft_official_pretrain \
  --save_weight dpo_official_sft_lr1e6 \
  --save_dir ../out \
  --hidden_size 512 \
  --num_hidden_layers 8 \
  --learning_rate 1e-6 \
  --epochs 1 \
  --use_wandb \
  --wandb_project MiniMind-Pretrain
```

### Config

| item | value |
|---|---|
| data_path | `dataset/dpo.jsonl` |
| samples | 17,166 preference pairs |
| from_weight | `full_sft_official_pretrain` |
| reference weight | `full_sft_official_pretrain` |
| save_weight | `dpo_official_sft`, `dpo_official_sft_lr1e6` |
| beta | 0.1 |
| epochs | 1 |
| batch_size | 4 |
| max_seq_len | 1024 |
| lr, first run | 4e-8 |
| lr, second run | 1e-6 |
| dtype | bfloat16 |
| peak GPU memory | 日志未记录 |
| W&B first run | `MiniMind-DPO-Epoch-1-BatchSize-4-LR-4e-08` |
| W&B second run | `MiniMind-DPO-Epoch-1-BatchSize-4-LR-1e-06` |

### Logs

第一次 `4e-8`，日志文件：`logs/day3_dpo_official_sft.log`

| step | loss | lr | gpu_mem | note |
|---|---:|---:|---|---|
| 100 | 0.6930 | 0.00000004 | 未记录 | 接近随机偏好基线 |
| 1000 | 0.6911 | 0.00000004 | 未记录 | 只有极小下降 |
| 2000 | 0.6927 | 0.00000002 | 未记录 | 基本无明显学习 |
| 3000 | 0.6932 | 0.00000001 | 未记录 | 回到 0.693 附近 |
| 4292 | 0.6927 | 0.00000000 | 未记录 | 最终仍近似 0.693 |

第二次 `1e-6`，日志来自 `trainer/wandb/run-20260429_113724-fd3r4vne/files/output.log`

| step | loss | lr | gpu_mem | note |
|---|---:|---:|---|---|
| 100 | 0.6931 | 0.00000100 | 未记录 | 起点仍接近 0.693 |
| 600 | 0.6870 | 0.00000096 | 未记录 | 开始低于 0.693 |
| 1000 | 0.6793 | 0.00000088 | 未记录 | 有正向训练信号 |
| 1800 | 0.6631 | 0.00000066 | 未记录 | 明显低于基线 |
| 2900 | 0.6447 | 0.00000031 | 未记录 | 单点最低之一 |
| 3400 | 0.7260 | 0.00000019 | 未记录 | 仍有 batch 波动 |
| 4292 | 0.6896 | 0.00000010 | 未记录 | 最终 summary 点 |

### Output

- DPO weight, default lr: `out/dpo_official_sft_512.pth`
- DPO weight, lr 1e-6: `out/dpo_official_sft_lr1e6_512.pth`

## 5. Evaluation Comparison

今天的评测不只看单条生成，而是补了两个脚本：

- `debug/eval_lora_identity.py`: 自动加载 SFT base 和 LoRA adapter，做身份类与通用 prompt 对比。
- `debug/eval_dpo_preference.py`: 按 DPO 训练口径加载 reference/policy，计算 chosen/rejected 的 logprob margin、preference accuracy 和离线 DPO loss。

### LoRA Identity Evaluation

运行命令：

```bash
python debug/eval_lora_identity.py \
  --compare_base \
  --max_new_tokens 128 \
  --output_prefix day3_eval_lora_identity_compare
```

报告：

- `logs/day3_eval_lora_identity_compare_20260429_113210.md`
- `logs/day3_eval_lora_identity_compare_20260429_113210.jsonl`

| prompt | SFT base | LoRA | observation |
|---|---|---|---|
| 你是谁？ | 自称由中国个人开发者开发的 `minimind`。 | 自称由 `Jingyao Gong` 开发的 AI 语言模型，但没有提到 MiniMind。 | LoRA 改变了身份回答，但身份字段不稳定。 |
| 你的身份是什么？ | 稳定回答 `minimind` 智能助手。 | 回答由 `jingyao Gong` 开发，身份是 `MiniMind`。 | 这是较理想的 LoRA 效果样例。 |
| 你是 ChatGPT 吗？ | 回答为 `minimind`，没有直接承认 ChatGPT。 | 回答“是的，我是 ChatGPT，我的名字是 MiniMind”。 | LoRA 引入了错误身份合并，说明不是单纯正向提升。 |
| 你是 OpenAI 开发的吗？ | 回答为个人开发者开发的 `minimind`。 | 回答“是的，我是 OpenAI 开发的”。 | LoRA 在反向身份拒绝上失败。 |
| 为什么天空是蓝色的？ | 能回答散射，但有重复和概念混乱。 | 仍能回答散射，但重复问题依旧。 | LoRA 没有明显改善通用能力。 |
| 请用 Python 写斐波那契函数 | 输出代码不可靠。 | 仍输出不可靠代码并重复。 | LoRA identity 数据不会解决代码能力。 |

LoRA 结论：adapter 确实影响了输出，尤其会把 `jingyao/Jingyao Gong` 注入到部分身份回答中；但它没有稳定地绑定 `MiniMind + creator`，并且在 ChatGPT/OpenAI 否认类问题上出现错误承认。因此只能说明 LoRA 起了作用，不能说明 identity 微调已经达标。

### DPO Preference Evaluation

运行命令：

```bash
python debug/eval_dpo_preference.py \
  --policy_weights dpo_official_sft dpo_official_sft_lr1e6 \
  --max_samples 512
```

报告：

- `logs/day3_eval_dpo_preference_20260429_144725.md`
- `logs/day3_eval_dpo_preference_20260429_144725.json`

| policy | pairs | pref_acc | avg_margin | avg_advantage_vs_ref | avg_dpo_loss_vs_ref | observation |
|---|---:|---:|---:|---:|---:|---|
| `dpo_official_sft` | 512 | 0.4746 | -0.0018 | 0.0017 | 0.6931 | 默认 `4e-8` 基本没有学到偏好方向。 |
| `dpo_official_sft_lr1e6` | 512 | 0.5234 | 0.1804 | 0.1839 | 0.6846 | 有轻微正向偏好学习，但幅度还不强。 |

指标解释：

- `pref_acc`: policy 是否给 chosen 比 rejected 更高的 assistant-token 平均 logprob。
- `avg_margin`: policy chosen logprob 减 rejected logprob，越高越偏向 chosen。
- `avg_advantage_vs_ref`: policy margin 相对 reference margin 的提升，正数说明相对 SFT base 朝 chosen 方向移动。
- `avg_dpo_loss_vs_ref`: 按训练口径计算的离线 DPO loss，接近 0.693 表示相对 reference 变化很小。

DPO 结论：`4e-8` 版本几乎无效；`1e-6` 版本有可观测但较弱的 preference 信号。当前结果还不能证明生成质量更好，只能证明在 512 条 sampled preference pair 上，policy 的 chosen/rejected 排序相对 reference 有轻微改善。

## 6. Key Takeaways

LoRA 学到的是：

- 学到了一部分身份注入信号，能够改变 SFT base 的回答分布。
- 由于数据只有 91 条且都是 identity 方向，它主要改变身份类表达，不会提升通用知识、代码能力或事实准确性。
- 当前 identity 约束不够稳，尤其 ChatGPT/OpenAI 否认类样本需要加强。

DPO 学到的是：

- 默认 `4e-8` 对这个实验太保守，loss 和离线指标都接近无训练。
- `1e-6` 后，loss 曲线和 preference eval 都出现正向信号。
- 但 `pref_acc=0.5234` 仍只是略高于 0.5，说明偏好学习很弱，不能直接等价为回答质量提升。

今日最明显的现象：

- SFT base 已经具备基本 assistant 形态，但通用回答仍会重复、事实混乱，代码能力弱。
- LoRA 可以用很小权重改变身份输出，但如果数据设计不严，会把身份、开发者、ChatGPT/OpenAI 等概念混在一起。
- DPO loss 在 0.693 附近不一定完全异常，但如果全程贴近 0.693 且离线 preference 指标无提升，就说明 policy 相对 reference 几乎没动。

今日踩到的坑：

- `train_lora.py` 参数名是 `--lora_name`，不是 `--save_weight`。
- DPO 训练时 `wandb` 曾出现 `module 'wandb' has no attribute 'init'`，本质是导入到错误的 `wandb` 模块或环境包异常，需要确认当前目录没有同名 `wandb.py`，并确认环境中安装的是官方 `wandb` 包。
- DPO 默认学习率 `4e-8` 太小，训练曲线和离线评测都显示效果接近无。
- 早版 DPO 评测脚本使用 `../model` 加载 tokenizer，在 repo root 运行时会被 transformers 当成非法 HuggingFace repo id；后来改为基于 repo root 显式解析 `model/` 和 `out/`。
- 当前 DPO 评测只是 preference logprob 评测，还没有做生成式 A/B 人工评分。

## 7. Tomorrow TODO

- [ ] 复盘 LoRA 参数冻结与 adapter 保存逻辑。
- [ ] 复盘 DPO chosen/rejected loss 计算，重点看 mask、平均 logprob、reference-policy margin。
- [ ] 整理 SFT / LoRA / DPO 的训练流程对比表。
- [ ] 准备迁移到 Transformers / PEFT / TRL 的对应实现。

