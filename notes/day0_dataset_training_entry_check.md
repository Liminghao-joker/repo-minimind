# Day 0 - Dataset and Training Entry Check

## 1. Dataset Status

| file | exists | lines | format | usage |
|---|---:|---:|---|---|
| pretrain_t2t_mini.jsonl | yes | 1,270,238 | text | Pretrain |
| sft_t2t_mini.jsonl | yes | 905,718 | conversations | SFT |
| dpo.jsonl | yes | 17,166 | chosen/rejected | DPO |
| rlaif.jsonl | yes | 19,502 | conversations | RLAIF |

## 2. Data Format Understanding

### Pretrain

- 字段：`text`
- 作用：训练模型的基础续写能力
- 对应脚本：`trainer/train_pretrain.py`

### SFT / LoRA

- 字段：`conversations`
- 作用：训练对话和指令响应能力
- 对应脚本：
  - `trainer/train_full_sft.py`
  - `trainer/train_lora.py`

### DPO

- 字段：`chosen` / `rejected`
- 作用：偏好对齐，让模型更倾向 chosen
- 对应脚本：`trainer/train_dpo.py`

### RLAIF

- 字段：`conversations`
- 作用：用于 PPO / GRPO / SPO 等基于 AI 反馈的强化学习训练
- 对应脚本：
  - `trainer/train_ppo.py`
  - `trainer/train_grpo.py`
  - `trainer/train_spo.py`

## 3. Training Script Map

| script | stage | dataset | output | key params |
|---|---|---|---|---|
| `trainer/train_pretrain.py` | Pretrain | `../dataset/pretrain_hq.jsonl` | `../out/pretrain_512.pth` | `batch_size=32`, `max_seq_len=340`, `learning_rate=5e-4`, `from_resume` supported |
| `trainer/train_full_sft.py` | SFT | `../dataset/sft_mini_512.jsonl` | `../out/full_sft_512.pth` | `batch_size=16`, `max_seq_len=340`, `learning_rate=1e-6`, `from_resume` supported |
| `trainer/train_lora.py` | LoRA | `../dataset/lora_identity.jsonl` | `../out/lora/lora_identity_512.pth` | `batch_size=32`, `max_seq_len=340`, `learning_rate=1e-4`, `from_resume` supported |
| `trainer/train_dpo.py` | DPO | `../dataset/dpo.jsonl` | `../out/dpo_512.pth` | `batch_size=4`, `max_seq_len=1024`, `learning_rate=4e-8`, `beta=0.1`, `from_resume` supported |


## 5. Problems

| phenomenon | possible cause | fix |
|---|---|---|
| Running `trainer/train_pretrain.py` with defaults will not find local data | Script default is `../dataset/pretrain_hq.jsonl`, but local file is `dataset/pretrain_t2t_mini.jsonl` | Run with `--data_path ../dataset/pretrain_t2t_mini.jsonl` or add the expected `pretrain_hq.jsonl` |
| Running `trainer/train_full_sft.py` with defaults will not find local data | Script default is `../dataset/sft_mini_512.jsonl`, but local file is `dataset/sft_t2t_mini.jsonl` | Run with `--data_path ../dataset/sft_t2t_mini.jsonl` or add the expected `sft_mini_512.jsonl` |
| Running `trainer/train_lora.py` with defaults will not find local data | Script default is `../dataset/lora_identity.jsonl`, and this file is missing locally | Add `dataset/lora_identity.jsonl` or pass another conversations-format dataset with `--data_path` |
| Local RLAIF file name does not match RL script defaults | Local file is `dataset/rlaif.jsonl`, while RLAIF scripts default to `../dataset/rlaif-mini.jsonl` | Run PPO / GRPO / SPO with `--data_path ../dataset/rlaif.jsonl` or rename/add `rlaif-mini.jsonl` |
