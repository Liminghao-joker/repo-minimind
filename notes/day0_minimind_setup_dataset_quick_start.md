# Day 0 - MiniMind Setup + Dataset + Quick Start

## 1. 今日目标

- 配置 MiniMind 环境
- 下载官方模型并完成 eval
- 准备 mini 数据集
- 检查 Pretrain / SFT / LoRA / DPO 数据格式
- 定位训练入口脚本

## 2. Environment

| item | value |
|---|---|
| GPU | NVIDIA GeForce RTX 3090 |
| CUDA | 12.4 |
| PyTorch | 2.6.0+cu124 |
| Python | 3.10.20 |
| bf 16 supported | yes |
| repo commit | `a94590e` |

## 3. Commands

```bash
git clone https://github.com/jingyaogong/minimind.git
cd minimind
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 加载评测 minimind-3 模型
python eval_llm.py --load_from ./minimind-3
```

## 4. Dataset Status

| file | exists | lines | format | usage |
|---|---:|---:|---|---|
| pretrain_t2t_mini.jsonl | yes | 1,270,238 | `{"text": ...}` | Pretrain |
| sft_t2t_mini.jsonl | yes | 905,718 | conversations | SFT |
| lora_identity.jsonl | no | 0 | conversations | LoRA |
| dpo.jsonl | yes | 17,166 | chosen/rejected | DPO |

## 5. Training Script Map

| script | stage | dataset | output | key params |
|---|---|---|---|---|
| `trainer/train_pretrain.py` | Pretrain | `../dataset/pretrain_hq.jsonl` | `../out/pretrain_512.pth` | `batch_size=32`, `max_seq_len=340`, `learning_rate=5e-4`, supports `--from_resume 1` |
| `trainer/train_full_sft.py` | SFT | `../dataset/sft_mini_512.jsonl` | `../out/full_sft_512.pth` | `batch_size=16`, `max_seq_len=340`, `learning_rate=1e-6`, supports `--from_resume 1` |
| `trainer/train_lora.py` | LoRA | `../dataset/lora_identity.jsonl` | `../out/lora/lora_identity_512.pth` | `batch_size=32`, `max_seq_len=340`, `learning_rate=1e-4`, supports `--from_resume 1` |
| `trainer/train_dpo.py` | DPO | `../dataset/dpo.jsonl` | `../out/dpo_512.pth` | `batch_size=4`, `max_seq_len=1024`, `learning_rate=4e-8`, `beta=0.1`, supports `--from_resume 1` |

## 6. Official Model Eval

| prompt | output summary | observation |
|---|---|---|
| 你是谁？ | 回答自己是 jingyaogong 创建的高效小参数 AI 模型，强调快速和精准信息支持。 | 身份类回答基本符合 MiniMind 项目语境。速度约 16.87 tokens/s。 |
| 请用三句话解释 Transformer。 | 提到自注意力和并行处理，但混入了 Adobe、Microsoft、Oreo 等错误来源描述。 | 存在明显事实错误，说明官方小模型在基础概念问答上仍可能幻觉。速度约 38.61 tokens/s。 |
| 请写一个 Python 函数计算 Fibonacci 数列。 | 生成了 `fibonacci_sequence(n)`，代码主体用列表迭代生成 Fibonacci 数列。 | 代码核心思路可用，但解释把 `n` 错写成整数列表，示例末尾变量名覆盖函数导致再调用会出错。速度约 38.80 tokens/s。 |

## 7. Key Takeaways

- Pretrain 数据是 text 格式，目标是 next token prediction。
- SFT 数据是 conversations 格式，目标是学习指令和对话模式。
- LoRA 数据格式接近 SFT，但训练参数量更少。
- DPO 数据包含 chosen/rejected，用于偏好对齐。
- `eval_llm.py --load_from ./minimind-3` 用于加载官方 transformers 格式模型。
- 后续训练主要从 `trainer/` 下四个脚本启动。
- 当前本地 mini 数据集文件名和部分训练脚本默认 `data_path` 不一致，正式训练时需要显式传 `--data_path`。

## 8. Problems & Fixes

| phenomenon | possible cause | fix |
|---|---|---|
| 官方模型 eval 有幻觉 | 小参数模型能力有限，部分知识问答和代码解释不稳定 | 评估时记录原始输出；后续通过 SFT / DPO 对齐改善指令和事实回答 |
| Fibonacci 示例代码存在变量覆盖问题 | 输出中把 `fibonacci_sequence` 函数名重新赋值为列表 | 正确写法应避免 `fibonacci_sequence = fibonacci_sequence(n)` 这种覆盖 |

## 9. Tomorrow Plan

- 正式跑 `train_pretrain.py`
- 开启 wandb, 记录 loss、显存、checkpoint
- 如果 pretrain 较慢，至少进入训练循环并保存日志
