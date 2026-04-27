# Day 1 - Pretrain Run

## 1. 今日目标

跑通 MiniMind 官方 Stage 1 Pretrain 最小训练闭环，观察数据进入训练、loss 输出、显存占用、checkpoint 保存、pretrain 权重加载测试等现象。

## 2. 运行环境

- GPU: NVIDIA GeForce RTX 3090
- PyTorch: 2.6.0+cu124
- CUDA: 12.4
- Python: 3.10.20
- Conda env: minimind
- Git branch: train-week1

## 3. 阅读文件

- `trainer/train_pretrain.py`
- `trainer/trainer_utils.py`
- `eval_llm.py`
- `debug/train_pretrain_mem_debug.py`
- 数据文件：`dataset/pretrain_t2t_mini.jsonl`

## 4. 关键配置

| item | value |
|---|---|
| batch_size | 32 |
| max_seq_len | 340 |
| learning_rate | 5e-4 |
| epochs | 1 |
| dtype | bfloat16 |
| save_interval | 1000 |
| log_interval | 100 |
| dataset | `../dataset/pretrain_t2t_mini.jsonl` |
| output dir | `../out` |
| hidden_size | 512 |
| num_hidden_layers | 8 |
| from_weight | `none` |

## 5. 训练命令

正式 pretrain：

```bash
cd /workspace/liminghao/Projects/repo-minimind/trainer

python train_pretrain.py \
  --data_path ../dataset/pretrain_t2t_mini.jsonl \
  --use_wandb \
  --wandb_project MiniMind-Pretrain \
  2>&1 | tee ../logs/day1_pretrain.log
```

显存 debug 短跑：

```bash
cd /workspace/liminghao/Projects/repo-minimind

python debug/train_pretrain_mem_debug.py \
  2>&1 | tee logs/pretrain_mem_debug.log
```

注意：从仓库根目录直接运行 debug 脚本时，`init_model()` 默认的 `../model` tokenizer 路径会解析到仓库外层，可能触发 HuggingFace repo id 校验错误。更稳的运行方式是进入 `debug/` 后运行，或给脚本补显式 tokenizer 路径参数。

## 6. 训练流程理解

```text
pretrain_t2t_mini.jsonl
-> text
-> tokenizer
-> input_ids / labels
-> dataloader
-> model forward
-> LM loss
-> backward
-> optimizer.step
-> checkpoint / pretrain weight
```

Pretrain 阶段没有加载已有模型权重。本次 `from_weight=none`，所以 tokenizer 来自本地 `model/`，模型参数由 `MiniMindForCausalLM(lm_config)` 随机初始化后开始训练。

## 7. 日志记录

正式训练日志：

| step | loss | lr | gpu_mem | note |
|---|---:|---:|---|---|
| 100 | 7.3730 | 0.00049999 | 未记录 | loss 开始从高位下降 |
| 1000 | 5.6562 | 0.00049930 | 未记录 | 第一次触发 checkpoint 保存 |
| 5000 | 3.1159 | 0.00048261 | 未记录 | loss 明显下降 |
| 10000 | 2.4995 左右 | 约 0.000431 | 未记录 | 进入较稳定区间 |
| 39695 | 2.1752 | 0.00005000 | 未记录 | 1 epoch 完成 |

显存 debug 日志：

| step | loss | lr | gpu_mem | note |
|---|---:|---:|---|---|
| 20 | 8.1757 | 0.00050000 | alloc=0.58GB, reserved=3.76GB, peak=3.54GB | 第一次显存日志 |
| 100 | 7.3727 | 0.00049999 | alloc=0.58GB, reserved=3.76GB, peak=3.54GB | 与正式训练初期接近 |
| 500 | 6.3311 | 0.00049982 | alloc=0.58GB, reserved=3.76GB, peak=3.54GB | reserved 基本稳定 |
| 1000 | 5.6623 | 0.00049930 | alloc=0.48GB, reserved=3.76GB, peak=3.54GB | PyTorch 缓存显存不等于实际张量占用 |
| 1500 | 4.5535 | 0.00049842 | alloc=0.58GB, reserved=3.76GB, peak=3.54GB | 短跑过程中无 OOM |

`alloc` 是当前 PyTorch 张量实际占用，`reserved` 是 PyTorch caching allocator 保留的显存，`peak` 是程序启动后最大实际占用。

## 8. 输出文件

- `out/pretrain_512.pth`: 普通模型权重，约 56 MB，用于推理测试和后续 SFT 初始化。
- `checkpoints/pretrain_512.pth`: checkpoint 目录下的普通权重副本，约 56 MB。
- `checkpoints/pretrain_512_resume.pth`: 完整续训状态，约 253 MB，包含 model、optimizer、scaler、epoch、step 等。
- `logs/day1_pretrain.log`: 正式训练日志。
- `logs/pretrain_mem_debug.log`: 显存 debug 日志。

保存触发条件来自 `train_pretrain.py`：

```python
if (step % args.save_interval == 0 or step == iters) and is_main_process():
```

因此中途只有到达 `save_interval` 的整数倍才会保存；如果提前 Ctrl+C，且没有到达保存 step，也没有跑到 epoch 末尾，就不会产生新的 checkpoint。

## 9. 现象观察

- loss 是否正常下降：是。正式训练从 step 100 的 7.3730 下降到 epoch 末尾的 2.1752。
- 是否出现 nan：没有在日志中观察到 nan。
- 是否出现 OOM：没有。
- GPU 利用率是否稳定：训练能稳定推进；显存 debug 中 peak 约 3.54GB，reserved 约 3.76GB。
- 是否生成 checkpoint：是。正式训练生成 `out/pretrain_512.pth` 和 `checkpoints/pretrain_512_resume.pth`。
- 预训练后推理效果：模型已经能生成流畅中文，但事实错误、重复、指令执行弱，符合只做 pretrain 后的状态。

## 10. 小幅改动测试

改动内容：在 debug 版 pretrain 脚本中给日志增加 GPU 显存字段。

修改文件：`debug/train_pretrain_mem_debug.py`

核心代码：

```python
def get_gpu_mem_log(device):
    if not torch.cuda.is_available() or "cuda" not in str(device):
        return "gpu_mem: cpu"

    device_obj = torch.device(device)
    device_idx = torch.cuda.current_device() if device_obj.index is None else device_obj.index
    allocated = torch.cuda.memory_allocated(device_idx) / 1024**3
    reserved = torch.cuda.memory_reserved(device_idx) / 1024**3
    max_allocated = torch.cuda.max_memory_allocated(device_idx) / 1024**3
    return f"gpu_mem: alloc={allocated:.2f}GB, reserved={reserved:.2f}GB, peak={max_allocated:.2f}GB"
```

运行命令：

```bash
python debug/train_pretrain_mem_debug.py 2>&1 | tee logs/pretrain_mem_debug.log
```

观察结果：

- 日志能同步打印 `gpu_mem: alloc=..., reserved=..., peak=...`。
- 当前配置下 peak 约 3.54GB，说明 25.83M 参数模型在 `batch_size=32`、`max_seq_len=340`、bf16 autocast 下显存压力不高。
- 短跑 debug 如果没有到达 `save_interval`，不会保存 checkpoint。

## 11. Pretrain 与 SFT 的关系

Pretrain 阶段主要学习文本续写和基础语言建模能力，模型目标是根据前文预测下一个 token；SFT 阶段则在 Pretrain 权重基础上学习对话格式、指令响应和 assistant 风格。因此，Pretrain 更像“打语言基础”，SFT 更像“教模型按人的指令回答”。

本次 pretrain 后的自动测试也验证了这一点：模型能输出中文、能接住问题主题，但还不能稳定遵循“写代码”“准确解释”“推荐中国美食”等指令，下一阶段应使用 `out/pretrain_512.pth` 作为初始化权重继续做 full SFT。
