# Day 2 - Full SFT Run

## 1. 今日目标

基于官方 `pretrain_512.pth`，运行 MiniMind 官方 Full SFT 阶段，使模型从续写式语言模型进一步学习对话格式和指令响应。

今天同时完成了 SFT 数据格式排查、工具调用样本清洗、W&B 记录接入、官方 pretrain 权重替换、完整 SFT 训练和 Pretrain/SFT 输出对比。

## 2. 输入与输出

### Input

- pretrain weight: `out/pretrain_512.pth`
  - 已将本地 mini pretrain 备份为 `out/pretrain_512_mini_local.pth`
  - 当前 `out/pretrain_512.pth` 为官方 `MiniMind2-PyTorch` 的 `pretrain_512.pth`
- original SFT dataset: `dataset/sft_t2t_mini.jsonl`
- cleaned SFT dataset: `dataset/sft_t2t_mini_no_tools.jsonl`
  - 原始数据: 905,718 条
  - 清洗后数据: 820,886 条
  - 因包含 `tools` / `tool_calls` / `functions` / `role=tool` 等字段清洗掉: 84,832 条，约 9.3663%

### Output

- full_sft weight: `out/full_sft_official_pretrain_512.pth`
- checkpoint: `checkpoints/full_sft_official_pretrain_512.pth`
- resume checkpoint: `checkpoints/full_sft_official_pretrain_512_resume.pth`
- W&B run: `MiniMind-Mini-SFT/runs/v2cisw0d`

## 3. SFT 数据格式

```json
{
  "conversations": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

今天第一次直接读取 `dataset/sft_t2t_mini.jsonl` 时，`datasets.load_dataset("json")` 在生成 Arrow 数据时失败。原因是部分样本的 `conversations` message 里额外包含 `tools`、`tool_calls`、`functions` 或 `role=tool`，导致 HuggingFace datasets 对嵌套字段推断出的 schema 不一致。

清洗脚本已放在：

```bash
debug/filter_sft_no_tools.py
```

核心处理逻辑是丢弃含工具调用字段或非 `system/user/assistant` role 的样本，并保留普通对话样本中的 `role`、`content` 和可选 `reasoning_content`。

清洗命令：

```bash
cd /workspace/liminghao/Projects/repo-minimind
python debug/filter_sft_no_tools.py
```

验证清洗结果：

```bash
rg -n '"tools"|"tool_calls"|"functions"|"role":\s*"tool"' dataset/sft_t2t_mini_no_tools.jsonl
```

期望无输出。

## 4. 运行命令

下载并替换官方 512 pretrain 权重：

```bash
cd /workspace/liminghao/Projects/repo-minimind

mv out/pretrain_512.pth out/pretrain_512_mini_local.pth

python -c "from huggingface_hub import hf_hub_download; import shutil; src = hf_hub_download(repo_id='jingyaogong/MiniMind2-Pytorch', filename='pretrain_512.pth'); shutil.copyfile(src, 'out/pretrain_512.pth'); print('saved to out/pretrain_512.pth')"
```

启动 SFT 训练：

```bash
cd /workspace/liminghao/Projects/repo-minimind/trainer

python train_full_sft.py \
  --data_path ../dataset/sft_t2t_mini_no_tools.jsonl \
  --from_weight pretrain \
  --save_weight full_sft_official_pretrain \
  --save_dir ../out \
  --hidden_size 512 \
  --num_hidden_layers 8 \
  --use_wandb \
  --wandb_project MiniMind-Mini-SFT \
  2>&1 | tee ../logs/day2_mini_sft_official_pretrain.log
```

评测 Pretrain：

```bash
cd /workspace/liminghao/Projects/repo-minimind

printf "0\n" | python eval_llm.py \
  --weight pretrain \
  --hidden_size 512 \
  --num_hidden_layers 8 \
  --max_new_tokens 128 \
  2>&1 | tee logs/day2_eval_pretrain_512.log
```

评测 SFT：

```bash
cd /workspace/liminghao/Projects/repo-minimind

printf "0\n" | python eval_llm.py \
  --weight full_sft_official_pretrain \
  --hidden_size 512 \
  --num_hidden_layers 8 \
  --max_new_tokens 128 \
  2>&1 | tee logs/day2_eval_full_sft_pretrain_512.log
```

## 5. 关键配置

| item            | value |
| --------------- | ----- |
| batch_size      | 16 |
| max_seq_len     | 340 |
| learning_rate   | 1e-6 |
| epochs          | 2 |
| dtype           | bfloat16 |
| hidden_size     | 512 |
| num_hidden_layers | 8 |
| use_moe         | 0 |
| accumulation_steps | 1 |
| grad_clip       | 1.0 |
| log_interval    | 100 |
| save_interval   | 1000 |
| pretrain weight | `out/pretrain_512.pth` |
| output dir      | `out/` |

代码路径对应关系：

- `trainer/train_full_sft.py`: 训练入口、参数、训练循环、保存权重
- `dataset/lm_dataset.py`: `SFTDataset` 读取 jsonl，渲染 chat template，生成 `input_ids` 和 `labels`
- `trainer/trainer_utils.py`: `init_model()` 根据 `--from_weight pretrain` 加载 `../out/pretrain_512.pth`
- `model/model_minimind.py`: forward 内部执行 next-token cross entropy，`ignore_index=-100`
- `eval_llm.py`: 使用 `--weight full_sft_official_pretrain` 加载 `out/full_sft_official_pretrain_512.pth`

## 6. 训练日志

训练日志来自 `logs/day2_mini_sft_official_pretrain.log`。本次完整跑完 2 个 epoch，每个 epoch 约 51,306 step 日志单位。

| step | loss | lr | gpu_mem | note |
| ---- | ---- | ---- | ------- | ---- |
| epoch1-100 | 3.5072 | 0.00000100 | 未记录 | 初始 SFT loss 较高 |
| epoch1-1000 | 2.3491 | 0.00000100 | 未记录 | 快速下降到 2.x |
| epoch1-10000 | 2.1363 | 0.00000098 | 未记录 | 进入震荡下降区间 |
| epoch1-51306 | 2.3337 | 0.00000055 | 未记录 | 第 1 个 epoch 结束 |
| epoch2-100 | 1.9317 | 0.00000055 | 未记录 | 第 2 个 epoch 起点低于第 1 个 epoch |
| epoch2-10000 | 1.9151 | 0.00000041 | 未记录 | 继续稳定在 2.0 附近 |
| epoch2-30000 | 1.6537 | 0.00000019 | 未记录 | 出现较低 loss 点 |
| epoch2-51306 | 2.4383 | 0.00000010 | 未记录 | 最后一个 batch 单点偏高，不代表整体趋势 |

窗口统计：

| window | mean loss | min | max | note |
| ------ | --------- | --- | --- | ---- |
| first 50 log points | 2.4921 | 2.0941 | 3.5072 | 训练早期 |
| middle 50 log points | 2.0856 | 1.7625 | 2.5562 | 中段明显低于早期 |
| last 50 log points | 2.0326 | 1.7809 | 2.6794 | 尾段仍有波动，但均值继续下降 |

W&B summary 最后一个记录点为 `loss=2.4383`、`learning_rate=1e-7`、`_step=1027`。这是最后 batch 的单点值；综合窗口均值看，loss 整体从 2.49 降到约 2.03。

## 7. Pretrain vs SFT 输出对比

评测日志：

- Pretrain: `logs/day2_eval_pretrain_512.log`
- SFT: `logs/day2_eval_full_sft_pretrain_512.log`

| prompt | Pretrain output summary | SFT output summary | observation |
| ------ | ----------------------- | ------------------ | ----------- |
| 你有什么特长？ | 像文本续写，回答很长，并继续“基于以上文本重新叙述一只聪明的机器人”。 | 明确自称 `minimind` AI 助手，回答简短，说明可以提供信息与帮助。 | SFT 明显学到 assistant 身份和对话风格。 |
| 为什么天空是蓝色的 | 能提到短波蓝光更容易散射，但开头像续写，表达重复。 | 使用较正式的解释结构，但把蓝天原因错误归因到日出日落太阳位置。 | SFT 改善格式，不保证事实性提升。 |
| 请用Python写一个计算斐波那契数列的函数 | 没有给出可靠代码，更像描述“如何写函数”，还出现错误序列。 | 输出了代码块，但代码逻辑错误，语法也不完整。 | 26M 模型和 mini SFT 数据不足以获得可靠代码能力。 |
| 推荐一些中国的美食 | 能列出火锅、粤菜、川菜等，但像续写文本，并混入奇怪内容。 | 语气更像助手，能推荐北京烤鸭、四川火锅、广东点心等，但仍有事实和命名错误。 | SFT 改善问答形态，内容质量仍受模型容量限制。 |

总体看，SFT 后模型从“续写文本”转向“回答用户问题”，但知识准确性、代码能力和复杂推理没有显著提升。

## 8. 现象观察

- loss 是否正常下降：是。早期 50 个日志点均值约 2.4921，尾部 50 个日志点均值约 2.0326。
- 是否出现 nan：否。日志中未观察到 `nan` 或 `inf`。
- 是否出现 OOM：否。本次官方 pretrain SFT 完整跑完。
- 是否成功加载 pretrain：是。`--from_weight pretrain` 与 `--hidden_size 512` 对应加载 `../out/pretrain_512.pth`。
- 是否生成 full_sft 权重：是。输出 `out/full_sft_official_pretrain_512.pth`。
- SFT 后是否更像 assistant：是。评测中 SFT 输出明显更短、更直接，更像助手回答。

需要注意：

- 今天第一次训练原始 `sft_t2t_mini.jsonl` 时因 tool 字段 schema 不一致失败，后续通过 `debug/filter_sft_no_tools.py` 清洗解决。
- 本地 mini pretrain 基础上的 SFT 曾启动并中途停止，最终有效产物以官方 `pretrain_512.pth` 基础上的 `full_sft_official_pretrain_512.pth` 为准。
- SFT 的主要收益是对话格式和回答风格，不应期待有限 mini SFT 数据显著注入新知识。

## 9. Pretrain 和 SFT 的关系总结

Pretrain 阶段学习基础语言建模能力，本质是根据前文预测下一个 token；SFT 阶段在 Pretrain 权重基础上，用 user-assistant 对话数据训练模型学习指令响应格式。因此，SFT 后模型应当从“续写文本”逐渐转向“回答用户问题”。

从今天的结果看，这个预期成立：`pretrain_512` 输出更像补全文本，`full_sft_official_pretrain_512` 输出更像 assistant。但由于模型只有 26M 参数，且 SFT 数据是 mini 规模，本轮训练不能显著提升知识覆盖、代码生成或事实准确性。

## 10. 明日 TODO

- 验证 `--from_resume 1`，确认 `checkpoints/full_sft_official_pretrain_512_resume.pth` 可恢复训练。
- 做 batch size / max_seq_len 显存对比，记录 `nvidia-smi` 峰值显存。
- 固化一组 Pretrain vs SFT 评测 prompt，并增加人工评分维度。
- 准备 LoRA 或 DPO 后训练。
- 如算力允许，尝试官方 `pretrain_768.pth` + `hidden_size=768` + `num_hidden_layers=16` 的 SFT 对比。
