# MiniMind - train_pretrain.py 代码拆解与训练配置
> 拆解 `trainer/train_pretrain.py`文件 `Pretrain`阶段
## 1. TL;DR

`trainer/train_pretrain.py` 是一个完整的 causal language modeling 预训练脚本：它从命令行参数得到训练配置，构造 `MiniMindConfig`，通过 `init_model()` 创建 `MiniMindForCausalLM` 和 tokenizer，用 `PretrainDataset` 读取 JSONL 中的 `text` 字段，再通过 `DataLoader` 送入模型做 next-token prediction。loss 在模型 forward 内部用 `F.cross_entropy(..., ignore_index=-100)` 计算，训练循环负责学习率调度、混合精度、梯度累积、梯度裁剪、优化器更新、日志打印和 checkpoint 保存。

当前工作区中的 `trainer/train_pretrain.py` 默认数据路径是 `../dataset/pretrain_t2t_mini.jsonl`，这是相对 git 版本的未提交改动；原仓库默认更常见的是 `../dataset/pretrain_hq.jsonl`。注意这些 `../dataset`、`../out`、`../checkpoints` 路径默认假设你从 `trainer/` 目录运行脚本。

## 2. 训练主流程

```text
命令行参数 / 默认配置
→ 模型配置 MiniMindConfig
→ tokenizer + MiniMindForCausalLM
→ PretrainDataset
→ SkipBatchSampler + DataLoader
→ model(input_ids, labels=labels)
→ Causal LM loss + aux_loss
→ loss / accumulation_steps
→ backward
→ grad clip
→ optimizer.step
→ cosine-like lr schedule
→ checkpoint
→ 日志输出
```

| step | 代码位置 | 输入 | 输出 | 说明 |
|---|---|---|---|---|
| 参数解析 | `trainer/train_pretrain.py:75-99` | CLI 参数或默认值 | `args` | 包括 `batch_size`、`learning_rate`、`max_seq_len`、`data_path`、`from_resume` 等。 |
| 分布式和随机种子 | `trainer/train_pretrain.py:101-104`，`trainer/trainer_utils.py:44-61` | 环境变量 `RANK/LOCAL_RANK`，seed | `local_rank`、确定性随机状态 | 非 DDP 时 `init_distributed_mode()` 直接返回 0。 |
| 模型配置 | `trainer/train_pretrain.py:106-109`，`model/model_minimind.py:8-79` | `hidden_size`、`num_hidden_layers`、`use_moe` | `lm_config` | 未显式传入的配置使用 `MiniMindConfig` 默认值，如 `vocab_size=6400`、`num_attention_heads=8`。 |
| tokenizer 和模型 | `trainer/train_pretrain.py:125-129`，`trainer/trainer_utils.py:119-131` | `lm_config`、`from_weight`、`device` | `model`、`tokenizer` | `AutoTokenizer.from_pretrained('../model')` 读取本地 tokenizer；`MiniMindForCausalLM(lm_config)` 构造模型。 |
| 数据集 | `trainer/train_pretrain.py:130-132`，`dataset/lm_dataset.py:33-59` | `args.data_path`、tokenizer、`max_seq_len` | `train_ds` | Hugging Face `load_dataset('json', ...)` 读取 JSONL；样本转为 `(input_ids, labels)`。 |
| batch sampler | `trainer/train_pretrain.py:151-157`，`trainer/trainer_utils.py:134-157` | 随机索引或分布式 sampler、`batch_size`、`skip` | `batch_sampler` | 支持 resume 时跳过已经训练过的 batch。 |
| DataLoader | `trainer/train_pretrain.py:157` | `train_ds`、`batch_sampler`、`num_workers` | `loader` | 输出 batch 形状大致为 `[batch_size, max_seq_len]`。 |
| forward | `trainer/train_pretrain.py:34-36`，`model/model_minimind.py:460-499` | `input_ids`、`labels` | `res.loss`、`res.aux_loss`、`logits` | 模型内部做 next-token prediction loss。 |
| backward | `trainer/train_pretrain.py:37-48` | `loss / accumulation_steps` | 梯度、参数更新 | 每 `accumulation_steps` 个 step 才执行一次 optimizer update。 |
| lr schedule | `trainer/train_pretrain.py:29-32`，`trainer/trainer_utils.py:40-41` | 当前 step、总 step、初始 lr | 当前 lr | 使用余弦形式衰减，范围约从 `lr` 到 `0.1 * lr`。 |
| checkpoint | `trainer/train_pretrain.py:60-69`，`trainer/trainer_utils.py:63-116` | model、optimizer、scaler、epoch、step | `../out/pretrain_512.pth` 和 `../checkpoints/pretrain_512_resume.pth` | 权重文件用于推理/后续训练；resume 文件包含优化器和进度。 |
| 日志 | `trainer/train_pretrain.py:50-58`，`trainer/trainer_utils.py:35-37` | loss、lr、耗时 | stdout | 只在主进程打印。 |

运行时常见输出来源：

| 输出 | 来源 | 解释 |
|---|---|---|
| `Model Params: 25.83M` | `trainer/trainer_utils.py:18-29`，由 `init_model()` 在 `trainer/trainer_utils.py:129` 调用 | 统计模型所有参数数量。 |
| `Trainable Params: 25.830M` | `trainer/trainer_utils.py:130` | 统计 `requires_grad=True` 的参数；pretrain 没冻结参数，所以等于总参数。 |
| `Generating train split: ...` | `dataset/lm_dataset.py:40` 调用 Hugging Face `datasets.load_dataset('json', ...)` | `datasets` 正在把 JSONL 解析成 Arrow dataset cache/split，不是训练循环日志。 |
| `loss: ...` | `trainer/train_pretrain.py:50-58` | 已经进入 `train_epoch()`，完成 forward/backward 后按 `log_interval` 打印。 |

## 3. 关键代码模块

### 3.1 `trainer/train_pretrain.py`

代码位置：`trainer/train_pretrain.py:1-18`

作用：导入训练入口所需模块。关键依赖包括 `MiniMindConfig`、`PretrainDataset`、`init_model`、`get_lr`、`lm_checkpoint`、`SkipBatchSampler`。

关键变量：`torch`、`dist`、`DataLoader`、`DistributedSampler`、`MiniMindConfig`、`PretrainDataset`。

我应该观察什么：如果 import 报错，优先确认是否在正确 conda 环境、仓库根或 `trainer/` 目录、依赖是否安装完整。

### 3.2 参数解析和训练配置

代码位置：`trainer/train_pretrain.py:75-99`

作用：定义所有 CLI 参数及默认值。

关键变量：`args.save_dir`、`args.batch_size`、`args.learning_rate`、`args.accumulation_steps`、`args.max_seq_len`、`args.data_path`、`args.from_resume`。

我应该观察什么：当前工作区默认 `data_path=../dataset/pretrain_t2t_mini.jsonl`，但路径是相对运行时 cwd；如果从仓库根目录执行 `python trainer/train_pretrain.py`，这个相对路径会指向仓库外层，需要显式传 `--data_path ./dataset/pretrain_t2t_mini.jsonl` 或进入 `trainer/` 后运行。

### 3.3 分布式、随机种子、目录

代码位置：`trainer/train_pretrain.py:101-109`，`trainer/trainer_utils.py:44-61`

作用：初始化 DDP，设置随机种子，创建输出目录，构造模型配置，按需读取 resume checkpoint。

关键变量：`local_rank`、`args.device`、`lm_config`、`ckp_data`。

我应该观察什么：单卡普通运行时 `dist.is_initialized()` 为 false；如果用 `torchrun`，会走 NCCL DDP 并设置每张卡的 device。

### 3.4 mixed precision / dtype

代码位置：`trainer/train_pretrain.py:111-114`，`trainer/train_pretrain.py:133`

作用：根据 `args.dtype` 设置 autocast；如果是 CPU 则不用 autocast；如果是 CUDA，则用 `torch.cuda.amp.autocast(dtype=dtype)`。

关键变量：`device_type`、`dtype`、`autocast_ctx`、`scaler`。

我应该观察什么：默认 `dtype=bfloat16`，`GradScaler` 只有在 `dtype == 'float16'` 时启用。RTX 3090 当前环境显示 bf16 supported 为 yes，但实际性能和稳定性仍建议通过 smoke test 观察。

### 3.5 初始化模型和 tokenizer

代码位置：`trainer/train_pretrain.py:125-129`，`trainer/trainer_utils.py:119-131`

作用：`init_model()` 加载 tokenizer，创建 `MiniMindForCausalLM`，可选加载已有权重，打印参数量，然后把模型移动到 device。

关键变量：`tokenizer_path='../model'`、`from_weight`、`save_dir='../out'`、`model`、`tokenizer`。

我应该观察什么：pretrain 默认 `from_weight='none'`，所以从随机初始化开始训练；如果 `from_weight` 不是 `none`，会从 `../out/{from_weight}_{hidden_size}.pth` 加载。

关键代码片段：

```python
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
model = MiniMindForCausalLM(lm_config)
if from_weight != 'none':
    weight_path = f'{save_dir}/{from_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
    weights = torch.load(weight_path, map_location=device)
    model.load_state_dict(weights, strict=False)
```

### 3.6 MiniMindConfig 和模型大小

代码位置：`model/model_minimind.py:8-79`

作用：定义模型结构默认值。

关键变量：`vocab_size=6400`、`hidden_size=512`、`num_hidden_layers=8`、`num_attention_heads=8`、`num_key_value_heads=2`、`intermediate_size=None`。

我应该观察什么：`intermediate_size` 初始为 `None`，第一次构造 FFN 时会在 `FeedForward` 中变成 1408，见 `model/model_minimind.py:224-233`。

### 3.7 PretrainDataset 如何读取 JSONL

代码位置：`dataset/lm_dataset.py:33-59`

作用：读取 JSONL 的 `text` 字段，tokenize，截断，添加 BOS/EOS，padding，构造 labels。

关键变量：`self.samples`、`tokens`、`input_ids`、`labels`。

我应该观察什么：`labels[input_ids == pad_token_id] = -100`，因此 padding token 不参与 loss。pretrain 的 label 基本等于 input，只是模型 forward 内部会 shift 一位。

关键代码片段：

```python
self.samples = load_dataset('json', data_files=data_path, split='train')
tokens = self.tokenizer(str(sample['text']), add_special_tokens=False,
                        max_length=self.max_length - 2, truncation=True).input_ids
tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
labels = input_ids.clone()
labels[input_ids == self.tokenizer.pad_token_id] = -100
```

### 3.8 DataLoader 和 resume 跳过 batch

代码位置：`trainer/train_pretrain.py:151-157`，`trainer/trainer_utils.py:134-157`

作用：每个 epoch 生成随机索引，构造 `SkipBatchSampler`，如果 resume 中有 `start_step`，跳过已训练 batch。

关键变量：`indices`、`skip`、`batch_sampler`、`loader`。

我应该观察什么：`setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()` 保证每个 epoch 有可复现 shuffle。resume 只按 batch 数跳过，数据集和 world size 改变时需要格外谨慎。

### 3.9 optimizer 和 learning rate scheduler

代码位置：`trainer/train_pretrain.py:29-32`，`trainer/train_pretrain.py:134`，`trainer/trainer_utils.py:40-41`

作用：使用 `optim.AdamW(model.parameters(), lr=args.learning_rate)`；每个 step 手动调用 `get_lr()` 更新 optimizer param group。

关键变量：`optimizer`、`lr`、`args.learning_rate`。

我应该观察什么：这里没有显式 `weight_decay` 参数，所以 AdamW 使用 PyTorch 默认 `weight_decay=0.01`。`get_lr()` 没有 warmup，初始 step 的 lr 接近 `learning_rate`，后续余弦衰减到约 `0.1 * learning_rate`。

### 3.10 model forward 和 loss

代码位置：`trainer/train_pretrain.py:34-37`，`model/model_minimind.py:460-499`

作用：`model(input_ids, labels=labels)` 返回 `CausalLMOutputWithPast`，其中 `loss` 是 next-token cross entropy。

关键变量：`res.loss`、`res.aux_loss`、`loss`、`shift_logits`、`shift_labels`。

我应该观察什么：非 MoE 时 `aux_loss` 是 0；最终训练 loss 是 `res.loss + res.aux_loss`，再除以 `accumulation_steps` 用于梯度累积。

关键代码片段：

```python
shift_logits = logits[..., :-1, :].contiguous()
shift_labels = labels[..., 1:].contiguous()
loss = F.cross_entropy(
    shift_logits.view(-1, shift_logits.size(-1)),
    shift_labels.view(-1),
    ignore_index=-100
)
```

### 3.11 backward、梯度累积和梯度裁剪

代码位置：`trainer/train_pretrain.py:37-48`

作用：把 loss 除以 `accumulation_steps`，每个 micro-batch backward，累计到指定步数后裁剪梯度并更新参数。

关键变量：`args.accumulation_steps`、`args.grad_clip`、`scaler`、`optimizer`。

我应该观察什么：默认 `batch_size=32`、`accumulation_steps=8`，单卡有效 batch size 约为 `32 * 8 = 256` 条样本；DDP 时还要乘以 GPU 数。

### 3.12 checkpoint 和 resume

代码位置：`trainer/train_pretrain.py:60-69`，`trainer/train_pretrain.py:136-143`，`trainer/trainer_utils.py:63-116`

作用：每到 `save_interval` 或 epoch 末尾保存普通权重和完整 resume checkpoint。

关键变量：`ckp`、`state_dict`、`resume_path`、`ckp_data`、`start_epoch`、`start_step`。

我应该观察什么：普通权重默认保存到 `../out/pretrain_512.pth`；resume 默认保存到 `../checkpoints/pretrain_512_resume.pth`。恢复时需要加 `--from_resume 1`。

### 3.13 日志打印

代码位置：`trainer/train_pretrain.py:50-58`，`trainer/trainer_utils.py:35-37`

作用：按 `log_interval` 打印 epoch、step、loss、logits_loss、aux_loss、lr 和剩余时间估计。

关键变量：`current_loss`、`current_aux_loss`、`current_logits_loss`、`current_lr`、`eta_min`。

我应该观察什么：默认 `log_interval=100`，大数据集开始训练后要等 100 个 batch 才打印第一条 loss；如果想 smoke test，建议调小到 1 或 5。

## 4. 关键配置表

| 参数名 | 当前默认值 | 代码位置 | 控制什么 | 改大/改小的影响 | 常见风险 |
|---|---:|---|---|---|---|
| `--save_dir` | `../out` | `trainer/train_pretrain.py:77` | 普通权重保存目录 | 改到实验目录便于隔离结果 | 路径相对 cwd，容易保存到意外位置 |
| `--save_weight` | `pretrain` | `trainer/train_pretrain.py:78` | 权重文件名前缀 | 改名可避免覆盖旧权重 | resume 也按这个名字查找 |
| `--epochs` | `1` | `trainer/train_pretrain.py:79` | 训练轮数 | 增大训练更久；减小更快验证流程 | 过多 epoch 可能过拟合或耗时过长 |
| `--batch_size` | `32` | `trainer/train_pretrain.py:80` | 单次 DataLoader batch 样本数 | 增大吞吐更高、loss 更平滑、显存更高；减小更省显存、loss 更抖 | OOM 或 GPU 利用率低 |
| `--learning_rate` | `5e-4` | `trainer/train_pretrain.py:81` | AdamW 初始学习率 | 增大收敛快但更容易 nan；减小更稳但训练慢 | loss nan、loss 不降 |
| `--device` | `cuda:0` 或 `cpu` | `trainer/train_pretrain.py:82` | 训练设备 | GPU 快；CPU 仅适合功能验证 | device 和张量不一致 |
| `--dtype` | `bfloat16` | `trainer/train_pretrain.py:83` | autocast 精度 | bf16 省显存且通常比 fp16 稳；fp16 启用 GradScaler | 不支持 bf16 的设备可能异常或性能差 |
| `--num_workers` | `8` | `trainer/train_pretrain.py:84` | DataLoader worker 数 | 增大可能提升供数速度；减小更稳更少 CPU 占用 | worker 太多导致卡顿或内存压力 |
| `--accumulation_steps` | `8` | `trainer/train_pretrain.py:85` | 梯度累积步数 | 增大有效 batch、更稳、省显存但更新更慢；减小更新更频繁 | 误判 step 和 update 次数 |
| `--grad_clip` | `1.0` | `trainer/train_pretrain.py:86` | 梯度裁剪阈值 | 降低可抑制梯度爆炸；升高约束更弱 | 太低可能训练慢，太高可能不稳 |
| `--log_interval` | `100` | `trainer/train_pretrain.py:87` | 日志间隔 | 减小更容易观察；增大减少 stdout 开销 | smoke test 看不到日志误以为卡住 |
| `--save_interval` | `1000` | `trainer/train_pretrain.py:88` | checkpoint 间隔 | 减小便于快速验证保存/resume；增大减少 IO | 太大时中断前没有 checkpoint |
| `--hidden_size` | `512` | `trainer/train_pretrain.py:89` | 模型 hidden 维度 | 增大参数量、表达能力和显存；减小相反 | 改后权重文件名和 shape 不兼容 |
| `--num_hidden_layers` | `8` | `trainer/train_pretrain.py:90` | Transformer block 层数 | 增大模型更深、更耗显存；减小更快 | 深模型更难稳定训练 |
| `--max_seq_len` | `340` | `trainer/train_pretrain.py:91` | 每条样本 token 长度 | 增大上下文更长但显存约随序列长度显著增长；减小更快 | 太小截断语义，太大 OOM |
| `--use_moe` | `0` | `trainer/train_pretrain.py:92` | 是否启用 MoE | 启用后有 aux_loss、参数量变化 | MoE 调参更复杂 |
| `--data_path` | `../dataset/pretrain_t2t_mini.jsonl` | `trainer/train_pretrain.py:93` | 预训练 JSONL 文件 | 换数据影响训练内容和数据量 | 路径错误、格式不是 `text` |
| `--from_weight` | `none` | `trainer/train_pretrain.py:94` | 是否从已有普通权重初始化 | 用已有权重可继续阶段训练；`none` 从头训 | shape 不匹配或路径错误 |
| `--from_resume` | `0` | `trainer/train_pretrain.py:95` | 是否读取完整训练状态 | 开启可恢复 optimizer/step/scaler | checkpoint 不存在则从头训 |
| `--use_wandb` | false | `trainer/train_pretrain.py:96` | 是否启用 swanlab 日志 | 开启可记录实验曲线 | 环境未配置会报错 |
| `--use_compile` | `0` | `trainer/train_pretrain.py:98` | 是否 `torch.compile` | 可能加速长训练 | 首次编译慢，debug 不建议开 |
| `vocab_size` | `6400` | `model/model_minimind.py:23` | tokenizer 词表大小和 lm_head 输出维度 | 增大表达更多 token 但参数更多 | 必须与 tokenizer 匹配 |
| `num_attention_heads` | `8` | `model/model_minimind.py:20` | attention head 数 | 改变 head_dim 和注意力结构 | hidden_size 必须能整除 |
| `num_key_value_heads` | `2` | `model/model_minimind.py:22` | GQA 的 KV head 数 | 更小更省 KV 相关计算 | 需整除 attention heads |
| `optimizer` | `AdamW` | `trainer/train_pretrain.py:134` | 参数更新算法 | AdamW 是 LLM 常用优化器 | 当前未显式设置 betas/weight_decay |
| `lr decay` | cosine-like，无 warmup | `trainer/trainer_utils.py:40-41` | 每 step 学习率 | 起始约 lr，结束约 0.1lr | 无 warmup 时大 lr 更易不稳 |

重点关系：

| 配置 | 影响 |
|---|---|
| `batch_size` | 直接影响激活显存和吞吐。batch 大时每 step 估计更稳定，loss 抖动小，但更容易 OOM；batch 小时更省显存，loss 曲线抖动更明显。 |
| `max_seq_len` | 决定每条样本 padding/truncation 后长度。attention 计算和激活显存对序列长度很敏感，增大它通常比小幅增大 batch 更容易涨显存。 |
| `learning_rate` | 控制每次参数更新幅度。过大时 loss 可能快速变 nan 或震荡，过小时 loss 下降很慢。当前调度没有 warmup，debug 时不要盲目增大。 |
| `accumulation_steps` | 有效 batch size 约等于 `batch_size * accumulation_steps * world_size`。它不降低单次 forward 的显存，但能在不增加 micro-batch 显存的情况下扩大有效 batch。 |
| `save_interval / log_interval` | debug run 应该调小，确保几分钟内能看到 loss 和 checkpoint；正式训练可以调大，减少 IO 和 stdout 开销。 |

## 5. 日志解释

### 5.1 为什么参数量是 25.83M？

当前默认配置实际构造出的参数量为 `25,829,888`，打印为 `25.83M`。来源是 `MiniMindForCausalLM(MiniMindConfig(hidden_size=512, num_hidden_layers=8))` 的所有参数求和，见 `trainer/trainer_utils.py:18-29`。

主要结构来自：

| 模块 | 代码位置 | 说明 |
|---|---|---|
| embedding / lm_head | `model/model_minimind.py:390`，`model/model_minimind.py:456-458` | `lm_head` 与 `embed_tokens.weight` 共享权重，所以不会双倍计数。 |
| 8 个 MiniMindBlock | `model/model_minimind.py:392` | 每层包括 GQA attention、RMSNorm、FFN。 |
| FFN intermediate size | `model/model_minimind.py:224-233` | 默认 `hidden_size=512` 时计算为 1408。 |
| final norm | `model/model_minimind.py:393` | 最后一层 RMSNorm。 |

### 5.2 为什么 Trainable Params 和 Model Params 一样？

pretrain 阶段没有冻结任何参数，也没有 LoRA 只训练 adapter 的逻辑。`trainer/trainer_utils.py:130` 统计的是所有 `p.requires_grad` 为 true 的参数；默认模型所有参数都是可训练的，所以 `Trainable Params` 等于 `Model Params`。

### 5.3 `Generating train split` 是什么？

它来自 `dataset/lm_dataset.py:40` 的 `load_dataset('json', data_files=data_path, split='train')`。Hugging Face `datasets` 正在把 JSONL 解析成内部 dataset split，通常会生成 Arrow cache。这个阶段还没有进入 PyTorch 训练循环。

### 5.4 从什么时候开始才算真正训练？

当执行到 `trainer/train_pretrain.py:160` 或 `trainer/train_pretrain.py:162` 调用 `train_epoch()` 后，进入 `trainer/train_pretrain.py:25` 的 `for step, (input_ids, labels) in enumerate(loader, ...)`，拿到第一个 batch，并执行 `model(input_ids, labels=labels)`、`backward()` 后，才算真正训练。

### 5.5 loss 打印后如何判断训练是否正常？

| 观察项 | 正常现象 | 异常信号 |
|---|---|---|
| loss 数值 | 有限数值，不是 `nan` 或 `inf` | 立刻 nan，多半是 lr、dtype、数据或梯度爆炸问题 |
| loss 趋势 | 短期可以抖动，较长窗口缓慢下降 | 长时间完全不动或持续上升 |
| lr | 从接近 `learning_rate` 逐步衰减 | lr 为 0、异常跳变或与预期不符 |
| aux_loss | 非 MoE 时应接近 0 | 非 MoE 却出现异常大 aux_loss 需要检查 |
| step 速度 | 稳定推进 | `Generating train split` 后无输出很久，可能还没到 `log_interval` 或 DataLoader 卡住 |

### 5.6 3090 显存只占几 GB 正常吗？

正常。当前模型只有 25.83M 参数，默认 `max_seq_len=340`，`batch_size=32`，并且用 bf16 autocast。参数本身很小，主要显存来自激活、梯度、优化器状态和 DataLoader batch。对于 24GB 的 RTX 3090，只占几 GB 是合理现象，不代表训练没用 GPU。判断 GPU 是否工作要结合 `nvidia-smi` 的显存、功耗、GPU-Util 和训练 step 速度。

## 6. 最小可观测训练实验

目标：20 到 30 分钟内验证训练链路、loss、显存、checkpoint、resume，而不是跑完整训练。

### 6.1 推荐命令

从仓库根目录执行：

```bash
mkdir -p logs out_smoke checkpoints_smoke
cd trainer
python train_pretrain.py \
  --data_path ../dataset/pretrain_t2t_mini.jsonl \
  --save_dir ../out_smoke \
  --save_weight pretrain_smoke \
  --epochs 1 \
  --batch_size 8 \
  --accumulation_steps 2 \
  --max_seq_len 128 \
  --learning_rate 1e-4 \
  --log_interval 1 \
  --save_interval 20 \
  --num_workers 2 \
  --dtype bfloat16 \
  2>&1 | tee ../logs/pretrain_smoke.log
```

说明：`lm_checkpoint()` 里的 resume 目录当前代码写死为 `../checkpoints`，不是参数化的 `checkpoints_smoke`。如果不改训练代码，resume 文件会写到 `../checkpoints/pretrain_smoke_512_resume.pth`。

### 6.2 推荐改哪些参数

| 参数 | 建议 | 原因 |
|---|---|---|
| `batch_size` | 改成 8 | 降低显存，便于 smoke test 稳定跑。 |
| `accumulation_steps` | 改成 2 | 有效 batch 仍有 16，更新频率更高，能更快看到 optimizer step。 |
| `max_seq_len` | 改成 128 | 大幅降低显存和 step 时间。 |
| `learning_rate` | 改成 `1e-4` | smoke test 优先稳定，不追求最快下降。 |
| `log_interval` | 改成 1 | 每个 step 都能看到 loss，便于判断是否卡住。 |
| `save_interval` | 改成 20 | 很快生成 checkpoint，便于验证 resume。 |
| `num_workers` | 改成 2 | debug 更稳，减少 worker 问题干扰。 |
| `hidden_size / num_hidden_layers / tokenizer` | 不改 | 保持权重文件名、参数量和模型结构与默认学习目标一致。 |

### 6.3 如何保存日志

使用 `tee` 同时输出到屏幕和文件：

```bash
2>&1 | tee ../logs/pretrain_smoke.log
```

如果只想追加日志：

```bash
2>&1 | tee -a ../logs/pretrain_smoke.log
```

### 6.4 如何记录显存

训练时另开一个终端：

```bash
nvidia-smi -l 5
```

如果要保存显存记录：

```bash
nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu,power.draw --format=csv -l 5 > logs/pretrain_smoke_nvidia_smi.csv
```

### 6.5 如何验证 resume

第一次训练至少等到 step 20 保存 checkpoint 后中断，再从 `trainer/` 目录运行：

```bash
python train_pretrain.py \
  --data_path ../dataset/pretrain_t2t_mini.jsonl \
  --save_dir ../out_smoke \
  --save_weight pretrain_smoke \
  --epochs 1 \
  --batch_size 8 \
  --accumulation_steps 2 \
  --max_seq_len 128 \
  --learning_rate 1e-4 \
  --log_interval 1 \
  --save_interval 20 \
  --num_workers 2 \
  --dtype bfloat16 \
  --from_resume 1
```

观察是否打印类似“跳过前 N 个 step，从 step N+1 开始”。

### 6.6 实验成功标准

| 检查项 | 成功标准 |
|---|---|
| 数据加载 | 出现 `Generating train split` 并完成，进入训练日志。 |
| loss | 多个 step 打印，且不是 nan/inf。 |
| GPU | `nvidia-smi` 显示 python 进程占用显存，GPU-Util/功耗有变化。 |
| checkpoint | 出现 `out_smoke/pretrain_smoke_512.pth` 和 `checkpoints/pretrain_smoke_512_resume.pth`。 |
| resume | 加 `--from_resume 1` 后能加载并跳过已训练 step。 |

### 6.7 实验记录模板

```markdown
## Pretrain Smoke Test

- date:
- commit:
- command:
- GPU:
- PyTorch / CUDA:
- data_path:
- batch_size:
- accumulation_steps:
- effective batch size:
- max_seq_len:
- learning_rate:
- dtype:
- log file:
- nvidia-smi file:
- first loss:
- last observed loss:
- max memory used:
- checkpoint generated:
- resume tested:
- conclusion:
- next action:
```

## 7. Debug 清单

| 现象 | 可能原因 | 如何定位 | 修复方式 |
|---|---|---|---|
| CUDA out of memory | `batch_size` 或 `max_seq_len` 太大；其他进程占用显存；DDP 每卡配置过高 | `nvidia-smi` 看显存；从日志确认最后进入哪个 step | 降低 `batch_size`、降低 `max_seq_len`、增大 `accumulation_steps` 替代大 batch、清理其他 GPU 进程 |
| loss nan | learning rate 太大；fp16 溢出；数据异常；梯度爆炸 | 看 nan 出现的 step、lr、是否 fp16；调 `log_interval=1` | 降低 `learning_rate`，用 `bfloat16`，保留 `grad_clip=1.0`，先用小 batch smoke test |
| loss 不下降 | 观察窗口太短；lr 太小；数据太少或重复；有效 batch 设置不合适 | 记录几百个 step 的移动平均，而不是看单点 | 延长观察窗口，检查 lr，适当调大 lr 或有效 batch |
| dataloader 很慢 | `num_workers` 不合适；磁盘慢；首次 datasets 解析/cache；JSONL 很大 | 看是否卡在 `Generating train split`；观察 CPU/IO | 首次等待 cache 完成；调小或调大 `num_workers`；把数据放到更快磁盘 |
| `Generating train split` 后长时间没输出 | 还没到 `log_interval`；DataLoader worker 卡住；第一个 batch 编译/初始化慢 | smoke test 设置 `--log_interval 1 --num_workers 0/2` | 调小 `log_interval`，降低 `num_workers`，先不用 `torch.compile` |
| checkpoint 没生成 | 没到 `save_interval`；非主进程不保存；输出路径理解错 | 检查 step 是否达到保存点；查 `../out` 和 `../checkpoints` | debug 时设置 `--save_interval 20`；确认从 `trainer/` 运行时路径 |
| resume 失败 | `save_weight` 不一致；hidden_size/use_moe 不一致；resume 文件不存在；运行 cwd 不一致 | 查 `../checkpoints/{save_weight}_{hidden_size}_resume.pth` 是否存在 | 保持相同 `save_weight`、`hidden_size`、`use_moe`、运行目录；加 `--from_resume 1` |
| tokenizer 路径错误 | `init_model()` 默认 `tokenizer_path='../model'`，cwd 不对 | 报错中通常包含 `../model` 找不到 | 从 `trainer/` 目录运行，或修改调用方式前先确认相对路径 |
| dataset 路径错误 | `--data_path` 相对 cwd；默认路径和本地文件名不一致 | `FileNotFoundError` 或 datasets 报 data_files 找不到 | 从 `trainer/` 运行并传 `--data_path ../dataset/pretrain_t2t_mini.jsonl` |
| 训练很慢但 GPU 利用率很低 | DataLoader/CPU/IO 成瓶颈；batch 太小；频繁日志或 checkpoint | 看 GPU-Util、CPU、磁盘 IO；调大 `log_interval` 对比 | 调整 `num_workers`，适当增大 batch，减少保存频率 |
| GPU 占用正常但日志不更新 | `log_interval` 太大；某个 step 很慢；stdout 被缓冲 | 设置 `--log_interval 1`；看 `nvidia-smi` 是否仍在跑 | smoke test 调小 log interval；用 `tee` 保存完整日志 |

## 8. 分块逐行讲解

### 8.1 import 和路径

```python
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.model_minimind import MiniMindConfig
from dataset.lm_dataset import PretrainDataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler
```

这段让脚本即使在 `trainer/` 目录下运行，也能 import 仓库根目录下的 `model`、`dataset` 和 `trainer` 包。核心训练组件不是在本文件里全部实现，而是拆到了 `model/model_minimind.py`、`dataset/lm_dataset.py`、`trainer/trainer_utils.py`。

### 8.2 `train_epoch()`

```python
for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
    input_ids = input_ids.to(args.device)
    labels = labels.to(args.device)
    lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
```

每个 batch 从 DataLoader 拿到 `input_ids` 和 `labels`，移动到 GPU，然后按全局 step 更新 lr。`iters` 是当前 epoch 的 batch 数，resume 时可能加上 skip 的 batch 数。

```python
with autocast_ctx:
    res = model(input_ids, labels=labels)
    loss = res.loss + res.aux_loss
    loss = loss / args.accumulation_steps
```

forward 在 autocast 下执行。`res.loss` 是 Causal LM cross entropy；`res.aux_loss` 主要给 MoE 用，默认非 MoE 为 0。除以 `accumulation_steps` 是为了让累积多个 micro-batch 后的梯度尺度保持合理。

```python
scaler.scale(loss).backward()
if step % args.accumulation_steps == 0:
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
```

每个 step 都 backward，但只有 step 能整除 `accumulation_steps` 时才更新参数。默认 bf16 下 `GradScaler` disabled，但调用接口仍然成立。

```python
if step % args.log_interval == 0 or step == iters:
    current_loss = loss.item() * args.accumulation_steps
    Logger(...)
```

日志打印的是乘回 `accumulation_steps` 后的 loss，便于看到真实 batch loss，而不是缩放后的反向传播 loss。

```python
if (step % args.save_interval == 0 or step == iters) and is_main_process():
    ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
    torch.save(...)
    lm_checkpoint(...)
```

保存两类文件：一个普通半精度权重文件，一个包含 optimizer/scaler/epoch/step 的 resume checkpoint。

### 8.3 main 入口

```python
parser = argparse.ArgumentParser(description="MiniMind Pretraining")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--learning_rate", type=float, default=5e-4)
parser.add_argument("--accumulation_steps", type=int, default=8)
parser.add_argument("--data_path", type=str, default="../dataset/pretrain_t2t_mini.jsonl")
args = parser.parse_args()
```

所有训练配置从这里进入。理解训练脚本时，先读参数默认值，再看每个参数在哪里被使用。

```python
lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
ckp_data = lm_checkpoint(...) if args.from_resume == 1 else None
```

这里只显式传了三个模型配置项，其余如 vocab、head 数、KV head 数、dropout、RoPE 等都来自 `MiniMindConfig` 默认值。

```python
model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
```

训练三件套在这里完成：模型/tokenizer、dataset、optimizer。`PretrainDataset` 会触发 `Generating train split`。

```python
if ckp_data:
    model.load_state_dict(ckp_data['model'])
    optimizer.load_state_dict(ckp_data['optimizer'])
    scaler.load_state_dict(ckp_data['scaler'])
    start_epoch = ckp_data['epoch']
    start_step = ckp_data.get('step', 0)
```

resume 恢复的是完整训练状态，不只是模型参数。这样学习率、optimizer 动量、scaler 状态、epoch/step 都能延续。

```python
for epoch in range(start_epoch, args.epochs):
    setup_seed(42 + epoch)
    indices = torch.randperm(len(train_ds)).tolist()
    skip = start_step if (epoch == start_epoch and start_step > 0) else 0
    batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
    loader = DataLoader(...)
    train_epoch(...)
```

每个 epoch 重新 shuffle 数据。resume 时用 `skip` 跳过已经训练过的 batch，然后继续训练。

## 9. 我需要继续追问的问题

- `get_lr()` 当前没有 warmup，这对从零 pretrain 是否足够稳定？是否需要加 warmup 做对比实验？
- `AdamW` 当前使用 PyTorch 默认 betas 和 weight_decay，MiniMind 作者是否有推荐值？
- `max_seq_len=340` 和 `pretrain_t2t_mini.jsonl` 的 token 长度分布是否匹配？是否需要统计截断比例？
- 默认 bf16 在 3090 上的稳定性和速度是否比 fp16 更好？是否需要做同参数 A/B smoke test？
- `num_workers=8` 在当前机器上是否最优？是否需要记录 DataLoader 吞吐和 GPU 利用率？
- resume 目前 checkpoint 目录写死为 `../checkpoints`，是否需要后续做实验目录隔离？
- 训练中保存的普通权重是 half CPU state_dict，是否会影响后续从该权重继续训练的精度？
