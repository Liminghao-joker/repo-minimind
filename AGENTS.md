---
description: MiniMind 项目协作约束
alwaysApply: true
---

# codex.md

## 1. 项目定位

本仓库基于 MiniMind 官方项目，用于开展 **LLM 训练 / 推理流程复现、训练实操、源码理解与实验记录沉淀**。

当前阶段的核心目标不是重写 MiniMind，也不是做大规模工程化改造，而是：

> 基于 MiniMind 官方 Quick Start 与 Model Training 流程，跑通小模型训练闭环，并形成可复盘、可展示、可追问的项目记录。

优先服务于以下能力建设：

- 跑通 MiniMind 官方推理与训练主链路；
- 理解 Pretrain、SFT、LoRA、DPO 的数据格式、训练入口和输出权重；
- 掌握从 `dataset → dataloader → forward → loss → backward → optimizer → checkpoint → eval` 的完整训练链路；
- 记录 loss、显存、checkpoint、推理输出和失败排查；
- 沉淀 README、实验记录、技术笔记和后续简历 / 面试表达材料。

---

## 2. 当前优先级

### P0：训练闭环跑通

优先完成 MiniMind 官方主线：

1. Quick Start 官方模型推理；
2. 数据集准备与格式检查；
3. Pretrain 最小训练；
4. SFT 最小训练；
5. Checkpoint / Resume 验证；
6. LoRA 或 DPO 二选一跑通；
7. Eval 推理对比与实验记录。

### P1：训练过程理解

重点理解并记录：

- Pretrain 数据：`{"text": ...}`；
- SFT / LoRA 数据：`conversations`；
- DPO 数据：`chosen / rejected`；
- `input_ids`、`labels`、loss mask 的构造；
- logits 与 labels 的对齐关系；
- loss、backward、optimizer、scheduler 的位置；
- checkpoint 与最终权重的区别；
- batch size、max seq length、gradient accumulation 对显存的影响。

### P2：源码与模块深化

在训练闭环完成后，再深化：

- Attention / GQA；
- RoPE；
- KV-cache；
- FFN / SwiGLU；
- LoRA 注入位置与参数冻结；
- DPO loss 与 reference model / policy model 关系。

---

## 3. 仓库区域约定

默认区分两类区域：

### 官方基线区域

MiniMind 官方仓库原有代码，主要用于：

- 官方训练流程运行；
- 源码阅读；
- 行为基线参考；
- 与个人实验对照。

默认不要随意大改官方主链路。

### 个人实验区域

推荐用于新增内容：

- `notes/`：学习笔记、源码走读、实验总结；
- `logs/`：训练日志、报错日志；
- `outputs/`：推理输出、before/after 对比；
- `debug/`：临时检查脚本、数据预览脚本、shape 检查脚本；
- `reimpl/`：个人手写复现、最小机制验证、小实验。

如果目录不存在，可按需创建。

---

## 4. Git 与环境约定

- `master` / `main`：保持接近官方基线，不做实验性修改；
- 开发分支：用于训练调试、笔记整理、实验脚本和小范围修改；
- 优先复用已有 Python / conda / uv 环境；
- 不随意重建环境，不无理由替换依赖管理方式；
- 不为了“更优雅”主动调整整体目录结构。

涉及训练前，应先确认：

```bash
python --version
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
nvidia-smi
````

---

## 5. 默认工作方式

### 5.1 先定位，再修改

遇到任务时，先判断它属于：

* 环境配置；
* 数据准备；
* 官方训练脚本运行；
* 源码理解；
* 调试 / 报错；
* 个人实验脚本；
* 文档 / README / 简历材料。

涉及代码修改时，先说明：

1. 当前文件在项目中的作用；
2. 任务属于官方基线还是个人实验；
3. 可能问题在哪里；
4. 计划改什么；
5. 如何验证改动有效。

然后再进行最小修改。

---

### 5.2 最小改动优先

优先采用：

* 小范围 patch；
* print / assert / shape 检查；
* 最小独立脚本；
* 单 batch 验证；
* 短训练 smoke test；
* README / notes 局部补全。

避免：

* 一次修改大量文件；
* 大规模重构；
* 盲目引入新依赖；
* 在根因不清时直接重写；
* 把实验代码侵入官方主链路；
* 只增加代码，不沉淀记录。

---

### 5.3 训练任务优先保留日志

所有训练或推理任务建议使用：

```bash
python xxx.py 2>&1 | tee logs/xxx.log
```

每次训练至少记录：

* 运行命令；
* 数据集；
* batch size；
* max seq length；
* learning rate；
* dtype；
* GPU 显存；
* loss 起止变化；
* checkpoint / output 权重路径；
* 是否出现 OOM / NaN / 路径错误；
* eval 输出样例。

---

## 6. 关键脚本约定

优先围绕以下官方入口工作：

```text
eval_llm.py
trainer/train_pretrain.py
trainer/train_full_sft.py
trainer/train_lora.py
trainer/train_dpo.py
```

默认理解：

| 脚本                  | 作用                           |
| ------------------- | ---------------------------- |
| `eval_llm.py`       | 模型推理与测试                      |
| `train_pretrain.py` | Pretrain，学习基础续写能力            |
| `train_full_sft.py` | SFT，学习指令与对话格式                |
| `train_lora.py`     | LoRA，参数高效领域适配                |
| `train_dpo.py`      | DPO，基于 chosen/rejected 的偏好优化 |

数据格式默认理解：

| 阶段       | 数据格式                |
| -------- | ------------------- |
| Pretrain | `{"text": ...}`     |
| SFT      | `conversations`     |
| LoRA     | `conversations`     |
| DPO      | `chosen / rejected` |

---

## 7. 调试原则

遇到报错时，按阶段定位：

1. 环境 / 依赖；
2. 数据下载 / 路径；
3. tokenizer / dataset；
4. model 初始化；
5. forward；
6. loss；
7. backward；
8. optimizer / scheduler；
9. checkpoint / resume；
10. eval / generate。

输出排查建议时，优先采用：

```text
现象 → 可能原因 → 验证方法 → 最小修复
```

常见问题优先检查：

* `dataset/` 文件是否存在；
* jsonl 格式是否正确；
* 权重路径是否匹配；
* `--weight` / `--load_from` 是否混用；
* CUDA 是否可用；
* batch size / seq length 是否导致 OOM；
* labels 是否全为 `-100`；
* checkpoint 是否与当前模型配置一致。

---

## 8. 文档产出要求

涉及学习、训练、调试、实验时，默认同步沉淀 Markdown。

优先产出：

```text
notes/
├── setup_and_dataset.md
├── pretrain_run.md
├── sft_run.md
├── lora_or_dpo_run.md
├── memory_ablation.md
├── training_pipeline_summary.md
└── interview_qa.md
```

实验记录应包含：

* 实验目的；
* 运行命令；
* 关键参数；
* 日志摘录；
* 显存记录；
* 输出文件；
* 推理样例；
* 问题与修复；
* 本次实验能证明什么；
* 本次实验还不能证明什么。

---

## 9. 输出风格

回答或协作时，默认按以下顺序：

1. 结论；
2. 当前任务在 MiniMind 项目中的作用；
3. 涉及文件；
4. 核心原理 / 问题原因；
5. 最小执行步骤；
6. 验收标准；
7. 产出文件；
8. 常见坑。

如果是报错问题，优先给：

```text
最可能根因 → 立即验证命令 → 最小修复方案 → 修复后如何确认
```

如果是文档任务，优先给可直接保存为 Markdown 的内容。

---

## 10. 时间粒度约束

默认单次任务应能在约 1–2 小时内形成可见结果。

如果任务过大，应主动收缩为：

* 一个训练入口；
* 一个数据格式检查；
* 一个最小 smoke test；
* 一个日志分析；
* 一个 README 小节；
* 一个可运行 debug 脚本。

不要默认展开成过长计划。

---

## 11. 默认不要做的事

除非明确要求，否则不要：

* 大规模重构 MiniMind 官方代码；
* 从零重写完整 LLM 框架；
* 修改大量无关文件；
* 无理由引入新依赖；
* 擅自改变目录结构；
* 直接改 `master/main` 做实验；
* 跳过 Pretrain / SFT 直接发散到 PPO、GRPO、CISPO；
* 把第一阶段重点放到 vLLM、Web UI、OpenAI API Server；
* 夸大项目完成度；
* 只给代码不解释、不验证、不记录。

---

## 12. 项目表述约束

总结本项目时，优先采用：

> 基于 MiniMind 官方仓库，对 decoder-only LLM 的训练与推理流程进行复现与拆解，重点完成 Quick Start 推理验证、Pretrain / SFT / LoRA / DPO 等训练阶段的最小闭环实践，并围绕数据格式、训练主循环、loss、checkpoint、显存调参和推理评估沉淀实验记录与技术笔记。

避免表述为：

* “从零完整实现大模型框架”；
* “完整复现 MiniMind 所有功能”；
* “独立完成全流程工业级训练系统”；
* “完整掌握所有后训练算法”。

所有项目表述必须能被追问且能用仓库中的代码、日志、笔记或实验输出支撑。

---

## 13. 参考资料

* MiniMind 官方仓库：
  `https://github.com/jingyaogong/minimind`

* MiniMind 官方文档：
  `https://minimind.readthedocs.io/en/latest/`

* learn-minimind 导学仓库：
  `https://github.com/bcefghj/learn-minimind`
