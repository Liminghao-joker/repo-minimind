# SFT

## 1. Core Problem

1. SFT 和 pretrain 的目标到底差在哪？
2. 为什么 prompt 部分通常不算 loss?
3. 为什么称 SFT 是从 base model 走向 chat / instruct model 的第一座桥。  


## 2. Conclusions

- SFT 的目标不是继续给模型灌通用知识，而是让模型学会按指令、按对话格式输出。
- 和 pretrain 最大的区别之一，是 SFT 通常只让 assistant 回复区间参与 loss。
>  prompt 部分如果也算 loss，模型会更倾向于“复述输入模板”，而不是学会生成回答。
- MiniMind 中 SFT 数据流是：`JSONL -> tokenizer -> padding/truncation -> loss mask -> DataLoader`。
- SFT 后面常接偏好优化，也就是 [DPO](./06_dpo.md) / RLHF 这一阶段。

## 3. Intuition

从训练目标看：

- pretrain 更像在学“语言本身”和“文本续写能力”；
- SFT 更像在学“收到这种输入时，应该怎样回答”。

所以这两步虽然都还是 causal LM 训练，但监督信号的重点已经不一样了。

通常 base model 已经能续写，但不一定会按对话格式答题。而 SFT 用“指令-回答”形式的数据，把模型往 chat / instruct 方向拉，这一步并不一定大量增加知识，但会明显影响模型输出的风格、结构。


## 4. MiniMind Implementation View

### 4.1 Dataflow


```text
JSONL 文件
-> 逐行读取 JSON
-> tokenizer 编码
-> padding / truncation
-> 构造 loss mask
-> DataLoader 批量输出
```

### 4.2 和 pretrain 的关键区别

结合 [Train Step](./04_train_step.md)，区别可概括成下面这张表：

| 维度 | Pretrain | SFT |
| --- | --- | --- |
| 目标 | 学 next-token distribution | 学指令跟随 / 对话输出 |
| 监督范围 | 大部分 token 都参与 | 通常只让 assistant 回复区间参与 |
| 数据形式 | 大规模通用文本 | 指令 / 问答 / 多轮对话 |
| 输出能力 | 更偏续写 | 更偏回答和格式对齐 |

### 4.3 loss mask

在工程实现上，它通常会写成：

- 需要学习的回复 token 对应正常 label；
- 不参与 loss 的位置改成 `-100`。

这一点和 [Train Step](./04_train_step.md) 是直接相通的，因为本质上还是同一个交叉熵，只是监督范围缩小了。
## 5. Training tricks
- 选择较小的学习率：SFT 学习率通常是预训练的 1/10 到 1/5
- 控制训练轮数：通常 1-3 个 epoch，防止过拟合
- 使用 LoRA
- 引入早停（early-stop）机制
- 适当使用 weight decay 和 正则化
> SFT 数据量的选择：如果只做格式对齐，几千到几万条**高质量**数据就已足够，但若需注入知识（mid-training），则需要更多的数据。在 MiniMind 项目中，使用了 14 GB 的全量 SFT 数据。

## 6. Questions

1. SFT 的数据流？它和 pretrain 最关键的差别在哪里，数据处理上有何不同？
2. 为什么可以说 SFT 是从 base model 走向 chat / instruct model 的第一座桥？
3. SFT 中为什么通常不让 prompt 部分参与 loss？如果把 prompt 也纳入监督，会带来什么副作用？
> 因为我们并不希望模型把“用户提问模板”也当成自己需要生成的目标。  
真正希望它学的是 assistant 回复区间，而不是把整段对话原封不动地复述出来。
4. SFT 和 [Train Step](./04_train_step.md) 里的 pretrain train step，在 loss 公式上和 label / mask 构造上分别有哪些异同？
5. 为什么说 SFT 主要在调整“输出方式和任务服从性”，而不一定等价于继续灌输通用知识？
6. SFT 和 prompt engineering 的区别是什么？为什么一个属于训练阶段，一个属于推理阶段？
7. 结合你目前的材料，SFT 这一篇已经确认了哪些内容，还有哪些部分仍然属于待补充？

## 7. Review

- SFT 在训练链路里的位置，不是继续做一轮通用预训练，而是把 base model 往 chat / instruct model 的方向推进。
- 它和 pretrain 都还属于 causal LM 框架，但真正不同的是“哪些 token 被当成监督信号”。
- pretrain 更像在学一般文本的 next-token distribution，SFT 更像在学“这种输入格式下应该怎样回答”。
- SFT 最关键的工程点不在 loss 公式本身，而在 label / loss mask 的构造方式。
- 通常只让 assistant 回复区间参与 loss，是为了让模型学会生成回答，而不是把用户提示词和模板也一并复述出来。
- 当前你已经整理清楚的一条主线是：`JSONL -> tokenizer -> padding/truncation -> loss mask -> DataLoader`。
- 这部分目前更像“训练机制和数据流的整理版”，还不是完整源码复盘版；哪些地方已经确认、哪些地方待补充要分开说。
- SFT 结束后，模型已经更会“按要求回答”，但还没有解决“多个回答里哪个更好”的问题，这也是它自然衔接到 [DPO](./06_dpo.md) 的原因。

## Related Notes

- [Train Step](./04_train_step.md)
- [DPO](./06_dpo.md)
- [LoRA](./07_lora.md)
