"""数据处理: Dataset 和 DataLoader 演示脚本

演示内容:
1. 自定义 PretrainDataset / SFTDataset，展示 loss mask 的核心区别
2. 自定义 collate_fn 处理变长序列
3. DataLoader 输出 batch，验证 shape 与模型 forward 对齐
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


# ============================================================
# 0. 准备示例数据（模拟 JSONL 中的若干条样本）
# ============================================================

# 预训练数据: 纯文本
PRETRAIN_SAMPLES = [
    {"text": "人工智能是计算机科学的一个分支，它企图了解智能的实质。"},
    {"text": "深度学习是机器学习的子领域，基于人工神经网络的研究。"},
    {"text": "自然语言处理是人工智能和语言学领域的分支学科。"},
    {"text": "PyTorch是一个开源的Python机器学习库，基于Torch。"},
    {"text": "Transformer模型由Google团队在2017年提出。"},
    {"text": "GPU"},
    {"text": "大规模语言模型（Large Language Model，简称LLM）是一种基于深度学习的自然语言处理模型，它通过在海量文本数据上进行预训练，学习语言的统计规律和语义知识。"},
]

# SFT 数据: 多轮对话
SFT_SAMPLES = [
    {
        "conversations": [
            {"role": "user", "content": "1+1等于几？"},
            {"role": "assistant", "content": "1+1等于2。"},
        ]
    },
    {
        "conversations": [
            {"role": "system", "content": "你是数学助手。"},
            {"role": "user", "content": "2×3=?"},
            {"role": "assistant", "content": "2×3=6。"},
        ]
    },
    {
        "conversations": [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好！有什么可以帮你的吗？"},
            {"role": "user", "content": "今天天气怎么样？"},
            {"role": "assistant", "content": "抱歉，我无法获取实时天气信息。"},
        ]
    },
]


# ============================================================
# 1. PretrainDataset — 所有 token 参与 loss（除 padding）
# ============================================================

# class PretrainDataset(Dataset):
#     """预训练数据集: 纯文本 -> tokenize -> 所有非 padding 位置计算 loss"""

#     def __init__(self, samples: list[dict], tokenizer, max_length: int = 64):
#         self.samples = samples
#         self.tokenizer = tokenizer
#         self.max_length = max_length

#     def __len__(self) -> int:
#         return len(self.samples)

#     def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
#         text = self.samples[index]["text"]

#         # tokenize，留 2 个位置给 BOS 和 EOS
#         tokens = self.tokenizer(
#             str(text),
#             add_special_tokens=False,
#             max_length=self.max_length - 2,
#             truncation=True,
#         ).input_ids

#         # 首尾加特殊 token
#         tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]

#         # 右侧 padding 到 max_length
#         pad_len = self.max_length - len(tokens)
#         input_ids = tokens + [self.tokenizer.pad_token_id] * pad_len
#         input_ids = torch.tensor(input_ids, dtype=torch.long)

#         # labels: 非 padding 位置保留原值，padding 位置设 -100（忽略）
#         labels = input_ids.clone()
#         labels[labels == self.tokenizer.pad_token_id] = -100

#         return input_ids, labels



# ============================================================
# 2. SFTDataset — 仅 assistant 回复部分参与 loss
# ============================================================

class SFTDataset(Dataset):
    """SFT 数据集: 多轮对话 -> chat template -> 仅 assistant 部分 loss"""

    def __init__(self, samples: list[dict], tokenizer, max_length: int = 128):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

        # 预计算 assistant 起止标记的 token 序列（用于滑动窗口匹配）
        self.bos_id = tokenizer(
            f"{tokenizer.bos_token}assistant\n", add_special_tokens=False
        ).input_ids
        self.eos_id = tokenizer(
            f"{tokenizer.eos_token}\n", add_special_tokens=False
        ).input_ids

    def __len__(self) -> int:
        return len(self.samples)

    def _generate_labels(self, input_ids: list[int]) -> list[int]:
        """生成 labels: 仅 assistant 回复部分保留 token id，其余为 -100"""
        labels = [-100] * len(input_ids)
        i = 0
        while i < len(input_ids):
            # 滑动窗口匹配 <|im_start|>assistant\n
            if input_ids[i : i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                # 找到对应的 <|im_end|>\n
                while end < len(input_ids):
                    if input_ids[end : end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                # assistant 内容（含 eos）设为真实 token id -> 参与 loss
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    labels[j] = input_ids[j]
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return labels

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        conversations = self.samples[index]["conversations"]

        # 应用 chat template 渲染对话
        prompt = self.tokenizer.apply_chat_template(
            conversations, tokenize=False, add_generation_prompt=False
        )

        # tokenize + 截断 + padding
        input_ids = self.tokenizer(prompt).input_ids[: self.max_length]
        pad_len = self.max_length - len(input_ids)
        input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len

        # 生成 labels（仅 assistant 部分）
        labels = self._generate_labels(input_ids)

        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(
            labels, dtype=torch.long
        )


# ============================================================
# 3. collate_fn — 将 list[tuple] 整合成 batch tensor
# ============================================================

def collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    """将多个 (input_ids, labels) 样本堆叠为一个 batch"""
    input_ids_list, labels_list = zip(*batch)
    input_ids_batch = torch.stack(input_ids_list)  # (B, L)
    labels_batch = torch.stack(labels_list)          # (B, L)
    return input_ids_batch, labels_batch


# ============================================================
# 4. 可视化工具函数
# ============================================================

def print_sample_detail(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    tokenizer,
    title: str,
) -> None:
    """逐 token 打印 input_ids 与 labels 的对齐情况"""
    ids = input_ids.tolist()
    lbs = labels.tolist()

    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    for i in range(len(ids)):
        tok = repr(tokenizer.decode([ids[i]]))
        if lbs[i] == -100:
            lbl_str = "-100 (ignore)"
        else:
            lbl_str = repr(tokenizer.decode([lbs[i]]))
        marker = " <-- LOSS" if lbs[i] != -100 else ""
        print(f"  [{i:3d}] id={ids[i]:>5d}  {tok:18s}  label={lbl_str}{marker}")

    # 统计
    total = len(lbs)
    loss_count = sum(1 for l in lbs if l != -100)
    pad_count = sum(1 for id_ in ids if id_ == tokenizer.pad_token_id)
    print(f"  --- 总 token: {total}, 参与 loss: {loss_count}, padding: {pad_count} ---")


# ============================================================
# 5. 主流程
# ============================================================

def main():
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained("./model")
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    print(f"BOS={repr(tokenizer.bos_token)}  EOS={repr(tokenizer.eos_token)}  PAD={repr(tokenizer.pad_token)}")

    # --------------------------------------------------
    # 5.1 预训练 Dataset + DataLoader
    # --------------------------------------------------
    print("\n" + "#" * 60)
    print("# Part A: PretrainDataset")
    print("#" * 60)

    pretrain_ds = PretrainDataset(PRETRAIN_SAMPLES, tokenizer, max_length=64)
    pretrain_loader = DataLoader(
        pretrain_ds,
        batch_size=3,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    print(f"\nDataset size: {len(pretrain_ds)}")
    print(f"DataLoader batches: {len(pretrain_loader)}")

    # 取第一个 batch
    input_ids_batch, labels_batch = next(iter(pretrain_loader))
    print(f"\nbatch input_ids shape: {input_ids_batch.shape}  # (B, L)")
    print(f"batch labels shape:    {labels_batch.shape}  # (B, L)")

    # 展示第 1 个样本的 detail
    print_sample_detail(input_ids_batch[0], labels_batch[0], tokenizer, title="Pretrain 样本 0")

    # 验证: 预训练中所有非 padding 位置都参与 loss
    sample_labels = labels_batch[0]
    non_pad_mask = input_ids_batch[0] != tokenizer.pad_token_id
    assert (sample_labels[non_pad_mask] != -100).all(), "预训练: 非 padding 位置应全部参与 loss"
    assert (sample_labels[~non_pad_mask] == -100).all(), "预训练: padding 位置应设为 -100"
    print("\n  [PASS] 预训练 loss mask 验证通过: 非 padding 全部参与 loss")

    # --------------------------------------------------
    # 5.2 SFT Dataset + DataLoader
    # --------------------------------------------------
    print("\n" + "#" * 60)
    print("# Part B: SFTDataset")
    print("#" * 60)

    sft_ds = SFTDataset(SFT_SAMPLES, tokenizer, max_length=128)
    sft_loader = DataLoader(
        sft_ds,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    print(f"\nDataset size: {len(sft_ds)}")
    print(f"DataLoader batches: {len(sft_loader)}")

    input_ids_batch, labels_batch = next(iter(sft_loader))
    print(f"\nbatch input_ids shape: {input_ids_batch.shape}  # (B, L)")
    print(f"batch labels shape:    {labels_batch.shape}  # (B, L)")

    # 展示 SFT 样本的 detail
    print_sample_detail(input_ids_batch[0], labels_batch[0], tokenizer, title="SFT 样本 0 (单轮对话)")
    print_sample_detail(input_ids_batch[1], labels_batch[1], tokenizer, title="SFT 样本 1 (带 system)")

    # 验证: SFT 中只有 assistant 部分参与 loss
    for idx in range(input_ids_batch.shape[0]):
        labels = labels_batch[idx]
        input_ids = input_ids_batch[idx]
        non_ignore = labels != -100
        assert non_ignore.any(), f"SFT 样本 {idx}: 应有 assistant 部分参与 loss"
        pad_mask = input_ids == tokenizer.pad_token_id
        assert (labels[pad_mask] == -100).all(), f"SFT 样本 {idx}: padding 不应参与 loss"
    print("\n  [PASS] SFT loss mask 验证通过: 仅 assistant 回复部分参与 loss")

    # --------------------------------------------------
    # 5.3 对比总结
    # --------------------------------------------------
    print("\n" + "=" * 60)
    print("  预训练 vs SFT loss mask 对比")
    print("=" * 60)

    _, pt_labels = pretrain_ds[0]
    _, sft_labels = sft_ds[0]

    pt_loss_ratio = (pt_labels != -100).sum().item() / len(pt_labels) * 100
    sft_loss_ratio = (sft_labels != -100).sum().item() / len(sft_labels) * 100

    print(f"  预训练样本 0: {pt_labels.shape[0]} tokens, {pt_loss_ratio:.1f}% 参与 loss")
    print(f"  SFT 样本 0:   {sft_labels.shape[0]} tokens, {sft_loss_ratio:.1f}% 参与 loss")
    print(f"\n  预训练 -> 几乎所有 token 都学（学习语言规律）")
    print(f"  SFT -> 只学 assistant 回复（学习如何回答）")


if __name__ == "__main__":
    main()
