# debug
# 数据检查脚本,看到每个数据文件打印出前 1–2 条样本。
import json
from pathlib import Path

files = [
    "dataset/pretrain_t2t_mini.jsonl",
    "dataset/sft_t2t_mini.jsonl",
    "dataset/dpo.jsonl",
]

def preview_jsonl(path: Path, n: int = 2):
    print("=" * 100)
    print(path)
    if not path.exists():
        print("[MISSING]")
        return

    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            obj = json.loads(line)
            print(f"\n--- sample {i} ---")
            print(json.dumps(obj, ensure_ascii=False, indent=2)[:1500])
            if i + 1 >= n:
                break

for file in files:
    preview_jsonl(Path(file))