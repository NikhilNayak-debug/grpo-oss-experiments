from __future__ import annotations

import json
from pathlib import Path

from datasets import load_dataset


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    out_path = repo_root / "data" / "gsm8k_main.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as handle:
        dataset = load_dataset("openai/gsm8k", "main")
        index = 0
        for split_name in ("train", "test"):
            for row in dataset[split_name]:
                payload = {
                    "id": f"gsm8k_{index}",
                    "source_split": split_name,
                    "question": row["question"],
                    "answer": row["answer"],
                }
                handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
                index += 1

    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
