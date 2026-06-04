# -*- coding: utf-8 -*-
"""데이터셋(JSONL) → OpenAI Fine-tuning chat 포맷으로 변환.

- 그림 의존 문항(has_figure=True)은 텍스트만으로 풀 수 없으므로 제외.
- 학습/검증 세트로 분할하여 data/finetune_train.jsonl, data/finetune_val.jsonl 생성.
- 각 예시는 {"messages": [system, user, assistant]} 형태이며,
  assistant 타깃은 정답 번호 한 글자(예: "4").

사용법:
    python training/prepare_data.py
"""
from __future__ import annotations
import json
import random

from config import (
    FULL_DATASET, DATA_DIR, FINETUNE_TRAIN, FINETUNE_VAL,
    SYSTEM_PROMPT, ANSWER_INSTRUCTION, VAL_RATIO, RANDOM_SEED,
)


def load_rows() -> list[dict]:
    rows = []
    with open(FULL_DATASET, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def format_user_prompt(row: dict) -> str:
    """문제 + 보기를 사용자 메시지 텍스트로 구성."""
    lines = [row["question"].strip(), ""]
    for i, opt in enumerate(row["options"], start=1):
        lines.append(f"{i}. {opt.strip()}")
    lines.append("")
    lines.append(ANSWER_INSTRUCTION)
    return "\n".join(lines)


def to_chat_example(row: dict) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": format_user_prompt(row)},
            {"role": "assistant", "content": str(row["answer"])},
        ]
    }


def main() -> None:
    rows = load_rows()
    usable = [r for r in rows if not r.get("has_figure", False)]
    skipped = len(rows) - len(usable)

    random.seed(RANDOM_SEED)
    random.shuffle(usable)

    n_val = max(1, int(len(usable) * VAL_RATIO))
    val_rows = usable[:n_val]
    train_rows = usable[n_val:]

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for path, subset in [(FINETUNE_TRAIN, train_rows), (FINETUNE_VAL, val_rows)]:
        with open(path, "w", encoding="utf-8") as f:
            for r in subset:
                f.write(json.dumps(to_chat_example(r), ensure_ascii=False) + "\n")

    print(f"전체 {len(rows)}문항 중 그림문항 {skipped}개 제외 → 사용 {len(usable)}개")
    print(f"  학습: {len(train_rows)}개 → {FINETUNE_TRAIN}")
    print(f"  검증: {len(val_rows)}개 → {FINETUNE_VAL}")


if __name__ == "__main__":
    main()
