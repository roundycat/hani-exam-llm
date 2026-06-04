# -*- coding: utf-8 -*-
"""모델 평가 — 한의학 문제를 풀게 해서 정답률(전체/과목별)을 비교.

- 베이스 모델(GPT-4, GPT-5)과 파인튜닝 모델을 동일 기준으로 평가.
- 그림 문항(has_figure=True)은 제외.
- 기본 평가셋은 검증 분할(data/finetune_val.jsonl 에 쓰인 것과 동일한 시드)이며,
  --all 로 전체 텍스트 문항을 평가할 수 있다.

사용법:
    python training/evaluate.py                 # 검증셋, .env 의 모든 모델
    python training/evaluate.py --all           # 전체 텍스트 문항
    python training/evaluate.py --models gpt-4o-2024-08-06 ft:gpt-4o-...:hani-exam:...
"""
from __future__ import annotations
import argparse
import json
import re
import random
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

from config import (
    FULL_DATASET, RESULTS_DIR, SYSTEM_PROMPT, ANSWER_INSTRUCTION,
    BASE_MODELS, GPT4_FINETUNED_MODEL, GPT5_FINETUNED_MODEL,
    VAL_RATIO, RANDOM_SEED, EVAL_CONCURRENCY, EVAL_MAX_TOKENS,
    EVAL_MAX_TOKENS_REASONING, EVAL_REASONING_EFFORT,
    REQUEST_TIMEOUT, is_reasoning_model,
)

client = OpenAI()

CIRCLED = {"①": 1, "②": 2, "③": 3, "④": 4, "⑤": 5}


def load_rows() -> list[dict]:
    rows = []
    with open(FULL_DATASET, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return [r for r in rows if not r.get("has_figure", False)]


def select_eval_rows(use_all: bool) -> list[dict]:
    rows = load_rows()
    if use_all:
        return rows
    # prepare_data 와 동일한 셔플/시드로 검증 분할을 재현
    random.seed(RANDOM_SEED)
    random.shuffle(rows)
    n_val = max(1, int(len(rows) * VAL_RATIO))
    return rows[:n_val]


def build_prompt(row: dict) -> str:
    lines = [row["question"].strip(), ""]
    for i, opt in enumerate(row["options"], start=1):
        lines.append(f"{i}. {opt.strip()}")
    lines.append("")
    lines.append(ANSWER_INSTRUCTION)
    return "\n".join(lines)


def parse_answer(text: str) -> int | None:
    """모델 출력에서 정답 번호(1~5)를 견고하게 추출.

    우선순위:
      1) "정답: N" / "정답 N" 패턴 (해설형/CoT 출력 대응) → 가장 신뢰도 높음
      2) 동그라미 숫자(①~⑤)
      3) 마지막에 등장하는 1~5 숫자 (해설 본문의 숫자에 휘둘리지 않도록 '마지막'을 택함)
    """
    if not text:
        return None
    # 1) 명시적 "정답" 표기
    m = re.search(r"정답\s*[:：]?\s*([1-5])", text)
    if m:
        return int(m.group(1))
    # 2) 동그라미 숫자
    for ch, num in CIRCLED.items():
        if ch in text:
            return num
    # 3) 마지막 1~5 숫자
    nums = re.findall(r"[1-5]", text)
    return int(nums[-1]) if nums else None


def ask(model_id: str, row: dict) -> int | None:
    """모델에게 한 문항을 물어 정답 번호(1~5)를 받아 파싱."""
    reasoning = is_reasoning_model(model_id)
    kwargs = dict(
        model=model_id,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(row)},
        ],
        # 추론형은 추론 토큰까지 포함되므로 예산을 넉넉히 준다.
        max_completion_tokens=EVAL_MAX_TOKENS_REASONING if reasoning else EVAL_MAX_TOKENS,
        timeout=REQUEST_TIMEOUT,
    )
    if reasoning:
        # 추론 강도 지정(비용 절감). 미지원 모델이면 제거 후 재시도.
        kwargs["reasoning_effort"] = EVAL_REASONING_EFFORT
    else:
        kwargs["temperature"] = 0  # 재현성

    try:
        resp = client.chat.completions.create(**kwargs)
    except Exception:
        # reasoning_effort 등 일부 파라미터를 모델이 거부하면 빼고 한 번 더 시도
        kwargs.pop("reasoning_effort", None)
        resp = client.chat.completions.create(**kwargs)
    return parse_answer(resp.choices[0].message.content or "")


def evaluate_model(model_id: str, rows: list[dict]) -> dict:
    print(f"\n=== 평가: {model_id} ({len(rows)}문항) ===")
    correct = 0
    by_subject = defaultdict(lambda: [0, 0])  # 과목 -> [맞음, 전체]
    details = []

    def worker(idx_row):
        idx, row = idx_row
        for attempt in range(3):
            try:
                pred = ask(model_id, row)
                return idx, pred, None
            except Exception as e:  # noqa: BLE001
                if attempt == 2:
                    return idx, None, str(e)
                time.sleep(2 * (attempt + 1))

    with ThreadPoolExecutor(max_workers=EVAL_CONCURRENCY) as ex:
        futures = [ex.submit(worker, (i, r)) for i, r in enumerate(rows)]
        done = 0
        for fut in as_completed(futures):
            idx, pred, err = fut.result()
            row = rows[idx]
            gold = row["answer"]
            ok = (pred == gold)
            correct += int(ok)
            subj = row.get("과목", "미상")
            by_subject[subj][0] += int(ok)
            by_subject[subj][1] += 1
            details.append({
                "source": row.get("source"), "과목": subj,
                "번호": row.get("번호"), "gold": gold,
                "pred": pred, "correct": ok, "error": err,
            })
            done += 1
            if done % 25 == 0 or done == len(rows):
                print(f"  진행 {done}/{len(rows)} | 누적 정답 {correct}")

    acc = correct / len(rows) if rows else 0.0
    subj_acc = {s: round(v[0] / v[1], 4) for s, v in sorted(by_subject.items())}
    return {
        "model": model_id, "n": len(rows), "correct": correct,
        "accuracy": round(acc, 4), "by_subject": subj_acc, "details": details,
    }


def resolve_models(cli_models: list[str] | None) -> list[str]:
    if cli_models:
        return cli_models
    models = [BASE_MODELS["gpt-4"], BASE_MODELS["gpt-5"]]
    if GPT4_FINETUNED_MODEL:
        models.append(GPT4_FINETUNED_MODEL)
    if GPT5_FINETUNED_MODEL:
        models.append(GPT5_FINETUNED_MODEL)
    # 중복 제거(순서 유지)
    seen, out = set(), []
    for m in models:
        if m and m not in seen:
            seen.add(m)
            out.append(m)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="전체 텍스트 문항 평가")
    ap.add_argument("--models", nargs="*", help="평가할 모델 ID 목록(직접 지정)")
    args = ap.parse_args()

    rows = select_eval_rows(args.all)
    models = resolve_models(args.models)
    print(f"평가셋 {len(rows)}문항 | 대상 모델: {models}")

    summary = []
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for mid in models:
        try:
            res = evaluate_model(mid, rows)
        except Exception as e:  # noqa: BLE001
            print(f"[오류] {mid}: {e}")
            continue
        summary.append(res)
        safe = mid.replace("/", "_").replace(":", "_")
        with open(RESULTS_DIR / f"eval_{safe}.json", "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)

    # 비교 표
    print("\n================ 정답률 요약 ================")
    print(f"{'모델':45s} {'정답률':>8s}  {'맞음/전체'}")
    for r in summary:
        print(f"{r['model']:45s} {r['accuracy']*100:7.2f}%  {r['correct']}/{r['n']}")

    with open(RESULTS_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump([{k: v for k, v in r.items() if k != "details"}
                   for r in summary], f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
