# -*- coding: utf-8 -*-
"""예측 + gold → 정답률 채점 (전체/회차별/과목별/교시별).

입력
----
- eval/gold.json                : {idx: {answer, 과목, source, 교시, 번호}}
- eval/predictions.json         : {model_name: {idx(str): pred(int|null)}}

출력
----
- results/eval_<model>.json     : 모델별 상세(문항별 정오 포함)
- results/summary.json          : 모델 비교 요약
- 표준출력 비교표

채점 기준은 repo training/evaluate.py 와 동일: pred(1~5) == gold.answer.
"""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
GOLD = ROOT / "gold.json"
PRED = ROOT / "predictions.json"
RESULTS = REPO / "results"


def wilson_ci(correct: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """이항비율 95% 신뢰구간(Wilson). n=0이면 (0,0)."""
    if n == 0:
        return (0.0, 0.0)
    p = correct / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def grade_model(model: str, preds: dict, gold: dict) -> dict:
    correct = 0
    answered = 0
    missing = 0
    by_subject = defaultdict(lambda: [0, 0])
    by_source = defaultdict(lambda: [0, 0])
    by_gyo = defaultdict(lambda: [0, 0])
    details = []

    for idx_str, g in gold.items():
        pred = preds.get(idx_str, None)
        gold_ans = g["answer"]
        if pred is None:
            missing += 1
            ok = False
        else:
            answered += 1
            ok = (int(pred) == gold_ans)
        correct += int(ok)
        subj = g.get("과목", "미상")
        src = g.get("source", "미상")
        gyo = f'{g.get("source","")}_{g.get("교시","")}교시'
        by_subject[subj][0] += int(ok); by_subject[subj][1] += 1
        by_source[src][0] += int(ok); by_source[src][1] += 1
        by_gyo[gyo][0] += int(ok); by_gyo[gyo][1] += 1
        details.append({
            "idx": int(idx_str), "source": src, "과목": subj,
            "번호": g.get("번호"), "gold": gold_ans, "pred": pred, "correct": ok,
        })

    n = len(gold)
    acc = correct / n if n else 0.0
    lo, hi = wilson_ci(correct, n)

    def fmt(d):
        return {k: {"correct": v[0], "n": v[1], "acc": round(v[0] / v[1], 4) if v[1] else 0.0}
                for k, v in sorted(d.items())}

    return {
        "model": model,
        "n": n,
        "answered": answered,
        "missing": missing,
        "correct": correct,
        "accuracy": round(acc, 4),
        "acc_ci95": [round(lo, 4), round(hi, 4)],
        "by_source": fmt(by_source),
        "by_gyo": fmt(by_gyo),
        "by_subject": fmt(by_subject),
        "details": details,
    }


def main() -> None:
    gold = json.loads(GOLD.read_text(encoding="utf-8"))
    preds_all = json.loads(PRED.read_text(encoding="utf-8"))
    RESULTS.mkdir(parents=True, exist_ok=True)

    summary = []
    for model, preds in preds_all.items():
        res = grade_model(model, preds, gold)
        summary.append(res)
        safe = model.replace("/", "_").replace(":", "_")
        (RESULTS / f"eval_{safe}.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    # 정답률 내림차순
    summary.sort(key=lambda r: r["accuracy"], reverse=True)

    print(f"\n{'='*64}")
    print(f"한의학 국가시험 정답률 (전체 {len(gold)}문항, 랜덤 베이스라인 20%)")
    print(f"{'='*64}")
    print(f"{'모델':22s} {'정답률':>8s}  {'95% CI':>16s}  {'맞음/전체':>10s}  {'미응답':>5s}")
    for r in summary:
        ci = f'[{r["acc_ci95"][0]*100:.1f},{r["acc_ci95"][1]*100:.1f}]'
        print(f'{r["model"]:22s} {r["accuracy"]*100:7.2f}%  {ci:>16s}  '
              f'{r["correct"]:>4d}/{r["n"]:<4d}  {r["missing"]:>5d}')

    # 회차별
    print(f"\n--- 회차(시험)별 정답률 ---")
    sources = sorted({s for r in summary for s in r["by_source"]})
    header = f'{"모델":22s}' + "".join(f'{s:>14s}' for s in sources)
    print(header)
    for r in summary:
        line = f'{r["model"]:22s}'
        for s in sources:
            v = r["by_source"].get(s)
            line += f'{(str(round(v["acc"]*100,1))+"%"):>14s}' if v else f'{"-":>14s}'
        print(line)

    (RESULTS / "summary.json").write_text(
        json.dumps([{k: v for k, v in r.items() if k != "details"} for r in summary],
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n결과 저장: {RESULTS}")


if __name__ == "__main__":
    main()
