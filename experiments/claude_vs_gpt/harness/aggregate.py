# -*- coding: utf-8 -*-
"""Claude + GPT 결과를 동일 지표로 통합 집계.

- Claude 모델: eval/predictions.json (키 'claude-*') + eval/gold.json  → idx 기반 채점
- GPT 모델   : results/eval_<model>.json (training/evaluate.py 산출) → details 기반 채점

두 경로 모두 같은 517개 텍스트 문항에서 평가되므로 직접 비교 가능.
지표: 전체 정답률 + Wilson 95% CI, 회차(source)별, 과목별.

출력
----
- results/combined_summary.json
- results/REPORT.md   (사람이 읽는 비교 리포트)
- 표준출력 통합 비교표
"""
from __future__ import annotations
import json
import glob
import os
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
RESULTS = REPO / "results"


def wilson_ci(correct: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 0.0)
    p = correct / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def stats_from_records(records):
    """records: list of (source, 과목, correct_bool)."""
    n = len(records)
    correct = sum(1 for _, _, c in records if c)
    by_source = defaultdict(lambda: [0, 0])
    by_subject = defaultdict(lambda: [0, 0])
    for src, subj, c in records:
        by_source[src][0] += int(c); by_source[src][1] += 1
        by_subject[subj][0] += int(c); by_subject[subj][1] += 1
    lo, hi = wilson_ci(correct, n)

    def fmt(d):
        return {k: {"correct": v[0], "n": v[1], "acc": round(v[0] / v[1], 4) if v[1] else 0.0}
                for k, v in sorted(d.items())}

    return {
        "n": n, "correct": correct,
        "accuracy": round(correct / n, 4) if n else 0.0,
        "acc_ci95": [round(lo, 4), round(hi, 4)],
        "by_source": fmt(by_source),
        "by_subject": fmt(by_subject),
    }


def collect_claude():
    gold_p = ROOT / "gold.json"
    pred_p = ROOT / "predictions.json"
    if not (gold_p.exists() and pred_p.exists()):
        return {}
    gold = json.loads(gold_p.read_text(encoding="utf-8"))
    preds = json.loads(pred_p.read_text(encoding="utf-8"))
    out = {}
    for model, pm in preds.items():
        records = []
        for idx_str, g in gold.items():
            pred = pm.get(idx_str)
            c = (pred is not None and int(pred) == g["answer"])
            records.append((g["source"], g.get("과목", "미상"), c))
        out[model] = stats_from_records(records)
    return out


def collect_gpt():
    out = {}
    for f in glob.glob(str(RESULTS / "eval_*.json")):
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        details = d.get("details")
        if not details or "gold" not in details[0]:
            continue
        records = []
        for x in details:
            c = bool(x.get("correct"))
            records.append((x.get("source", "미상"), x.get("과목", "미상"), c))
        out[d["model"]] = stats_from_records(records)
    return out


def main():
    models = {}
    models.update(collect_claude())
    models.update(collect_gpt())
    if not models:
        print("결과 없음.")
        return

    order = sorted(models.items(), key=lambda kv: kv[1]["accuracy"], reverse=True)
    sources = sorted({s for _, st in order for s in st["by_source"]})

    lines = []
    def P(s=""):
        print(s); lines.append(s)

    P(f"# 한의학 국가시험 LLM 정답률 비교 (전체 517 텍스트 문항)")
    P()
    P(f"- 평가셋: 그림 비의존 순수 텍스트 문항 517개 (한의사 81회 276 · 한약사 27회 241)")
    P(f"- 랜덤 베이스라인: 20% (5지선다)")
    P(f"- 채점: 예측 보기번호(1~5) == 정답. 95% CI는 Wilson 구간.")
    P()
    P("## 전체 정답률")
    P()
    P("| 모델 | 정답률 | 95% CI | 맞음/전체 |")
    P("|---|---:|:---:|---:|")
    for m, st in order:
        ci = f'{st["acc_ci95"][0]*100:.1f}–{st["acc_ci95"][1]*100:.1f}%'
        P(f'| {m} | **{st["accuracy"]*100:.2f}%** | {ci} | {st["correct"]}/{st["n"]} |')
    P()
    P("## 회차(시험)별 정답률")
    P()
    P("| 모델 | " + " | ".join(sources) + " |")
    P("|---|" + "|".join(["---:"] * len(sources)) + "|")
    for m, st in order:
        cells = []
        for s in sources:
            v = st["by_source"].get(s)
            cells.append(f'{v["acc"]*100:.1f}% ({v["correct"]}/{v["n"]})' if v else "-")
        P(f"| {m} | " + " | ".join(cells) + " |")
    P()

    # 콘솔 요약 표(정렬)
    print("\n================ 통합 비교 ================")
    print(f"{'모델':28s} {'정답률':>8s}  {'맞음/전체':>10s}")
    for m, st in order:
        print(f'{m:28s} {st["accuracy"]*100:7.2f}%  {st["correct"]:>4d}/{st["n"]:<4d}')

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "combined_summary.json").write_text(
        json.dumps({m: st for m, st in order}, ensure_ascii=False, indent=2), encoding="utf-8")
    (RESULTS / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n저장: {RESULTS/'combined_summary.json'}, {RESULTS/'REPORT.md'}")


if __name__ == "__main__":
    main()
