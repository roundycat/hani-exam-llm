# -*- coding: utf-8 -*-
"""내과학 앙상블 결합·측정.

입력:
  - eval/naegwa_gpt5_sc.json   {nidx: {votes, majority, gold, 과목}}
  - eval/naegwa_claude_sc.json {opus:{nidx:[votes]}, sonnet:{nidx:[votes]}}
  - eval/naegwa_questions.jsonl (gold/과목)
전략: 모델별 다수결 → (A) 3모델 투표(동률 GPT-5) / (B) 전체 샘플 풀드 다수결.
출력: results/naegwa_ensemble.json + 비교표.
"""
import json
from collections import Counter

rows = [json.loads(l) for l in open("eval/naegwa_questions.jsonl", encoding="utf-8") if l.strip()]
gold = {r["nidx"]: r["answer"] for r in rows}
subj = {r["nidx"]: r["과목"] for r in rows}
N2 = [r["nidx"] for r in rows if r["과목"] == "내과학2"]

g5 = json.load(open("eval/naegwa_gpt5_sc.json", encoding="utf-8"))
cl = json.load(open("eval/naegwa_claude_sc.json", encoding="utf-8"))
g5 = {int(k): v for k, v in g5.items()}
opus = {int(k): v for k, v in cl["opus"].items()}
sonnet = {int(k): v for k, v in cl["sonnet"].items()}

def maj(votes):
    v = [x for x in (votes or []) if x]
    return Counter(v).most_common(1)[0][0] if v else None

def acc(pred_of):
    ok = sum(1 for n in gold if pred_of(n) == gold[n])
    ok2 = sum(1 for n in N2 if pred_of(n) == gold[n])
    return ok, len(gold), ok2, len(N2)

# 모델별 자기일관성 다수결
g5_maj = {n: maj(g5.get(n, {}).get("votes")) for n in gold}
op_maj = {n: maj(opus.get(n)) for n in gold}
so_maj = {n: maj(sonnet.get(n)) for n in gold}

def ensemble_vote(n):
    cand = [g5_maj[n], op_maj[n], so_maj[n]]
    cand = [x for x in cand if x]
    if not cand: return None
    cnt = Counter(cand); top, c = cnt.most_common(1)[0]
    if c >= 2: return top
    return g5_maj[n] or op_maj[n] or so_maj[n]  # 동률 → GPT-5 우선

def ensemble_pool(n):
    allv = (g5.get(n, {}).get("votes") or []) + (opus.get(n) or []) + (sonnet.get(n) or [])
    return maj(allv)

methods = {
    "GPT-5 단일(베이스라인)": lambda n: None,  # 채움 아래
    "GPT-5 자기일관성": lambda n: g5_maj[n],
    "Opus 자기일관성": lambda n: op_maj[n],
    "Sonnet 자기일관성": lambda n: so_maj[n],
    "앙상블(3모델 투표)": ensemble_vote,
    "앙상블(전체 풀드)": ensemble_pool,
}

# 베이스라인(단일패스)은 기존 results에서
import glob
base = {}
for f in glob.glob("results/eval_gpt-5.json"):
    d = json.load(open(f, encoding="utf-8"))
    for x in d.get("details", []):
        if x.get("과목") in ("내과학1", "내과학2"):
            base[(x["source"], x["과목"], x["번호"])] = x["pred"]
row_key = {r["nidx"]: (r["source"], r["과목"], r["번호"]) for r in rows}
def base_pred(n): return base.get(row_key[n])

print(f"{'방법':22s} {'내과학 96':>14s} {'내과학2 30':>14s}")
print("-"*54)
out = {}
for name, fn in methods.items():
    pf = base_pred if name.startswith("GPT-5 단일") else fn
    ok, n, ok2, n2 = acc(pf)
    out[name] = {"내과학": [ok, n, round(ok/n*100, 1)], "내과학2": [ok2, n2, round(ok2/n2*100, 1)]}
    print(f"{name:22s} {ok}/{n}={ok/n*100:5.1f}%   {ok2}/{n2}={ok2/n2*100:5.0f}%")

json.dump(out, open("results/naegwa_ensemble.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("\n저장: results/naegwa_ensemble.json")
