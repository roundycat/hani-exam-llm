# -*- coding: utf-8 -*-
"""내과학2 RAG 결합·측정. RAG(웹근거) 자기일관성 vs 베이스라인/no-RAG 앙상블, RAG+앙상블.

입력:
  eval/naegwa2_rag.json   {votes:{nidx:[pred...]}, evidence:{...}}
  eval/naegwa_gpt5_sc.json, eval/naegwa_claude_sc.json (no-RAG 모델별 votes)
  eval/naegwa_questions.jsonl (gold)
출력: results/naegwa2_rag.json + 표
"""
import json
from collections import Counter

def wilson(c, n, z=1.96):
    if n == 0: return (0, 0)
    p = c / n; d = 1 + z*z/n; ctr = (p + z*z/(2*n))/d
    h = z*((p*(1-p)/n + z*z/(4*n*n))**0.5)/d
    return round((ctr-h)*100, 1), round((ctr+h)*100, 1)

def maj(v):
    v = [x for x in (v or []) if x]
    return Counter(v).most_common(1)[0][0] if v else None

rows = [json.loads(l) for l in open("eval/naegwa_questions.jsonl", encoding="utf-8") if l.strip()]
n2 = [r for r in rows if r["과목"] == "내과학2"]
gold = {r["nidx"]: r["answer"] for r in n2}

rag = json.load(open("eval/naegwa2_rag.json", encoding="utf-8"))
rv = {int(k): v for k, v in rag["votes"].items()}

g5 = json.load(open("eval/naegwa_gpt5_sc.json", encoding="utf-8")); g5 = {int(k): v for k, v in g5.items()}
cl = json.load(open("eval/naegwa_claude_sc.json", encoding="utf-8"))
op = {int(k): v for k, v in cl["opus"].items()}; so = {int(k): v for k, v in cl["sonnet"].items()}

rag_maj = {n: maj(rv.get(n)) for n in gold}
g5_maj = {n: maj(g5.get(n, {}).get("votes")) for n in gold}
op_maj = {n: maj(op.get(n)) for n in gold}
so_maj = {n: maj(so.get(n)) for n in gold}

# 베이스라인(GPT-5 단일패스)
import glob
base = {}
d = json.load(open("results/eval_gpt-5.json", encoding="utf-8"))
for x in d.get("details", []):
    if x.get("과목") == "내과학2":
        base[(x["source"], x["번호"])] = x["pred"]
rk = {r["nidx"]: (r["source"], r["번호"]) for r in n2}

def acc(pf):
    ok = sum(1 for n in gold if pf(n) == gold[n]); return ok, len(gold)

def vote(cands, tiebreak):
    cands = [x for x in cands if x]
    if not cands: return None
    cnt = Counter(cands); top, c = cnt.most_common(1)[0]
    if c >= 2: return top
    for t in tiebreak:
        if t: return t
    return cands[0]

methods = {
    "GPT-5 단일(베이스라인)": lambda n: base.get(rk[n]),
    "no-RAG 앙상블(3모델)": lambda n: vote([g5_maj[n], op_maj[n], so_maj[n]], [g5_maj[n]]),
    "RAG (Opus 웹근거 자기일관성)": lambda n: rag_maj[n],
    "RAG + 앙상블(RAG+3모델 투표)": lambda n: vote([rag_maj[n], g5_maj[n], op_maj[n], so_maj[n]], [rag_maj[n], g5_maj[n]]),
}

print(f"{'방법':28s} {'내과학2 30':>14s} {'95% CI':>14s}")
print("-" * 60)
out = {}
for name, fn in methods.items():
    ok, n = acc(fn); lo, hi = wilson(ok, n)
    out[name] = {"correct": ok, "n": n, "acc": round(ok/n*100, 1), "ci": [lo, hi]}
    star = "**" if name.startswith("RAG +") else ""
    print(f"{name:28s} {star}{ok}/{n}={ok/n*100:5.1f}%{star}   {lo}-{hi}%")

json.dump(out, open("results/naegwa2_rag.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("\n저장: results/naegwa2_rag.json")
