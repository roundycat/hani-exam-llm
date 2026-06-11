# -*- coding: utf-8 -*-
"""견고한 Qwen 학습 전후 평가 — base/FT 각각 전용 엔드포인트를 순차로 띄워 평가.

- 전용 엔드포인트는 반드시 endpoint.name(고유 서빙 이름)으로 호출.
- 순차 실행(동시 X)로 동시 과금 최소화. 각 단계 끝나면 즉시 삭제(finally).
- 워밍업 재시도로 전용 엔드포인트 초기 500/미준비를 흡수.
검증 51문항(held-out). 측정: 정답률(전체/과목별), 출력토큰/문항, 지연/문항.
"""
import sys, time, json, os
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
sys.path.insert(0, "training")
from dotenv import load_dotenv
load_dotenv(dotenv_path=".env")
from together import Together
from config import SYSTEM_PROMPT
import evaluate as ev

c = Together()
BASE = "Qwen/Qwen2.5-7B-Instruct"
HW = "2x_nvidia_h100_80gb_sxm"
FT = open("eval/ft_model_together.txt").read().strip()


def deploy(model, label):
    ep = c.endpoints.create(model=model, hardware=HW,
                            autoscaling={"min_replicas": 1, "max_replicas": 1},
                            display_name=label, inactive_timeout=60, state="STARTED")
    eid, name = ep.id, ep.name
    print(f"[deploy] {label} id={eid} name={name}", flush=True)
    for _ in range(150):
        st = str(getattr(c.endpoints.retrieve(eid), "state", "")).upper()
        if st in ("STARTED", "RUNNING", "READY"):
            break
        time.sleep(20)
    # 워밍업(전용 엔드포인트 초기 500/준비지연 흡수)
    for i in range(20):
        try:
            r = c.chat.completions.create(model=name,
                messages=[{"role": "user", "content": "정답은 3. 숫자만 답: ?"}],
                max_tokens=8, temperature=0)
            print(f"  warmup OK ({i}): {r.choices[0].message.content!r}", flush=True)
            return eid, name
        except Exception as e:
            print(f"  warmup {i}: {str(e)[:70]}", flush=True)
            time.sleep(15)
    print("  warmup 실패 — 그래도 평가 시도", flush=True)
    return eid, name


def ask(name, row):
    t0 = time.time()
    r = c.chat.completions.create(model=name,
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": ev.build_prompt(row)}],
        max_tokens=16, temperature=0)
    dt = time.time() - t0
    txt = r.choices[0].message.content or ""
    ctok = getattr(getattr(r, "usage", None), "completion_tokens", None)
    return ev.parse_answer(txt), (ctok if ctok is not None else 0), dt


def eval_via(name, rows, workers=4):
    res = [None] * len(rows)
    def work(i):
        for a in range(5):
            try: return i, ask(name, rows[i])
            except Exception as e:
                if a == 4: print(f"   문항{i} 실패: {str(e)[:70]}", flush=True); return i, (None, 0, 0.0)
                time.sleep(3 * (a + 1))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for fut in as_completed([ex.submit(work, i) for i in range(len(rows))]):
            i, o = fut.result(); res[i] = o
    correct = 0; toks = []; lats = []; bysub = defaultdict(lambda: [0, 0]); details = []
    for row, (pred, ctok, dt) in zip(rows, res):
        gold = row["answer"]; ok = (pred == gold)
        correct += int(ok); toks.append(ctok); lats.append(dt)
        s = row.get("과목", "미상").replace(" ", "")
        bysub[s][0] += int(ok); bysub[s][1] += 1
        details.append({"source": row.get("source"), "과목": s, "번호": row.get("번호"),
                        "gold": gold, "pred": pred, "correct": ok, "ctok": ctok, "lat": round(dt, 3)})
    n = len(rows)
    return {"model": name, "n": n, "correct": correct, "accuracy": round(correct / n, 4),
            "answered": sum(1 for x in details if x["pred"] is not None),
            "mean_ctok": round(sum(toks) / n, 2), "mean_lat": round(sum(lats) / n, 3),
            "by_subject": {k: {"correct": v[0], "n": v[1], "acc": round(v[0] / v[1], 4)}
                           for k, v in sorted(bysub.items())}, "details": details}


def run_one(model, label, rows):
    eid, name = deploy(model, label)
    try:
        return eval_via(name, rows)
    finally:
        try: c.endpoints.delete(eid); print(f"[delete] {eid}", flush=True)
        except Exception as e: print(f"[delete 실패] {eid}: {str(e)[:70]} — 콘솔 확인!", flush=True)


def main():
    rows = ev.select_eval_rows(use_all=False)
    print(f"검증 {len(rows)}문항 (held-out)\n", flush=True)
    out = {}
    print("=== 학습 전(베이스) ===", flush=True)
    out["before_base"] = run_one(BASE, "hani-base-v2", rows)
    b = out["before_base"]
    print(f"  정답률 {b['accuracy']*100:.1f}% ({b['correct']}/{b['n']}, 응답 {b['answered']}) 토큰 {b['mean_ctok']} 지연 {b['mean_lat']}s", flush=True)
    print("\n=== 학습 후(파인튜닝) ===", flush=True)
    out["after_ft"] = run_one(FT, "hani-ft-v2", rows)
    a = out["after_ft"]
    print(f"  정답률 {a['accuracy']*100:.1f}% ({a['correct']}/{a['n']}, 응답 {a['answered']}) 토큰 {a['mean_ctok']} 지연 {a['mean_lat']}s", flush=True)

    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/together_beforeafter.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("\n========== 학습 전후 (Qwen2.5-7B, 검증 51) ==========", flush=True)
    print(f"{'지표':14s} {'전':>9s} {'후':>9s} {'변화':>9s}")
    print(f"{'정답률':12s} {b['accuracy']*100:8.1f}% {a['accuracy']*100:8.1f}% {(a['accuracy']-b['accuracy'])*100:+8.1f}p")
    print(f"{'출력토큰/문항':10s} {b['mean_ctok']:9.2f} {a['mean_ctok']:9.2f} {a['mean_ctok']-b['mean_ctok']:+9.2f}")
    print(f"{'지연(초)':12s} {b['mean_lat']:9.3f} {a['mean_lat']:9.3f} {a['mean_lat']-b['mean_lat']:+9.3f}")
    print("\n저장: results/together_beforeafter.json", flush=True)


if __name__ == "__main__":
    main()
