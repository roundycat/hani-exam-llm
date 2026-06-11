# -*- coding: utf-8 -*-
"""GPT-5 자기일관성(self-consistency) + 변증추론 — 내과학 96문항.

각 문항을 변증 추론 프롬프트로 k회 샘플링 → '정답: N' 파싱 → 다수결.
결과: eval/naegwa_gpt5_sc.json {nidx: {votes, majority, gold, 과목}}
"""
import sys, json, re, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, "training")
from dotenv import load_dotenv
load_dotenv(dotenv_path=".env")
from openai import OpenAI

c = OpenAI()
K = 3
MODEL = "gpt-5"

SYS = ("당신은 한방내과 전문의이자 한의사·한약사 국가시험 풀이 전문가입니다. "
       "한의학 이론(팔강·장부변증, 치법, 방제)에 근거해 단계적으로 추론합니다.")

def build(row):
    opts = "\n".join(f"{i}. {o.strip()}" for i, o in enumerate(row["options"], 1))
    # 추론은 내부(reasoning) 토큰으로 하고, content에는 정답만 → 잘림 방지
    return (f"다음 내과학 5지선다를 한의학 변증(팔강·장부)·치법에 근거해 신중히 추론한 뒤 정답을 고르세요.\n\n"
            f"[문제]\n{row['question'].strip()}\n\n[보기]\n{opts}\n\n"
            f"풀이 과정은 출력하지 말고, 마지막에 '정답: N' (N은 1~5) 한 줄만 출력하세요.")

CIRCLED = {"①":1,"②":2,"③":3,"④":4,"⑤":5}
def parse(t):
    if not t: return None
    m = re.search(r"정답\s*[:：]?\s*([1-5])", t)
    if m: return int(m.group(1))
    for ch,n in CIRCLED.items():
        if ch in t: return n
    nums = re.findall(r"[1-5]", t)
    return int(nums[-1]) if nums else None

def ask(row):
    for a in range(3):
        try:
            r = c.chat.completions.create(
                model=MODEL,
                messages=[{"role":"system","content":SYS},{"role":"user","content":build(row)}],
                max_completion_tokens=6000, reasoning_effort="medium", timeout=180)
            return parse(r.choices[0].message.content or "")
        except Exception as e:
            if a==2: return None
            time.sleep(3*(a+1))

def main():
    rows = [json.loads(l) for l in open("eval/naegwa_questions.jsonl",encoding="utf-8") if l.strip()]
    print(f"내과학 {len(rows)}문항 × GPT-5 k={K} 자기일관성", flush=True)
    tasks = [(r, s) for r in rows for s in range(K)]
    votes = {r["nidx"]: [] for r in rows}
    def work(t):
        row, s = t
        return row["nidx"], ask(row)
    done=0
    with ThreadPoolExecutor(max_workers=8) as ex:
        for fut in as_completed([ex.submit(work,t) for t in tasks]):
            nidx, pred = fut.result()
            votes[nidx].append(pred)
            done+=1
            if done%30==0 or done==len(tasks): print(f"  {done}/{len(tasks)}", flush=True)
    out={}
    gold={r["nidx"]:r["answer"] for r in rows}
    subj={r["nidx"]:r["과목"] for r in rows}
    correct=0
    for r in rows:
        v=[x for x in votes[r["nidx"]] if x]
        maj=Counter(v).most_common(1)[0][0] if v else None
        out[r["nidx"]]={"votes":votes[r["nidx"]],"majority":maj,"gold":gold[r["nidx"]],"과목":subj[r["nidx"]]}
        correct+=int(maj==gold[r["nidx"]])
    json.dump(out, open("eval/naegwa_gpt5_sc.json","w",encoding="utf-8"), ensure_ascii=False, indent=2)
    n=len(rows)
    n2=[r for r in rows if r["과목"]=="내과학2"]; c2=sum(1 for r in n2 if out[r["nidx"]]["majority"]==r["answer"])
    print(f"\nGPT-5 SC 내과학: {correct}/{n}={correct/n*100:.1f}% | 내과학2 {c2}/{len(n2)}={c2/len(n2)*100:.0f}%", flush=True)
    print("저장: eval/naegwa_gpt5_sc.json", flush=True)

if __name__=="__main__":
    main()
