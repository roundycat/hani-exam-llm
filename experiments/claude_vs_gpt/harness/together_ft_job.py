# -*- coding: utf-8 -*-
"""Together AI에서 Qwen2.5-7B-Instruct LoRA 파인튜닝 작업 생성 + 완료 모니터링.

완료 시 출력 모델명을 eval/ft_model_together.txt 에 기록한다.
업로드된 파일 id는 eval/together_files.txt 에서 읽는다.
"""
import time, sys
from dotenv import load_dotenv
load_dotenv(dotenv_path=".env")
from together import Together

c = Together()
BASE = "Qwen/Qwen2.5-7B-Instruct"

tr_id, va_id = [l.strip() for l in open("eval/together_files.txt") if l.strip()][:2]
print(f"[파일] train={tr_id} val={va_id}", flush=True)

job = c.fine_tuning.create(
    training_file=tr_id,
    validation_file=va_id,
    model=BASE,
    n_epochs=3,
    n_evals=3,
    lora=True,
    suffix="hani-exam",
    random_seed=42,
)
jid = job.id
print(f"[job 생성] id={jid} status={getattr(job,'status',None)}", flush=True)

seen = set()
last_status = None
while True:
    job = c.fine_tuning.retrieve(jid)
    st = str(getattr(job, "status", ""))
    try:
        evs = c.fine_tuning.list_events(jid)
        for ev in getattr(evs, "data", []) or []:
            msg = getattr(ev, "message", str(ev))
            key = (getattr(ev, "created_at", ""), msg)
            if key not in seen:
                seen.add(key)
                print(f"  [{st}] {msg}", flush=True)
    except Exception as e:
        if st != last_status:
            print(f"  status={st}", flush=True)
    last_status = st
    if st in ("completed", "error", "cancelled", "failed"):
        break
    time.sleep(20)

print(f"[종료] status={st}", flush=True)
# 출력 모델명 추출
out = (getattr(job, "output_name", None) or getattr(job, "model_output_name", None)
       or getattr(job, "fine_tuned_model", None))
print(f"[출력 모델명] {out}", flush=True)
if st == "completed" and out:
    open("eval/ft_model_together.txt", "w").write(out + "\n")
    print("저장: eval/ft_model_together.txt", flush=True)
else:
    print("파인튜닝 미완료/실패. 사후 평가 보류.", flush=True)
    sys.exit(1)
