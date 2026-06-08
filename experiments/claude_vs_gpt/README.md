# 한의학 국가시험 LLM 평가 실험 — Claude vs GPT (+ Qwen 파인튜닝)

**실험일: 2026-06-08**

국시원 공개 기출(제81회 한의사·제27회 한약사)에서 추출한 **순수 텍스트 5지선다 517문항**을
Claude 3티어와 GPT 3종에게 **폐쇄형(closed-book)** 으로 풀게 해 정답률을 비교하고,
추가로 **Qwen2.5-7B를 LoRA 파인튜닝**해 학습 전후를 검증셋에서 비교했다.

> 📘 **최종 종합 보고서(논문 형식): [FINAL_REPORT.md](FINAL_REPORT.md)** — 벤치마크 + 파인튜닝 +
> 내과학 개선(변증·자기일관성·앙상블·RAG)까지 전부 포함, "무엇이 나아졌는가" 정리.
> 〔보조 문서: [PAPER_RESULTS.md](PAPER_RESULTS.md)(벤치마크+파인튜닝), [NAEGWA_IMPROVEMENT.md](NAEGWA_IMPROVEMENT.md)(내과학 개선), [results_tables.md](results_tables.md)(표 단일 진실원)〕
>
> **핵심 개선:** 내과학 74.0%→83.3%(+9.3%p) · 최난도 내과학2 56.7%→70.0%→**83.3%**(RAG, 총 +26.6%p).

> 데이터 출처·저작권은 상위 레포 README 참고. 국시원 기출은 저작권 보호 대상이며 개인 학습·연구용.

---

## 1. 결과 요약 (전체 517문항)

| 순위 | 모델 | 정답률 | 95% CI (Wilson) | 맞음/전체 |
|---:|---|---:|:---:|---:|
| 1 | **gpt-5** | **84.91%** | 81.6–87.7% | 439/517 |
| 2 | **claude-sonnet** | **83.75%** | 80.3–86.7% | 433/517 |
| 3 | **claude-opus** | **81.24%** | 77.6–84.4% | 420/517 |
| 4 | gpt-4o (2024-08-06) | 77.95% | 74.2–81.3% | 403/517 |
| 5 | claude-haiku | 71.95% | 67.9–75.6% | 372/517 |
| 6 | gpt-4o-mini | 59.57% | 55.3–63.7% | 308/517 |

- 랜덤 베이스라인 20%(5지선다) → **전 모델이 크게 상회**.
- 상위 3개(gpt-5 / sonnet / opus)는 **95% CI가 서로 겹친다** → 1~3위 차이는 통계적으로 유의하다고
  보기 어렵다(사실상 동급, ~81~85%). 반면 mini는 명확히 하위.

### 회차(시험)별

| 모델 | 한약사 27회 (n=241) | 한의사 81회 (n=276) |
|---|---:|---:|
| gpt-5 | 89.6% | 80.8% |
| claude-sonnet | 89.2% | 79.0% |
| claude-opus | 88.0% | 75.4% |
| gpt-4o | 83.0% | 73.6% |
| claude-haiku | 78.4% | 66.3% |
| gpt-4o-mini | 69.3% | 51.1% |

→ 모든 모델이 **한의사(81회)를 한약사(27회)보다 어려워한다**(평균 약 10%p 낮음).

### Qwen2.5-7B 파인튜닝 학습 전후 (검증 51문항, held-out)

| 지표 | 학습 전(베이스) | 학습 후(FT) | 변화 |
|---|---:|---:|---:|
| 정답률 | 51.0% (26/51) | 49.0% (25/51) | **−2.0%p** |
| 95% CI | 37.7–64.1% | 35.9–62.3% | 크게 겹침 |
| 출력토큰/문항 | 2.00 | 2.00 | 0 |
| 지연(초)/문항 | 0.386 | 0.378 | ≈0 |

→ **정직한 음성 결과**: 466문항 LoRA SFT(3에폭)는 정답률을 높이지 못했고(오히려 −1문항, 통계적 노이즈),
연산량(토큰)도 줄지 않았다. 학습 중 검증손실 0.318→0.340 상승(과적합). 소규모 객관식 SFT는 도메인
지식을 주입하지 못한 채 형식만 학습하며, "번호만 출력"은 베이스도 이미 수행해 절감 여지가 없었다.
자세한 분석은 [PAPER_RESULTS.md](PAPER_RESULTS.md) §2.3 참조.

### 과목별 난이도 (6모델 평균 정답률, 낮은 순 일부)

| 과목 | 평균 정답률 | 문항수 |
|---|---:|---:|
| 내과학2 | **43.9%** | 30 |
| 외과학 | 61.1% | 6 |
| 보건의약관계법규 | 66.7% | 20 |
| 본초학 | 70.5% | 13 |
| … | … | … |
| 예방의학 | 82.6% | 22 |
| 한방생리학 | 86.5% | 16 |
| 한의학 기초 | 92.4% | 107 |

→ **내과학2(43.9%)** 가 압도적으로 어려움(전 모델 공통). 임상 추론·감별이 많은 영역으로 추정.
   반대로 한의학 기초·생리학 등 개념 영역은 90% 안팎으로 쉬움.

전체 과목·모델별 수치는 [`results/combined_summary.json`](results/combined_summary.json) 참고.

---

## 2. 평가셋

- 원천: 상위 레포 `dataset/한의학_문제_전체.jsonl` (587문항).
- **그림 의존 문항 70개(`has_figure=true`) 제외** → 텍스트만으로 풀 수 있는 **517문항**.
  (한의사 81회 276 · 한약사 27회 241)
- 무결성: 517개 전부 보기 5개·정답 1~5·정답누락 0 확인.

## 3. 누수(leakage) 방지

모델이 정답을 미리 못 보도록 데이터를 분리:
- [`data/questions_noanswer.jsonl`](data/questions_noanswer.jsonl) — **정답 제거** 문항(`idx, source, 교시, 과목, 번호, question, options`). 모델은 이것만 본다.
- [`data/gold.json`](data/gold.json) — 정답은 따로 보관, 채점에만 사용.
- (회차·교시·번호) 안정 정렬 후 `idx 0..516` 부여 → 채점·재현 기준 고정.
- 검증: gold ↔ 원본 데이터셋 정답 **불일치 0**.

(주의: 본 실험은 **베이스 모델 벤치마크**로 파인튜닝이 없으므로 train/val 누수는 애초에 발생하지 않는다.)

## 4. 방법

### Claude 3티어 (Opus·Sonnet·Haiku)
- Claude Code의 Workflow 오케스트레이션으로 실행(별도 API 키 불필요).
- 517문항을 13개씩 **40배치**로 나눠 **배치마다 독립 에이전트** → 모델당 40, 총 **120 에이전트**.
- 각 에이전트: 정답제거 파일의 자기 구간만 Read, **폐쇄형(웹·도구 금지, 지식만)**,
  출력은 스키마로 강제한 `{idx, pred(1~5)}` → 파싱오류 0.
- 미응답 자동 재시도 포함. 결과: 3모델 모두 **517/517 응답, 미응답 0, 오류 0**.
- 사전 스모크 테스트(Opus, 앞 13문항)에서 13/13 → 하니스·무누수 확인 후 본 실행.
- 하니스: [`harness/claude_eval_workflow.js`](harness/claude_eval_workflow.js),
  [`harness/claude_smoke_workflow.js`](harness/claude_smoke_workflow.js).

### GPT 3종 (gpt-4o-mini · gpt-4o · gpt-5)
- 상위 레포의 [`training/evaluate.py`](../../training/evaluate.py)를 **수정 없이** 사용.
- 문항당 1회 API 호출, 전체 517(`--all`), 동시요청 8.
- 비추론형(mini·4o) `temperature=0`(재현성), 추론형(gpt-5) `reasoning_effort=low`.
- gpt-5는 추론 토큰 소진으로 **빈응답 5개** 발생 → 해당 5개만 `max_completion_tokens=8000`으로
  **재요청해 복구**(3개 정답). 복구 후 최종 84.91%.

### 채점 (양쪽 동일)
- 기준: **예측 보기번호(1~5) == 정답**.
- 지표: 전체 정답률 + **Wilson 95% CI**, 회차별, 과목별.
- Claude는 idx 기반([`harness/grade.py`](harness/grade.py)), GPT는 evaluate.py details 기반,
  최종 통합은 [`harness/aggregate.py`](harness/aggregate.py)가 **동일 지표**로 계산.
- 독립 재계산으로 모든 수치 교차검증 통과(저장값과 일치).

## 5. 한계 (정직한 기재)

1. **프로토콜 차이**: Claude·gpt-5는 내부 추론 허용, gpt-4o/mini는 번호만 즉답(레포 기본).
   완전 동일 조건은 아니므로 "각 모델의 실전 성능" 비교로 해석하는 것이 적절.
2. **상위권 동률**: 1~3위는 CI가 겹쳐 통계적 우열을 단정하기 어렵다.
3. **Claude 배치-13 공유 컨텍스트**: 한 에이전트가 13문항을 함께 보므로 미세한 문맥 영향 가능
   (시험지를 한 번에 보는 것과 유사). GPT는 문항당 1호출.
4. **closed-book 보장**: 에이전트에 도구·웹 금지를 지시했으나 100% 강제는 아님. 단 문항 난이도·
   소요시간(120에이전트 ~11분)상 외부 검색이 아닌 지식 기반 풀이로 판단됨.
5. **데이터 품질**: 비전 전사 기반이라 드물게 한자 오탈자 가능(상위 레포 주의사항).

## 6. 재현 방법

```bash
# (전제) 상위 레포 루트에서 .env 에 OPENAI_API_KEY 설정, pip install -r requirements.txt

# 1) 평가셋·gold 생성은 dataset 에서 파생 (questions_noanswer.jsonl / gold.json 제공됨)
# 2) GPT 평가 (레포 원본 코드)
python training/evaluate.py --all --models gpt-4o-mini-2024-07-18 gpt-4o-2024-08-06 gpt-5
# 3) Claude 평가는 Claude Code Workflow 로 실행 (harness/claude_eval_workflow.js)
# 4) 통합 집계
python experiments/claude_vs_gpt/harness/aggregate.py
```

## 7. 파일 안내

```
experiments/claude_vs_gpt/
├── README.md                  # 이 문서
├── harness/
│   ├── claude_eval_workflow.js   # Claude 3티어 평가 오케스트레이션
│   ├── claude_smoke_workflow.js  # 하니스 스모크 테스트
│   ├── grade.py                  # Claude 채점(idx 기반) + CI
│   └── aggregate.py              # Claude+GPT 통합 집계
├── data/
│   ├── questions_noanswer.jsonl  # 정답 제거 평가 문항(모델 입력)
│   ├── gold.json                 # 정답(채점용)
│   └── predictions.json          # Claude 원시 예측(idx→번호)
└── results/
    ├── eval_claude-{opus,sonnet,haiku}.json   # 모델별 상세(문항별 정오)
    ├── eval_gpt-{4o-mini,4o,gpt-5}.json
    └── combined_summary.json                  # 통합 지표
```
