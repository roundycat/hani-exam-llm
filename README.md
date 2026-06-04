# 한의학 국가시험 문제 데이터셋 & 학습 코드

한국보건의료인국가시험원(국시원) 공개 기출문제에서 추출한 **객관식 5지선다 + 정답** 데이터셋과,
이를 **OpenAI API로 파인튜닝(SFT)** 하고 **GPT‑4 / GPT‑5 성능을 평가**하는 코드입니다.

> ⚠️ **저작권 유의**: 국시원 기출문제는 저작권법 보호 대상입니다. **개인 학습·연구용**으로만
> 사용하고 외부 재배포는 주의하세요.

---

## 목차
1. [데이터셋](#-데이터셋-dataset)
2. [전체 흐름](#-전체-흐름)
3. [빠른 시작](#-빠른-시작)
4. [코드 구성](#-코드-구성-training)
5. [파인튜닝 모드(번호만 vs 해설)](#-파인튜닝-모드)
6. [해설 필드 생성](#-해설-필드-생성)
7. [GitHub Actions 자동 평가](#-github-actions-자동-평가)
8. [비용 가이드](#-비용-가이드)
9. [트러블슈팅](#-트러블슈팅)
10. [데이터 추출 파이프라인](#-데이터-추출-파이프라인-scripts)

---

## 📦 데이터셋 (`dataset/`)

| 파일 | 내용 | 문항 수 |
|---|---|---|
| `한의학_문제.jsonl` | **학습용 메인.** 그림 비의존 순수 텍스트 문항 | 517 |
| `한의학_문제_전체.jsonl` | 그림생략(70) 포함 전체 + 메타데이터 | 587 |
| `한의학_문제_해설.jsonl` | 위에 `해설` 필드를 추가한 버전 *(생성 시)* | — |
| `stats.json` | 회차·교시별 통계 | — |

수집 회차: **제81회 한의사(2026)** 337문항 · **제27회 한약사(2026)** 250문항.

**스키마**
```jsonc
// 한의학_문제.jsonl (메인)
{"question": "...", "options": ["...","...","...","...","..."], "answer": 4, "answer_text": "..."}

// 한의학_문제_전체.jsonl (메타 포함)
{"source":"한의사_81회","교시":1,"과목":"내과학","번호":1,
 "question":"...","options":[...],"answer":4,"answer_text":"...","has_figure":false}

// 한의학_문제_해설.jsonl (add_explanations.py 산출물)
{... 위 필드 ..., "해설":"정답 4번은 ... 때문에 옳다. 2번은 ... 이유로 틀렸다."}
```
- `answer`: 정답 보기 번호(1~5). `answer_text`: 해당 보기 텍스트(100% 일치 검증 완료).
- `has_figure`: 그림/도표 의존 문항 여부(텍스트만으로 풀 수 없는 70문항). 학습·평가에서 자동 제외.

---

## 🔄 전체 흐름

```
                          dataset/한의학_문제_전체.jsonl  (587문항, 정답 포함)
                                        │
            ┌───────────────────────────┼───────────────────────────┐
            ▼                           ▼                           ▼
  add_explanations.py          prepare_data.py                evaluate.py
  (정답 근거 해설 생성)      (chat JSONL로 변환·분할)        (모델에 풀게 해 채점)
            │                           │                           │
            ▼                           ▼                           ▼
  한의학_문제_해설.jsonl      data/finetune_train.jsonl       results/eval_*.json
                              data/finetune_val.jsonl         results/summary.json
                                        │
                                        ▼
                                  finetune.py  ──►  ft:gpt-4o-...:hani-exam  (파인튜닝 모델)
                                                          │
                                                          ▼  (다시 evaluate.py 로 학습 전후 비교)
```

---

## 🚀 빠른 시작

```bash
# 1) 의존성 설치
pip install -r requirements.txt

# 2) API 키 설정
cp .env.example .env          # .env 에 OPENAI_API_KEY 입력 (필요 시 모델 ID 수정)

# 3) 파인튜닝용 데이터 변환 (그림문항 제외, train/val 분할)
python training/prepare_data.py

# 4) 파인튜닝 (GPT-4 / GPT-5)
python training/finetune.py --model all
#   완료되면 출력된 ft:... 모델 ID를 .env 의 *_FINETUNED_MODEL 에 기입

# 5) 평가 (베이스 + 파인튜닝 모델 정답률 비교)
python training/evaluate.py            # 검증셋(51문항)
python training/evaluate.py --all      # 전체 텍스트 문항(517)

# (선택) 변환 → 베이스 평가 → 파인튜닝 → 사후 평가 한 번에
python training/run_all.py
```

---

## 🧩 코드 구성 (`training/`)

| 파일 | 역할 | 주요 옵션 |
|---|---|---|
| `config.py` | 경로·모델 ID·프롬프트·하이퍼파라미터 중앙 관리. `.env` 로 오버라이드 | — |
| `prepare_data.py` | 데이터셋 → OpenAI chat 포맷 변환 + train/val 분할 | `--with-rationale` |
| `add_explanations.py` | 정답 기반 **해설 생성**(재개 가능) → `한의학_문제_해설.jsonl` | `--limit`, `--skip-figure` |
| `finetune.py` | 파일 업로드 → fine‑tuning job 생성 → 진행 모니터링 | `--model {gpt-4,gpt-5,all}`, `--model-id` |
| `evaluate.py` | 문제를 풀게 해 **전체/과목별 정답률** 산출, 모델 비교표·JSON 저장 | `--all`, `--models ...` |
| `run_all.py` | 위 단계 순차 실행 오케스트레이터 | `--skip-finetune`, `--eval-all` |

모든 모듈은 상단 docstring과 인라인 주석으로 동작/주의점을 설명합니다.

---

## 🎓 파인튜닝 모드

`prepare_data.py` 는 두 가지 학습 타깃을 만들 수 있습니다.

| 모드 | 명령 | assistant 타깃 | 특징 |
|---|---|---|---|
| 번호만(기본) | `prepare_data.py` | `"4"` | 가볍고 빠름. 정답 선택만 학습 |
| 해설 포함(CoT) | `prepare_data.py --with-rationale` | `"{해설}\n\n정답: 4"` | 근거를 먼저 쓰고 답을 내도록 학습. `한의학_문제_해설.jsonl` 필요 |

> 해설 모드는 해설 파일이 없으면 자동으로 번호만 모드로 폴백합니다(경고 출력).

학습/평가 데이터는 **고정 시드(42)** 로 분할되어, `evaluate.py` 의 검증셋과 정확히 동일합니다 →
학습에 쓴 문항으로 평가하는 **데이터 누수(leakage)가 없습니다.**

---

## ✍️ 해설 필드 생성

원본 국시원 자료에는 해설이 없어 LLM으로 생성합니다. 단, **정답을 모델에 알려준 상태**에서
"왜 그 답이 옳은지"를 서술하게 하여(모델이 직접 풀게 하지 않음) 사실 오류 위험을 낮춥니다.

```bash
python training/add_explanations.py --limit 5    # 먼저 5개로 품질 확인
python training/add_explanations.py              # 전체 생성(재개 가능)
```
- **재개 가능**: 중간에 멈춰도 다시 실행하면 이미 만든 해설은 건너뛰고 이어서 생성.
- 결과는 `dataset/한의학_문제_해설.jsonl` 에 저장되며, 이후 `prepare_data.py --with-rationale` 로 학습에 활용.
- ⚠️ 생성 해설은 **참고용**입니다. 학습/배포 전 표본 검수를 권장합니다.

---

## 🤖 GitHub Actions 자동 평가

`.github/workflows/evaluate.yml` — **수동 실행(workflow_dispatch)** 기반(API 비용 때문에 자동 트리거 OFF).

**설정 1회**: 저장소 → Settings → Secrets and variables → Actions → **New repository secret**
→ 이름 `OPENAI_API_KEY`, 값에 API 키 입력.

**실행**: 저장소 **Actions** 탭 → *Evaluate (한의학 문제 정답률)* → **Run workflow**
- `models`: 평가할 모델 ID(공백 구분), 예) `gpt-4o-2024-08-06 ft:gpt-4o-...:hani-exam:...`
- `eval_all`: 전체 문항(true) / 검증셋만(false)

결과는 **Artifacts(`eval-results`)** 로 다운로드되고, 요약은 워크플로 **Summary** 에 표로 표시됩니다.
정기 평가가 필요하면 워크플로의 `schedule:` 주석을 해제하세요(비용 주의).

---

## 💰 비용 가이드

대략적인 호출 수(1회 실행 기준):

| 작업 | API 호출 수 | 비고 |
|---|---|---|
| 검증셋 평가 | 모델당 ~51회 | 짧은 응답(번호) |
| 전체 평가(`--all`) | 모델당 ~517회 | |
| 해설 생성 | ~517~587회 | 응답이 길어 토큰 사용량 ↑ |
| 파인튜닝 | 학습 토큰량 기반 과금 | 모델·에폭에 비례 |

> 실제 비용은 모델 단가에 따라 다릅니다. 먼저 `--limit`/검증셋으로 소규모 확인 후 전체를 돌리세요.
> 추론형 모델(gpt‑5/o‑계열)은 `reasoning_effort=low` 와 넉넉한 출력 토큰으로 평가합니다(빈 응답 방지).

---

## 🛠 트러블슈팅

| 증상 | 원인/해결 |
|---|---|
| `openai.AuthenticationError` | `.env` 의 `OPENAI_API_KEY` 누락/오타 |
| 평가에서 정답이 전부 `None` | 추론형 모델인데 출력 토큰이 부족 → `config.EVAL_MAX_TOKENS_REASONING` 상향 |
| 파인튜닝 `model not found / not fine-tunable` | 해당 모델 ID가 파인튜닝 미지원 → `.env` 의 모델 ID를 파인튜닝 가능 스냅샷으로 변경 |
| GPT‑5 관련 파라미터 오류 | `config.REASONING_MODEL_HINTS` 로 추론형을 감지해 `temperature` 등을 자동 생략. 새 모델명은 힌트에 추가 |
| 한글이 깨져 보임(콘솔) | 파일은 UTF‑8 정상. Windows 콘솔 표시 문제 → `chcp 65001` 또는 `PYTHONUTF8=1` |

---

## 🔧 데이터 추출 파이프라인 (`scripts/`)

국시원 PDF는 **텍스트가 전부 벡터(곡선)로 변환**되어(복사 방지) 일반 텍스트 추출이 불가 →
**고해상도 렌더 + 비전 전사**가 유일한 방법. A3 한 페이지 통째로는 작은 한자가 뭉개지므로
**2단 컬럼 타일링**(장변 ≤1980px)으로 가독성 확보. 사진·도표 문항은 국시원이 비공개 처리하여
`[그림 생략]` 표기 후 메인 학습셋에서 제외.

```powershell
python scripts/discover.py --max-pages 30                 # 1) 게시물·첨부 탐색 → manifest.json
python scripts/fetch_render.py --manifest manifest.json   # 2) PDF 다운로드 + 페이지 PNG 렌더
python scripts/tile_render.py --pdf "..." --out "..."     # 3) 2단 컬럼 타일 분할
python scripts/transcribe_api.py questions --slug ...      # 4) 비전 API 전사 (정답표 포함)
python scripts/assemble.py                                # 5) 전사본 + 정답표 조인 → dataset/*.jsonl
```

> 게시판에는 **최신 회차만** 공개됩니다(과년도는 내려감). 새 회차가 올라오면 동일 파이프라인 재실행.
> 원본 PDF/이미지(`raw/`, `img/`, `tiles/`)는 용량·저작권상 레포에서 제외됩니다.

### 품질 메모
- 587문항 전부 정답 매칭, 전부 보기 5개(정답누락 0 / 보기오류 0).
- 비전 전사 특성상 일부 한자·작은 글씨에 국소적 오탈자 가능 → 학습 전 표본 검수 권장.

---

## 📁 디렉터리
```
.
├── .github/workflows/evaluate.yml   # 수동 실행 평가 워크플로
├── dataset/                         # 최종 JSONL 데이터셋
├── training/                        # 파인튜닝 + 평가 + 해설생성 코드
│   ├── config.py
│   ├── prepare_data.py
│   ├── add_explanations.py
│   ├── finetune.py
│   ├── evaluate.py
│   └── run_all.py
├── scripts/                         # PDF→데이터셋 추출 파이프라인
├── requirements.txt
├── .env.example
└── README.md
```
