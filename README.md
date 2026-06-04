# 한의학 국가시험 문제 데이터셋 & 학습 코드

한국보건의료인국가시험원(국시원) 공개 기출문제에서 추출한 **객관식 5지선다 + 정답** 데이터셋과,
이를 **OpenAI API로 파인튜닝(SFT)** 하고 **GPT‑4 / GPT‑5 성능을 평가**하는 코드.

> ⚠️ **저작권 유의**: 국시원 기출문제는 저작권법 보호 대상입니다. **개인 학습·연구용**으로만
> 사용하고 외부 재배포는 주의하세요.

---

## 📦 데이터셋 (`dataset/`)

| 파일 | 내용 | 문항 수 |
|---|---|---|
| `한의학_문제.jsonl` | **학습용 메인.** 그림 비의존 순수 텍스트 문항. `{question, options, answer, answer_text}` | 517 |
| `한의학_문제_전체.jsonl` | 그림생략(70) 포함 전체 + 메타(`source,교시,과목,번호,has_figure`) | 587 |
| `stats.json` | 회차·교시별 통계 | — |

수집 회차: **제81회 한의사(2026)** 337문항, **제27회 한약사(2026)** 250문항.
`answer` 는 정답 보기 번호(1~5), `answer_text` 는 해당 보기 텍스트(100% 일치 검증 완료).

```jsonc
// 한의학_문제_전체.jsonl
{"source":"한의사_81회","교시":1,"과목":"내과학","번호":1,
 "question":"...","options":["...","...","...","...","..."],"answer":4,"answer_text":"...","has_figure":false}
```

---

## 🚀 학습/평가 빠른 시작

```bash
# 1) 의존성 설치
pip install -r requirements.txt

# 2) API 키 설정
cp .env.example .env          # .env 에 OPENAI_API_KEY 입력 (필요시 모델 ID 수정)

# 3) 파인튜닝용 데이터 변환 (그림문항 제외, train/val 분할)
python training/prepare_data.py

# 4) 파인튜닝 (GPT-4 / GPT-5)
python training/finetune.py --model all
#   완료되면 출력된 ft:... 모델 ID를 .env 의 *_FINETUNED_MODEL 에 기입

# 5) 평가 (베이스 + 파인튜닝 모델 정답률 비교)
python training/evaluate.py            # 검증셋
python training/evaluate.py --all      # 전체 텍스트 문항

# (선택) 변환 → 베이스 평가 → 파인튜닝 → 사후 평가 한 번에
python training/run_all.py
```

### 코드 구성 (`training/`)

| 파일 | 역할 |
|---|---|
| `config.py` | 경로·모델 ID·프롬프트·하이퍼파라미터. `.env` 로 오버라이드 |
| `prepare_data.py` | 데이터셋 → OpenAI chat 포맷(`{"messages":[...]}`) 변환 + train/val 분할 |
| `finetune.py` | 파일 업로드 → fine‑tuning job 생성 → 모니터링 → 결과 모델 ID 출력 |
| `evaluate.py` | 문제를 풀게 해 **전체/과목별 정답률** 산출, 모델 간 비교표 |
| `run_all.py` | 위 단계를 순차 실행하는 오케스트레이터 |

### 학습 방식
- **포맷**: 시스템 프롬프트 + (문제+보기) → 정답 번호(1글자) 를 맞히는 지도학습(SFT).
- **분할**: `VAL_RATIO=0.1`, 고정 시드(42)로 재현 가능. 평가셋은 학습에 쓰지 않은 검증 분할.
- **그림 문항 제외**: 텍스트만으로 풀 수 없는 `has_figure=true` 70문항은 자동 제외.

### 모델 ID 참고
- **GPT‑4**: 파인튜닝 가능 스냅샷(예: `gpt-4o-2024-08-06`)이 기본값.
- **GPT‑5**: 모델 ID·파인튜닝 지원 여부가 **계정 권한에 따라 다름** → 콘솔 확인 후 `.env`(`GPT5_MODEL`)에 입력.
  추론형 모델은 `temperature` 등을 받지 않으므로 `config.is_reasoning_model()` 이 평가 시 자동 생략.

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

> 게시판에는 **최신 회차만** 공개됩니다(과년도는 내려감). 새 회차가 올라오면 동일 파이프라인을 재실행.
> 원본 PDF/이미지(`raw/`, `img/`)는 용량·저작권상 레포에서 제외됩니다.

### 품질 메모
- 587문항 전부 정답 매칭, 전부 보기 5개(정답누락 0 / 보기오류 0).
- 비전 전사 특성상 일부 한자·작은 글씨에 국소적 오탈자 가능 → 학습 전 표본 검수 권장.

---

## 📁 디렉터리
```
.
├── dataset/      # 최종 JSONL 데이터셋
├── training/     # 파인튜닝 + 평가 코드 (OpenAI API)
├── scripts/      # PDF→데이터셋 추출 파이프라인
├── requirements.txt
├── .env.example
└── README.md
```
