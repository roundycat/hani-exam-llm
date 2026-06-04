# -*- coding: utf-8 -*-
"""공통 설정 — 경로, 모델 ID, 프롬프트, 하이퍼파라미터."""
from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# 경로
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "dataset"
DATA_DIR = ROOT / "data"          # 파인튜닝용 변환 데이터 (gitignore)
RESULTS_DIR = ROOT / "results"    # 평가 결과 (gitignore)

# 587문항 전체(과목/그림여부 메타 포함). 그림 문항은 학습/평가에서 제외한다.
FULL_DATASET = DATASET_DIR / "한의학_문제_전체.jsonl"

FINETUNE_TRAIN = DATA_DIR / "finetune_train.jsonl"
FINETUNE_VAL = DATA_DIR / "finetune_val.jsonl"

# ---------------------------------------------------------------------------
# 모델 ID
#   - GPT-4 계열은 fine-tuning 가능한 스냅샷을 기본값으로 둔다.
#   - GPT-5 계열은 계정 권한에 따라 다르므로 .env(GPT5_MODEL)로 덮어쓰는 것을 권장.
# ---------------------------------------------------------------------------
GPT4_MODEL = os.getenv("GPT4_MODEL", "gpt-4o-2024-08-06")
GPT5_MODEL = os.getenv("GPT5_MODEL", "gpt-5")

# 파인튜닝 결과 모델 ID(있으면 평가에 포함)
GPT4_FINETUNED_MODEL = os.getenv("GPT4_FINETUNED_MODEL", "").strip()
GPT5_FINETUNED_MODEL = os.getenv("GPT5_FINETUNED_MODEL", "").strip()

# 베이스 모델 매핑(파인튜닝/평가에서 이름으로 참조)
BASE_MODELS = {
    "gpt-4": GPT4_MODEL,
    "gpt-5": GPT5_MODEL,
}

# ---------------------------------------------------------------------------
# 프롬프트
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "당신은 한의사·한약사 국가시험 문제를 푸는 한의학 전문가입니다. "
    "주어진 5지선다 문제를 읽고 정답을 고르세요."
)

# 파인튜닝 정답 형식: 번호만 출력하도록 학습한다.
ANSWER_INSTRUCTION = "정답 번호(1~5) 하나만 숫자로 출력하세요. 설명은 하지 마세요."

# ---------------------------------------------------------------------------
# 하이퍼파라미터 / 실행 옵션
# ---------------------------------------------------------------------------
VAL_RATIO = 0.1          # 학습/검증 분할 비율
RANDOM_SEED = 42
EVAL_CONCURRENCY = 8     # 평가 시 동시 요청 수
EVAL_MAX_TOKENS = 8      # 번호만 받으면 되므로 작게
REQUEST_TIMEOUT = 60     # 초

# 일부 추론형 모델(gpt-5 등)은 temperature 등 샘플링 파라미터를 받지 않는다.
# 이름에 아래 토큰이 들어가면 샘플링 파라미터를 생략한다.
REASONING_MODEL_HINTS = ("gpt-5", "o1", "o3", "o4")


def is_reasoning_model(model_id: str) -> bool:
    m = model_id.lower()
    return any(h in m for h in REASONING_MODEL_HINTS)
