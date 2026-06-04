# -*- coding: utf-8 -*-
"""전체 파이프라인 오케스트레이터.

1) 데이터 변환 (prepare_data)
2) 학습 전(베이스) 평가
3) GPT-4 / GPT-5 파인튜닝
4) 학습 후(파인튜닝) 평가

각 단계는 독립 실행도 가능하지만, 이 스크립트로 한 번에 돌릴 수 있다.
파인튜닝은 시간이 오래 걸리므로(수십 분~) --skip-finetune 으로 건너뛸 수 있다.

사용법:
    python training/run_all.py
    python training/run_all.py --skip-finetune     # 변환 + 베이스 평가만
"""
from __future__ import annotations
import argparse

import prepare_data
import finetune
import evaluate
from config import BASE_MODELS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-finetune", action="store_true")
    ap.add_argument("--eval-all", action="store_true",
                    help="평가를 검증셋이 아닌 전체 텍스트 문항으로")
    args = ap.parse_args()

    print("\n########## 1. 데이터 변환 ##########")
    prepare_data.main()

    print("\n########## 2. 학습 전(베이스) 평가 ##########")
    base_rows = evaluate.select_eval_rows(args.eval_all)
    for mid in [BASE_MODELS["gpt-4"], BASE_MODELS["gpt-5"]]:
        try:
            evaluate.evaluate_model(mid, base_rows)
        except Exception as e:  # noqa: BLE001
            print(f"[베이스 평가 오류] {mid}: {e}")

    if args.skip_finetune:
        print("\n--skip-finetune: 파인튜닝/사후 평가 생략")
        return

    print("\n########## 3. 파인튜닝 (GPT-4, GPT-5) ##########")
    ft_models = []
    for mid in [BASE_MODELS["gpt-4"], BASE_MODELS["gpt-5"]]:
        try:
            ft = finetune.run_finetune(mid)
            if ft:
                ft_models.append(ft)
        except Exception as e:  # noqa: BLE001
            print(f"[파인튜닝 오류] {mid}: {e}")

    if not ft_models:
        print("파인튜닝된 모델이 없어 사후 평가를 건너뜁니다.")
        return

    print("\n########## 4. 학습 후(파인튜닝) 평가 ##########")
    for mid in ft_models:
        try:
            evaluate.evaluate_model(mid, base_rows)
        except Exception as e:  # noqa: BLE001
            print(f"[사후 평가 오류] {mid}: {e}")

    print("\n완료. 파인튜닝 모델 ID:", ft_models)
    print("정확한 정답률 비교 표는 evaluate.py 를 직접 실행해 확인하세요.")


if __name__ == "__main__":
    main()
