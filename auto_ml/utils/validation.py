"""시스템 경계(데이터 로드, 스코어링 입력) 에서 사용되는 입력 검증 유틸.

내부 코드 사이에서는 타입 힌트를 신뢰하고 추가 검증을 두지 않는다.
오로지 외부에서 들어온 데이터(파일, 사용자 입력) 의 가정 위반만 검사한다.
"""
from __future__ import annotations

import pandas as pd


def validate_binary_target(y: pd.Series) -> None:
    """타깃이 이진(0/1) 인지 검증한다.

    Args:
        y: 타깃 시리즈.

    Raises:
        ValueError: 0/1 외 값이 존재하거나 단일 클래스만 있는 경우.
    """
    unique = set(pd.unique(y.dropna()))
    if not unique.issubset({0, 1}):
        raise ValueError(
            f"Target must be binary (0/1). Found unique values: {sorted(unique)}"
        )
    if len(unique) < 2:
        # 단일 클래스만 있으면 분류기 학습 자체가 불가하다.
        raise ValueError("Target contains only one class; cannot train a classifier.")


def validate_schema(df: pd.DataFrame, required_columns: list[str]) -> None:
    """필수 컬럼이 DataFrame 에 모두 존재하는지 검증한다.

    Args:
        df: 검증 대상 DataFrame.
        required_columns: 반드시 존재해야 하는 컬럼 목록.

    Raises:
        ValueError: 누락 컬럼이 1개 이상인 경우.
    """
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")
