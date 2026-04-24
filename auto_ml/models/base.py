"""모델 래퍼 공통 인터페이스.

LGBM / XGBoost / CatBoost 의 학습·예측 API 가 조금씩 다르기 때문에,
trainer / scorer 가 모델 종류와 무관하게 동일한 호출 방식을 사용할 수
있도록 본 추상 클래스로 통일한다.

각 래퍼는 다음을 보장해야 한다:
    - ``fit(X, y, X_valid, y_valid)`` : 학습 (조기종료 옵션 포함)
    - ``predict_proba(X)``           : 양성 클래스 확률 1차원 ndarray 반환
    - ``predict(X, threshold)``      : threshold 기반 0/1 예측
    - ``feature_importance()``       : {feature_name: importance} dict
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd


class BaseModel(ABC):
    """모든 모델 래퍼가 따르는 공통 인터페이스.

    Attributes:
        name: 모델 식별 문자열 ("lgbm" | "xgb" | "catboost").
        params: 라이브러리 원본 하이퍼파라미터.
        categorical_columns: 범주형으로 취급할 컬럼.
    """

    name: str = "base"

    def __init__(
        self,
        params: dict[str, Any] | None = None,
        categorical_columns: list[str] | None = None,
        random_state: int = 42,
    ) -> None:
        self.params = dict(params or {})
        self.categorical_columns = list(categorical_columns or [])
        self.random_state = random_state

        # 학습 후 채워질 멤버
        self.model: Any = None
        self.feature_columns: list[str] = []
        self.best_iteration: int | None = None

    @abstractmethod
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame | None = None,
        y_valid: pd.Series | None = None,
        early_stopping_rounds: int | None = None,
    ) -> "BaseModel":
        """모델을 학습한다."""

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """양성 클래스(=1) 확률을 1차원 ndarray 로 반환한다."""

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """threshold 를 기준으로 0/1 예측을 반환한다."""
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    @abstractmethod
    def feature_importance(self) -> dict[str, float]:
        """feature 별 중요도를 dict 로 반환한다 (정렬되지 않은 형태)."""
