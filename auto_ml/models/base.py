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

    # 부스팅 iteration 개수를 지정하는 라이브러리 파라미터 이름.
    # iteration capping (final_fit_strategy=iteration_capping) 이 이 키를 통해
    # CV fold best_iter 평균으로 트리 수를 고정한다. 부스팅이 아닌 모델
    # (ElasticNet) 은 None 으로 두어 capping 대상에서 제외됨을 표시한다.
    ITER_PARAM_NAME: str | None = None

    def __init__(
        self,
        params: dict[str, Any] | None = None,
        categorical_columns: list[str] | None = None,
        random_state: int = 42,
        loss: str = "logloss",
    ) -> None:
        self.params = dict(params or {})
        self.categorical_columns = list(categorical_columns or [])
        self.random_state = random_state
        # 손실 함수 식별자 ("logloss" | "focal"). 래퍼별 _resolved_params 가
        # 본 값에 따라 custom objective 를 주입하고 _focal_active 플래그를 켠다.
        self.loss = loss

        # 학습 후 채워질 멤버
        self.model: Any = None
        self.feature_columns: list[str] = []
        self.best_iteration: int | None = None
        # focal 활성 여부 — predict_proba 의 후처리 (sigmoid) 분기에 사용
        self._focal_active: bool = False

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

    @abstractmethod
    def shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """건별·변수별 SHAP 기여도 (raw-margin / logit 도메인).

        Returns:
            ``(n_samples, n_features + 1)`` shape 의 ndarray. 컬럼 순서는
            ``self.feature_columns`` 와 동일하며, **마지막 열은 base value**
            (모델의 기대 출력 = 평균 raw margin) 다.

            가법성: ``shap[:, :-1].sum(axis=1) + shap[:, -1]`` 가 모델의
            raw margin 예측과 (수치오차 이내로) 일치한다.

            확률 도메인으로의 변환은 호출자(Explainer) 가 담당한다.
        """
