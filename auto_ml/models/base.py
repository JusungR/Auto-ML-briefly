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

from auto_ml.utils.logger import get_logger

logger = get_logger("models")


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

    # 스레드 수를 지정하는 라이브러리 파라미터 이름 (첫 번째가 주입에 쓰는 이름, 나머지는
    # 별칭). 사용자가 fixed_params 에 이 중 하나를 직접 쓰면 n_jobs 설정보다 우선한다.
    # 스레드 개념이 없는 모델은 빈 튜플.
    THREAD_PARAM_NAMES: tuple[str, ...] = ()

    # categorical_encoding="onehot" 일 때 넣을 라이브러리 파라미터 (없으면 이 모델은 무시).
    # 범주 하나씩 분할되는 경계가 라이브러리마다 달라 ONEHOT_OFFSET 으로 맞춘다
    # (실측: LightGBM·XGBoost 는 범주 수 + 1, CatBoost 는 범주 수).
    ONEHOT_PARAM_NAME: str | None = None
    ONEHOT_OFFSET: int = 0
    ONEHOT_PARAM_MAX: int | None = None
    # onehot 인데 범주 수가 이보다 많으면 범주값 정리를 권하는 WARNING
    ONEHOT_WARN_CARDINALITY: int = 50

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
        # best 지점까지의 트리 개수 (1부터 세는 개수). 조기 종료가 동작하지 않았거나
        # 부스팅 모델이 아니면 None. 라이브러리마다 원래 의미가 달라(LightGBM 은 개수,
        # XGBoost·CatBoost 는 0부터 세는 번호, LightGBM 은 미발동 시 0) 래퍼에서 통일한다.
        self.best_iteration: int | None = None
        # focal 활성 여부 — predict_proba 의 후처리 (sigmoid) 분기에 사용
        self._focal_active: bool = False
        # CPU 스레드 수 (None = 라이브러리 기본 = 전체 코어). set_n_jobs 로 바꾼다.
        self.n_jobs: int | None = None
        # 범주형 분할 방식 ("native" | "onehot"). set_categorical_encoding 으로 바꾼다.
        self.categorical_encoding: str = "native"

    # ------------------------------------------------------------------
    # 범주형 분할 방식
    # ------------------------------------------------------------------
    def set_categorical_encoding(self, encoding: str | None) -> "BaseModel":
        """``"native"`` (라이브러리 기본) 또는 ``"onehot"`` (범주 하나씩 분할). 학습 전에 부른다."""
        encoding = encoding or "native"
        if encoding not in ("native", "onehot"):
            raise ValueError(f"categorical_encoding must be 'native' or 'onehot', got {encoding!r}")
        self.categorical_encoding = encoding
        return self

    def _onehot_params(self, X: pd.DataFrame) -> dict[str, Any]:
        """``categorical_encoding == "onehot"`` 이면 범주 하나씩 분할하게 하는 파라미터.

        학습 데이터의 범주형 변수 중 가장 많은 범주 수 k 에 맞춰 값을 정한다. 사용자가
        ``fixed_params`` (또는 튜닝 공간) 에 같은 파라미터를 직접 지정했으면 비운다.
        """
        name = self.ONEHOT_PARAM_NAME
        if getattr(self, "categorical_encoding", "native") != "onehot" or name is None:
            return {}
        if name in self.params:
            logger.info(
                "%s: categorical_encoding=onehot but %s=%s is set in params — using it",
                type(self).__name__, name, self.params[name],
            )
            return {}
        cols = [c for c in self.categorical_columns if c in X.columns]
        if not cols:
            return {}
        k = max(int(X[c].nunique(dropna=True)) for c in cols)
        if k > self.ONEHOT_WARN_CARDINALITY:
            logger.warning(
                "%s: categorical_encoding=onehot with up to %d categories per column — "
                "consider preprocessing.categorical_levels to keep only meaningful values",
                type(self).__name__, k,
            )
        value = max(k, 1) + self.ONEHOT_OFFSET
        if self.ONEHOT_PARAM_MAX is not None and value > self.ONEHOT_PARAM_MAX:
            logger.warning(
                "%s: %s capped at %d (needs %d) — columns with more categories keep the "
                "library's default handling", type(self).__name__, name,
                self.ONEHOT_PARAM_MAX, value,
            )
            value = self.ONEHOT_PARAM_MAX
        return {name: value}

    # ------------------------------------------------------------------
    # 스레드 수
    # ------------------------------------------------------------------
    def set_n_jobs(self, n_jobs: int | None) -> "BaseModel":
        """학습·예측에 쓸 스레드 수를 정한다 (``auto_ml.utils.threads.resolve_n_jobs`` 규칙).

        학습 전에 부르면 ``fit`` 이 그 값으로 학습하고, 학습된 모델에 부르면 이후 예측·SHAP
        에 적용된다 (예측 결과는 바뀌지 않는다). ``None`` 이면 아무것도 바꾸지 않는다.
        과거 artifact 의 객체에도 쓸 수 있다 (클래스 코드는 참조로 불러온다).
        """
        from auto_ml.utils.threads import resolve_n_jobs

        n = resolve_n_jobs(n_jobs)
        if n is None:
            return self
        self.n_jobs = n
        if getattr(self, "model", None) is not None:
            self._apply_n_jobs_to_fitted(n)
        return self

    def _apply_n_jobs_to_fitted(self, n_jobs: int) -> None:
        """학습된 라이브러리 객체에 스레드 수를 반영한다. 기본은 아무것도 하지 않음."""

    def _thread_params(self) -> dict[str, Any]:
        """``_resolved_params`` 에 더할 스레드 인자. 사용자가 직접 지정했으면 비운다."""
        n = getattr(self, "n_jobs", None)  # 과거 artifact 객체에는 속성이 없다
        if n is None or not self.THREAD_PARAM_NAMES:
            return {}
        if any(k in self.params for k in self.THREAD_PARAM_NAMES):
            return {}
        return {self.THREAD_PARAM_NAMES[0]: n}

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
