"""Elastic Net (L1+L2 규제 로지스틱 회귀) 모델 래퍼.

sklearn LogisticRegression(penalty='elasticnet', solver='saga') 를 사용한다.
범주형 피처는 OneHotEncoding 으로 변환하며, SHAP 는 LinearExplainer 로 계산한다.

주요 설계 결정:
    - early_stopping_rounds: 무시함 — sklearn 에는 해당 개념이 없다.
    - OHE handle_unknown='ignore': 스코어링 시 미학습 범주를 0-벡터로 처리.
    - shap_values(): OHE-확장 공간이 아닌 원본 피처 공간으로 집계해 반환.
      Explainer 가 len(selected_features)+1 을 기대하기 때문이다.
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

# sklearn 1.8 에서 penalty 파라미터가 deprecated → 버전에 따라 분기
_SKLEARN_GE_18 = tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 8)

from auto_ml.models.base import BaseModel


class ElasticNetModel(BaseModel):
    """Elastic Net 이진분류 래퍼."""

    name = "elasticnet"
    # 부스팅 모델이 아니므로 iteration capping 대상에서 제외 (None).
    ITER_PARAM_NAME = None

    DEFAULT_PARAMS: dict[str, Any] = {
        "C": 1.0,
        "l1_ratio": 0.5,
        "max_iter": 2000,
    }

    def __init__(
        self,
        params: dict[str, Any] | None = None,
        categorical_columns: list[str] | None = None,
        random_state: int = 42,
        loss: str = "logloss",
    ) -> None:
        super().__init__(params, categorical_columns, random_state, loss)
        self._ohe: OneHotEncoder | None = None
        self._numeric_cols: list[str] = []
        self._cat_cols: list[str] = []
        # mapping: cat_col -> list of OHE output column indices in expanded space
        self._cat_to_ohe: dict[str, list[int]] = {}
        # background for SHAP (mean of OHE-encoded training data)
        self._background: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    def _encode_fit(self, X: pd.DataFrame) -> pd.DataFrame:
        """OHE 를 fit 하고 인코딩된 DataFrame 을 반환한다."""
        self._numeric_cols = [c for c in X.columns if c not in self.categorical_columns]
        self._cat_cols = [c for c in X.columns if c in self.categorical_columns]
        self._cat_to_ohe = {}

        if self._cat_cols:
            self._ohe = OneHotEncoder(
                drop="if_binary",
                sparse_output=False,
                handle_unknown="ignore",
            )
            self._ohe.fit(X[self._cat_cols].astype(str))

            # OHE 내부 구조로 positional mapping 생성 (문자열 파싱보다 안전)
            drop_idx = getattr(self._ohe, "drop_idx_", None)
            offset = 0
            for i, cat_col in enumerate(self._cat_cols):
                n = len(self._ohe.categories_[i])
                if drop_idx is not None and drop_idx[i] is not None:
                    n -= 1
                self._cat_to_ohe[cat_col] = list(range(offset, offset + n))
                offset += n

            return self._encode_transform(X)

        return X[self._numeric_cols].astype(float).copy()

    def _encode_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """저장된 OHE 로 변환한다."""
        X = X[self.feature_columns]
        if self._ohe is not None and self._cat_cols:
            ohe_data = self._ohe.transform(X[self._cat_cols].astype(str))
            ohe_names = list(self._ohe.get_feature_names_out(self._cat_cols))
            X_ohe = pd.DataFrame(ohe_data, columns=ohe_names, index=X.index)
            X_num = X[self._numeric_cols].astype(float)
            return pd.concat([X_num, X_ohe], axis=1)
        return X[self._numeric_cols].astype(float).copy()

    # ------------------------------------------------------------------
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame | None = None,
        y_valid: pd.Series | None = None,
        early_stopping_rounds: int | None = None,
    ) -> "ElasticNetModel":
        self.feature_columns = list(X_train.columns)
        X_enc = self._encode_fit(X_train)
        # SHAP background: 학습 데이터의 column-wise mean (단일 행)
        self._background = X_enc.mean(axis=0).to_frame().T

        merged = {**self.DEFAULT_PARAMS, **self.params}
        merged.pop("focal_gamma", None)
        merged.pop("focal_alpha", None)

        # sklearn 1.8+ 에서는 penalty / n_jobs 파라미터가 deprecated
        lr_kwargs: dict[str, Any] = dict(
            solver="saga",
            random_state=self.random_state,
            **merged,
        )
        if not _SKLEARN_GE_18:
            lr_kwargs["penalty"] = "elasticnet"
            lr_kwargs["n_jobs"] = -1

        self.model = LogisticRegression(**lr_kwargs)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            self.model.fit(X_enc, y_train)

        self.best_iteration = None
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_enc = self._encode_transform(X[self.feature_columns])
        return self.model.predict_proba(X_enc)[:, 1]

    def feature_importance(self) -> dict[str, float]:
        """원본 피처별 |계수| 를 반환한다. 범주형은 OHE 더미 중 max 값으로 대표."""
        coef = np.abs(self.model.coef_[0])
        result: dict[str, float] = {}

        for i, col in enumerate(self._numeric_cols):
            result[col] = float(coef[i])

        if self._ohe is not None:
            ohe_coef = coef[len(self._numeric_cols):]
            for cat_col, indices in self._cat_to_ohe.items():
                result[cat_col] = float(ohe_coef[indices].max()) if indices else 0.0

        return result

    def shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """raw-margin SHAP 를 원본 피처 공간으로 집계해 반환한다.

        Returns:
            (n_samples, n_original_features + 1).
            마지막 열은 base value (raw-margin 도메인).
        """
        import shap  # optional heavy dep; import deferred

        X_enc = self._encode_transform(X[self.feature_columns])
        explainer = shap.LinearExplainer(self.model, self._background)
        shap_expanded = np.asarray(explainer.shap_values(X_enc))  # (n, n_expanded)

        n_samples = len(X_enc)
        n_original = len(self.feature_columns)
        n_numeric = len(self._numeric_cols)
        result = np.zeros((n_samples, n_original + 1), dtype=float)

        # 수치형: 원본 피처 위치에 그대로 복사
        for i, col in enumerate(self._numeric_cols):
            orig_idx = self.feature_columns.index(col)
            result[:, orig_idx] = shap_expanded[:, i]

        # 범주형: OHE 더미 SHAP 를 원본 컬럼으로 합산
        if self._ohe is not None:
            for cat_col, ohe_indices in self._cat_to_ohe.items():
                orig_idx = self.feature_columns.index(cat_col)
                expanded_idx = [n_numeric + j for j in ohe_indices]
                if expanded_idx:
                    result[:, orig_idx] = shap_expanded[:, expanded_idx].sum(axis=1)

        # 마지막 열: base value (expected_value, raw-margin 도메인)
        result[:, -1] = float(explainer.expected_value)
        return result
