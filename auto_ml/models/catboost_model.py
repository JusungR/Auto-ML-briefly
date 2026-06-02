"""CatBoost 모델 래퍼.

CatBoost 는 범주형을 매우 잘 다루는 부스팅 모델이다. ``cat_features`` 인자에
범주형 컬럼 인덱스를 전달하면 내부에서 target encoding 변형을 자동 적용한다.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from auto_ml.models.base import BaseModel
from auto_ml.models.losses import CatBoostFocalObjective


class CatBoostModel(BaseModel):
    """CatBoost 분류기 래퍼."""

    name = "catboost"
    ITER_PARAM_NAME = "iterations"

    DEFAULT_PARAMS: dict[str, Any] = {
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "learning_rate": 0.05,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "iterations": 2000,
        # 학습 로그를 줄여 운영 콘솔 가독성 확보
        "verbose": False,
        "allow_writing_files": False,
    }

    def _resolved_params(self) -> dict[str, Any]:
        """기본값 ⊕ 사용자 지정값 ⊕ 시드. ``self.loss == "focal"`` 분기 포함.

        CatBoost 는 user-defined loss 에 ``calc_ders_range`` 를 가진 객체를 받는다.
        ``predict_proba`` 는 custom loss 일 때도 sigmoid 적용된 2D 확률을 반환하므로
        (실측) 후처리는 불필요하다.
        """
        merged = {**self.DEFAULT_PARAMS, **self.params}
        merged.setdefault("random_seed", self.random_state)

        if self.loss == "focal":
            gamma = float(merged.pop("focal_gamma", 2.0))
            alpha = float(merged.pop("focal_alpha", 0.25))
            merged["loss_function"] = CatBoostFocalObjective(gamma=gamma, alpha=alpha)
            self._focal_active = True
        else:
            merged.pop("focal_gamma", None)
            merged.pop("focal_alpha", None)
            self._focal_active = False
        return merged

    def _to_pool(self, X: pd.DataFrame, y: pd.Series | None = None) -> Pool:
        """CatBoost 의 Pool 객체로 감싼다 (cat_features 자동 지정)."""
        cat_features = [c for c in self.categorical_columns if c in X.columns]
        # CatBoost 는 string/None 도 받지만, NaN 보다 명시적 문자열이 안전하다.
        X_in = X.copy()
        for col in cat_features:
            X_in[col] = X_in[col].astype(str).fillna("MISSING")
        return Pool(data=X_in, label=y, cat_features=cat_features)

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame | None = None,
        y_valid: pd.Series | None = None,
        early_stopping_rounds: int | None = None,
    ) -> "CatBoostModel":
        self.feature_columns = list(X_train.columns)
        params = self._resolved_params()
        if early_stopping_rounds:
            params.setdefault("early_stopping_rounds", early_stopping_rounds)

        train_pool = self._to_pool(X_train, y_train)
        eval_pool = self._to_pool(X_valid, y_valid) if X_valid is not None and y_valid is not None else None

        self.model = CatBoostClassifier(**params)
        self.model.fit(train_pool, eval_set=eval_pool, use_best_model=eval_pool is not None)
        self.best_iteration = self.model.get_best_iteration()
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        pool = self._to_pool(X[self.feature_columns])
        return self.model.predict_proba(pool)[:, 1]

    def feature_importance(self) -> dict[str, float]:
        importances = self.model.get_feature_importance()
        return dict(zip(self.feature_columns, map(float, importances)))

    def shap_values(self, X: pd.DataFrame) -> np.ndarray:
        # CatBoost 는 get_feature_importance(type='ShapValues', data=Pool) 로
        # (n, n_features + 1) 의 raw-margin SHAP 을 직접 반환. 마지막 열 = base value.
        pool = self._to_pool(X[self.feature_columns])
        contrib = self.model.get_feature_importance(type="ShapValues", data=pool)
        return np.asarray(contrib)
