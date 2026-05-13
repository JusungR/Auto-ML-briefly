"""XGBoost 모델 래퍼.

XGBoost 도 2.x 부터 ``enable_categorical=True`` 로 native 범주형 처리가
가능하지만, 폐쇄망에서 버전이 더 낮을 가능성을 고려해 LabelEncoder 로
범주형을 정수로 변환한 뒤 학습한다. 학습 시 fit 된 인코더를 보관해
스코어링 시에도 동일 매핑을 적용한다.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

from auto_ml.models.base import BaseModel
from auto_ml.models.losses import xgb_focal_objective


class XGBModel(BaseModel):
    """XGBoost 분류기 래퍼."""

    name = "xgb"

    DEFAULT_PARAMS: dict[str, Any] = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "min_child_weight": 5,
        "tree_method": "hist",  # 대용량에서 가장 일관된 성능
        "n_estimators": 2000,
    }

    # 학습 시 학습된 컬럼별 LabelEncoder (스코어링 시 재사용)
    _encoders: dict[str, LabelEncoder]
    # 미등록 카테고리(unseen) 매핑용 — 안전하게 -1 로 치환
    UNKNOWN_CATEGORY_CODE: int = -1

    def __init__(
        self,
        params: dict[str, Any] | None = None,
        categorical_columns: list[str] | None = None,
        random_state: int = 42,
        loss: str = "logloss",
    ) -> None:
        super().__init__(
            params=params,
            categorical_columns=categorical_columns,
            random_state=random_state,
            loss=loss,
        )
        self._encoders = {}

    def _resolved_params(self) -> dict[str, Any]:
        """기본값 ⊕ 사용자 지정값 ⊕ 시드. ``self.loss == "focal"`` 분기 포함.

        XGBoost 의 ``predict_proba`` 는 custom objective 일 때도 sigmoid 적용된
        2D 확률을 반환한다 (실측). 따라서 ``predict_proba`` 후처리는 불필요하다.
        다만 일부 버전에서 custom obj 시 ``base_score`` 자동 추정이 동작하지 않을
        수 있으므로 0.5 (margin 0) 으로 명시한다.
        """
        merged = {**self.DEFAULT_PARAMS, **self.params}
        merged.setdefault("random_state", self.random_state)

        if self.loss == "focal":
            gamma = float(merged.pop("focal_gamma", 2.0))
            alpha = float(merged.pop("focal_alpha", 0.25))
            merged["objective"] = xgb_focal_objective(gamma=gamma, alpha=alpha)
            merged.setdefault("base_score", 0.5)
            self._focal_active = True
        else:
            merged.pop("focal_gamma", None)
            merged.pop("focal_alpha", None)
            self._focal_active = False
        return merged

    def _encode_fit(self, X: pd.DataFrame) -> pd.DataFrame:
        """범주형 컬럼에 LabelEncoder 를 fit 하고 정수로 변환한다."""
        out = X.copy()
        self._encoders = {}
        for col in self.categorical_columns:
            if col not in out.columns:
                continue
            enc = LabelEncoder()
            values = out[col].astype(str).fillna("MISSING").values
            out[col] = enc.fit_transform(values)
            self._encoders[col] = enc
        return out

    def _encode_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """학습 시 fit 된 인코더로 동일 매핑을 적용한다 (unseen 은 -1)."""
        out = X.copy()
        for col, enc in self._encoders.items():
            if col not in out.columns:
                continue
            values = out[col].astype(str).fillna("MISSING").values
            known = set(enc.classes_)
            mapped = np.array(
                [enc.transform([v])[0] if v in known else self.UNKNOWN_CATEGORY_CODE for v in values]
            )
            out[col] = mapped
        return out

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame | None = None,
        y_valid: pd.Series | None = None,
        early_stopping_rounds: int | None = None,
    ) -> "XGBModel":
        self.feature_columns = list(X_train.columns)
        params = self._resolved_params()
        n_estimators = params.pop("n_estimators", 2000)

        X_tr = self._encode_fit(X_train)
        eval_set = None
        if X_valid is not None and y_valid is not None:
            eval_set = [(self._encode_transform(X_valid), y_valid)]

        # XGBoost 2.x 의 sklearn API: early_stopping_rounds 는 생성자 인자
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            early_stopping_rounds=early_stopping_rounds if eval_set else None,
            **params,
        )
        self.model.fit(X_tr, y_train, eval_set=eval_set, verbose=False)
        self.best_iteration = getattr(self.model, "best_iteration", None)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_in = self._encode_transform(X[self.feature_columns])
        return self.model.predict_proba(X_in)[:, 1]

    def feature_importance(self) -> dict[str, float]:
        # gain 기준이 의미 해석이 가장 명확
        booster = self.model.get_booster()
        score = booster.get_score(importance_type="gain")
        # XGBoost 는 사용되지 않은 feature 를 dict 에서 빠뜨리므로 0 으로 채워준다.
        return {name: float(score.get(name, 0.0)) for name in self.feature_columns}
