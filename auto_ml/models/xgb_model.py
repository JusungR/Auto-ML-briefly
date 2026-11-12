"""XGBoost 모델 래퍼.

XGBoost 2.x 의 native 범주형 처리(``enable_categorical=True``) 를 사용한다.
범주형 컬럼은 학습 때 본 범주 목록으로 고정한 pandas ``category`` 로 변환해
넘기므로, 추론 입력의 범주 구성이 달라도 같은 코드가 유지되고 학습 때 없던
범주는 결측(NaN) 으로 처리된다. 범주 사이에 순서를 가정하지 않는다
(partition 기반 분할).

과거 버전은 LabelEncoder 로 정수 코드를 만들어 수치처럼 분할했다. 그 방식으로
학습된 artifact 는 ``_encoders`` 를 가지고 있으므로 같은 방식으로 예측한다.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

from auto_ml.models.base import BaseModel
from auto_ml.models.losses import xgb_focal_objective
from auto_ml.utils.categorical import normalize_category_series


class XGBModel(BaseModel):
    """XGBoost 분류기 래퍼."""

    name = "xgb"
    ITER_PARAM_NAME = "n_estimators"
    THREAD_PARAM_NAMES = ("n_jobs", "nthread")
    # 범주 수 < max_cat_to_onehot 이면 one-hot 분할 — 실측(2.1.2): 범주 6개면 7 부터
    ONEHOT_PARAM_NAME = "max_cat_to_onehot"
    ONEHOT_OFFSET = 1

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

    # (과거 artifact 전용) LabelEncoder 방식의 컬럼별 인코더와 unseen 코드
    _encoders: dict[str, LabelEncoder]
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
        # 학습 때 본 범주 목록 (컬럼 → 정렬된 문자열 리스트). native 범주형 경로.
        self._categories: dict[str, list[str]] | None = None

    def _resolved_params(self) -> dict[str, Any]:
        """기본값 ⊕ 사용자 지정값 ⊕ 시드. ``self.loss == "focal"`` 분기 포함.

        XGBoost 의 ``predict_proba`` 는 custom objective 일 때도 sigmoid 적용된
        2D 확률을 반환한다 (실측). 따라서 ``predict_proba`` 후처리는 불필요하다.
        다만 일부 버전에서 custom obj 시 ``base_score`` 자동 추정이 동작하지 않을
        수 있으므로 0.5 (margin 0) 으로 명시한다.
        """
        merged = {**self.DEFAULT_PARAMS, **self._thread_params(), **self.params}
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

    # ------------------------------------------------------------------
    # native 범주형 (현재 방식)
    # ------------------------------------------------------------------
    def _fit_categories(self, X: pd.DataFrame) -> None:
        self._categories = {}
        for col in self.categorical_columns:
            if col not in X.columns:
                continue
            values = normalize_category_series(X[col]).dropna().unique()
            self._categories[col] = sorted(str(v) for v in values)

    def _to_model_input(self, X: pd.DataFrame) -> pd.DataFrame:
        """모델 입력 변환. 범주형은 학습 범주로 고정한 category dtype (unseen → NaN).

        과거 LabelEncoder 방식 artifact 는 기존 정수 매핑을 그대로 적용한다.
        """
        if getattr(self, "_categories", None) is None:
            return self._encode_transform(X)
        out = X.copy()
        for col, cats in self._categories.items():
            if col not in out.columns:
                continue
            out[col] = pd.Categorical(
                normalize_category_series(out[col]), categories=cats,
            )
        return out

    # ------------------------------------------------------------------
    # LabelEncoder (과거 artifact 호환용)
    # ------------------------------------------------------------------
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
            values = out[col].astype(str).fillna("MISSING")
            lookup = {cls: i for i, cls in enumerate(enc.classes_)}
            out[col] = values.map(lookup).fillna(self.UNKNOWN_CATEGORY_CODE).astype(int)
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
        params.update(self._onehot_params(X_train))
        n_estimators = params.pop("n_estimators", 2000)

        self._fit_categories(X_train)
        X_tr = self._to_model_input(X_train)
        eval_set = None
        if X_valid is not None and y_valid is not None:
            eval_set = [(self._to_model_input(X_valid), y_valid)]

        # XGBoost 2.x 의 sklearn API: early_stopping_rounds 는 생성자 인자
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            early_stopping_rounds=early_stopping_rounds if eval_set else None,
            enable_categorical=True,
            **params,
        )
        self.model.fit(X_tr, y_train, eval_set=eval_set, verbose=False)
        # XGBoost 의 best_iteration 은 0부터 세는 번호 → 트리 개수로 변환 (조기 종료 미발동 시 None)
        best_idx = self._booster_best_index()
        self.best_iteration = None if best_idx is None else best_idx + 1
        return self

    def _apply_n_jobs_to_fitted(self, n_jobs: int) -> None:
        # sklearn API 의 set_params 는 학습된 booster 의 nthread 까지 바꾼다.
        self.model.set_params(n_jobs=n_jobs)

    def _booster_best_index(self) -> int | None:
        """조기 종료가 동작했으면 best 지점의 0부터 세는 번호, 아니면 None.

        래퍼의 ``best_iteration`` 이 아니라 학습된 XGBoost 객체에서 직접 읽으므로
        (번호 체계가 달랐던) 과거 artifact 에도 그대로 맞다.
        """
        try:
            idx = self.model.best_iteration
        except AttributeError:
            return None
        return None if idx is None else int(idx)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_in = self._to_model_input(X[self.feature_columns])
        return self.model.predict_proba(X_in)[:, 1]

    def feature_importance(self) -> dict[str, float]:
        # gain 기준이 의미 해석이 가장 명확
        booster = self.model.get_booster()
        score = booster.get_score(importance_type="gain")
        # XGBoost 는 사용되지 않은 feature 를 dict 에서 빠뜨리므로 0 으로 채워준다.
        return {name: float(score.get(name, 0.0)) for name in self.feature_columns}

    def shap_values(self, X: pd.DataFrame) -> np.ndarray:
        # Booster.predict(pred_contribs=True) 는 (n, n_features + 1) 반환.
        # 마지막 열은 bias term (= base value, raw-margin 도메인).
        X_in = self._to_model_input(X[self.feature_columns])
        if getattr(self, "_categories", None) is None:
            # 과거 artifact: 정수 인코딩 행렬
            dmat = xgb.DMatrix(X_in.values, feature_names=list(X_in.columns))
        else:
            dmat = xgb.DMatrix(X_in, enable_categorical=True)
        # predict_proba(sklearn API) 는 조기 종료 시 best 지점까지의 트리만 쓴다.
        # SHAP 도 같은 범위로 계산해야 합이 실제 예측과 일치한다 (지정하지 않으면
        # best 이후 early_stopping_rounds 만큼의 트리까지 포함된다).
        best_idx = self._booster_best_index()
        iteration_range = (0, 0) if best_idx is None else (0, best_idx + 1)
        contrib = self.model.get_booster().predict(
            dmat, pred_contribs=True, iteration_range=iteration_range,
        )
        return np.asarray(contrib)
