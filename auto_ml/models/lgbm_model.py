"""LightGBM 모델 래퍼.

LightGBM 은 범주형 컬럼을 native 로 처리할 수 있다. ``category`` dtype 으로
변환된 컬럼을 ``categorical_feature`` 인자로 넘기면 인코딩 없이도 학습
가능하다.
"""
from __future__ import annotations

from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from auto_ml.models.base import BaseModel
from auto_ml.models.losses import _sigmoid, lgb_focal_objective


class LGBMModel(BaseModel):
    """LightGBM 분류기 래퍼."""

    name = "lgbm"
    ITER_PARAM_NAME = "n_estimators"

    # 운영에서 보수적으로 동작하도록 한 기본 하이퍼파라미터.
    # 사용자 params 가 우선이며, 본 dict 는 그 위에 덮어써지는 fallback 이다.
    DEFAULT_PARAMS: dict[str, Any] = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 5,
        "min_data_in_leaf": 50,
        "verbose": -1,
        "n_estimators": 2000,
    }

    def _resolved_params(self) -> dict[str, Any]:
        """기본값 ⊕ 사용자 지정값 ⊕ 시드 를 합친 최종 파라미터를 만든다.

        ``self.loss == "focal"`` 이면 LightGBM 의 ``objective`` 를 callable 로
        덮어쓴다. ``focal_gamma`` / ``focal_alpha`` 는 라이브러리 원본 키가 아니므로
        반드시 pop 해서 ``LGBMClassifier`` 생성자에 전달되지 않게 한다.
        """
        merged = {**self.DEFAULT_PARAMS, **self.params}
        merged.setdefault("random_state", self.random_state)

        if self.loss == "focal":
            gamma = float(merged.pop("focal_gamma", 2.0))
            alpha = float(merged.pop("focal_alpha", 0.25))
            # callable 클래스 인스턴스 — joblib pickle 가능하고 LightGBM 의
            # signature 검사가 2-인자로 정확히 잡힘.
            merged["objective"] = lgb_focal_objective(gamma=gamma, alpha=alpha)
            self._focal_active = True
        else:
            # 잔여 키 방어적 pop (사용자가 잘못 넣어둔 경우 라이브러리 경고 회피)
            merged.pop("focal_gamma", None)
            merged.pop("focal_alpha", None)
            self._focal_active = False
        return merged

    def _to_categorical(self, X: pd.DataFrame) -> pd.DataFrame:
        """범주형 컬럼을 category dtype 으로 변환한다 — LightGBM native 처리용."""
        if not self.categorical_columns:
            return X
        out = X.copy()
        for col in self.categorical_columns:
            if col in out.columns:
                out[col] = out[col].astype("category")
        return out

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame | None = None,
        y_valid: pd.Series | None = None,
        early_stopping_rounds: int | None = None,
    ) -> "LGBMModel":
        self.feature_columns = list(X_train.columns)
        params = self._resolved_params()
        n_estimators = params.pop("n_estimators", 2000)

        X_tr = self._to_categorical(X_train)
        eval_set = None
        if X_valid is not None and y_valid is not None:
            eval_set = [(self._to_categorical(X_valid), y_valid)]

        callbacks = []
        if early_stopping_rounds and eval_set is not None:
            callbacks.append(lgb.early_stopping(early_stopping_rounds, verbose=False))
        # 학습 진행 로그를 너무 많이 찍지 않도록 100 라운드마다만 출력
        callbacks.append(lgb.log_evaluation(period=0))

        self.model = lgb.LGBMClassifier(n_estimators=n_estimators, **params)
        self.model.fit(
            X_tr,
            y_train,
            eval_set=eval_set,
            callbacks=callbacks,
            categorical_feature=self.categorical_columns or "auto",
        )
        # 조기종료가 동작했다면 best_iteration_ 가 설정된다.
        self.best_iteration = getattr(self.model, "best_iteration_", None)
        # LightGBM 은 leaf-wise 성장이라 max_depth 파라미터가 실제 깊이를 보장하지 않는다.
        # 학습 완료 후 실제 트리 구조에서 최대 깊이를 계산한다.
        tree_df = self.model.booster_.trees_to_dataframe()
        self.actual_max_depth = int(tree_df["node_depth"].max())
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_in = self._to_categorical(X[self.feature_columns])
        if self._focal_active:
            # LightGBM 은 custom objective 를 쓰면 predict_proba 가 (n,) 모양의
            # raw margin 을 반환하고 경고를 출력한다. raw_score=True 를 명시해
            # 동일 결과를 깨끗하게 받아 sigmoid 로 [0,1] 확률로 환산한다.
            raw = self.model.predict(X_in, raw_score=True)
            return _sigmoid(np.asarray(raw))
        # LGBMClassifier.predict_proba 는 (n_samples, 2) 모양이라 양성 컬럼만 추출
        return self.model.predict_proba(X_in)[:, 1]

    def feature_importance(self) -> dict[str, float]:
        # importance_type='gain' 이 분포 왜곡이 적어 운영 해석에 더 유용
        importances = self.model.booster_.feature_importance(importance_type="gain")
        return dict(zip(self.feature_columns, map(float, importances)))

    def shap_values(self, X: pd.DataFrame) -> np.ndarray:
        # Booster.predict(pred_contrib=True) 는 (n, n_features + 1) 의
        # raw-margin 도메인 SHAP 을 반환한다. 마지막 열이 expected value.
        # focal loss 활성 시에도 동일 — Booster 출력은 항상 raw margin.
        X_in = self._to_categorical(X[self.feature_columns])
        contrib = self.model.booster_.predict(X_in, pred_contrib=True)
        return np.asarray(contrib)
