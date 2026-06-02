"""가중 평균 앙상블 모델.

학습된 BaseModel 인스턴스들을 내부에 보유하고,
predict_proba / shap_values 를 가중 평균으로 결합한다.

registry.py 에 등록하지 않는다 — 이름+파라미터만으로 인스턴스를
생성할 수 없기 때문이다 (Trainer 가 직접 생성).

가중치 산출: test_metrics[primary_metric] 에 비례 (높을수록 더 큰 가중치).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from auto_ml.models.base import BaseModel

if TYPE_CHECKING:
    from auto_ml.models.trainer import ModelResult


class EnsembleModel(BaseModel):
    """학습된 모델들의 가중 평균 앙상블."""

    name = "ensemble"

    def __init__(
        self,
        base_models: dict[str, BaseModel],
        weights: dict[str, float],
    ) -> None:
        first = next(iter(base_models.values()))
        super().__init__(
            params={},
            categorical_columns=list(first.categorical_columns),
            random_state=42,
            loss="logloss",
        )
        self.base_models = base_models      # {model_name: fitted BaseModel}
        self.weights = weights               # {model_name: normalized weight}
        self.feature_columns = list(first.feature_columns)

    # ------------------------------------------------------------------
    @classmethod
    def from_results(
        cls,
        results: dict[str, "ModelResult"],
        primary_metric: str,
    ) -> "EnsembleModel":
        """전체 모델의 점수 비례 가중 평균 앙상블."""
        scores = {name: mr.test_metrics[primary_metric] for name, mr in results.items()}
        total = sum(scores.values())
        if total > 0:
            weights = {name: s / total for name, s in scores.items()}
        else:
            weights = {name: 1.0 / len(scores) for name in scores}
        base_models = {name: mr.model for name, mr in results.items()}
        return cls(base_models=base_models, weights=weights)

    @classmethod
    def from_elasticnet_plus_best(
        cls,
        results: dict[str, "ModelResult"],
        primary_metric: str,
        elasticnet_weight: float | None = None,
    ) -> "EnsembleModel":
        """elasticnet + 나머지 best 모델 2개 앙상블.

        elasticnet 은 무조건 포함하고, 나머지 활성 모델 중
        test primary_metric 이 가장 높은 1개를 결합한다.

        Args:
            elasticnet_weight: elasticnet 의 가중치 (0 < w < 1).
                               None 이면 점수 비례로 자동 계산.
        """
        if "elasticnet" not in results:
            raise ValueError(
                "strategy='elasticnet_plus_best' 를 사용하려면 elasticnet 모델이 "
                "활성화되어 있어야 합니다."
            )
        others = {n: mr for n, mr in results.items() if n != "elasticnet"}
        if not others:
            raise ValueError(
                "elasticnet 외에 최소 1개의 활성 모델이 필요합니다."
            )
        best_name = max(others, key=lambda n: others[n].test_metrics[primary_metric])

        en_score = results["elasticnet"].test_metrics[primary_metric]
        best_score = others[best_name].test_metrics[primary_metric]

        if elasticnet_weight is not None:
            w_en = float(elasticnet_weight)
            if not (0.0 < w_en < 1.0):
                raise ValueError(f"elasticnet_weight 는 (0, 1) 범위여야 합니다. 입력값: {w_en}")
        else:
            total = en_score + best_score
            w_en = en_score / total if total > 0 else 0.5
        w_best = 1.0 - w_en

        base_models = {
            "elasticnet": results["elasticnet"].model,
            best_name: others[best_name].model,
        }
        weights = {"elasticnet": w_en, best_name: w_best}
        return cls(base_models=base_models, weights=weights)

    @classmethod
    def from_cv_folds(
        cls,
        base_name: str,
        fold_models: list[BaseModel],
    ) -> "EnsembleModel":
        """동일 알고리즘의 K 개 fold 모델을 균등 가중 앙상블로 묶는다.

        CV-bagging (``final_fit_strategy=cv_bagging``) 용. 최종 재학습 대신 CV
        루프에서 학습된 fold 모델들을 그대로 평균한다. 테스트셋을 학습 신호로
        쓰지 않으므로 일반화 추정이 정직하다.

        Args:
            base_name: 원본 모델 이름 (예: ``"lgbm"``). 서브모델 키 prefix.
            fold_models: CV fold 별로 학습 완료된 BaseModel 인스턴스 리스트.

        Returns:
            ``EnsembleModel`` — 각 fold 에 1/K 균등 가중.
        """
        if not fold_models:
            raise ValueError("from_cv_folds: fold_models is empty")
        base_models = {f"{base_name}__fold{i}": m for i, m in enumerate(fold_models)}
        w = 1.0 / len(fold_models)
        weights = {k: w for k in base_models}
        return cls(base_models=base_models, weights=weights)

    # ------------------------------------------------------------------
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame | None = None,
        y_valid: pd.Series | None = None,
        early_stopping_rounds: int | None = None,
    ) -> "EnsembleModel":
        """No-op: 서브모델은 이미 학습 완료."""
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        proba = np.zeros(len(X), dtype=float)
        for name, model in self.base_models.items():
            proba += self.weights[name] * model.predict_proba(X)
        return proba

    def feature_importance(self) -> dict[str, float]:
        """서브모델 feature importance 의 가중 평균."""
        combined: dict[str, float] = {}
        for name, model in self.base_models.items():
            w = self.weights[name]
            for feat, val in model.feature_importance().items():
                combined[feat] = combined.get(feat, 0.0) + w * val
        return combined

    def shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """서브모델 raw-margin SHAP 의 가중 평균.

        각 서브모델은 원본 피처 공간 (n, n_features+1) 을 반환하므로
        앙상블도 동일한 shape 를 유지한다.
        """
        result: np.ndarray | None = None
        for name, model in self.base_models.items():
            sv = model.shap_values(X)
            if result is None:
                result = self.weights[name] * sv
            else:
                result = result + self.weights[name] * sv
        assert result is not None
        return result
