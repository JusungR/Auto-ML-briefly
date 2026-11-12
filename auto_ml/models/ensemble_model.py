"""가중 평균 앙상블 모델.

학습된 BaseModel 인스턴스들을 내부에 보유하고,
predict_proba / shap_values 를 가중 평균으로 결합한다. 예측은 확률의 가중 평균이며,
확률 도메인 SHAP(``shap_values_proba``) 도 같은 방식으로 결합해 합이 예측과 일치한다.

registry.py 에 등록하지 않는다 — 이름+파라미터만으로 인스턴스를
생성할 수 없기 때문이다 (Trainer 가 직접 생성).

가중치 산출: test_metrics[primary_metric] 에 비례 (높을수록 더 큰 가중치).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from auto_ml.models.base import BaseModel
from auto_ml.utils.shap_domain import margin_shap_to_probability

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
        """elasticnet + 나머지 활성 모델 중 best 1개 앙상블.

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

    def set_n_jobs(self, n_jobs: int | None) -> "EnsembleModel":
        """스레드 수를 모든 서브모델에 전달한다 (서브모델은 순서대로 예측한다)."""
        for model in self.base_models.values():
            model.set_n_jobs(n_jobs)
        if n_jobs is not None:
            from auto_ml.utils.threads import resolve_n_jobs

            self.n_jobs = resolve_n_jobs(n_jobs)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        proba = np.zeros(len(X), dtype=float)
        for name, model in self.base_models.items():
            proba += self.weights[name] * model.predict_proba(X)
        return proba

    def feature_importance(self) -> dict[str, float]:
        """서브모델 feature importance 의 가중 평균.

        서브모델마다 중요도 단위가 다르다 (LightGBM/XGBoost gain 은 수천 이상,
        CatBoost 는 합 100, ElasticNet 은 |계수| ~1). 그대로 가중합하면 큰 단위 모델이
        사실상 전부를 차지하므로, 서로 다른 종류가 섞이면 각 서브모델 중요도를 합이 1 인
        비율로 바꾼 뒤 가중합한다 (결과도 합 1). 같은 종류만 있으면 (cv_bagging 의
        fold 모델들) 단위가 같으므로 원래 값의 가중 평균을 유지한다.
        """
        per_model = {name: model.feature_importance() for name, model in self.base_models.items()}
        mixed_kinds = len({type(m) for m in self.base_models.values()}) > 1

        combined: dict[str, float] = {}
        for name, imp in per_model.items():
            w = self.weights[name]
            if mixed_kinds:
                total = float(sum(abs(v) for v in imp.values()))
                if total <= 0:
                    continue  # 정보 없는 서브모델은 기여 0
                imp = {feat: val / total for feat, val in imp.items()}
            for feat, val in imp.items():
                combined[feat] = combined.get(feat, 0.0) + w * val
        # 모든 피처 키를 유지 (기여 0 인 피처 포함)
        for imp in per_model.values():
            for feat in imp:
                combined.setdefault(feat, 0.0)
        return combined

    def shap_values_proba(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """확률 도메인 SHAP — 서브모델별로 확률 도메인으로 바꾼 뒤 가중 평균한다.

        앙상블 예측은 서브모델 **확률**의 가중 평균이므로, SHAP 도 확률 도메인에서
        같은 가중치로 평균해야 합이 실제 예측과 일치한다:

            base     = Σ w_m · p_base_m
            shap[i]  = Σ w_m · shap_prob_m[i]
            base + Σ shap = Σ w_m · p_pred_m = predict_proba(X)

        Returns:
            ``(shap_prob (n, n_features), base_prob (n,))``. Explainer 가 우선 사용한다.
        """
        shap_sum: np.ndarray | None = None
        base_sum: np.ndarray | None = None
        for name, model in self.base_models.items():
            w = self.weights[name]
            if hasattr(model, "shap_values_proba"):   # 중첩 앙상블
                sp, bp = model.shap_values_proba(X)
            else:
                sp, bp, _ = margin_shap_to_probability(model.shap_values(X))
            shap_sum = w * sp if shap_sum is None else shap_sum + w * sp
            base_sum = w * bp if base_sum is None else base_sum + w * bp
        assert shap_sum is not None and base_sum is not None
        return shap_sum, base_sum

    def shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """서브모델 raw-margin SHAP 의 가중 평균 (logit 도메인).

        각 서브모델은 원본 피처 공간 (n, n_features+1) 을 반환하므로
        앙상블도 동일한 shape 를 유지한다.

        주의: 앙상블 예측은 확률 평균이므로 이 값의 합에 sigmoid 를 씌워도
        ``predict_proba`` 와 일치하지 않는다. 확률 해석에는 ``shap_values_proba`` 를 쓴다.
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
