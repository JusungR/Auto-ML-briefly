"""다중 모델 학습 / 비교기.

수행 흐름:
    1) ``StratifiedKFold`` 로 학습 데이터를 fold 분할 → 각 모델의 OOF 확률 수집
    2) holdout 평가 — 학습 데이터 전체로 다시 fit 한 모델로 hold-out 점수 계산
    3) ``primary_metric`` 기준으로 best 모델 선정

OOF 결과는 리포트의 ROC / PR 차트에 사용되어 보다 안정적인 평가가 가능하다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from auto_ml.config import AutoMLConfig
from auto_ml.models.base import BaseModel
from auto_ml.models.registry import build_model
from auto_ml.reporting.metrics import compute_metrics
from auto_ml.utils.logger import get_logger

logger = get_logger("trainer")


@dataclass
class ModelResult:
    """단일 모델 학습 결과."""

    name: str
    model: BaseModel                                  # holdout 평가까지 끝난 최종 fit 모델
    oof_proba: np.ndarray                             # 학습 데이터의 fold 별 OOF 확률
    holdout_proba: np.ndarray                         # holdout 데이터에 대한 확률
    cv_metrics: dict[str, float]                      # OOF 기반 지표
    holdout_metrics: dict[str, float]                 # holdout 기반 지표
    feature_importance: dict[str, float]              # gain 기준 중요도
    fold_best_iterations: list[int | None] = field(default_factory=list)


@dataclass
class TrainingResult:
    """전체 학습 결과 묶음."""

    results: dict[str, ModelResult]                   # 모델 이름 → 결과
    best_model_name: str
    primary_metric: str
    holdout_y: np.ndarray                             # 리포트 차트용
    feature_columns: list[str]

    @property
    def best(self) -> ModelResult:
        return self.results[self.best_model_name]


class Trainer:
    """3개 모델을 동시에 학습하고 best 를 선정하는 오케스트레이터."""

    def __init__(self, config: AutoMLConfig) -> None:
        self.config = config

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_holdout: pd.DataFrame,
        y_holdout: pd.Series,
    ) -> TrainingResult:
        """학습 + 평가 + best 선정 한 번에 수행.

        Args:
            X_train, y_train: 전처리 완료된 학습 데이터.
            X_holdout, y_holdout: 전처리 완료된 hold-out 데이터.

        Returns:
            ``TrainingResult`` — 모든 모델의 결과와 best 정보.
        """
        enabled_models = [name for name, mc in self.config.models.items() if mc.enabled]
        if not enabled_models:
            raise ValueError("No model is enabled in config.models")

        results: dict[str, ModelResult] = {}
        for name in enabled_models:
            logger.info("Training model: %s", name)
            results[name] = self._train_single(
                name=name,
                X_train=X_train,
                y_train=y_train,
                X_holdout=X_holdout,
                y_holdout=y_holdout,
            )

        # primary_metric 기준 best 선정 (holdout 점수 사용 — overfitting 방지)
        primary = self.config.training.primary_metric
        best_name = max(
            results.keys(), key=lambda n: results[n].holdout_metrics[primary]
        )
        logger.info(
            "Best model: %s (%s=%.4f)",
            best_name,
            primary,
            results[best_name].holdout_metrics[primary],
        )

        return TrainingResult(
            results=results,
            best_model_name=best_name,
            primary_metric=primary,
            holdout_y=y_holdout.to_numpy(),
            feature_columns=list(X_train.columns),
        )

    # ------------------------------------------------------------------
    def _train_single(
        self,
        name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_holdout: pd.DataFrame,
        y_holdout: pd.Series,
    ) -> ModelResult:
        """단일 모델에 대해 OOF + holdout 학습을 수행한다."""
        cfg = self.config
        kf = StratifiedKFold(
            n_splits=cfg.training.cv_folds,
            shuffle=True,
            random_state=cfg.training.random_state,
        )

        oof_proba = np.zeros(len(X_train), dtype=float)
        fold_best_iters: list[int | None] = []

        for fold_idx, (tr_idx, va_idx) in enumerate(kf.split(X_train, y_train)):
            X_tr = X_train.iloc[tr_idx]
            y_tr = y_train.iloc[tr_idx]
            X_va = X_train.iloc[va_idx]
            y_va = y_train.iloc[va_idx]

            model = build_model(
                name=name,
                params=cfg.models[name].params,
                categorical_columns=cfg.categorical_columns,
                random_state=cfg.training.random_state + fold_idx,
            )
            model.fit(
                X_tr, y_tr, X_va, y_va,
                early_stopping_rounds=cfg.training.early_stopping_rounds,
            )
            oof_proba[va_idx] = model.predict_proba(X_va)
            fold_best_iters.append(model.best_iteration)
            logger.info(
                "  fold %d/%d done (best_iter=%s)",
                fold_idx + 1, cfg.training.cv_folds, model.best_iteration,
            )

        # 최종 모델: 전체 train 으로 다시 fit. 조기종료 위해 holdout 을 valid 로 사용.
        final_model = build_model(
            name=name,
            params=cfg.models[name].params,
            categorical_columns=cfg.categorical_columns,
            random_state=cfg.training.random_state,
        )
        final_model.fit(
            X_train, y_train, X_holdout, y_holdout,
            early_stopping_rounds=cfg.training.early_stopping_rounds,
        )
        holdout_proba = final_model.predict_proba(X_holdout)

        return ModelResult(
            name=name,
            model=final_model,
            oof_proba=oof_proba,
            holdout_proba=holdout_proba,
            cv_metrics=compute_metrics(y_train.to_numpy(), oof_proba),
            holdout_metrics=compute_metrics(y_holdout.to_numpy(), holdout_proba),
            feature_importance=final_model.feature_importance(),
            fold_best_iterations=fold_best_iters,
        )
