"""다중 모델 학습 / 비교기.

수행 흐름 (모델당):
    1) 튜닝 (옵션) — Optuna TPE 로 ``primary_metric`` 을 최대화하는 파라미터 탐색
       - 평가는 학습 데이터의 KFold OOF (테스트 데이터는 사용하지 않음)
    2) CV (StratifiedKFold) — 1) 의 best params 로 OOF 예측 수집 → 안정적 평가
    3) 최종 fit — 학습 전체 + 테스트 데이터를 valid 로 사용한 조기종료 fit
    4) 테스트 평가 — 최종 모델로 테스트 데이터에 대한 지표 계산

best 모델 선정은 테스트 지표 기준 (모든 모델이 동일한 테스트 데이터에 대해 평가됨).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from auto_ml.config import AutoMLConfig
from auto_ml.models.base import BaseModel
from auto_ml.models.registry import build_model
from auto_ml.reporting.metrics import compute_metrics
from auto_ml.tuning import HyperparameterOptimizer, TuningResult
from auto_ml.utils.logger import get_logger

logger = get_logger("trainer")


@dataclass
class ModelResult:
    """단일 모델 학습 결과."""

    name: str
    model: BaseModel                                  # 테스트 평가까지 끝난 최종 fit 모델
    params: dict[str, Any]                            # 실제로 사용된 파라미터 (튜닝 결과 포함)
    oof_proba: np.ndarray                             # 학습 데이터의 fold 별 OOF 확률
    test_proba: np.ndarray                            # 테스트 데이터에 대한 확률
    train_proba: np.ndarray                           # 학습 데이터에 대한 in-sample 확률 (overfit 점검용)
    cv_metrics: dict[str, float]                      # OOF 기반 지표
    test_metrics: dict[str, float]                    # 테스트 기반 지표
    train_metrics: dict[str, float]                   # in-sample (final_model fit→predict on X_train) 지표
    feature_importance: dict[str, float]              # gain 기준 중요도
    fold_best_iterations: list[int | None] = field(default_factory=list)
    tuning: TuningResult | None = None                # 튜닝을 수행한 경우의 결과


@dataclass
class TrainingResult:
    """전체 학습 결과 묶음."""

    results: dict[str, ModelResult]                   # 모델 이름 → 결과
    best_model_name: str
    primary_metric: str
    train_y: np.ndarray                               # 리포트 overfit 점검용
    test_y: np.ndarray                                # 리포트 차트용
    feature_columns: list[str]

    @property
    def best(self) -> ModelResult:
        return self.results[self.best_model_name]


class Trainer:
    """3개 모델을 동시에 학습하고 best 를 선정하는 오케스트레이터."""

    def __init__(self, config: AutoMLConfig) -> None:
        self.config = config
        self.optimizer = HyperparameterOptimizer(config)

    # ------------------------------------------------------------------
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> TrainingResult:
        """학습 + 평가 + best 선정 한 번에 수행.

        Args:
            X_train, y_train: 전처리 완료된 학습 데이터.
            X_test, y_test:  전처리 완료된 테스트 데이터.

        Returns:
            ``TrainingResult`` — 모든 모델의 결과와 best 정보.
        """
        enabled_models = [name for name, mc in self.config.models.items() if mc.enabled]
        if not enabled_models:
            raise ValueError("No model is enabled in config.models")

        results: dict[str, ModelResult] = {}
        for name in enabled_models:
            logger.info("=== Training pipeline: %s ===", name)
            results[name] = self._train_single(
                name=name,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
            )

        # best 선정 — 사용자 override 가 있으면 그 모형, 아니면 primary_metric 최대값.
        primary = self.config.training.primary_metric
        override = self.config.training.best_model
        if override is not None:
            if override not in results:
                raise ValueError(
                    f"Unknown best_model override: {override!r}. "
                    f"Available: {sorted(results.keys())}"
                )
            best_name = override
            logger.info(
                "Best model: %s (user override; %s=%.4f)",
                best_name, primary, results[best_name].test_metrics[primary],
            )
        else:
            best_name = max(
                results.keys(), key=lambda n: results[n].test_metrics[primary]
            )
            logger.info(
                "Best model: %s (auto-selected by %s=%.4f)",
                best_name, primary, results[best_name].test_metrics[primary],
            )

        return TrainingResult(
            results=results,
            best_model_name=best_name,
            primary_metric=primary,
            train_y=y_train.to_numpy(),
            test_y=y_test.to_numpy(),
            feature_columns=list(X_train.columns),
        )

    # ------------------------------------------------------------------
    def _resolve_params(
        self,
        name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        categorical_columns: list[str],
    ) -> tuple[dict[str, Any], TuningResult | None]:
        """튜닝 활성 여부에 따라 모델별 파라미터를 결정한다.

        Returns:
            (params, tuning_result) — 튜닝을 생략하면 tuning_result 는 None.
        """
        cfg = self.config
        model_cfg = cfg.models[name]
        if cfg.tuning.enabled and model_cfg.search_space:
            tuning = self.optimizer.optimize(
                model_name=name,
                X=X_train,
                y=y_train,
                search_space=model_cfg.search_space,
                fixed_params=model_cfg.fixed_params,
                categorical_columns=categorical_columns,
            )
            return tuning.best_params, tuning
        # 튜닝 비활성 또는 search_space 미지정 → fixed_params 만 사용
        return dict(model_cfg.fixed_params), None

    # ------------------------------------------------------------------
    def _train_single(
        self,
        name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> ModelResult:
        """단일 모델: 튜닝 → CV(OOF) → 최종 fit → 테스트 평가."""
        cfg = self.config

        # feature selection 이 컬럼을 줄였을 수 있으므로, 실제 데이터에
        # 남아 있는 범주형 컬럼만 모델에 전달한다.
        actual_columns = set(X_train.columns)
        cat_cols = [c for c in cfg.categorical_columns if c in actual_columns]

        params, tuning_result = self._resolve_params(
            name, X_train, y_train, cat_cols,
        )

        # ----- CV(OOF) 평가 -------------------------------------------------
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
                params=params,
                categorical_columns=cat_cols,
                random_state=cfg.training.random_state + fold_idx,
                loss=cfg.models[name].loss,
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

        # ----- 최종 모델: 학습 전체 + 테스트 데이터를 조기종료 valid 로 사용 -----
        # 주의: 테스트 셋을 early-stopping 신호로만 사용하고, best 모델 선정은 동일 테스트
        # 점수로 한다. 폐쇄 운영 환경에서 단일 데이터셋 사용 일관성을 우선시한다.
        final_model = build_model(
            name=name,
            params=params,
            categorical_columns=cat_cols,
            random_state=cfg.training.random_state,
            loss=cfg.models[name].loss,
        )
        final_model.fit(
            X_train, y_train, X_test, y_test,
            early_stopping_rounds=cfg.training.early_stopping_rounds,
        )
        test_proba = final_model.predict_proba(X_test)
        # In-sample 점수 — 학습 데이터에 fit 한 모델로 같은 학습 데이터를 다시 스코어링.
        # OOF 와 달리 unbiased 가 아니라 의도적으로 over-optimistic 한 추정치이며,
        # test_metrics 와의 gap 이 메모리제이션(overfit) 강도 신호가 된다.
        train_proba = final_model.predict_proba(X_train)

        return ModelResult(
            name=name,
            model=final_model,
            params=params,
            oof_proba=oof_proba,
            test_proba=test_proba,
            train_proba=train_proba,
            cv_metrics=compute_metrics(y_train.to_numpy(), oof_proba),
            test_metrics=compute_metrics(y_test.to_numpy(), test_proba),
            train_metrics=compute_metrics(y_train.to_numpy(), train_proba),
            feature_importance=final_model.feature_importance(),
            fold_best_iterations=fold_best_iters,
            tuning=tuning_result,
        )
