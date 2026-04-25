"""모델별 베이지안 하이퍼파라미터 최적화 실행기.

Optuna 의 TPE 샘플러를 사용해 ``primary_metric`` 을 최대화하는 파라미터를 찾는다.
탐색 단계의 평가는 학습 데이터에 대한 KFold OOF 점수의 평균이다 — 테스트
데이터는 본 단계에서 절대 사용되지 않는다 (정보 누설 방지).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from auto_ml.config import AutoMLConfig
from auto_ml.models.registry import build_model
from auto_ml.reporting.metrics import score_metric
from auto_ml.tuning.space import sample_params, validate_search_space
from auto_ml.utils.logger import get_logger

logger = get_logger("tuning")

# Optuna 자체 로그가 매우 시끄러우므로 WARNING 으로 낮춘다.
optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class TuningResult:
    """단일 모델의 튜닝 결과."""

    model_name: str
    best_params: dict[str, Any]   # fixed_params 와 병합된 최종 파라미터
    best_value: float             # 최고 OOF 점수 (primary_metric 기준)
    n_trials: int


class HyperparameterOptimizer:
    """``AutoMLConfig`` 와 학습 데이터를 받아 모델별 best params 를 찾는 객체."""

    def __init__(self, config: AutoMLConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    def optimize(
        self,
        model_name: str,
        X: pd.DataFrame,
        y: pd.Series,
        search_space: dict[str, dict[str, Any]],
        fixed_params: dict[str, Any],
        categorical_columns: list[str],
    ) -> TuningResult:
        """주어진 모델에 대해 Optuna 스터디를 1회 실행한다.

        Args:
            model_name: ``"lgbm" | "xgb" | "catboost"``.
            X: 전처리 완료된 학습 feature.
            y: 학습 타깃.
            search_space: 모델 라이브러리 원본 파라미터 이름 기준 search space.
            fixed_params: 탐색에서 제외하고 항상 적용할 파라미터.
            categorical_columns: 모델 래퍼에 전달할 범주형 컬럼.

        Returns:
            TuningResult — fixed_params 와 best params 를 병합한 최종 파라미터 포함.
        """
        validate_search_space(search_space)

        sampler = optuna.samplers.TPESampler(seed=self.config.tuning.random_state)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        objective = self._make_objective(
            model_name=model_name,
            X=X,
            y=y,
            search_space=search_space,
            fixed_params=fixed_params,
            categorical_columns=categorical_columns,
        )
        logger.info(
            "Tuning %s — n_trials=%d, cv_folds=%d, metric=%s",
            model_name,
            self.config.tuning.n_trials,
            self.config.tuning.cv_folds,
            self.config.training.primary_metric,
        )
        study.optimize(
            objective,
            n_trials=self.config.tuning.n_trials,
            timeout=self.config.tuning.timeout,
            show_progress_bar=False,
        )

        best_params = {**fixed_params, **study.best_params}
        logger.info(
            "Tuning %s done — best %s=%.4f, params=%s",
            model_name,
            self.config.training.primary_metric,
            study.best_value,
            study.best_params,
        )
        return TuningResult(
            model_name=model_name,
            best_params=best_params,
            best_value=float(study.best_value),
            n_trials=len(study.trials),
        )

    # ------------------------------------------------------------------
    def _make_objective(
        self,
        model_name: str,
        X: pd.DataFrame,
        y: pd.Series,
        search_space: dict[str, dict[str, Any]],
        fixed_params: dict[str, Any],
        categorical_columns: list[str],
    ):
        """Optuna 가 반복 호출할 objective 함수를 클로저로 만든다."""
        cfg = self.config
        primary = cfg.training.primary_metric
        kf = StratifiedKFold(
            n_splits=cfg.tuning.cv_folds,
            shuffle=True,
            random_state=cfg.tuning.random_state,
        )

        def objective(trial: optuna.Trial) -> float:
            sampled = sample_params(trial, search_space)
            params = {**fixed_params, **sampled}

            scores: list[float] = []
            for tr_idx, va_idx in kf.split(X, y):
                X_tr = X.iloc[tr_idx]
                y_tr = y.iloc[tr_idx]
                X_va = X.iloc[va_idx]
                y_va = y.iloc[va_idx]
                model = build_model(
                    name=model_name,
                    params=params,
                    categorical_columns=categorical_columns,
                    random_state=cfg.tuning.random_state,
                )
                model.fit(
                    X_tr, y_tr, X_va, y_va,
                    early_stopping_rounds=cfg.training.early_stopping_rounds,
                )
                proba = model.predict_proba(X_va)
                scores.append(score_metric(primary, y_va.to_numpy(), proba))
            return float(np.mean(scores))

        return objective
