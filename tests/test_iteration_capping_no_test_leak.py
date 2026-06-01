"""iteration_capping 전략: 최종 fit 이 테스트셋을 보지 않는지 검증.

핵심 회귀 가드 — 기본 early_stop_on_test 는 최종 fit 의 early-stopping 검증셋으로
테스트셋을 사용하지만, iteration_capping 은 CV fold best_iter 로 트리 수를 고정하고
테스트셋 없이 train 전체로 재학습해야 한다.
"""
from __future__ import annotations

import math

import pandas as pd

from auto_ml.config import (
    AutoMLConfig,
    FeatureSpec,
    ModelConfig,
    TrainingConfig,
    TuningConfig,
)
from auto_ml.config import EnsembleConfig
from auto_ml.models import trainer as trainer_mod
from auto_ml.models.trainer import Trainer


def _make_config(strategy: str, **training_kw) -> AutoMLConfig:
    """lgbm 단독, 튜닝 off, 작은 트리 수의 빠른 단위 테스트용 config."""
    return AutoMLConfig(
        target_column="y",
        features=[
            FeatureSpec(name="num1", type="numeric"),
            FeatureSpec(name="num2", type="numeric"),
            FeatureSpec(name="cat1", type="categorical"),
        ],
        training=TrainingConfig(
            cv_folds=3,
            early_stopping_rounds=10,
            final_fit_strategy=strategy,
            **training_kw,
        ),
        tuning=TuningConfig(enabled=False),
        models={"lgbm": ModelConfig(fixed_params={"n_estimators": 50, "verbose": -1})},
        ensemble=EnsembleConfig(enabled=False),
    )


def test_iteration_capping_final_fit_sees_no_test(tiny_binary_data, monkeypatch):
    X, y = tiny_binary_data
    X_tr, y_tr = X.iloc[:240], y.iloc[:240]
    X_te, y_te = X.iloc[240:], y.iloc[240:]

    # 모든 fit 호출의 (X_valid is None, early_stopping_rounds) 를 기록한다.
    fit_calls: list[dict] = []
    real_build = trainer_mod.build_model

    def spy_build(*args, **kwargs):
        model = real_build(*args, **kwargs)
        real_fit = model.fit

        def wrapped_fit(X_train, y_train, X_valid=None, y_valid=None,
                        early_stopping_rounds=None):
            fit_calls.append({
                "n": len(X_train),
                "valid_is_none": X_valid is None,
                "esr": early_stopping_rounds,
                "n_estimators": model.params.get("n_estimators"),
            })
            return real_fit(X_train, y_train, X_valid, y_valid, early_stopping_rounds)

        model.fit = wrapped_fit
        return model

    monkeypatch.setattr(trainer_mod, "build_model", spy_build)

    cfg = _make_config("iteration_capping", iteration_cap_headroom=1.0)
    result = Trainer(cfg).train(X_tr, y_tr, X_te, y_te)

    # 마지막 fit = 최종 fit. 테스트셋 미노출 + early stopping 없음.
    final_call = fit_calls[-1]
    assert final_call["valid_is_none"] is True
    assert final_call["esr"] is None
    # train 전체로 학습 (fold subset 이 아님)
    assert final_call["n"] == len(X_tr)

    # 트리 수가 fold best_iter 평균(ceil)으로 고정됐는지 확인.
    mr = result.results["lgbm"]
    fold_iters = [bi for bi in mr.fold_best_iterations if bi is not None]
    if fold_iters:
        expected = max(1, math.ceil(sum(fold_iters) / len(fold_iters)))
        assert final_call["n_estimators"] == expected
    # 최종 모델은 단일 LGBM (앙상블 아님). early stop 미발동이므로 best_iteration 은
    # falsy (LightGBM 버전에 따라 None 또는 0).
    assert mr.model.name == "lgbm"
    assert not mr.model.best_iteration


def test_default_strategy_still_uses_test_for_early_stop(tiny_binary_data, monkeypatch):
    """기본 전략 회귀: 최종 fit 이 여전히 테스트셋을 early-stopping valid 로 사용."""
    X, y = tiny_binary_data
    X_tr, y_tr = X.iloc[:240], y.iloc[:240]
    X_te, y_te = X.iloc[240:], y.iloc[240:]

    fit_calls: list[dict] = []
    real_build = trainer_mod.build_model

    def spy_build(*args, **kwargs):
        model = real_build(*args, **kwargs)
        real_fit = model.fit

        def wrapped_fit(X_train, y_train, X_valid=None, y_valid=None,
                        early_stopping_rounds=None):
            fit_calls.append({"valid_is_none": X_valid is None,
                              "valid_n": None if X_valid is None else len(X_valid)})
            return real_fit(X_train, y_train, X_valid, y_valid, early_stopping_rounds)

        model.fit = wrapped_fit
        return model

    monkeypatch.setattr(trainer_mod, "build_model", spy_build)

    cfg = _make_config("early_stop_on_test")
    Trainer(cfg).train(X_tr, y_tr, X_te, y_te)

    final_call = fit_calls[-1]
    assert final_call["valid_is_none"] is False
    assert final_call["valid_n"] == len(X_te)
