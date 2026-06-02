"""cv_bagging 전략: CV fold 모델을 균등 가중 앙상블로 묶는지 검증."""
from __future__ import annotations

import logging

import numpy as np
import pytest

from auto_ml.config import (
    AutoMLConfig,
    EnsembleConfig,
    FeatureSpec,
    ModelConfig,
    TrainingConfig,
    TuningConfig,
)
from auto_ml.models.ensemble_model import EnsembleModel
from auto_ml.models.trainer import Trainer
from auto_ml.utils.io import ArtifactMetadata, load_artifact, save_artifact


def _make_config(strategy: str, ensemble_enabled: bool = False) -> AutoMLConfig:
    return AutoMLConfig(
        target_column="y",
        features=[
            FeatureSpec(name="num1", type="numeric"),
            FeatureSpec(name="num2", type="numeric"),
            FeatureSpec(name="cat1", type="categorical"),
        ],
        training=TrainingConfig(
            cv_folds=3, early_stopping_rounds=10, final_fit_strategy=strategy,
        ),
        tuning=TuningConfig(enabled=False),
        models={"lgbm": ModelConfig(fixed_params={"n_estimators": 40, "verbose": -1})},
        ensemble=EnsembleConfig(enabled=ensemble_enabled),
    )


def test_cv_bagging_builds_uniform_ensemble(tiny_binary_data):
    X, y = tiny_binary_data
    X_tr, y_tr = X.iloc[:240], y.iloc[:240]
    X_te, y_te = X.iloc[240:], y.iloc[240:]

    cfg = _make_config("cv_bagging")
    result = Trainer(cfg).train(X_tr, y_tr, X_te, y_te)

    mr = result.results["lgbm"]
    model = mr.model
    # 최종 모델 = 3개 fold 모델을 품은 EnsembleModel
    assert isinstance(model, EnsembleModel)
    assert len(model.base_models) == 3
    # 균등 가중
    weights = list(model.weights.values())
    assert all(abs(w - 1.0 / 3) < 1e-9 for w in weights)
    assert abs(sum(weights) - 1.0) < 1e-9

    # test_proba 가 fold 별 predict_proba 의 수동 평균과 일치
    manual = np.mean(
        [m.predict_proba(X_te) for m in model.base_models.values()], axis=0
    )
    np.testing.assert_allclose(mr.test_proba, manual, rtol=1e-9, atol=1e-12)


def test_cv_bagging_artifact_roundtrip(tiny_binary_data, tmp_path):
    X, y = tiny_binary_data
    X_tr, y_tr = X.iloc[:240], y.iloc[:240]
    X_te, y_te = X.iloc[240:], y.iloc[240:]

    cfg = _make_config("cv_bagging")
    result = Trainer(cfg).train(X_tr, y_tr, X_te, y_te)
    model = result.results["lgbm"].model

    # cloudpickle 로 K fold 모델을 품은 앙상블이 저장/복원되는지 확인.
    # preprocessor 는 본 테스트 범위 밖이라 None 대용 객체로 대체.
    meta = ArtifactMetadata(
        target_column="y",
        feature_columns=["num1", "num2", "cat1"],
        selected_features=["num1", "num2", "cat1"],
        categorical_columns=["cat1"],
        id_columns=[],
        model_name="lgbm",
        primary_metric="roc_auc",
        metric_value=0.5,
        extra={"training_mode": "cv_bagging"},
    )
    path = tmp_path / "bag.joblib"
    save_artifact(path, preprocessor=None, model=model, metadata=meta)
    loaded = load_artifact(path)

    proba_before = model.predict_proba(X_te)
    proba_after = loaded["model"].predict_proba(X_te)
    np.testing.assert_allclose(proba_before, proba_after, rtol=1e-9, atol=1e-12)


def test_cv_bagging_disables_external_ensemble(tiny_binary_data, caplog):
    """cv_bagging + ensemble.enabled=True → 외부 앙상블 자동 비활성 + WARN."""
    X, y = tiny_binary_data
    X_tr, y_tr = X.iloc[:240], y.iloc[:240]
    X_te, y_te = X.iloc[240:], y.iloc[240:]

    # 2개 모델 활성 (앙상블 조건 충족) + ensemble.enabled=True
    cfg = AutoMLConfig(
        target_column="y",
        features=[
            FeatureSpec(name="num1", type="numeric"),
            FeatureSpec(name="num2", type="numeric"),
            FeatureSpec(name="cat1", type="categorical"),
        ],
        training=TrainingConfig(
            cv_folds=3, early_stopping_rounds=10, final_fit_strategy="cv_bagging",
        ),
        tuning=TuningConfig(enabled=False),
        models={
            "lgbm": ModelConfig(fixed_params={"n_estimators": 40, "verbose": -1}),
            "xgb": ModelConfig(fixed_params={"n_estimators": 40}),
        },
        ensemble=EnsembleConfig(enabled=True, strategy="weighted_average"),
    )

    # auto_ml 로거는 propagate=False 라 caplog 핸들러를 직접 부착해 캡처한다.
    auto_ml_logger = logging.getLogger("auto_ml")
    auto_ml_logger.addHandler(caplog.handler)
    try:
        with caplog.at_level("WARNING"):
            result = Trainer(cfg).train(X_tr, y_tr, X_te, y_te)
    finally:
        auto_ml_logger.removeHandler(caplog.handler)

    # 외부 앙상블 결과가 없어야 한다.
    assert "ensemble" not in result.results
    assert any("cv_bagging already bags" in r.getMessage() for r in caplog.records)
