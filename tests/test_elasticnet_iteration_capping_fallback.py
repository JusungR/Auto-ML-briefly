"""ElasticNet + iteration_capping: 부스팅이 아니므로 일반 refit 으로 면제."""
from __future__ import annotations

import logging

from auto_ml.config import (
    AutoMLConfig,
    EnsembleConfig,
    FeatureSpec,
    ModelConfig,
    TrainingConfig,
    TuningConfig,
)
from auto_ml.models.elasticnet_model import ElasticNetModel
from auto_ml.models.trainer import Trainer


def test_elasticnet_iteration_capping_is_exempt(tiny_binary_data, caplog):
    X, y = tiny_binary_data
    X_tr, y_tr = X.iloc[:240], y.iloc[:240]
    X_te, y_te = X.iloc[240:], y.iloc[240:]

    cfg = AutoMLConfig(
        target_column="y",
        features=[
            FeatureSpec(name="num1", type="numeric"),
            FeatureSpec(name="num2", type="numeric"),
            FeatureSpec(name="cat1", type="categorical"),
        ],
        training=TrainingConfig(
            cv_folds=3, final_fit_strategy="iteration_capping",
        ),
        tuning=TuningConfig(enabled=False),
        models={"elasticnet": ModelConfig(fixed_params={"max_iter": 200})},
        ensemble=EnsembleConfig(enabled=False),
    )

    # auto_ml 로거는 propagate=False 라 caplog 핸들러를 직접 부착해 캡처한다.
    auto_ml_logger = logging.getLogger("auto_ml")
    auto_ml_logger.addHandler(caplog.handler)
    try:
        with caplog.at_level("INFO"):
            result = Trainer(cfg).train(X_tr, y_tr, X_te, y_te)
    finally:
        auto_ml_logger.removeHandler(caplog.handler)

    mr = result.results["elasticnet"]
    # 앙상블이 아닌 일반 ElasticNet 모델로 학습되어야 한다.
    assert isinstance(mr.model, ElasticNetModel)
    assert mr.model.best_iteration is None
    # 면제 로그가 남아야 한다.
    assert any("not applicable to elasticnet" in r.getMessage() for r in caplog.records)
