"""핵심 버그 수정 회귀 테스트.

규약:
- XGBoost 의 SHAP 합(raw margin)은 조기 종료 후에도 실제 예측과 일치한다.
- ``best_iteration`` 은 세 부스팅 모델 모두 "best 지점까지의 트리 개수" 이며,
  조기 종료가 동작하지 않으면 None 이다 (LightGBM 의 0 포함).
- PDF 생성이 실패해도 학습 결과(artifact)와 HTML 리포트는 저장된다.
- 타깃에 결측이 있으면 명확한 메시지로 거부한다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import auto_ml.reporting.report as report_mod
from auto_ml.config import (
    AutoMLConfig,
    FeatureSpec,
    LoggingConfig,
    ModelConfig,
    ReportingConfig,
    TrainingConfig,
    TuningConfig,
)
from auto_ml.models.catboost_model import CatBoostModel
from auto_ml.models.lgbm_model import LGBMModel
from auto_ml.models.xgb_model import XGBModel
from auto_ml.pipeline import AutoMLPipeline
from auto_ml.utils.validation import validate_binary_target


def _split(tiny_binary_data):
    X, y = tiny_binary_data
    return X.iloc[:200], y.iloc[:200], X.iloc[200:], y.iloc[200:]


def test_xgb_shap_matches_prediction_after_early_stopping(tiny_binary_data):
    Xt, yt, Xv, yv = _split(tiny_binary_data)
    m = XGBModel(
        params={"n_estimators": 500, "learning_rate": 0.3}, categorical_columns=["cat1"],
    ).fit(Xt, yt, Xv, yv, early_stopping_rounds=20)
    n_trees = m.model.get_booster().num_boosted_rounds()
    assert m.best_iteration < n_trees  # 조기 종료 이후 트리가 남아 있는 상황

    shap_sum = m.shap_values(Xv).sum(axis=1)
    p = m.predict_proba(Xv)
    margin = np.log(p / (1 - p))
    np.testing.assert_allclose(shap_sum, margin, atol=1e-4)


def test_best_iteration_is_tree_count(tiny_binary_data):
    Xt, yt, Xv, yv = _split(tiny_binary_data)
    xgbm = XGBModel(
        params={"n_estimators": 500, "learning_rate": 0.3}, categorical_columns=["cat1"],
    ).fit(Xt, yt, Xv, yv, early_stopping_rounds=20)
    assert xgbm.best_iteration == xgbm.model.best_iteration + 1

    cat = CatBoostModel(
        params={"iterations": 500, "learning_rate": 0.3, "verbose": False,
                "allow_writing_files": False},
        categorical_columns=["cat1"],
    ).fit(Xt, yt, Xv, yv, early_stopping_rounds=20)
    assert cat.best_iteration == cat.model.tree_count_  # use_best_model 로 잘린 트리 수

    lgb = LGBMModel(
        params={"n_estimators": 500, "learning_rate": 0.3, "verbose": -1},
        categorical_columns=["cat1"],
    ).fit(Xt, yt, Xv, yv, early_stopping_rounds=20)
    assert lgb.best_iteration == lgb.model.booster_.best_iteration >= 1


@pytest.mark.parametrize(
    "model",
    [
        LGBMModel(params={"n_estimators": 20, "verbose": -1}, categorical_columns=["cat1"]),
        XGBModel(params={"n_estimators": 20}, categorical_columns=["cat1"]),
        CatBoostModel(
            params={"iterations": 20, "verbose": False, "allow_writing_files": False},
            categorical_columns=["cat1"],
        ),
    ],
    ids=["lgbm", "xgb", "catboost"],
)
def test_best_iteration_none_without_early_stopping(model, tiny_binary_data):
    X, y = tiny_binary_data
    model.fit(X, y)
    assert model.best_iteration is None


def _pipeline_config(tmp_path, tiny_binary_data, **training) -> AutoMLConfig:
    X, y = tiny_binary_data
    df = X.copy()
    df["target"] = y.values
    df.iloc[:220].to_parquet(tmp_path / "train.parquet")
    df.iloc[220:].to_parquet(tmp_path / "test.parquet")
    return AutoMLConfig(
        train_data_path=str(tmp_path / "train.parquet"),
        test_data_path=str(tmp_path / "test.parquet"),
        target_column="target",
        features=[
            FeatureSpec(name="num1", type="numeric"),
            FeatureSpec(name="num2", type="numeric"),
            FeatureSpec(name="cat1", type="categorical"),
        ],
        artifact_dir=str(tmp_path / "artifacts" / "models"),
        training=TrainingConfig(cv_folds=2, **training),
        tuning=TuningConfig(enabled=False),
        models={
            "lgbm": ModelConfig(fixed_params={"n_estimators": 50, "verbose": -1}),
            "xgb": ModelConfig(fixed_params={"n_estimators": 50}),
        },
        reporting=ReportingConfig(
            output_dir=str(tmp_path / "artifacts" / "reports"),
            generate_html=True,
            generate_pdf=True,
        ),
        logging=LoggingConfig(log_dir=str(tmp_path / "logs"), to_file=False),
    )


def test_iteration_capping_without_early_stopping_keeps_trees(tmp_path, tiny_binary_data):
    """조기 종료를 끈 iteration_capping 에서 LightGBM 이 트리 1개로 줄어들지 않는다."""
    cfg = _pipeline_config(
        tmp_path, tiny_binary_data,
        final_fit_strategy="iteration_capping", early_stopping_rounds=None,
    )
    AutoMLPipeline(cfg).run()
    from auto_ml.utils.io import load_artifact

    lgbm = load_artifact(tmp_path / "artifacts/models/models/lgbm.joblib")["model"]
    assert lgbm.model.get_params()["n_estimators"] == 50


def test_pdf_failure_keeps_artifacts_and_html(tmp_path, tiny_binary_data, monkeypatch, capsys):
    def broken_pdf(*args, **kwargs):
        raise OSError("cannot load library 'libpango-1.0-0'")

    monkeypatch.setattr(report_mod.ReportBuilder, "_html_to_pdf", staticmethod(broken_pdf))
    cfg = _pipeline_config(tmp_path, tiny_binary_data)
    outputs = AutoMLPipeline(cfg).run()

    assert outputs["artifact"].exists()
    assert (tmp_path / "artifacts/models/models/lgbm.joblib").exists()
    assert outputs["report_html"].exists()
    assert "report_pdf" not in outputs
    # run() 이 로거를 재구성(propagate=False)하므로 caplog 대신 stdout 로그를 본다.
    log = capsys.readouterr().out
    assert log.count("PDF report skipped") == 1  # 메인·sub 리포트마다 반복하지 않는다


def test_target_with_missing_values_is_rejected():
    with pytest.raises(ValueError, match="'default' has 2 missing value"):
        validate_binary_target(pd.Series([0, 1, np.nan, None, 1], name="default"))
    validate_binary_target(pd.Series([0, 1], dtype="Int64"))  # 정상 입력은 통과
