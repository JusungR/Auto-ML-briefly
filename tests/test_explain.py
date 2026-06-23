"""SHAP 해석 (auto-ml-explain) 회귀 / 단위 테스트.

검증 범위:
1. 세 모델 (LGBM/XGB/CatBoost) 의 ``shap_values`` 가 올바른 shape 반환
2. raw-margin 도메인 가법성: shap[:, :-1].sum + shap[:, -1] ≈ raw margin
3. Explainer 의 probability 정규화: base_value + sum(shap_*) == score 모든 행
4. End-to-end: 학습 → artifact → Explainer.explain → wide DataFrame
5. id_columns 우선순위 (Scorer 와 동일 규약)
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from auto_ml.config import (
    AutoMLConfig,
    ExplainConfig,
    FeatureSpec,
    ModelConfig,
    ReportingConfig,
    ScoringConfig,
    TrainingConfig,
    TuningConfig,
)
from auto_ml.explain.explainer import Explainer
from auto_ml.explain.runner import run_explain
from auto_ml.models.catboost_model import CatBoostModel
from auto_ml.models.lgbm_model import LGBMModel
from auto_ml.models.xgb_model import XGBModel
from auto_ml.pipeline import AutoMLPipeline


# ───────────────────────────────────────────────────────────────────────
# 1) 모델별 shap_values: shape + raw-margin 가법성
# ───────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("model_cls", [LGBMModel, XGBModel, CatBoostModel])
def test_shap_values_shape_and_raw_additivity(tiny_binary_data, model_cls):
    X, y = tiny_binary_data
    if model_cls is LGBMModel:
        params = {"n_estimators": 30, "verbose": -1}
    elif model_cls is XGBModel:
        params = {"n_estimators": 30}
    else:
        params = {"iterations": 30, "verbose": False, "allow_writing_files": False}
    m = model_cls(params=params, categorical_columns=["cat1"])
    m.fit(X, y)

    shap = m.shap_values(X)
    assert shap.shape == (len(X), len(m.feature_columns) + 1)

    # raw margin 도메인 가법성: 라이브러리별 raw 예측 API
    if model_cls is LGBMModel:
        X_in = m._to_categorical(X[m.feature_columns])
        raw = m.model.predict(X_in, raw_score=True)
        tol = 1e-10
    elif model_cls is XGBModel:
        X_in = m._encode_transform(X[m.feature_columns])
        dmat = xgb.DMatrix(X_in.values, feature_names=list(X_in.columns))
        raw = m.model.get_booster().predict(dmat, output_margin=True)
        tol = 1e-4   # XGBoost DMatrix 는 float32 — 수치 오차 더 큼
    else:  # CatBoostModel
        pool = m._to_pool(X[m.feature_columns])
        raw = m.model.predict(pool, prediction_type="RawFormulaVal")
        tol = 1e-10

    shap_sum = shap[:, :-1].sum(axis=1) + shap[:, -1]
    assert np.max(np.abs(shap_sum - raw)) < tol


# ───────────────────────────────────────────────────────────────────────
# 2) End-to-end + 확률 도메인 가법성
# ───────────────────────────────────────────────────────────────────────

def _build_config_and_train(
    tmp_path: Path,
    tiny_binary_data,
    *,
    explain_id_columns: list[str] | None = None,
    top_id_columns: list[str] | None = None,
) -> AutoMLConfig:
    """학습까지 끝낸 cfg 반환 — 이후 run_explain 호출이 가능."""
    X, y = tiny_binary_data
    df = X.copy()
    df["target"] = y.values
    df["client_id"] = np.arange(1000, 1000 + len(df))
    df["batch_id"] = (np.arange(len(df)) // 50)

    tr = df.iloc[:240].reset_index(drop=True)
    te = df.iloc[240:].reset_index(drop=True)
    tr.to_parquet(tmp_path / "train.parquet", index=False)
    te.to_parquet(tmp_path / "test.parquet", index=False)
    score_input = te.drop(columns=["target"]).copy()
    score_input.to_parquet(tmp_path / "score_input.parquet", index=False)

    cfg = AutoMLConfig(
        train_data_path=str(tmp_path / "train.parquet"),
        test_data_path=str(tmp_path / "test.parquet"),
        target_column="target",
        features=[
            FeatureSpec(name="num1", type="numeric"),
            FeatureSpec(name="num2", type="numeric"),
            FeatureSpec(name="cat1", type="categorical"),
        ],
        id_columns=list(top_id_columns) if top_id_columns is not None else [],
        artifact_dir=str(tmp_path / "artifacts" / "models"),
        training=TrainingConfig(cv_folds=2, early_stopping_rounds=5),
        tuning=TuningConfig(enabled=False),
        models={
            "lgbm": ModelConfig(
                fixed_params={"n_estimators": 20, "verbose": -1},
                search_space={},
            ),
        },
        reporting=ReportingConfig(
            output_dir=str(tmp_path / "artifacts" / "reports"),
            generate_html=False,
            generate_pdf=False,
        ),
        scoring=ScoringConfig(
            input_path=str(tmp_path / "score_input.parquet"),
            output_path=str(tmp_path / "artifacts" / "scores" / "scores.parquet"),
            id_columns=[],
            threshold=0.5,
        ),
        explain=ExplainConfig(
            input_path=str(tmp_path / "score_input.parquet"),
            output_path=str(tmp_path / "artifacts" / "explain" / "shap.parquet"),
            id_columns=list(explain_id_columns) if explain_id_columns is not None else [],
        ),
    )
    AutoMLPipeline(cfg).run()
    return cfg


def test_explain_end_to_end_probability_additivity(tmp_path, tiny_binary_data):
    """run_explain 출력에서 base_value + sum(shap_*) == score 가 모든 행에서 성립."""
    cfg = _build_config_and_train(
        tmp_path, tiny_binary_data, explain_id_columns=["client_id"],
    )
    run_explain(cfg)
    out = pd.read_parquet(cfg.explain.output_path)

    # 컬럼 구성 확인
    assert "client_id" in out.columns
    assert "base_value" in out.columns
    assert "score" in out.columns
    shap_cols = [c for c in out.columns if c.startswith("shap_")]
    # 학습 시 변수 선택을 끄면 feature 3개 그대로 모델 입력
    assert len(shap_cols) == 3
    expected_shap = {"shap_num1", "shap_num2", "shap_cat1"}
    assert set(shap_cols) == expected_shap

    # 컬럼 순서: id → shap_* → base_value → score
    assert list(out.columns) == ["client_id"] + shap_cols + ["base_value", "score"]

    # 가법성 (확률 도메인) — 모든 행에서 base + sum(shap) == score
    reconstructed = out["base_value"].values + out[shap_cols].sum(axis=1).values
    assert np.allclose(reconstructed, out["score"].values, atol=1e-12)


def test_explain_uses_top_level_id_columns_when_explain_empty(tmp_path, tiny_binary_data):
    """explain.id_columns 가 비어 있으면 top-level id_columns 를 사용한다."""
    cfg = _build_config_and_train(
        tmp_path, tiny_binary_data,
        top_id_columns=["client_id"],
        explain_id_columns=[],
    )
    run_explain(cfg)
    out = pd.read_parquet(cfg.explain.output_path)
    assert "client_id" in out.columns


def test_explain_id_columns_explicit_overrides_top_level(tmp_path, tiny_binary_data):
    """explain.id_columns 가 명시되면 top-level id_columns 보다 우선한다."""
    cfg = _build_config_and_train(
        tmp_path, tiny_binary_data,
        top_id_columns=["client_id", "batch_id"],
        explain_id_columns=["client_id"],  # batch_id 제외
    )
    run_explain(cfg)
    out = pd.read_parquet(cfg.explain.output_path)
    assert "client_id" in out.columns
    assert "batch_id" not in out.columns


def test_explain_fallback_to_artifact_metadata_when_both_empty(tmp_path, tiny_binary_data):
    """top-level / explain 둘 다 비어 있으면 artifact metadata 의 id_columns 사용."""
    cfg = _build_config_and_train(
        tmp_path, tiny_binary_data,
        top_id_columns=["client_id"],
    )
    cfg.id_columns = []
    cfg.explain.id_columns = []
    run_explain(cfg)
    out = pd.read_parquet(cfg.explain.output_path)
    assert "client_id" in out.columns


def test_warn_when_id_column_missing_from_explain_input(tmp_path, tiny_binary_data):
    """설정된 id 컬럼이 입력 parquet 에 없으면 WARN 발화."""
    cfg = _build_config_and_train(
        tmp_path, tiny_binary_data,
        top_id_columns=["client_id"],
    )
    explainer = Explainer.from_artifact(Path(cfg.artifact_dir) / "best.joblib")
    df = pd.read_parquet(cfg.explain.input_path)

    records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = _Capture(level=logging.WARNING)
    explainer_logger = logging.getLogger("auto_ml.explainer")
    explainer_logger.addHandler(handler)
    try:
        out = explainer.explain(df, id_columns=["client_id", "missing_col"])
    finally:
        explainer_logger.removeHandler(handler)

    assert "client_id" in out.columns
    assert "missing_col" not in out.columns
    warn_msgs = " ".join(r.getMessage() for r in records if r.levelno == logging.WARNING)
    assert "missing_col" in warn_msgs


def test_explainer_score_matches_scorer(tmp_path, tiny_binary_data):
    """Explainer 의 score 와 Scorer 의 score 가 (같은 입력에서) 일치해야 한다."""
    from auto_ml.scoring.scorer import Scorer

    cfg = _build_config_and_train(
        tmp_path, tiny_binary_data, explain_id_columns=["client_id"],
    )
    artifact = Path(cfg.artifact_dir) / "best.joblib"
    df = pd.read_parquet(cfg.explain.input_path)

    scorer = Scorer.from_artifact(artifact)
    explainer = Explainer.from_artifact(artifact)
    scored = scorer.score(df, id_columns=["client_id"])
    explained = explainer.explain(df, id_columns=["client_id"])

    assert np.allclose(scored["score"].values, explained["score"].values, atol=1e-9)
