"""스코어링 출력에 id_columns 가 보존되는지 회귀 테스트.

규약:
- ``ScoringConfig.id_columns`` 가 비어 있어도 top-level ``AutoMLConfig.id_columns``
  가, 그것도 비어 있으면 artifact metadata 의 학습 시점 id_columns 가 자동
  사용된다 (운영 실수 가장 흔한 'scoring 블록에 적어주는 걸 깜빡' 케이스 방지).
- 입력 parquet 에 설정된 id 컬럼이 없으면 WARN.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from auto_ml.config import (
    AutoMLConfig,
    FeatureSpec,
    ModelConfig,
    ReportingConfig,
    ScoringConfig,
    TrainingConfig,
    TuningConfig,
)
from auto_ml.pipeline import AutoMLPipeline
from auto_ml.scoring.runner import run_scoring


def _build_config_and_train(
    tmp_path: Path,
    tiny_binary_data,
    *,
    top_id_columns: list[str],
    scoring_id_columns: list[str],
) -> AutoMLConfig:
    """학습까지 끝낸 cfg 반환 — 이후 run_scoring 호출이 가능."""
    X, y = tiny_binary_data
    df = X.copy()
    df["target"] = y.values
    df["client_id"] = np.arange(1000, 1000 + len(df))
    df["batch_id"] = (np.arange(len(df)) // 50)

    tr = df.iloc[:240].reset_index(drop=True)
    te = df.iloc[240:].reset_index(drop=True)
    tr.to_parquet(tmp_path / "train.parquet", index=False)
    te.to_parquet(tmp_path / "test.parquet", index=False)
    # 스코어링 input 은 target 없는 형태로
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
        id_columns=list(top_id_columns),
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
            generate_html=True,
            generate_pdf=False,
        ),
        scoring=ScoringConfig(
            input_path=str(tmp_path / "score_input.parquet"),
            output_path=str(tmp_path / "artifacts" / "scores" / "scores.parquet"),
            id_columns=list(scoring_id_columns),
            threshold=0.5,
        ),
    )
    AutoMLPipeline(cfg).run()
    return cfg


def test_scoring_uses_top_level_id_columns_when_scoring_empty(
    tmp_path, tiny_binary_data,
):
    """scoring.id_columns 가 비어 있어도 top-level id_columns 가 사용된다."""
    cfg = _build_config_and_train(
        tmp_path, tiny_binary_data,
        top_id_columns=["client_id", "batch_id"],
        scoring_id_columns=[],     # 스코어링 블록 비움 — 운영 실수 시뮬레이션
    )
    run_scoring(cfg)
    out = pd.read_parquet(cfg.scoring.output_path)
    assert list(out.columns) == ["client_id", "batch_id", "score", "prediction"]


def test_scoring_id_columns_override_takes_precedence(tmp_path, tiny_binary_data):
    """scoring.id_columns 가 명시되면 top-level 보다 우선한다 (override)."""
    cfg = _build_config_and_train(
        tmp_path, tiny_binary_data,
        top_id_columns=["client_id", "batch_id"],
        scoring_id_columns=["client_id"],  # batch_id 제외
    )
    run_scoring(cfg)
    out = pd.read_parquet(cfg.scoring.output_path)
    assert list(out.columns) == ["client_id", "score", "prediction"]


def test_scoring_fallback_to_artifact_metadata_when_both_empty(
    tmp_path, tiny_binary_data,
):
    """top-level / scoring 둘 다 비어 있어도 artifact metadata 의 id_columns 사용.

    실 운영 시나리오: 학습 시 id_columns 만 지정, score 시점 config 는 그대로
    그 cfg 를 재사용 — 빠진 곳이 없어도 동작해야 한다.
    """
    cfg = _build_config_and_train(
        tmp_path, tiny_binary_data,
        top_id_columns=["client_id"],
        scoring_id_columns=[],
    )
    # 학습 후 cfg.id_columns 를 강제로 비워 본 — artifact metadata 만 남는 상황.
    cfg.id_columns = []
    cfg.scoring.id_columns = []
    run_scoring(cfg)
    out = pd.read_parquet(cfg.scoring.output_path)
    # metadata 의 id_columns 가 fallback 으로 사용되어 client_id 가 들어가야 함
    assert "client_id" in out.columns


def test_warn_when_id_column_missing_from_input(tmp_path, tiny_binary_data):
    """설정된 id 컬럼이 입력 parquet 에 없으면 WARN 발화.

    ``auto_ml`` 루트 로거가 propagate=False 라 caplog 가 못 잡으므로 직접
    핸들러를 붙여 records 를 수집한다.
    """
    from auto_ml.scoring.scorer import Scorer

    cfg = _build_config_and_train(
        tmp_path, tiny_binary_data,
        top_id_columns=["client_id"],
        scoring_id_columns=[],
    )
    scorer = Scorer.from_artifact(Path(cfg.artifact_dir) / "best.joblib")
    df = pd.read_parquet(cfg.scoring.input_path)

    records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record):  # noqa: D401
            records.append(record)

    handler = _Capture(level=logging.WARNING)
    scorer_logger = logging.getLogger("auto_ml.scorer")
    scorer_logger.addHandler(handler)
    try:
        out = scorer.score(df, id_columns=["client_id", "missing_col"])
    finally:
        scorer_logger.removeHandler(handler)

    assert "client_id" in out.columns
    assert "missing_col" not in out.columns
    warn_msgs = " ".join(
        r.getMessage() for r in records if r.levelno == logging.WARNING
    )
    assert "missing_col" in warn_msgs
