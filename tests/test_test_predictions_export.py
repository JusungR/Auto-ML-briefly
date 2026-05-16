"""학습 시 test 예측 export 회귀 테스트.

학습 직후 ``artifacts/<name>/predictions/test_predictions.parquet`` 에
``<id_columns> + score + prediction + <target_column>`` 스키마로 저장되는지 확인.
"""
from __future__ import annotations

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


@pytest.fixture
def quick_config_with_ids(tmp_path: Path, tiny_binary_data):
    """tiny 데이터 + client_id 컬럼을 가진 임시 train/test parquet 을 생성."""
    X, y = tiny_binary_data
    n = len(X)
    df = X.copy()
    df["target"] = y.values
    df["client_id"] = np.arange(1000, 1000 + n)

    tr = df.iloc[:240].reset_index(drop=True)
    te = df.iloc[240:].reset_index(drop=True)
    tr_path = tmp_path / "train.parquet"
    te_path = tmp_path / "test.parquet"
    tr.to_parquet(tr_path, index=False)
    te.to_parquet(te_path, index=False)

    return AutoMLConfig(
        train_data_path=str(tr_path),
        test_data_path=str(te_path),
        target_column="target",
        features=[
            FeatureSpec(name="num1", type="numeric"),
            FeatureSpec(name="num2", type="numeric"),
            FeatureSpec(name="cat1", type="categorical"),
        ],
        id_columns=["client_id"],
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
        scoring=ScoringConfig(threshold=0.5),
    )


def test_train_writes_test_predictions(quick_config_with_ids: AutoMLConfig):
    cfg = quick_config_with_ids
    outputs = AutoMLPipeline(cfg).run()

    assert "test_predictions" in outputs, "pipeline should expose test_predictions path"
    path = outputs["test_predictions"]
    assert path.exists(), f"missing predictions parquet at {path}"

    pred = pd.read_parquet(path)
    # 컬럼 순서/이름 규약: id + score + prediction + target
    assert list(pred.columns) == ["client_id", "score", "prediction", "target"]

    # 행 수는 test 데이터와 동일
    test_df = pd.read_parquet(cfg.test_data_path)
    assert len(pred) == len(test_df)

    # 키 보존: test_df 의 client_id 와 동일 (정렬은 같은 행 순서를 가정 — pipeline 이 변경 안 함)
    np.testing.assert_array_equal(
        pred["client_id"].to_numpy(), test_df["client_id"].to_numpy()
    )
    # 점수 [0,1], prediction 0/1, target 0/1
    assert ((pred["score"] >= 0.0) & (pred["score"] <= 1.0)).all()
    assert set(pred["prediction"].unique()).issubset({0, 1})
    assert set(pred[cfg.target_column].unique()).issubset({0, 1})

    # threshold 일치
    expected_pred = (pred["score"] >= cfg.scoring.threshold).astype(int)
    np.testing.assert_array_equal(pred["prediction"].to_numpy(), expected_pred.to_numpy())


def test_train_predictions_without_id_columns(
    tmp_path: Path, tiny_binary_data,
):
    """id_columns 설정이 비어 있어도 export 가 score+prediction+target 만으로 동작."""
    X, y = tiny_binary_data
    df = X.copy()
    df["target"] = y.values
    tr_path = tmp_path / "train.parquet"
    te_path = tmp_path / "test.parquet"
    df.iloc[:240].to_parquet(tr_path, index=False)
    df.iloc[240:].to_parquet(te_path, index=False)

    cfg = AutoMLConfig(
        train_data_path=str(tr_path),
        test_data_path=str(te_path),
        target_column="target",
        features=[
            FeatureSpec(name="num1", type="numeric"),
            FeatureSpec(name="num2", type="numeric"),
            FeatureSpec(name="cat1", type="categorical"),
        ],
        id_columns=[],  # 빈 리스트 — id 없는 입력
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
    )
    outputs = AutoMLPipeline(cfg).run()
    pred = pd.read_parquet(outputs["test_predictions"])
    assert list(pred.columns) == ["score", "prediction", "target"]
