"""리포트 변경사항 회귀 테스트:
- feature importance 의 relative 정규화 (sum=1)
- 변수선택 결과 노출 개수 기본 = 전체, 설정 가능
- feature_importance.csv / feature_selection.csv 전체 데이터 저장
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from auto_ml.config import (
    AutoMLConfig,
    FeatureSpec,
    FeatureSelectionConfig,
    ModelConfig,
    ReportingConfig,
    TrainingConfig,
    TuningConfig,
)
from auto_ml.feature_selection import SelectionResult
from auto_ml.pipeline import AutoMLPipeline
from auto_ml.reporting.report import ReportBuilder


# ----- 정규화 단위 테스트 (no IO) -----------------------------------------

def test_prepare_importance_normalizes_to_sum_one():
    fi = {"a": 10.0, "b": 30.0, "c": 60.0}
    rows = ReportBuilder._prepare_importance(fi)
    # 정렬 내림차순
    assert [r["feature"] for r in rows] == ["c", "b", "a"]
    # gain 보존
    assert rows[0]["importance_gain"] == 60.0
    # relative 합 = 1
    total = sum(r["importance_relative"] for r in rows)
    assert total == pytest.approx(1.0, abs=1e-10)
    # 각 값 = gain / total_gain
    assert rows[0]["importance_relative"] == pytest.approx(0.6)
    assert rows[1]["importance_relative"] == pytest.approx(0.3)
    assert rows[2]["importance_relative"] == pytest.approx(0.1)


def test_prepare_importance_zero_total_safe():
    """모든 gain 이 0 일 때 분모 0 회피 — relative 도 0 으로."""
    rows = ReportBuilder._prepare_importance({"a": 0.0, "b": 0.0})
    assert all(r["importance_relative"] == 0.0 for r in rows)


def test_prepare_selection_orders_and_marks_selected():
    sel = SelectionResult(
        selected_features=["b", "a"],
        frequencies={"a": 0.8, "b": 0.9, "c": 0.3, "d": 0.7},
        n_subsamples=100,
        threshold=0.7,
        base_estimator="lasso",
        fallback_used=False,
    )
    rows = ReportBuilder._prepare_selection(sel)
    assert [r["feature"] for r in rows] == ["b", "a", "d", "c"]
    assert rows[0]["selected"] is True   # b
    assert rows[1]["selected"] is True   # a
    assert rows[2]["selected"] is False  # d (freq 0.7, but not in selected_features)
    assert rows[3]["selected"] is False  # c


# ----- end-to-end fixture: 작은 학습 + 리포트 빌드 ---------------------

@pytest.fixture
def quick_pipeline(tmp_path: Path, tiny_binary_data):
    X, y = tiny_binary_data
    df = X.copy()
    df["target"] = y.values
    df["client_id"] = np.arange(1000, 1000 + len(df))
    df.iloc[:240].to_parquet(tmp_path / "train.parquet", index=False)
    df.iloc[240:].to_parquet(tmp_path / "test.parquet", index=False)

    cfg = AutoMLConfig(
        train_data_path=str(tmp_path / "train.parquet"),
        test_data_path=str(tmp_path / "test.parquet"),
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
    )
    return cfg


def test_pipeline_writes_feature_importance_csv(quick_pipeline: AutoMLConfig):
    outputs = AutoMLPipeline(quick_pipeline).run()
    assert "feature_importance_csv" in outputs
    csv_path: Path = outputs["feature_importance_csv"]
    assert csv_path.exists()

    with open(csv_path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    # 컬럼 헤더 검증
    assert set(rows[0].keys()) == {"feature", "importance_gain", "importance_relative"}
    # 모든 feature 가 포함되어야 함 (3개)
    feature_names = [r["feature"] for r in rows]
    assert set(feature_names) == {"num1", "num2", "cat1"}
    # relative 의 합이 1 (모두 0 인 비정상 케이스가 아니면)
    rels = [float(r["importance_relative"]) for r in rows]
    if sum(float(r["importance_gain"]) for r in rows) > 0:
        assert sum(rels) == pytest.approx(1.0, abs=1e-6)


def test_pipeline_no_selection_csv_when_disabled(quick_pipeline: AutoMLConfig):
    """feature_selection.enabled=False (기본) 면 selection CSV 는 생성되지 않는다."""
    outputs = AutoMLPipeline(quick_pipeline).run()
    assert "feature_selection_csv" not in outputs
    assert not (Path(quick_pipeline.reporting.output_dir) / "feature_selection.csv").exists()


def test_pipeline_writes_feature_selection_csv_when_enabled(
    tmp_path: Path, tiny_binary_data,
):
    X, y = tiny_binary_data
    df = X.copy()
    df["target"] = y.values
    df.iloc[:240].to_parquet(tmp_path / "train.parquet", index=False)
    df.iloc[240:].to_parquet(tmp_path / "test.parquet", index=False)

    cfg = AutoMLConfig(
        train_data_path=str(tmp_path / "train.parquet"),
        test_data_path=str(tmp_path / "test.parquet"),
        target_column="target",
        features=[
            FeatureSpec(name="num1", type="numeric"),
            FeatureSpec(name="num2", type="numeric"),
            FeatureSpec(name="cat1", type="categorical"),
        ],
        artifact_dir=str(tmp_path / "artifacts" / "models"),
        training=TrainingConfig(cv_folds=2, early_stopping_rounds=5),
        tuning=TuningConfig(enabled=False),
        feature_selection=FeatureSelectionConfig(
            enabled=True,
            base_estimator="lasso",
            n_subsamples=10,   # 빠르게
            subsample_ratio=0.7,
            threshold=0.3,
            min_selected=1,
        ),
        models={
            "lgbm": ModelConfig(
                fixed_params={"n_estimators": 10, "verbose": -1},
                search_space={},
            ),
        },
        reporting=ReportingConfig(
            output_dir=str(tmp_path / "artifacts" / "reports"),
            generate_html=True,
            generate_pdf=False,
            # 노출 개수 = 1 로 좁혀 봐도 CSV 는 전체가 저장되는지 확인
            top_selection_features=1,
        ),
    )
    outputs = AutoMLPipeline(cfg).run()
    assert "feature_selection_csv" in outputs
    csv_path: Path = outputs["feature_selection_csv"]
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8-sig")))
    assert set(rows[0].keys()) == {"feature", "frequency", "selected"}
    # 전체 feature (3개) 가 CSV 에 있어야 함 — 화면 노출 개수와 독립
    assert len(rows) == 3

    # HTML 표는 top_selection_features=1 로 단 1개만 노출
    html = (Path(cfg.reporting.output_dir) / "report.html").read_text()
    # selection 섹션 메타 메시지에 "표 노출 1 / 전체 3" 가 나타나야 함
    assert "표 노출 1 / 전체 3" in html


def test_top_selection_features_default_is_all_in_html(
    tmp_path: Path, tiny_binary_data,
):
    """top_selection_features=None (기본) 이면 HTML 표가 모든 변수를 노출한다."""
    X, y = tiny_binary_data
    df = X.copy()
    df["target"] = y.values
    df.iloc[:240].to_parquet(tmp_path / "train.parquet", index=False)
    df.iloc[240:].to_parquet(tmp_path / "test.parquet", index=False)

    cfg = AutoMLConfig(
        train_data_path=str(tmp_path / "train.parquet"),
        test_data_path=str(tmp_path / "test.parquet"),
        target_column="target",
        features=[
            FeatureSpec(name="num1", type="numeric"),
            FeatureSpec(name="num2", type="numeric"),
            FeatureSpec(name="cat1", type="categorical"),
        ],
        artifact_dir=str(tmp_path / "artifacts" / "models"),
        training=TrainingConfig(cv_folds=2, early_stopping_rounds=5),
        tuning=TuningConfig(enabled=False),
        feature_selection=FeatureSelectionConfig(
            enabled=True,
            base_estimator="lasso",
            n_subsamples=10,
            threshold=0.3,
            min_selected=1,
        ),
        models={
            "lgbm": ModelConfig(
                fixed_params={"n_estimators": 10, "verbose": -1},
                search_space={},
            ),
        },
        reporting=ReportingConfig(
            output_dir=str(tmp_path / "artifacts" / "reports"),
            generate_html=True,
            generate_pdf=False,
            # 기본값 그대로 (None) — 모든 변수 노출
        ),
    )
    AutoMLPipeline(cfg).run()
    html = (Path(cfg.reporting.output_dir) / "report.html").read_text()
    assert "표 노출 3 / 전체 3" in html
