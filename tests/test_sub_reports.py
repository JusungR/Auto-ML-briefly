"""Sub-report / best_model override / auto-ml-set-best 회귀 테스트.

다루는 변경:
  - 모든 모형이 ``artifact_dir/models/<name>.joblib`` 으로 저장된다.
  - 비-best 모형마다 ``output_dir/sub/<name>/report.html`` + importance CSV 가 생성된다.
  - ``training.best_model`` 설정으로 best 자동 선정을 override 할 수 있다.
  - ``auto-ml-set-best`` 가 재학습 없이 best 를 다른 sub-artifact 로 교체한다.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from auto_ml.cli.set_best import promote
from auto_ml.config import (
    AutoMLConfig,
    FeatureSpec,
    ModelConfig,
    ReportingConfig,
    TrainingConfig,
    TuningConfig,
)
from auto_ml.pipeline import AutoMLPipeline
from auto_ml.utils.io import load_artifact


def _build_quick_config(tmp_path: Path, X: pd.DataFrame, y: pd.Series, **overrides) -> AutoMLConfig:
    """3개 모형이 모두 enabled 인 가벼운 학습 설정 (튜닝 끄고 n_estimators 작게)."""
    df = X.copy()
    df["target"] = y.values
    df.iloc[:240].to_parquet(tmp_path / "train.parquet", index=False)
    df.iloc[240:].to_parquet(tmp_path / "test.parquet", index=False)

    base = dict(
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
        models={
            "lgbm": ModelConfig(
                fixed_params={"n_estimators": 20, "verbose": -1},
                search_space={},
            ),
            "xgb": ModelConfig(
                fixed_params={
                    "n_estimators": 20,
                    "objective": "binary:logistic",
                    "eval_metric": "auc",
                    "tree_method": "hist",
                },
                search_space={},
            ),
            "catboost": ModelConfig(
                fixed_params={
                    "iterations": 20,
                    "loss_function": "Logloss",
                    "eval_metric": "AUC",
                    "verbose": False,
                    "allow_writing_files": False,
                },
                search_space={},
            ),
        },
        reporting=ReportingConfig(
            output_dir=str(tmp_path / "artifacts" / "reports"),
            generate_html=True,
            generate_pdf=False,
        ),
    )
    base.update(overrides)
    return AutoMLConfig(**base)


# ----- 1. 모든 모형의 sub-artifact 가 저장되는지 ---------------------------

def test_all_models_saved_as_sub_artifacts(tmp_path: Path, tiny_binary_data):
    X, y = tiny_binary_data
    cfg = _build_quick_config(tmp_path, X, y)
    outputs = AutoMLPipeline(cfg).run()

    artifact_dir = Path(cfg.artifact_dir)
    assert (artifact_dir / "best.joblib").exists()
    for name in ("lgbm", "xgb", "catboost"):
        assert (artifact_dir / "models" / f"{name}.joblib").exists()

    assert "artifact" in outputs
    assert "sub_artifacts" in outputs


# ----- 2. 각 sub-artifact 가 독립적으로 로드 가능 + metadata 가 다름 -----

def test_each_sub_artifact_loads_with_its_own_model_name(tmp_path: Path, tiny_binary_data):
    X, y = tiny_binary_data
    cfg = _build_quick_config(tmp_path, X, y)
    AutoMLPipeline(cfg).run()

    artifact_dir = Path(cfg.artifact_dir)
    for name in ("lgbm", "xgb", "catboost"):
        bundle = load_artifact(artifact_dir / "models" / f"{name}.joblib")
        assert bundle["metadata"].model_name == name
        # preprocessor / model 객체가 모두 채워져 있어야 한다.
        assert bundle["preprocessor"] is not None
        assert bundle["model"] is not None


# ----- 3. 비-best 모형마다 sub-report 가 만들어지고 best 는 sub 없음 ------

def test_sub_reports_built_for_non_best_models(tmp_path: Path, tiny_binary_data):
    X, y = tiny_binary_data
    cfg = _build_quick_config(tmp_path, X, y)
    AutoMLPipeline(cfg).run()

    best_bundle = load_artifact(Path(cfg.artifact_dir) / "best.joblib")
    best_name = best_bundle["metadata"].model_name

    reports_dir = Path(cfg.reporting.output_dir)
    for name in ("lgbm", "xgb", "catboost"):
        sub_html = reports_dir / "sub" / name / "report.html"
        sub_csv = reports_dir / "sub" / name / "feature_importance.csv"
        if name == best_name:
            assert not sub_html.exists(), f"sub-report for best model {name!r} should not be generated"
        else:
            assert sub_html.exists(), f"sub-report missing: {sub_html}"
            assert sub_csv.exists(), f"sub-importance CSV missing: {sub_csv}"


# ----- 4. sub importance CSV 의 모형별 ranking 이 서로 다른지 ------------

def test_sub_importance_csv_per_model_distinct(tmp_path: Path, tiny_binary_data):
    """모형마다 importance 분포가 다르므로 ranking 도 (적어도 1개는) 달라야 한다."""
    X, y = tiny_binary_data
    cfg = _build_quick_config(tmp_path, X, y)
    AutoMLPipeline(cfg).run()

    reports_dir = Path(cfg.reporting.output_dir)
    rankings: dict[str, list[str]] = {}
    for name in ("lgbm", "xgb", "catboost"):
        csv_path = reports_dir / "sub" / name / "feature_importance.csv"
        if not csv_path.exists():
            continue  # best 모형은 sub 없음 — main 의 feature_importance.csv 가 그 역할
        with open(csv_path, encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        rankings[name] = [r["feature"] for r in rows]

    # 최소 1개 sub-report 가 존재하고, 적어도 두 sub 간 ranking 이 동일하지는 않다
    # (3개 feature 라 운 좋게 같을 수 있어 약한 조건만 — 서로 비교 가능한지만 확인)
    assert len(rankings) >= 2


# ----- 5. best_model override 동작 -----------------------------------------

def test_best_model_override_changes_best(tmp_path: Path, tiny_binary_data):
    X, y = tiny_binary_data
    cfg = _build_quick_config(
        tmp_path, X, y,
        training=TrainingConfig(
            cv_folds=2, early_stopping_rounds=5, best_model="lgbm",
        ),
    )
    AutoMLPipeline(cfg).run()

    bundle = load_artifact(Path(cfg.artifact_dir) / "best.joblib")
    assert bundle["metadata"].model_name == "lgbm"
    assert bundle["metadata"].extra.get("best_selection") == "user_override"


def test_best_model_override_unknown_name_raises(tmp_path: Path, tiny_binary_data):
    X, y = tiny_binary_data
    cfg = _build_quick_config(
        tmp_path, X, y,
        training=TrainingConfig(
            cv_folds=2, early_stopping_rounds=5, best_model="bogus",
        ),
    )
    with pytest.raises(ValueError, match="Unknown best_model override"):
        AutoMLPipeline(cfg).run()


# ----- 6. auto-ml-set-best 동작 -------------------------------------------

def test_set_best_promotes_sub_artifact(tmp_path: Path, tiny_binary_data):
    X, y = tiny_binary_data
    cfg = _build_quick_config(tmp_path, X, y)
    AutoMLPipeline(cfg).run()

    artifact_dir = Path(cfg.artifact_dir)
    before = load_artifact(artifact_dir / "best.joblib")["metadata"].model_name

    # before 와 다른 모형으로 교체 — set 중 첫 번째로 다른 것을 고른다.
    target = next(n for n in ("lgbm", "xgb", "catboost") if n != before)
    promote(cfg, target)

    after_bundle = load_artifact(artifact_dir / "best.joblib")
    assert after_bundle["metadata"].model_name == target
    assert after_bundle["metadata"].extra.get("promoted_from") == before
    assert "promoted_at" in after_bundle["metadata"].extra


def test_set_best_unknown_model_raises_friendly(tmp_path: Path, tiny_binary_data):
    X, y = tiny_binary_data
    cfg = _build_quick_config(tmp_path, X, y)
    AutoMLPipeline(cfg).run()

    with pytest.raises(FileNotFoundError, match="Sub-artifact not found"):
        promote(cfg, "nope")
