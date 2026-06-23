"""``final_fit_strategy`` 등 신규 학습 전략 옵션의 설정 검증."""
from __future__ import annotations

import pytest

from auto_ml.config import AutoMLConfig, TrainingConfig, load_config


def test_invalid_primary_metric_raises():
    with pytest.raises(ValueError, match="primary_metric must be one of"):
        TrainingConfig(primary_metric="bogus_metric")


@pytest.mark.parametrize(
    "metric", ["roc_auc", "pr_auc", "f1", "accuracy", "precision", "recall", "ks"]
)
def test_valid_primary_metrics_accepted(metric):
    cfg = TrainingConfig(primary_metric=metric)
    assert cfg.primary_metric == metric


def test_default_strategy_is_early_stop_on_test():
    """기본 전략은 현행 동작(early_stop_on_test) — 비파괴 회귀 가드."""
    assert TrainingConfig().final_fit_strategy == "early_stop_on_test"
    assert TrainingConfig().iteration_cap_aggregation == "mean"
    assert TrainingConfig().iteration_cap_headroom == 1.0


def test_invalid_strategy_raises():
    with pytest.raises(ValueError, match="final_fit_strategy must be one of"):
        TrainingConfig(final_fit_strategy="bogus")


def test_invalid_aggregation_raises():
    with pytest.raises(ValueError, match="iteration_cap_aggregation must be"):
        TrainingConfig(iteration_cap_aggregation="bogus")


def test_invalid_headroom_raises():
    with pytest.raises(ValueError, match="iteration_cap_headroom must be"):
        TrainingConfig(iteration_cap_headroom=0.0)


@pytest.mark.parametrize(
    "strategy", ["early_stop_on_test", "iteration_capping", "cv_bagging"]
)
def test_valid_strategies_accepted(strategy):
    cfg = TrainingConfig(final_fit_strategy=strategy)
    assert cfg.final_fit_strategy == strategy


def test_yaml_roundtrip(tmp_path):
    """YAML → load_config 경로로 신규 키가 정상 전달되는지 확인."""
    yaml_text = """
target_column: y
features_csv: ""
features:
  - {name: num1, type: numeric}
training:
  final_fit_strategy: iteration_capping
  iteration_cap_aggregation: median
  iteration_cap_headroom: 1.2
"""
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml_text)
    cfg = load_config(str(p))
    assert isinstance(cfg, AutoMLConfig)
    assert cfg.training.final_fit_strategy == "iteration_capping"
    assert cfg.training.iteration_cap_aggregation == "median"
    assert cfg.training.iteration_cap_headroom == 1.2
