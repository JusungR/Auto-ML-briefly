"""배포 안전장치 회귀 테스트.

규약: 모델이 쓰는 변수가 스코어링 입력에 없으면 배포(학습 후 자동 배포·set_best·rollback) 를
기본 거부하고, ``allow_missing_features`` 로만 진행한다.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from test_versioning import _config, _prod, _train

from auto_ml import versioning
from auto_ml.cli.rollback import rollback
from auto_ml.cli.set_best import promote
from auto_ml.pipeline import AutoMLPipeline
from auto_ml.utils.io import load_artifact

ROOT = Path(__file__).resolve().parents[1]


# ───────────────────────────────────────────────────────────────────────
# ① 모델이 쓰는 변수가 입력에 없으면 배포 거부
# ───────────────────────────────────────────────────────────────────────

def _drop_num2(cfg) -> None:
    """운영 입력에서 num2 가 빠지고, 그에 맞춰 num2 없이 학습하는 상황."""
    path = Path(cfg.scoring.input_path)
    pd.read_parquet(path).drop(columns=["num2"]).to_parquet(path)
    cfg.features = [f for f in cfg.features if f.name != "num2"]


def test_rollback_to_model_using_missing_feature_is_refused(tmp_path, tiny_binary_data):
    cfg = _config(tmp_path, tiny_binary_data)
    r1 = _train(cfg, 1)                      # num2 를 쓰는 모델
    _drop_num2(cfg)
    _train(cfg, 2)                           # num2 없는 모델 (운영)
    art = Path(cfg.artifact_dir)
    before = (art / "best.joblib").read_bytes()
    n_log = len(versioning.read_deployments(art))

    with pytest.raises(versioning.DeploymentError, match=r"missing from .*\['num2'\]"):
        rollback(cfg)
    assert (art / "best.joblib").read_bytes() == before
    assert len(versioning.read_deployments(art)) == n_log

    rollback(cfg, allow_missing_features=True)  # 명시하면 진행
    assert _prod(cfg)["metadata"].extra["run_id"] == r1


def test_auto_deploy_refused_when_scoring_input_lacks_a_used_feature(tmp_path, tiny_binary_data):
    cfg = _config(tmp_path, tiny_binary_data)
    path = Path(cfg.scoring.input_path)
    pd.read_parquet(path).drop(columns=["num2"]).to_parquet(path)  # 학습엔 있고 운영 입력엔 없음

    with pytest.raises(versioning.DeploymentError, match="saved but NOT deployed") as err:
        AutoMLPipeline(cfg).run()
    runs = versioning.list_runs(cfg.artifact_dir)
    assert len(runs) == 1 and runs[0].name in str(err.value)
    assert "--allow-missing-features" in str(err.value)
    assert not (Path(cfg.artifact_dir) / "best.joblib").exists()
    promote(cfg, "lgbm", run_id=runs[0].name, allow_missing_features=True)
    assert (Path(cfg.artifact_dir) / "best.joblib").exists()


def test_check_compatibility_missing_used_feature(tmp_path, tiny_binary_data):
    cfg = _config(tmp_path, tiny_binary_data)
    _train(cfg, 1)
    bundle = load_artifact(Path(cfg.artifact_dir) / "best.joblib")
    pd.read_parquet(cfg.scoring.input_path).drop(columns=["num1"]).to_parquet(
        cfg.scoring.input_path,
    )
    with pytest.raises(versioning.DeploymentError, match="num1"):
        versioning.check_compatibility(bundle, cfg)
    result = versioning.check_compatibility(bundle, cfg, allow_missing_features=True)
    assert result["missing_features"] == ["num1"] and result["scored_rows"] == 80
