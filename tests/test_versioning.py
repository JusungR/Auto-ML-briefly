"""실행 이력(runs) · 배포 이력 · 롤백 회귀 테스트.

규약:
- 학습마다 ``artifact_dir/runs/<run_id>/`` 에 실행 전체가 남고, ``models/`` 는 최신 실행 사본.
- 운영 모델 교체(학습 자동 배포 / set_best / rollback) 는 ``deployments.jsonl`` 에 기록된다.
- ``rollback`` 은 현재 ↔ 직전 운영 모델을 전환한다.
- 보존 정책은 최근 N 개 + 현재·직전 운영 실행을 남긴다.
- 운영 모델 교체는 원자적이며, 시험 스코어링에 실패하면 운영 모델이 바뀌지 않는다.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from auto_ml import versioning
from auto_ml.cli.rollback import rollback
from auto_ml.cli.set_best import describe, promote
from auto_ml.config import (
    AutoMLConfig,
    FeatureSpec,
    LoggingConfig,
    ModelConfig,
    ReportingConfig,
    ScoringConfig,
    TrainingConfig,
    TuningConfig,
    VersioningConfig,
)
from auto_ml.pipeline import AutoMLPipeline
from auto_ml.scoring.scorer import Scorer
from auto_ml.stages.inference import artifact_path_of
from auto_ml.utils.io import load_artifact, save_artifact


def _config(tmp_path: Path, tiny_binary_data, **versioning_kw) -> AutoMLConfig:
    X, y = tiny_binary_data
    df = X.copy()
    df["target"] = y.values
    df.iloc[:220].to_parquet(tmp_path / "train.parquet")
    df.iloc[220:].to_parquet(tmp_path / "test.parquet")
    df.iloc[220:].drop(columns=["target"]).to_parquet(tmp_path / "score_input.parquet")
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
        training=TrainingConfig(cv_folds=2),
        tuning=TuningConfig(enabled=False),
        models={
            "lgbm": ModelConfig(fixed_params={"n_estimators": 30, "verbose": -1}),
            "xgb": ModelConfig(fixed_params={"n_estimators": 30}),
        },
        reporting=ReportingConfig(
            output_dir=str(tmp_path / "artifacts" / "reports"), generate_pdf=False,
        ),
        scoring=ScoringConfig(
            input_path=str(tmp_path / "score_input.parquet"),
            output_path=str(tmp_path / "artifacts" / "scores.parquet"),
        ),
        logging=LoggingConfig(log_dir=str(tmp_path / "logs"), to_file=False),
        versioning=VersioningConfig(**versioning_kw),
    )


def _train(cfg: AutoMLConfig, seed: int) -> str:
    """seed 를 바꿔 실행마다 다른 모델을 만들고 run_id 를 반환한다."""
    cfg.models["lgbm"].fixed_params["seed"] = seed
    cfg.models["xgb"].fixed_params["random_state"] = seed
    cfg.models["lgbm"].fixed_params["n_estimators"] = 20 + seed
    return AutoMLPipeline(cfg).run()["run_dir"].name


def _prod(cfg: AutoMLConfig):
    return load_artifact(Path(cfg.artifact_dir) / "best.joblib")


def _scores(bundle, cfg: AutoMLConfig) -> np.ndarray:
    df = pd.read_parquet(cfg.scoring.input_path)
    return Scorer(bundle["preprocessor"], bundle["model"], bundle["metadata"]).score(df)[
        "score"
    ].to_numpy()


def test_each_training_keeps_a_full_run(tmp_path, tiny_binary_data):
    cfg = _config(tmp_path, tiny_binary_data)
    art = Path(cfg.artifact_dir)
    r1 = _train(cfg, 1)
    first = sorted(p.name for p in (art / "runs" / r1 / "models").glob("*.joblib"))
    r2 = _train(cfg, 2)

    assert [d.name for d in versioning.list_runs(art)] == [r1, r2]
    latest = sorted(p.name for p in (art / "models").glob("*.joblib"))
    assert latest == sorted(p.name for p in (art / "runs" / r2 / "models").glob("*.joblib"))
    assert sorted(p.name for p in (art / "runs" / r1 / "models").glob("*.joblib")) == first
    for run in (r1, r2):
        d = art / "runs" / run
        assert (d / "reports" / "report.html").exists()
        assert (d / "predictions" / "test_predictions.parquet").exists()
        info = json.loads((d / "run_info.json").read_text())
        assert info["run_id"] == run and info["best_model"] in info["models"]
        assert info["data"]["train"]["rows"] == 220
    # 후보 모델 메타데이터에 run_id
    assert load_artifact(art / "models" / latest[0])["metadata"].extra["run_id"] == r2


def test_deployment_history_and_rollback_toggle(tmp_path, tiny_binary_data):
    cfg = _config(tmp_path, tiny_binary_data)
    r1 = _train(cfg, 1)
    s1 = _scores(_prod(cfg), cfg)
    r2 = _train(cfg, 2)
    s2 = _scores(_prod(cfg), cfg)
    assert not np.allclose(s1, s2)

    log = versioning.read_deployments(cfg.artifact_dir)
    assert [e["action"] for e in log] == ["train", "train"]
    assert log[1]["previous"]["run_id"] == r1 and log[0]["previous"] is None

    rollback(cfg)
    assert _prod(cfg)["metadata"].extra["run_id"] == r1
    np.testing.assert_allclose(_scores(_prod(cfg), cfg), s1)
    assert _prod(cfg)["metadata"].extra["best_selection"] == "rollback"

    rollback(cfg)  # 한 번 더 → 원래 모델 (전환)
    assert _prod(cfg)["metadata"].extra["run_id"] == r2
    np.testing.assert_allclose(_scores(_prod(cfg), cfg), s2)
    assert [e["action"] for e in versioning.read_deployments(cfg.artifact_dir)][-2:] == [
        "rollback", "rollback",
    ]


def test_set_best_from_earlier_run(tmp_path, tiny_binary_data):
    cfg = _config(tmp_path, tiny_binary_data)
    r1 = _train(cfg, 1)
    _train(cfg, 2)
    promote(cfg, "xgb", run_id=r1)
    meta = _prod(cfg)["metadata"]
    assert (meta.model_name, meta.extra["run_id"], meta.extra["best_selection"]) == (
        "xgb", r1, "promoted",
    )
    last = versioning.current_deployment(cfg.artifact_dir)
    assert last["action"] == "promote" and last["source"] == f"runs/{r1}/models/xgb.joblib"
    with pytest.raises(FileNotFoundError, match="Run not found"):
        promote(cfg, "xgb", run_id="nope")
    assert "PRODUCTION: xgb" in describe(cfg)


def test_retention_keeps_recent_and_protected_runs(tmp_path, tiny_binary_data):
    cfg = _config(tmp_path, tiny_binary_data, keep_last_runs=2)
    r1 = _train(cfg, 1)                      # 운영 모델 = r1
    cfg.versioning.auto_deploy = False       # 이후 실행은 배포하지 않는다
    r2, r3, r4 = (_train(cfg, s) for s in (2, 3, 4))

    remaining = {d.name for d in versioning.list_runs(cfg.artifact_dir)}
    assert remaining == {r1, r3, r4}         # 최근 2개 + 운영 중인 r1 (오래됐어도 보호)
    assert r2 not in remaining
    assert _prod(cfg)["metadata"].extra["run_id"] == r1


def test_auto_deploy_false_leaves_production_unset(tmp_path, tiny_binary_data):
    cfg = _config(tmp_path, tiny_binary_data, auto_deploy=False)
    run = _train(cfg, 1)
    art = Path(cfg.artifact_dir)
    assert (art / "runs" / run / "models").exists()
    assert not (art / "best.joblib").exists()
    with pytest.raises(FileNotFoundError, match=f"--run {run}"):
        artifact_path_of(cfg)
    promote(cfg, "lgbm", run_id=run)
    assert artifact_path_of(cfg).exists()


def test_legacy_layout_is_migrated_and_rollback_restores_it(tmp_path, tiny_binary_data):
    cfg = _config(tmp_path, tiny_binary_data)
    art = Path(cfg.artifact_dir)
    _train(cfg, 1)
    # 버전 관리 이전 구조로 되돌린다: models/ + best.joblib 만 남김
    shutil.rmtree(art / "runs")
    (art / "deployments.jsonl").unlink()
    legacy_scores = _scores(_prod(cfg), cfg)
    trained_at = _prod(cfg)["metadata"].trained_at
    (art / "best.joblib").touch()  # 복사·checkout 처럼 수정 시각이 바뀌어도

    _train(cfg, 2)
    runs = versioning.list_runs(art)
    assert runs[0].name.startswith("legacy_") and len(runs) == 2
    # 실행 이름·시각은 파일 수정 시각이 아니라 메타데이터의 학습 시각을 따른다
    from datetime import datetime, timezone

    local = datetime.fromisoformat(trained_at).replace(tzinfo=timezone.utc).astimezone()
    assert runs[0].name == f"legacy_{local.strftime('%Y%m%d_%H%M%S')}"
    assert "legacy import" in describe(cfg)  # --list 가 legacy 실행에서도 동작
    log = versioning.read_deployments(art)
    assert [e["action"] for e in log] == ["legacy", "train"]

    rollback(cfg)
    np.testing.assert_allclose(_scores(_prod(cfg), cfg), legacy_scores)
    assert _prod(cfg)["metadata"].extra["run_id"] == runs[0].name


def test_save_artifact_is_atomic(tmp_path, monkeypatch):
    import cloudpickle

    path = tmp_path / "best.joblib"
    save_artifact(path, "pp", "model-v1", {"v": 1})

    def broken_dump(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(cloudpickle, "dump", broken_dump)
    with pytest.raises(OSError, match="disk full"):
        save_artifact(path, "pp", "model-v2", {"v": 2})
    assert load_artifact(path)["model"] == "model-v1"
    assert [p.name for p in tmp_path.iterdir()] == ["best.joblib"]  # 임시 파일 정리됨


def test_failed_compatibility_check_keeps_production(tmp_path, tiny_binary_data):
    cfg = _config(tmp_path, tiny_binary_data)
    r1 = _train(cfg, 1)
    _train(cfg, 2)
    art = Path(cfg.artifact_dir)
    before = (art / "best.joblib").read_bytes()
    n_log = len(versioning.read_deployments(art))

    previous_src = art / versioning.current_deployment(art)["previous"]["source"]
    assert r1 in str(previous_src)
    previous_src.write_bytes(b"not a model")  # 열리지 않는 옛 모델
    with pytest.raises(versioning.DeploymentError, match="Cannot load"):
        rollback(cfg)
    assert (art / "best.joblib").read_bytes() == before
    assert len(versioning.read_deployments(art)) == n_log
