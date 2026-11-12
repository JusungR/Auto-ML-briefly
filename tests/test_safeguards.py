"""롤백·배포 안전장치와 CPU 스레드 설정 회귀 테스트.

규약:
- 모델이 쓰는 변수가 스코어링 입력에 없으면 배포(학습 후 자동 배포·set_best·rollback) 를
  기본 거부하고, ``allow_missing_features`` 로만 진행한다.
- 학습·배포는 ``artifact_dir/.lock`` 으로 한 번에 하나만 실행된다. 살아 있는 작업의
  잠금이 있으면 즉시 실패하고, 죽은 프로세스의 잠금은 넘겨받는다.
- ``training.n_jobs`` 는 각 라이브러리 스레드 인자로 들어가고(사용자 지정이 우선),
  ``scoring.n_jobs`` 는 학습된 모델의 예측 스레드를 바꾸되 예측 결과는 바꾸지 않는다.
"""
from __future__ import annotations

import json
import socket
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from test_versioning import _config, _prod, _train

from auto_ml import versioning
from auto_ml.cli.rollback import rollback
from auto_ml.cli.set_best import promote
from auto_ml.models.catboost_model import CatBoostModel
from auto_ml.models.elasticnet_model import ElasticNetModel
from auto_ml.models.ensemble_model import EnsembleModel
from auto_ml.models.lgbm_model import LGBMModel
from auto_ml.models.registry import build_model
from auto_ml.models.xgb_model import XGBModel
from auto_ml.pipeline import AutoMLPipeline
from auto_ml.utils.io import load_artifact
from auto_ml.utils.threads import available_cpus, resolve_n_jobs

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


# ───────────────────────────────────────────────────────────────────────
# ② artifact_dir 잠금
# ───────────────────────────────────────────────────────────────────────

def _write_lock(cfg, pid: int, host: str | None = None) -> Path:
    lock = Path(cfg.artifact_dir) / versioning.LOCK_FILE
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text(json.dumps({
        "pid": pid, "host": host or socket.gethostname(),
        "started_at": "2026-01-01T00:00:00+00:00", "purpose": "test",
    }))
    return lock


def _dead_pid() -> int:
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


def test_live_lock_blocks_training_and_deploys_immediately(tmp_path, tiny_binary_data):
    cfg = _config(tmp_path, tiny_binary_data)
    _train(cfg, 1)
    _train(cfg, 2)
    sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        _write_lock(cfg, sleeper.pid)
        n_runs = len(versioning.list_runs(cfg.artifact_dir))
        for action in (lambda: AutoMLPipeline(cfg).run(), lambda: promote(cfg, "xgb"),
                       lambda: rollback(cfg)):
            with pytest.raises(versioning.ArtifactLockError, match=str(sleeper.pid)):
                action()
        assert len(versioning.list_runs(cfg.artifact_dir)) == n_runs  # 학습이 시작되지 않음
    finally:
        sleeper.kill()
        sleeper.wait()


def test_stale_lock_is_taken_over_and_released(tmp_path, tiny_binary_data):
    cfg = _config(tmp_path, tiny_binary_data)
    lock = _write_lock(cfg, _dead_pid())
    _train(cfg, 1)                                  # 죽은 프로세스의 잠금 → 넘겨받아 진행
    assert not lock.exists()                        # 끝나면 해제

    _write_lock(cfg, _dead_pid(), host="another-host")  # 다른 호스트는 판단하지 않는다
    with pytest.raises(versioning.ArtifactLockError, match="delete"):
        promote(cfg, "xgb")


def test_lock_released_after_failure_and_reentrant(tmp_path, tiny_binary_data):
    cfg = _config(tmp_path, tiny_binary_data)
    lock = Path(cfg.artifact_dir) / versioning.LOCK_FILE
    cfg.train_data_path = str(tmp_path / "missing.parquet")
    with pytest.raises(FileNotFoundError):
        AutoMLPipeline(cfg).run()
    assert not lock.exists()

    with versioning.artifact_lock(cfg.artifact_dir, "outer"):
        with versioning.artifact_lock(cfg.artifact_dir, "inner"):  # 같은 프로세스 중첩 허용
            assert lock.exists()
        assert lock.exists()
    assert not lock.exists()


# ───────────────────────────────────────────────────────────────────────
# ③ 스레드 수
# ───────────────────────────────────────────────────────────────────────

def test_resolve_n_jobs_rules():
    cpus = available_cpus()
    assert resolve_n_jobs(None) is None
    assert resolve_n_jobs(1) == 1
    assert resolve_n_jobs(10_000) == cpus
    assert resolve_n_jobs(-1) == cpus
    assert resolve_n_jobs(-10_000) == 1
    with pytest.raises(ValueError):
        resolve_n_jobs(0)


@pytest.mark.parametrize(
    "name,key", [("lgbm", "n_jobs"), ("xgb", "n_jobs"), ("catboost", "thread_count")],
)
def test_training_n_jobs_injected_unless_user_sets_threads(name, key):
    assert build_model(name, n_jobs=1)._resolved_params()[key] == 1
    assert key not in build_model(name)._resolved_params()      # 기본 = 라이브러리 기본
    user_key = {"lgbm": "num_threads", "xgb": "nthread", "catboost": "thread_count"}[name]
    params = build_model(name, params={user_key: 3}, n_jobs=1)._resolved_params()
    assert params[user_key] == 3 and (key == user_key or key not in params)


def test_training_n_jobs_reaches_trained_models(tmp_path, tiny_binary_data):
    cfg = _config(tmp_path, tiny_binary_data)
    cfg.training.n_jobs = 1
    _train(cfg, 1)
    lgbm = load_artifact(Path(cfg.artifact_dir) / "models" / "lgbm.joblib")["model"]
    xgb = load_artifact(Path(cfg.artifact_dir) / "models" / "xgb.joblib")["model"]
    assert lgbm.model.get_params()["n_jobs"] == 1
    assert xgb.model.get_params()["n_jobs"] == 1


def _xgb_booster_nthread(model: XGBModel) -> int:
    conf = json.loads(model.model.get_booster().save_config())
    return int(conf["learner"]["generic_param"]["nthread"])


def test_set_n_jobs_changes_threads_not_predictions(tiny_binary_data):
    X, y = tiny_binary_data
    models = {
        "lgbm": LGBMModel(params={"n_estimators": 30, "verbose": -1}, categorical_columns=["cat1"]),
        "xgb": XGBModel(params={"n_estimators": 30}, categorical_columns=["cat1"]),
        "catboost": CatBoostModel(
            params={"iterations": 30, "verbose": False, "allow_writing_files": False},
            categorical_columns=["cat1"],
        ),
        "elasticnet": ElasticNetModel(params={}, categorical_columns=["cat1"]),
    }
    for m in models.values():
        m.fit(X, y)
    ens = EnsembleModel(dict(models), {k: 0.25 for k in models})
    before = {k: m.predict_proba(X) for k, m in models.items()} | {"ens": ens.predict_proba(X)}

    ens.set_n_jobs(1)  # 서브모델 전체에 전파
    assert models["lgbm"].model.get_params()["n_jobs"] == 1
    assert _xgb_booster_nthread(models["xgb"]) == 1
    assert models["catboost"]._predict_threads() == 1
    after = {k: m.predict_proba(X) for k, m in models.items()} | {"ens": ens.predict_proba(X)}
    for k in before:
        np.testing.assert_allclose(after[k], before[k], err_msg=k)


@pytest.mark.parametrize("name", ["lgbm", "xgb", "catboost", "ensemble"])
def test_set_n_jobs_on_legacy_artifacts(name):
    """기능 이전에 학습된 실제 artifact 객체에도 동작하고 점수가 같다."""
    from auto_ml.scoring.scorer import Scorer

    path = ROOT / f"artifacts/credit/models/models/{name}.joblib"
    data = ROOT / "examples/credit/data/score_input.parquet"
    if not path.exists() or not data.exists():
        pytest.skip("legacy artifact not present")
    df = pd.read_parquet(data).head(300)
    base = Scorer.from_artifact(path).score(df)["score"].to_numpy()
    limited = Scorer.from_artifact(path, n_jobs=1)
    assert getattr(limited.model, "n_jobs", None) == 1
    np.testing.assert_allclose(limited.score(df)["score"].to_numpy(), base)


def test_scoring_n_jobs_flows_through_runner(tmp_path, tiny_binary_data, monkeypatch):
    from auto_ml.scoring.runner import run_scoring
    from auto_ml.scoring.scorer import Scorer

    cfg = _config(tmp_path, tiny_binary_data)
    _train(cfg, 1)
    cfg.scoring.n_jobs = 1
    seen = {}
    original = Scorer.from_artifact.__func__

    def spy(cls, path, n_jobs=None):
        seen["n_jobs"] = n_jobs
        return original(cls, path, n_jobs=n_jobs)

    monkeypatch.setattr(Scorer, "from_artifact", classmethod(spy))
    run_scoring(cfg)
    assert seen["n_jobs"] == 1
    assert len(pd.read_parquet(cfg.scoring.output_path)) == 80


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
