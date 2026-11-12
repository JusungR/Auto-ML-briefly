"""CPU 스레드 설정(training.n_jobs / scoring.n_jobs) 회귀 테스트.

규약: ``training.n_jobs`` 는 각 라이브러리 스레드 인자로 들어가고(fixed_params 의 스레드 인자가
우선), ``scoring.n_jobs`` 는 학습된 모델의 예측 스레드를 바꾸되 예측 결과는 바꾸지 않는다.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from test_versioning import _config, _train

from auto_ml.models.catboost_model import CatBoostModel
from auto_ml.models.elasticnet_model import ElasticNetModel
from auto_ml.models.ensemble_model import EnsembleModel
from auto_ml.models.lgbm_model import LGBMModel
from auto_ml.models.registry import build_model
from auto_ml.models.xgb_model import XGBModel
from auto_ml.utils.io import load_artifact
from auto_ml.utils.threads import available_cpus, resolve_n_jobs

ROOT = Path(__file__).resolve().parents[1]


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

