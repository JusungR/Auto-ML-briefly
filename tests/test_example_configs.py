"""저장소에 포함된 예시 설정 / 예시 데이터 회귀 테스트.

규약:
- 예시 설정이 ``ensemble.strategy: elasticnet_plus_best`` 를 쓰면 ElasticNet 도 켜져
  있어야 한다. 아니면 앙상블이 WARNING 만 남기고 조용히 생략된다.
- ``examples/make_dummy_data.py`` 의 train / test / score 는 같은 분포에서 나와야 하고
  (학습한 모델이 test 에서도 통해야 함) 식별자가 파일 간에 겹치지 않아야 한다.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from auto_ml.config import load_config

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_CONFIGS = [
    "configs/example.yaml",
    "examples/credit/config.yaml",
    "examples/sample/config.yaml",
    "examples/titanic/config.yaml",
]


@pytest.mark.parametrize("path", EXAMPLE_CONFIGS)
def test_example_config_ensemble_is_buildable(path, monkeypatch):
    monkeypatch.chdir(ROOT)  # features_csv 상대경로 해석
    cfg = load_config(path)
    if not cfg.ensemble.enabled or cfg.ensemble.strategy != "elasticnet_plus_best":
        pytest.skip("ensemble 미사용 또는 elasticnet 불필요 strategy")
    assert cfg.models["elasticnet"].enabled, (
        f"{path}: strategy=elasticnet_plus_best 인데 elasticnet 이 꺼져 있어 앙상블이 생략된다"
    )


def test_make_dummy_data_splits_share_distribution(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "make_dummy_data", ROOT / "examples" / "make_dummy_data.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    monkeypatch.chdir(tmp_path)
    mod.main()
    tr = pd.read_parquet(tmp_path / "data" / "train.parquet")
    te = pd.read_parquet(tmp_path / "data" / "test.parquet")
    sc = pd.read_parquet(tmp_path / "data" / "score_input.parquet")

    assert (len(tr), len(te), len(sc)) == (4000, 1000, 1000)
    assert "target" not in sc.columns
    ids = [set(df["user_id"]) for df in (tr, te, sc)]
    assert not (ids[0] & ids[1]) and not (ids[1] & ids[2]) and not (ids[0] & ids[2])

    # 같은 분포라면 train 으로 학습한 단순 모델이 test 에서도 충분히 맞아야 한다.
    # (파일별로 따로 생성하던 이전 버전은 0.6 안팎까지 떨어졌다.)
    num = [c for c in tr.columns if c.startswith("num_")]
    clf = LogisticRegression(max_iter=1000).fit(tr[num], tr["target"])
    auc = roc_auc_score(te["target"], clf.predict_proba(te[num])[:, 1])
    assert auc > 0.85, auc
