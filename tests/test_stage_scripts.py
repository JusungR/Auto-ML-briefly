"""단계 분리 실행 (train/*.py, inference/*.py) 회귀 테스트.

규약:
- 단계 산출물을 work_dir 에 저장·로드해 따로 실행해도 한 번에 실행한 결과
  (``run_scoring``) 와 스코어가 동일하다.
- 스크립트는 패키지 설치 없이 저장소 루트 밖의 cwd 에서도 실행된다.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from auto_ml.config import load_config
from auto_ml.scoring.runner import run_scoring
from auto_ml.stages import inference as stages_inference
from auto_ml.stages import load_stage, save_stage
from auto_ml.stages import train as stages_train

ROOT = Path(__file__).resolve().parents[1]


def _write_config(tmp_path: Path, tiny_binary_data) -> Path:
    X, y = tiny_binary_data
    df = X.copy()
    df["target"] = y.values
    df["client_id"] = np.arange(1000, 1000 + len(df))
    df.iloc[:240].to_parquet(tmp_path / "train.parquet", index=False)
    df.iloc[240:].to_parquet(tmp_path / "test.parquet", index=False)
    df.iloc[240:].drop(columns=["target"]).to_parquet(
        tmp_path / "score_input.parquet", index=False,
    )
    pd.DataFrame(
        {
            "name": ["num1", "num2", "cat1"],
            "type": ["continuous", "continuous", "category"],
            "used": ["true", "true", "true"],
        }
    ).to_csv(tmp_path / "features.csv", index=False)

    out = tmp_path / "artifacts"
    cfg = {
        "train_data_path": str(tmp_path / "train.parquet"),
        "test_data_path": str(tmp_path / "test.parquet"),
        "target_column": "target",
        "features_csv": str(tmp_path / "features.csv"),
        "id_columns": ["client_id"],
        "artifact_dir": str(out / "models"),
        "training": {"cv_folds": 2, "early_stopping_rounds": 5},
        "tuning": {"enabled": False},
        "models": {
            "lgbm": {"fixed_params": {"n_estimators": 20, "verbose": -1}, "search_space": {}},
        },
        "reporting": {
            "output_dir": str(out / "reports"),
            "generate_html": True,
            "generate_pdf": False,
        },
        "scoring": {
            "input_path": str(tmp_path / "score_input.parquet"),
            "output_path": str(out / "scores" / "scores.parquet"),
            "threshold": 0.5,
        },
        "logging": {"log_dir": str(out / "logs"), "to_file": False},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_stagewise_matches_run_scoring(tmp_path, tiny_binary_data):
    """단계별 save/load 를 거친 결과가 run_scoring 결과와 같다."""
    cfg = load_config(_write_config(tmp_path, tiny_binary_data))
    work = tmp_path / "work"

    # 학습: 전처리 → 학습 → 후처리 (단계마다 디스크 왕복)
    save_stage(stages_train.preprocess(cfg), work, stages_train.PREPROCESS_OUTPUT)
    prep = load_stage(work, stages_train.PREPROCESS_OUTPUT)
    save_stage(stages_train.train(cfg, prep), work, stages_train.TRAIN_OUTPUT)
    result = load_stage(work, stages_train.TRAIN_OUTPUT)
    outputs = stages_train.postprocess(cfg, prep, result)
    assert outputs["artifact"].exists()
    assert (Path(cfg.artifact_dir) / "models" / "lgbm.joblib").exists()

    # 추론: 전처리 → 예측 → 후처리 (단계마다 디스크 왕복)
    save_stage(stages_inference.preprocess(cfg), work, stages_inference.PREPROCESS_OUTPUT)
    iprep = load_stage(work, stages_inference.PREPROCESS_OUTPUT)
    save_stage(stages_inference.predict(iprep), work, stages_inference.PREDICT_OUTPUT)
    ipred = load_stage(work, stages_inference.PREDICT_OUTPUT)
    stagewise = pd.read_parquet(stages_inference.postprocess(cfg, ipred))

    oneshot = pd.read_parquet(run_scoring(cfg))
    assert list(stagewise.columns) == ["client_id", "score", "prediction"]
    pd.testing.assert_frame_equal(stagewise, oneshot)


def test_load_stage_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="previous stage"):
        load_stage(tmp_path, stages_train.PREPROCESS_OUTPUT)


def test_scripts_run_without_install(tmp_path, tiny_binary_data):
    """6개 단계 스크립트를 저장소 밖 cwd 에서 순서대로 실행한다."""
    config_path = _write_config(tmp_path, tiny_binary_data)
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    for script in (
        "train/preprocess.py",
        "train/train.py",
        "train/postprocess.py",
        "inference/preprocess.py",
        "inference/predict.py",
        "inference/postprocess.py",
    ):
        proc = subprocess.run(
            [sys.executable, str(ROOT / script), "--config", str(config_path)],
            cwd=tmp_path, env=env, capture_output=True, text=True,
        )
        assert proc.returncode == 0, f"{script} failed:\n{proc.stderr[-2000:]}"

    out = pd.read_parquet(tmp_path / "artifacts" / "scores" / "scores.parquet")
    assert list(out.columns) == ["client_id", "score", "prediction"]
    assert len(out) == 60
    assert (tmp_path / "artifacts" / "work" / "train" / "train_result.pkl.gz").exists()
