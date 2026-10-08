"""단계 간 중간 산출물 저장 / 로드 유틸리티."""
from __future__ import annotations

import gzip
import pickle
from pathlib import Path
from typing import Any

import cloudpickle

from auto_ml.config import AutoMLConfig


def default_work_dir(config: AutoMLConfig, mode: str) -> Path:
    """단계 간 중간 산출물 기본 경로.

    ``artifact_dir`` 의 부모 아래 ``work/<mode>`` 를 사용한다
    (예: ``./artifacts/credit/models`` → ``./artifacts/credit/work/train``).

    Args:
        config: AutoMLConfig.
        mode: ``"train"`` 또는 ``"inference"``.
    """
    return Path(config.artifact_dir).parent / "work" / mode


def save_stage(obj: Any, work_dir: str | Path, name: str) -> Path:
    """단계 산출물을 ``<work_dir>/<name>.pkl.gz`` 로 저장한다.

    모델 객체(focal loss 클로저 등) 를 포함할 수 있어 artifact 와 동일하게
    cloudpickle + gzip 으로 직렬화한다.
    """
    path = Path(work_dir) / f"{name}.pkl.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as f:
        cloudpickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_stage(work_dir: str | Path, name: str) -> Any:
    """``save_stage`` 로 저장한 단계 산출물을 로드한다."""
    path = Path(work_dir) / f"{name}.pkl.gz"
    if not path.exists():
        raise FileNotFoundError(
            f"Stage output not found: {path}. Run the previous stage first."
        )
    with gzip.open(path, "rb") as f:
        return pickle.load(f)
