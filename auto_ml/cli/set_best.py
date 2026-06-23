"""``auto-ml-set-best`` CLI 진입점.

학습이 끝난 뒤 ``artifact_dir/models/<name>.joblib`` 중 하나를
``artifact_dir/best.joblib`` 으로 승격(복사)한다. 재학습 없이 운영 best 를
교체할 수 있어, 자동 선정된 모형이 운영 중 부적합하다고 판단될 때 빠르게
대응할 수 있다.

사용 예::

    auto-ml-set-best --config configs/example.yaml --model xgb
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from auto_ml.config import AutoMLConfig, load_config
from auto_ml.utils.io import ARTIFACT_FILENAME, load_artifact, save_artifact
from auto_ml.utils.logger import get_logger, setup_logging

logger = get_logger("set_best")
SUB_ARTIFACT_DIRNAME = "models"


def promote(config: AutoMLConfig, model_name: str) -> Path:
    """``model_name`` 의 sub-artifact 를 ``best.joblib`` 으로 복사한다.

    Args:
        config: AutoMLConfig — ``artifact_dir`` 만 사용.
        model_name: 승격할 모형 이름 (sub-artifact 파일 stem).

    Returns:
        새로 저장된 ``best.joblib`` 경로.

    Raises:
        FileNotFoundError: 지정한 sub-artifact 가 없을 때.
    """
    artifact_dir = Path(config.artifact_dir)
    sub_path = artifact_dir / SUB_ARTIFACT_DIRNAME / f"{model_name}.joblib"
    best_path = artifact_dir / ARTIFACT_FILENAME

    if not sub_path.exists():
        available: list[str] = []
        sub_dir = artifact_dir / SUB_ARTIFACT_DIRNAME
        if sub_dir.exists():
            available = sorted(p.stem for p in sub_dir.glob("*.joblib"))
        raise FileNotFoundError(
            f"Sub-artifact not found: {sub_path}. "
            f"Available sub-artifacts: {available or '(none — train first)'}"
        )

    bundle = load_artifact(sub_path)
    preprocessor = bundle["preprocessor"]
    model = bundle["model"]
    metadata = bundle["metadata"]

    # 이전 best 의 모형 이름을 로그에 남긴다 — 운영 추적용.
    previous_name: str | None = None
    if best_path.exists():
        try:
            prev = load_artifact(best_path)
            previous_name = getattr(prev["metadata"], "model_name", None)
        except Exception as exc:  # 손상된 best 파일이어도 승격은 진행
            logger.warning("Failed to inspect existing best.joblib: %s", exc)

    metadata.extra = dict(metadata.extra)
    metadata.extra["promoted_at"] = datetime.utcnow().isoformat()
    metadata.extra["promoted_from"] = previous_name

    save_artifact(best_path, preprocessor, model, metadata)
    primary_metric = getattr(metadata, "primary_metric", "?")
    metric_value = getattr(metadata, "metric_value", float("nan"))
    logger.info(
        "Promoted '%s' to %s (%s=%.4f, was=%s)",
        model_name, best_path, primary_metric, metric_value,
        previous_name if previous_name is not None else "(none)",
    )
    return best_path


def cli_set_best() -> None:
    """``auto-ml-set-best`` 진입점."""
    parser = argparse.ArgumentParser(
        description="Promote a sub-artifact (artifact_dir/models/<name>.joblib) "
                    "to artifact_dir/best.joblib without retraining."
    )
    parser.add_argument("--config", required=True, help="설정 YAML 경로")
    parser.add_argument(
        "--model", required=True,
        help="승격할 모형 이름 (sub-artifact 파일 stem). 예: lgbm | xgb | catboost",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(
        log_dir=config.logging.log_dir,
        level=config.logging.level,
        to_stdout=config.logging.to_stdout,
        to_file=config.logging.to_file,
        stage="set_best",
    )
    promote(config, args.model)


if __name__ == "__main__":
    cli_set_best()
