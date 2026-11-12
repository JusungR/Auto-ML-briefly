"""운영 모델 롤백 CLI 진입점.

배포 이력(``artifact_dir/deployments.jsonl``) 의 마지막 항목이 가리키는 **직전 운영 모델**
로 ``best.joblib`` 을 되돌린다. 롤백도 배포 이력에 기록되므로 한 번 더 실행하면 원래
모델로 돌아온다 (``cd -`` 처럼 현재 ↔ 직전 전환). 더 이전 실행으로 가려면
``python -m auto_ml.cli.set_best --run <id> --model <name>`` 을 쓴다.

교체는 :func:`auto_ml.versioning.deploy` 를 거치므로 대상 모델이 열리지 않거나 시험
스코어링에 실패하면 운영 모델은 바뀌지 않는다.

사용 예::

    python -m auto_ml.cli.rollback --config configs/example.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

from auto_ml import versioning
from auto_ml.config import AutoMLConfig, load_config
from auto_ml.utils.logger import get_logger, setup_logging

logger = get_logger("rollback")


def rollback(config: AutoMLConfig, allow_missing_features: bool = False) -> Path:
    """직전 운영 모델로 ``best.joblib`` 을 되돌린다.

    Args:
        config: AutoMLConfig.
        allow_missing_features: 되돌릴 모델이 쓰는 변수가 스코어링 입력에 없어도 진행한다.

    Raises:
        RuntimeError: 배포 이력이 없거나 직전 운영 모델 정보가 없을 때.
        FileNotFoundError: 직전 운영 모델 파일(실행 폴더) 이 삭제된 경우.
        auto_ml.versioning.DeploymentError: 열리지 않거나, 모델이 쓰는 변수가 입력에
            없거나, 시험 스코어링에 실패한 경우.
        auto_ml.versioning.ArtifactLockError: 다른 학습·배포가 진행 중일 때.
    """
    with versioning.artifact_lock(config.artifact_dir, "rollback"):
        return _rollback(config, allow_missing_features)


def _rollback(config: AutoMLConfig, allow_missing_features: bool) -> Path:
    artifact_dir = Path(config.artifact_dir)
    current = versioning.current_deployment(artifact_dir)
    if current is None:
        raise RuntimeError(
            f"No deployment history in {versioning.deploy_log_path(artifact_dir)} — "
            "nothing to roll back to."
        )
    target = current.get("previous")
    if not target or not target.get("source"):
        raise RuntimeError(
            f"The current production model ({current.get('model')}, run "
            f"{current.get('run_id')}) has no previous deployment to roll back to."
        )
    source = artifact_dir / target["source"]
    if not source.exists():
        raise FileNotFoundError(
            f"Previous production model file is gone: {source}. Use "
            "`python -m auto_ml.cli.set_best --list` to see available runs."
        )
    logger.info(
        "Rolling back production model: %s (run %s) -> %s (run %s)",
        current.get("model"), current.get("run_id"), target.get("model"), target.get("run_id"),
    )
    return versioning.deploy(
        config, source, action="rollback", best_selection="rollback",
        allow_missing_features=allow_missing_features,
    )


def cli_rollback() -> None:
    """``python -m auto_ml.cli.rollback`` 진입점."""
    parser = argparse.ArgumentParser(
        description="Restore the previous production model (toggle current <-> previous)."
    )
    parser.add_argument("--config", required=True, help="설정 YAML 경로")
    parser.add_argument(
        "--allow-missing-features", action="store_true",
        help="되돌릴 모델이 쓰는 변수가 scoring.input_path 에 없어도 진행한다 "
             "(그 변수는 학습 기본값으로 채워져 예측된다)",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    setup_logging(
        log_dir=config.logging.log_dir,
        level=config.logging.level,
        to_stdout=config.logging.to_stdout,
        to_file=config.logging.to_file,
        stage="rollback",
    )
    rollback(config, allow_missing_features=args.allow_missing_features)


if __name__ == "__main__":
    cli_rollback()
