"""``train/*.py`` · ``inference/*.py`` 스크립트 공통 인자 / 로깅 처리."""
from __future__ import annotations

import argparse
from pathlib import Path

from auto_ml.config import AutoMLConfig, load_config
from auto_ml.stages.common import default_work_dir
from auto_ml.utils.logger import get_logger, setup_logging

logger = get_logger("stages.cli")


def parse_stage_args(
    description: str,
    mode: str,
    argv: list[str] | None = None,
) -> tuple[AutoMLConfig, Path]:
    """``--config`` / ``--work-dir`` 를 파싱해 (config, work_dir) 를 반환한다.

    Args:
        description: argparse 설명.
        mode: ``"train"`` 또는 ``"inference"`` — work_dir 기본값 결정에 사용.
        argv: 테스트용 인자 목록. None 이면 ``sys.argv``.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, help="설정 YAML 경로")
    parser.add_argument(
        "--work-dir",
        default=None,
        help="단계 간 중간 산출물 디렉토리 (기본: <artifact_dir 의 부모>/work/<mode>)",
    )
    args = parser.parse_args(argv)
    config = load_config(args.config)
    work_dir = Path(args.work_dir) if args.work_dir else default_work_dir(config, mode)
    return config, work_dir


def start_stage_logging(config: AutoMLConfig, stage: str) -> None:
    """이번 실행 전용 로그 파일을 만들고 시작 배너를 남긴다."""
    log_file = setup_logging(
        log_dir=config.logging.log_dir,
        level=config.logging.level,
        to_stdout=config.logging.to_stdout,
        to_file=config.logging.to_file,
        stage=stage,
    )
    logger.info("=" * 60)
    logger.info("Auto-ML stage started: %s", stage)
    logger.info("=" * 60)
    if log_file is not None:
        logger.info("Log file: %s", log_file)
