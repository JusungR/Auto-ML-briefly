"""주기 실행용 CLI 진입점.

cron / Airflow / 사내 스케줄러에서 다음과 같이 호출한다::

    python inference/run_all.py --config configs/example.yaml

학습 산출물(artifact) 경로는 설정 YAML 의 ``artifact_dir`` 와
관례적인 파일명(``best.joblib``) 으로 결정한다. 입출력은 모두 Parquet.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from auto_ml.config import AutoMLConfig, load_config
from auto_ml.scoring.scorer import Scorer
from auto_ml.stages import inference as stages_inference
from auto_ml.utils.logger import get_logger, setup_logging

logger = get_logger("scoring.runner")


def run_scoring(config: AutoMLConfig) -> Path:
    """설정에 따라 배치 스코어링을 1회 수행한다.

    Args:
        config: AutoMLConfig.

    Returns:
        결과 Parquet 경로.
    """
    # 이번 실행 전용 로그 파일을 새로 만든다 (stage="score")
    log_file = setup_logging(
        log_dir=config.logging.log_dir,
        level=config.logging.level,
        to_stdout=config.logging.to_stdout,
        to_file=config.logging.to_file,
        stage="score",
    )
    started_at = time.perf_counter()
    logger.info("=" * 60)
    logger.info("Auto-ML scoring started")
    logger.info("=" * 60)
    if log_file is not None:
        logger.info("Log file: %s", log_file)

    # 전처리 → 예측 → 후처리 (단계별 구현은 auto_ml.stages.inference)
    artifact_path = stages_inference.artifact_path_of(config)
    logger.info("Loading artifact: %s", artifact_path)
    scorer = Scorer.from_artifact(artifact_path)
    prep = stages_inference.preprocess(config, scorer=scorer)
    pred = stages_inference.predict(prep, scorer=scorer)
    output_path = stages_inference.postprocess(config, pred)

    elapsed = time.perf_counter() - started_at
    logger.info("=" * 60)
    logger.info("Scoring completed in %.1fs", elapsed)
    logger.info("=" * 60)
    return output_path


def cli_score() -> None:
    """``python -m auto_ml.scoring.runner`` 진입점."""
    parser = argparse.ArgumentParser(description="Auto-ML batch scoring runner")
    parser.add_argument("--config", required=True, help="설정 YAML 경로")
    args = parser.parse_args()
    config = load_config(args.config)
    run_scoring(config)


if __name__ == "__main__":
    cli_score()
