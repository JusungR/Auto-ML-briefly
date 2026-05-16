"""주기 실행용 CLI 진입점.

cron / Airflow / 사내 스케줄러에서 다음과 같이 호출한다::

    auto-ml-score --config configs/score.yaml

학습 산출물(artifact) 경로는 설정 YAML 의 ``artifact_dir`` 와
관례적인 파일명(``best.joblib``) 으로 결정한다. 입출력은 모두 Parquet.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from auto_ml.config import AutoMLConfig, load_config
from auto_ml.scoring.scorer import Scorer
from auto_ml.utils.io import summarize_dataframe, warn_degenerate_columns
from auto_ml.utils.logger import get_logger, setup_logging

logger = get_logger("scoring.runner")

ARTIFACT_FILENAME = "best.joblib"


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

    artifact_path = Path(config.artifact_dir) / ARTIFACT_FILENAME
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"Artifact not found at {artifact_path}. Train first with auto-ml-train."
        )

    input_path = Path(config.scoring.input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Scoring input not found: {input_path}")

    logger.info("Loading artifact: %s", artifact_path)
    scorer = Scorer.from_artifact(artifact_path)

    logger.info("Reading input parquet: %s", input_path)
    df = pd.read_parquet(input_path)
    # 스코어링 입력은 target 이 없으므로 클래스 비율 절은 생략된다.
    logger.info("Input: %s", summarize_dataframe(df))
    # 모델 feature 컬럼 한정 — 스코어링 입력이 학습 시점과 분포가 깨졌는지 조기 감지.
    warn_degenerate_columns(
        df,
        columns=list(scorer.metadata.feature_columns),
        logger=logger,
        context="score_input",
    )

    out = scorer.score(
        df,
        threshold=config.scoring.threshold,
        id_columns=config.scoring.id_columns,
    )

    output_path = Path(config.scoring.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)
    logger.info("Wrote scored output: %s (rows=%d)", output_path, len(out))

    elapsed = time.perf_counter() - started_at
    logger.info("=" * 60)
    logger.info("Scoring completed in %.1fs", elapsed)
    logger.info("=" * 60)
    return output_path


def cli_score() -> None:
    """``auto-ml-score`` 진입점."""
    parser = argparse.ArgumentParser(description="Auto-ML batch scoring runner")
    parser.add_argument("--config", required=True, help="설정 YAML 경로")
    args = parser.parse_args()
    config = load_config(args.config)
    run_scoring(config)


if __name__ == "__main__":
    cli_score()
