"""``auto-ml-explain`` CLI 진입점.

스코어링과 동일한 구조로, 학습 산출물(``best.joblib``) 을 로드해
입력 Parquet 의 모든 행에 대해 건별·변수별 SHAP wide 표를 계산하고
출력 Parquet 으로 저장한다.

사용 예::

    auto-ml-explain --config configs/example.yaml
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from auto_ml.config import AutoMLConfig, load_config
from auto_ml.explain.explainer import Explainer
from auto_ml.utils.io import ARTIFACT_FILENAME, summarize_dataframe, warn_degenerate_columns
from auto_ml.utils.logger import get_logger, setup_logging

logger = get_logger("explain.runner")


def run_explain(config: AutoMLConfig) -> Path:
    """설정에 따라 건별·변수별 SHAP 산출을 1회 수행한다.

    Args:
        config: AutoMLConfig.

    Returns:
        결과 Parquet 경로.
    """
    log_file = setup_logging(
        log_dir=config.logging.log_dir,
        level=config.logging.level,
        to_stdout=config.logging.to_stdout,
        to_file=config.logging.to_file,
        stage="explain",
    )
    started_at = time.perf_counter()
    logger.info("=" * 60)
    logger.info("Auto-ML explanation started")
    logger.info("=" * 60)
    if log_file is not None:
        logger.info("Log file: %s", log_file)

    artifact_path = Path(config.artifact_dir) / ARTIFACT_FILENAME
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"Artifact not found at {artifact_path}. Train first with auto-ml-train."
        )

    input_path = Path(config.explain.input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Explain input not found: {input_path}")

    logger.info("Loading artifact: %s", artifact_path)
    explainer = Explainer.from_artifact(artifact_path)

    logger.info("Reading input parquet: %s", input_path)
    df = pd.read_parquet(input_path)
    logger.info("Input: %s", summarize_dataframe(df))
    warn_degenerate_columns(
        df,
        columns=list(explainer.metadata.feature_columns),
        logger=logger,
        context="explain_input",
    )

    # id_columns 우선순위: explain.id_columns > top-level id_columns > metadata
    id_cols = config.explain.id_columns or config.id_columns or None
    out = explainer.explain(df, id_columns=id_cols)

    output_path = Path(config.explain.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)
    logger.info("Wrote SHAP output: %s (rows=%d, cols=%d)", output_path, len(out), len(out.columns))

    elapsed = time.perf_counter() - started_at
    logger.info("=" * 60)
    logger.info("Explanation completed in %.1fs", elapsed)
    logger.info("=" * 60)
    return output_path


def cli_explain() -> None:
    """``auto-ml-explain`` 진입점."""
    parser = argparse.ArgumentParser(description="Auto-ML SHAP explanation runner")
    parser.add_argument("--config", required=True, help="설정 YAML 경로")
    args = parser.parse_args()
    config = load_config(args.config)
    run_explain(config)


if __name__ == "__main__":
    cli_explain()
