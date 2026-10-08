"""학습 파이프라인 오케스트레이션.

전체 흐름:
    1) 전처리 — Parquet 로드 + 검증, 전처리 fit/transform, (선택) 변수 선택
    2) 학습   — 모델 학습 (튜닝 → CV → 최종 fit) + best 선정
    3) 후처리 — HTML / PDF 리포트, artifact 저장, test 예측 export

단계별 구현은 ``auto_ml.stages.train`` 에 있다. 단계를 따로 실행하려면
``train/preprocess.py`` → ``train/train.py`` → ``train/postprocess.py`` 를 사용한다.

사용:
    >>> from auto_ml import AutoMLPipeline, load_config
    >>> AutoMLPipeline(load_config("configs/example.yaml")).run()
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from auto_ml.config import AutoMLConfig, load_config
from auto_ml.stages import train as stages_train
from auto_ml.utils.logger import get_logger, setup_logging

logger = get_logger("pipeline")


class AutoMLPipeline:
    """학습 전 과정을 한 번의 ``run()`` 호출로 수행하는 오케스트레이터."""

    def __init__(self, config: AutoMLConfig) -> None:
        self.config = config

    def run(self) -> dict[str, Path]:
        """전체 파이프라인을 실행한다.

        Returns:
            ``{"artifact": Path, "report_html": Path, "report_pdf": Path,
              "log_file": Path}`` (해당 산출물이 비활성이면 키 누락)
        """
        cfg = self.config

        # 0) 로깅 초기화 — 이번 실행 전용 로그 파일을 새로 만든다
        log_file = setup_logging(
            log_dir=cfg.logging.log_dir,
            level=cfg.logging.level,
            to_stdout=cfg.logging.to_stdout,
            to_file=cfg.logging.to_file,
            stage="train",
        )
        outputs: dict[str, Path] = {}
        if log_file is not None:
            outputs["log_file"] = log_file
            logger.info("Log file: %s", log_file)

        started_at = time.perf_counter()
        logger.info("=" * 60)
        logger.info("Auto-ML training pipeline started")
        logger.info("=" * 60)

        prep = stages_train.preprocess(cfg)
        result = stages_train.train(cfg, prep)
        outputs.update(stages_train.postprocess(cfg, prep, result))
        best_result = result.best

        # ----- 종료 요약 ------------------------------------------------
        elapsed = time.perf_counter() - started_at
        logger.info("=" * 60)
        logger.info(
            "Pipeline completed in %.1fs — best=%s, %s=%.4f",
            elapsed,
            result.best_model_name,
            result.primary_metric,
            best_result.test_metrics[result.primary_metric],
        )
        for key, path in outputs.items():
            logger.info("  %-12s : %s", key, path)
        logger.info("=" * 60)

        return outputs


# ----------------------------------------------------------------------
def cli_train() -> None:
    """``python -m auto_ml.pipeline`` 진입점."""
    parser = argparse.ArgumentParser(description="Auto-ML training pipeline")
    parser.add_argument("--config", required=True, help="설정 YAML 경로")
    args = parser.parse_args()
    config = load_config(args.config)
    AutoMLPipeline(config).run()


if __name__ == "__main__":
    cli_train()
