"""학습 파이프라인 오케스트레이션.

전체 흐름:
    1) Parquet 로드 + 스키마/타깃 검증
    2) holdout split (StratifiedShuffleSplit)
    3) 전처리 fit_transform (학습) / transform (holdout)
    4) 3개 모델 학습 + best 선정 (Trainer)
    5) HTML / PDF 리포트 생성 (ReportBuilder)
    6) artifact 저장 (preprocessor + best_model + metadata)

사용:
    >>> from auto_ml import AutoMLPipeline, load_config
    >>> AutoMLPipeline(load_config("configs/example.yaml")).run()
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from auto_ml.config import AutoMLConfig, load_config
from auto_ml.models.trainer import Trainer, TrainingResult
from auto_ml.preprocessing import PreprocessingPipeline
from auto_ml.reporting import ReportBuilder
from auto_ml.utils.io import ArtifactMetadata, save_artifact
from auto_ml.utils.logger import get_logger
from auto_ml.utils.validation import validate_binary_target, validate_schema

logger = get_logger("pipeline")

ARTIFACT_FILENAME = "best.joblib"


class AutoMLPipeline:
    """학습 전 과정을 한 번의 ``run()`` 호출로 수행하는 오케스트레이터."""

    def __init__(self, config: AutoMLConfig) -> None:
        self.config = config

    def run(self) -> dict[str, Path]:
        """전체 파이프라인을 실행한다.

        Returns:
            ``{"artifact": Path, "report_html": Path, "report_pdf": Path}`` (포맷 비활성 시 키 누락)
        """
        cfg = self.config

        # 1) 데이터 로드 + 검증 -----------------------------------------
        logger.info("Loading training data: %s", cfg.train_data_path)
        df = pd.read_parquet(cfg.train_data_path)
        validate_schema(df, [cfg.target_column])
        validate_binary_target(df[cfg.target_column])

        feature_columns, numeric_columns, categorical_columns = self._resolve_columns(df)
        logger.info(
            "Resolved columns — features=%d, numeric=%d, categorical=%d",
            len(feature_columns), len(numeric_columns), len(categorical_columns),
        )

        # 2) holdout 분할 ---------------------------------------------
        X = df[feature_columns]
        y = df[cfg.target_column].astype(int)
        X_train, X_holdout, y_train, y_holdout = train_test_split(
            X, y,
            test_size=cfg.training.test_size,
            random_state=cfg.training.random_state,
            stratify=y,
        )
        logger.info("Split — train=%d, holdout=%d", len(X_train), len(X_holdout))

        # 3) 전처리 ----------------------------------------------------
        preprocessor = PreprocessingPipeline(
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
            config=cfg.preprocessing,
        )
        X_train_p = preprocessor.fit_transform(X_train)
        X_holdout_p = preprocessor.transform(X_holdout)

        # 4) 학습 + best 선정 -----------------------------------------
        trainer = Trainer(cfg)
        result: TrainingResult = trainer.train(X_train_p, y_train, X_holdout_p, y_holdout)

        # 5) 리포트 ----------------------------------------------------
        outputs: dict[str, Path] = {}
        report_paths = ReportBuilder(cfg).build(result)
        if "html" in report_paths:
            outputs["report_html"] = report_paths["html"]
        if "pdf" in report_paths:
            outputs["report_pdf"] = report_paths["pdf"]

        # 6) artifact 저장 ---------------------------------------------
        best_result = result.best
        metadata = ArtifactMetadata(
            target_column=cfg.target_column,
            feature_columns=feature_columns,
            categorical_columns=categorical_columns,
            id_columns=cfg.id_columns,
            model_name=best_result.name,
            primary_metric=cfg.training.primary_metric,
            metric_value=best_result.holdout_metrics[cfg.training.primary_metric],
            extra={
                "best_iteration": best_result.model.best_iteration,
                "fold_best_iterations": best_result.fold_best_iterations,
            },
        )
        artifact_path = Path(cfg.artifact_dir) / ARTIFACT_FILENAME
        save_artifact(artifact_path, preprocessor, best_result.model, metadata)
        outputs["artifact"] = artifact_path
        logger.info("Saved artifact: %s", artifact_path)

        return outputs

    # ------------------------------------------------------------------
    def _resolve_columns(self, df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
        """설정과 실제 데이터를 비교해 (feature, numeric, categorical) 목록을 결정한다."""
        cfg = self.config
        # feature_columns 가 None 이면 target/id 를 제외한 모든 컬럼을 feature 로 사용
        if cfg.feature_columns is None:
            feature_columns = [
                c for c in df.columns
                if c != cfg.target_column and c not in cfg.id_columns
            ]
        else:
            feature_columns = list(cfg.feature_columns)

        # 사용자가 지정한 categorical 중 실제 데이터에 있는 것만
        categorical_columns = [c for c in cfg.categorical_columns if c in feature_columns]
        numeric_columns = [c for c in feature_columns if c not in categorical_columns]
        return feature_columns, numeric_columns, categorical_columns


# ----------------------------------------------------------------------
def cli_train() -> None:
    """``auto-ml-train`` 진입점."""
    parser = argparse.ArgumentParser(description="Auto-ML training pipeline")
    parser.add_argument("--config", required=True, help="설정 YAML 경로")
    args = parser.parse_args()
    config = load_config(args.config)
    AutoMLPipeline(config).run()


if __name__ == "__main__":
    cli_train()
