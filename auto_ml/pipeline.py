"""학습 파이프라인 오케스트레이션.

전체 흐름:
    1) Parquet 로드 (train / test 별도 파일) + 스키마/타깃 검증
    2) 전처리 fit_transform (학습) / transform (테스트)
    3) (선택) Stability Selection 변수 선택 — feature_selection.enabled=True 일 때
    4) 3개 모델 학습 (튜닝 → CV → 최종 fit) + best 선정
    5) HTML / PDF 리포트 생성
    6) artifact 저장 (preprocessor + best_model + metadata)

사용:
    >>> from auto_ml import AutoMLPipeline, load_config
    >>> AutoMLPipeline(load_config("configs/example.yaml")).run()
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from auto_ml.config import AutoMLConfig, load_config
from auto_ml.feature_selection import SelectionResult, StabilitySelector
from auto_ml.models.trainer import Trainer, TrainingResult
from auto_ml.preprocessing import PreprocessingPipeline
from auto_ml.reporting.report import ReportBuilder
from auto_ml.utils.io import (
    ArtifactMetadata,
    save_artifact,
    summarize_dataframe,
    warn_degenerate_columns,
)
from auto_ml.utils.logger import get_logger, setup_logging
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

        # 1) 데이터 로드 + 검증 ---------------------------------------------
        logger.info("Loading train data: %s", cfg.train_data_path)
        df_train = pd.read_parquet(cfg.train_data_path)
        logger.info("Train: %s", summarize_dataframe(df_train, cfg.target_column))
        logger.info("Loading test data:  %s", cfg.test_data_path)
        df_test = pd.read_parquet(cfg.test_data_path)
        logger.info("Test:  %s", summarize_dataframe(df_test, cfg.target_column))

        validate_schema(df_train, [cfg.target_column])
        validate_schema(df_test, [cfg.target_column])
        validate_binary_target(df_train[cfg.target_column])
        validate_binary_target(df_test[cfg.target_column])

        if not cfg.features:
            raise ValueError(
                "config.features is empty. Specify the feature list with name/type "
                "(numeric or categorical)."
            )
        feature_columns = cfg.feature_columns
        numeric_columns = cfg.numeric_columns
        categorical_columns = cfg.categorical_columns
        logger.info(
            "Features — total=%d, numeric=%d, categorical=%d",
            len(feature_columns), len(numeric_columns), len(categorical_columns),
        )

        # 모델에 의미 없는 컬럼 (all-NaN / all-zero / constant) 조기 경고.
        # 학습 데이터 기준 — test 만 상수인 경우는 정상적일 수 있어 train 만 점검.
        warn_degenerate_columns(
            df_train, columns=feature_columns, logger=logger, context="train",
        )

        # 학습/테스트 데이터 양쪽에 모든 feature 가 있어야 한다.
        validate_schema(df_train, feature_columns)
        validate_schema(df_test, feature_columns)

        # 2) 전처리 ---------------------------------------------------------
        X_train = df_train[feature_columns]
        y_train = df_train[cfg.target_column].astype(int)
        X_test = df_test[feature_columns]
        y_test = df_test[cfg.target_column].astype(int)
        logger.info("Sizes — train=%d, test=%d", len(X_train), len(X_test))

        logger.info("Step 1/4: preprocessing (null -> outlier -> scaling)")
        preprocessor = PreprocessingPipeline(
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
            config=cfg.preprocessing,
        )
        X_train_p = preprocessor.fit_transform(X_train)
        X_test_p = preprocessor.transform(X_test)

        # 3) (선택) 변수 선택 -----------------------------------------------
        # 변수 선택을 끄면 selected_features = feature_columns 로 두어 이후 단계에서
        # 통일된 인터페이스(metadata.selected_features) 로 다룬다.
        selected_features = list(feature_columns)
        selection: SelectionResult | None = None
        if cfg.feature_selection.enabled:
            logger.info(
                "Step 2/4: stability selection (base=%s, n_subsamples=%d, threshold=%.2f)",
                cfg.feature_selection.base_estimator,
                cfg.feature_selection.n_subsamples,
                cfg.feature_selection.threshold,
            )
            selector = StabilitySelector(
                config=cfg.feature_selection,
                numeric_columns=numeric_columns,
                categorical_columns=categorical_columns,
            )
            selection = selector.fit_select(X_train_p, y_train)
            selected_features = selection.selected_features
            logger.info(
                "Stability selection kept %d / %d features%s",
                len(selected_features), len(feature_columns),
                " (fallback used)" if selection.fallback_used else "",
            )
        else:
            logger.info("Step 2/4: feature selection disabled — using all features")

        X_train_used = X_train_p[selected_features]
        X_test_used = X_test_p[selected_features]

        # 4) 학습 + best 선정 ----------------------------------------------
        logger.info("Step 3/4: model training (3 algorithms)")
        trainer = Trainer(cfg)
        result: TrainingResult = trainer.train(X_train_used, y_train, X_test_used, y_test)

        # 5) 리포트 ---------------------------------------------------------
        logger.info("Step 4/4: report generation")
        report_paths = ReportBuilder(cfg).build(result, selection=selection)
        if "html" in report_paths:
            outputs["report_html"] = report_paths["html"]
        if "pdf" in report_paths:
            outputs["report_pdf"] = report_paths["pdf"]
        if "feature_importance_csv" in report_paths:
            outputs["feature_importance_csv"] = report_paths["feature_importance_csv"]
        if "feature_selection_csv" in report_paths:
            outputs["feature_selection_csv"] = report_paths["feature_selection_csv"]

        # 6) artifact 저장 --------------------------------------------------
        best_result = result.best
        extra = {
            "best_iteration": best_result.model.best_iteration,
            "fold_best_iterations": best_result.fold_best_iterations,
            "best_params": best_result.params,
            "tuning": (
                {
                    "best_value": best_result.tuning.best_value,
                    "n_trials": best_result.tuning.n_trials,
                }
                if best_result.tuning is not None
                else None
            ),
        }
        if selection is not None:
            extra["feature_selection"] = {
                "base_estimator": selection.base_estimator,
                "threshold": selection.threshold,
                "n_subsamples": selection.n_subsamples,
                "fallback_used": selection.fallback_used,
                "frequencies": selection.frequencies,
            }
        metadata = ArtifactMetadata(
            target_column=cfg.target_column,
            feature_columns=feature_columns,
            selected_features=selected_features,
            categorical_columns=categorical_columns,
            id_columns=cfg.id_columns,
            model_name=best_result.name,
            primary_metric=cfg.training.primary_metric,
            metric_value=best_result.test_metrics[cfg.training.primary_metric],
            extra=extra,
        )
        artifact_path = Path(cfg.artifact_dir) / ARTIFACT_FILENAME
        save_artifact(artifact_path, preprocessor, best_result.model, metadata)
        outputs["artifact"] = artifact_path
        logger.info("Saved artifact: %s", artifact_path)

        # 7) test 예측 export ----------------------------------------------
        # 외부 분석용 (BI / 추가 진단) — best 모델의 holdout 예측을 키와 함께 저장.
        # 스키마: <id_columns> + score + prediction + <target_column>.
        # 운영 스코어링(auto-ml-score) 출력과 동일한 컬럼 규약을 유지하되 actual 추가.
        predictions_dir = Path(cfg.artifact_dir).parent / "predictions"
        predictions_dir.mkdir(parents=True, exist_ok=True)
        predictions_path = predictions_dir / "test_predictions.parquet"
        threshold = cfg.scoring.threshold
        keep_id_cols = [c for c in cfg.id_columns if c in df_test.columns]
        pred_df = pd.DataFrame(index=df_test.index)
        for col in keep_id_cols:
            pred_df[col] = df_test[col].values
        pred_df["score"] = best_result.test_proba
        pred_df["prediction"] = (best_result.test_proba >= threshold).astype(int)
        pred_df[cfg.target_column] = y_test.values
        pred_df.to_parquet(predictions_path, index=False)
        outputs["test_predictions"] = predictions_path
        logger.info(
            "Wrote test predictions: %s (rows=%d, id_cols=%s)",
            predictions_path, len(pred_df), keep_id_cols or "(none)",
        )

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
    """``auto-ml-train`` 진입점."""
    parser = argparse.ArgumentParser(description="Auto-ML training pipeline")
    parser.add_argument("--config", required=True, help="설정 YAML 경로")
    args = parser.parse_args()
    config = load_config(args.config)
    AutoMLPipeline(config).run()


if __name__ == "__main__":
    cli_train()
