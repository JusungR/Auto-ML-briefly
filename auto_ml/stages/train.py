"""학습 3단계: 전처리 → 학습 → 후처리.

    1) ``preprocess``  — Parquet 로드 + 검증, 전처리 fit/transform,
                         (선택) Stability Selection 변수 선택
    2) ``train``       — 모델 학습 (튜닝 → CV → 최종 fit) + best 선정
    3) ``postprocess`` — HTML/PDF 리포트, artifact 저장, test 예측 export

``AutoMLPipeline.run()`` 은 세 함수를 메모리에서 연달아 호출하고,
``train/*.py`` 스크립트는 단계 사이 산출물을 work_dir 에 저장해 따로 실행한다.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from auto_ml.config import AutoMLConfig
from auto_ml.feature_selection import SelectionResult, StabilitySelector
from auto_ml.models.trainer import Trainer, TrainingResult
from auto_ml.preprocessing import PreprocessingPipeline
from auto_ml.reporting.report import ReportBuilder
from auto_ml.utils.io import (
    ARTIFACT_FILENAME,
    ArtifactMetadata,
    save_artifact,
    summarize_dataframe,
    warn_degenerate_columns,
)
from auto_ml.utils.logger import get_logger
from auto_ml.utils.validation import validate_binary_target, validate_schema

logger = get_logger("stages.train")

# work_dir 내 단계 산출물 이름
PREPROCESS_OUTPUT = "train_preprocessed"
TRAIN_OUTPUT = "train_result"


@dataclass
class TrainPreprocessed:
    """전처리 단계 산출물 — 학습 단계 입력."""

    preprocessor: PreprocessingPipeline
    selection: SelectionResult | None
    feature_columns: list[str]
    selected_features: list[str]
    categorical_columns: list[str]
    X_train: pd.DataFrame          # 전처리 + 변수 선택 후 모델 입력
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    test_ids: pd.DataFrame         # test 예측 export 용 id 컬럼 (없으면 컬럼 0개)


# ----------------------------------------------------------------------
# 1) 전처리
# ----------------------------------------------------------------------
def preprocess(cfg: AutoMLConfig) -> TrainPreprocessed:
    """데이터 로드·검증 → 전처리 → (선택) 변수 선택."""
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

    keep_id_cols = [c for c in cfg.id_columns if c in df_test.columns]
    return TrainPreprocessed(
        preprocessor=preprocessor,
        selection=selection,
        feature_columns=feature_columns,
        selected_features=selected_features,
        categorical_columns=categorical_columns,
        X_train=X_train_p[selected_features],
        y_train=y_train,
        X_test=X_test_p[selected_features],
        y_test=y_test,
        test_ids=df_test[keep_id_cols].copy(),
    )


# ----------------------------------------------------------------------
# 2) 학습
# ----------------------------------------------------------------------
def train(cfg: AutoMLConfig, prep: TrainPreprocessed) -> TrainingResult:
    """활성화된 모델 학습 (튜닝 → CV → 최종 fit) + best 선정."""
    logger.info("Step 3/4: model training")
    return Trainer(cfg).train(prep.X_train, prep.y_train, prep.X_test, prep.y_test)


# ----------------------------------------------------------------------
# 3) 후처리
# ----------------------------------------------------------------------
def _selection_extra(selection: SelectionResult | None) -> dict[str, Any]:
    if selection is None:
        return {}
    return {
        "feature_selection": {
            "base_estimator": selection.base_estimator,
            "threshold": selection.threshold,
            "n_subsamples": selection.n_subsamples,
            "fallback_used": selection.fallback_used,
            "frequencies": selection.frequencies,
        }
    }


def postprocess(
    cfg: AutoMLConfig,
    prep: TrainPreprocessed,
    result: TrainingResult,
) -> dict[str, Path]:
    """리포트 생성 → artifact 저장 → test 예측 export.

    Returns:
        산출물 이름 → 경로 dict (비활성 산출물은 키 누락).
    """
    outputs: dict[str, Path] = {}

    # 리포트 ------------------------------------------------------------
    logger.info("Step 4/4: report generation")
    report_paths = ReportBuilder(cfg).build(result, selection=prep.selection)
    for src, dst in (
        ("html", "report_html"),
        ("pdf", "report_pdf"),
        ("feature_importance_csv", "feature_importance_csv"),
        ("feature_selection_csv", "feature_selection_csv"),
    ):
        if src in report_paths:
            outputs[dst] = report_paths[src]

    # artifact 저장 -----------------------------------------------------
    # 모든 모형을 sub-artifact (artifact_dir/models/<name>.joblib) 로 독립 저장한 뒤,
    # best 의 동일 내용을 best.joblib 으로 한 번 더 저장한다. 이렇게 두면
    # set_best 가 재학습 없이 best 를 교체할 수 있고, 각 sub-artifact
    # 만으로도 Scorer/Explainer 가 독립적으로 동작한다.
    sub_dir = Path(cfg.artifact_dir) / "models"
    primary_metric_name = cfg.training.primary_metric
    selection_extra = _selection_extra(prep.selection)
    for name, mr in result.results.items():
        sub_extra: dict[str, Any] = {
            "best_iteration": mr.model.best_iteration,
            "fold_best_iterations": mr.fold_best_iterations,
            "best_params": mr.params,
            "training_mode": cfg.training.final_fit_strategy,
            "tuning": (
                {"best_value": mr.tuning.best_value, "n_trials": mr.tuning.n_trials}
                if mr.tuning is not None
                else None
            ),
        }
        if cfg.training.final_fit_strategy == "iteration_capping":
            sub_extra["iteration_cap"] = {
                "aggregation": cfg.training.iteration_cap_aggregation,
                "headroom": cfg.training.iteration_cap_headroom,
            }
        elif cfg.training.final_fit_strategy == "cv_bagging":
            sub_extra["cv_bagging"] = {
                "n_folds": len(mr.fold_best_iterations) or cfg.training.cv_folds,
                "weights": "uniform",
            }
        sub_extra.update(selection_extra)
        sub_metadata = ArtifactMetadata(
            target_column=cfg.target_column,
            feature_columns=prep.feature_columns,
            selected_features=prep.selected_features,
            categorical_columns=prep.categorical_columns,
            id_columns=cfg.id_columns,
            model_name=mr.name,
            primary_metric=primary_metric_name,
            metric_value=mr.test_metrics[primary_metric_name],
            extra=sub_extra,
        )
        sub_path = sub_dir / f"{name}.joblib"
        save_artifact(sub_path, prep.preprocessor, mr.model, sub_metadata)
        logger.info("Saved sub-artifact (%s): %s", name, sub_path)

    # best 는 sub 중 하나의 복사본.
    best_result = result.best
    best_metadata = ArtifactMetadata(
        target_column=cfg.target_column,
        feature_columns=prep.feature_columns,
        selected_features=prep.selected_features,
        categorical_columns=prep.categorical_columns,
        id_columns=cfg.id_columns,
        model_name=best_result.name,
        primary_metric=primary_metric_name,
        metric_value=best_result.test_metrics[primary_metric_name],
        extra={
            "best_iteration": best_result.model.best_iteration,
            "fold_best_iterations": best_result.fold_best_iterations,
            "best_params": best_result.params,
            "training_mode": cfg.training.final_fit_strategy,
            "tuning": (
                {
                    "best_value": best_result.tuning.best_value,
                    "n_trials": best_result.tuning.n_trials,
                }
                if best_result.tuning is not None
                else None
            ),
            "best_selection": (
                "user_override" if cfg.training.best_model is not None
                else "auto_by_metric"
            ),
            **selection_extra,
        },
    )
    artifact_path = Path(cfg.artifact_dir) / ARTIFACT_FILENAME
    save_artifact(artifact_path, prep.preprocessor, best_result.model, best_metadata)
    outputs["artifact"] = artifact_path
    outputs["sub_artifacts"] = sub_dir
    logger.info("Saved artifact: %s", artifact_path)

    # test 예측 export --------------------------------------------------
    # 외부 분석용 (BI / 추가 진단) — best 모델의 holdout 예측을 키와 함께 저장.
    # 스키마: <id_columns> + score + prediction + <target_column>.
    # 운영 스코어링 출력과 동일한 컬럼 규약을 유지하되 actual 추가.
    predictions_dir = Path(cfg.artifact_dir).parent / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = predictions_dir / "test_predictions.parquet"
    threshold = cfg.scoring.threshold
    keep_id_cols = list(prep.test_ids.columns)
    pred_df = pd.DataFrame(index=prep.test_ids.index)
    for col in keep_id_cols:
        pred_df[col] = prep.test_ids[col].values
    pred_df["score"] = best_result.test_proba
    pred_df["prediction"] = (best_result.test_proba >= threshold).astype(int)
    pred_df[cfg.target_column] = prep.y_test.values
    pred_df.to_parquet(predictions_path, index=False)
    outputs["test_predictions"] = predictions_path
    logger.info(
        "Wrote test predictions: %s (rows=%d, id_cols=%s)",
        predictions_path, len(pred_df), keep_id_cols or "(none)",
    )

    logger.info(
        "Best model: %s, %s=%.4f",
        result.best_model_name,
        result.primary_metric,
        best_result.test_metrics[result.primary_metric],
    )
    return outputs
