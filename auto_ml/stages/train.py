"""학습 3단계: 전처리 → 학습 → 후처리.

    1) ``preprocess``  — Parquet 로드 + 검증, 전처리 fit/transform,
                         (선택) Stability Selection 변수 선택
    2) ``train``       — 모델 학습 (튜닝 → CV → 최종 fit) + best 선정
    3) ``postprocess`` — HTML/PDF 리포트, artifact 저장, test 예측 export

``AutoMLPipeline.run()`` 은 세 함수를 메모리에서 연달아 호출하고,
``train/*.py`` 스크립트는 단계 사이 산출물을 work_dir 에 저장해 따로 실행한다.
"""
from __future__ import annotations

import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from auto_ml import versioning
from auto_ml.config import AutoMLConfig
from auto_ml.feature_selection import SelectionResult, StabilitySelector
from auto_ml.models.trainer import Trainer, TrainingResult
from auto_ml.preprocessing import PreprocessingPipeline
from auto_ml.reporting.report import ReportBuilder
from auto_ml.utils.categorical import log_unseen_categories
from auto_ml.utils.io import (
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
    log_unseen_categories(preprocessor, X_test_p, logger, context="test")

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
            n_jobs=cfg.training.n_jobs,
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
def _remove_stale_sub_reports(cfg: AutoMLConfig, result: TrainingResult) -> None:
    """현재 실행에 없는 모델의 sub-report 폴더를 ``reporting.output_dir/sub`` 에서 지운다.

    이전 실행의 리포트는 그 실행 폴더(``runs/<id>/reports``) 에 사본이 있으므로 지워도
    잃지 않는다. 남겨 두면 현재 실행에 없는 모델의 리포트가 섞여 보인다.
    sub-report 는 best 를 제외한 모델마다 만들어진다 (ReportBuilder 와 같은 규칙).
    """
    sub_reports = Path(cfg.reporting.output_dir) / "sub"
    if not sub_reports.exists():
        return
    keep = set(result.results) - {result.best_model_name}
    stale = [d for d in sub_reports.iterdir() if d.is_dir() and d.name not in keep]
    for d in stale:
        shutil.rmtree(d)
    if stale:
        logger.info(
            "Removed sub-report folder(s) of models not in this run: %s "
            "(earlier copies remain under artifact_dir/runs/<run_id>/reports)",
            sorted(d.name for d in stale),
        )


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
    """실행 저장 → test 예측 → 리포트 → (자동) 배포 → 오래된 실행 정리.

    1. 후보 모델을 실행 스냅샷 ``artifact_dir/runs/<run_id>/models/`` 에 저장하고
       ``artifact_dir/models/`` 를 그 사본으로 교체한다. 모델을 가장 먼저 저장하므로
       이후 단계가 실패해도 학습 결과는 남는다.
    2. test 예측·리포트를 기존 위치에 쓰고 실행 폴더에도 사본을 남긴다.
    3. ``versioning.auto_deploy`` 이면 best 를 운영 모델(best.joblib) 로 배포한다 —
       실행이 완전히 끝난 뒤에만 운영 모델이 바뀐다.
    4. ``versioning.keep_last_runs`` 를 넘는 오래된 실행을 지운다 (운영·직전 운영 제외).

    Returns:
        산출물 이름 → 경로 dict (비활성 산출물은 키 누락).
    """
    outputs: dict[str, Path] = {}
    artifact_dir = Path(cfg.artifact_dir)

    # 실행 폴더 + 후보 모델 저장 ----------------------------------------------
    versioning.migrate_legacy(cfg)  # runs/ 가 없는 기존 폴더는 legacy 실행으로 먼저 담는다
    run_id, run_dir = versioning.new_run_dir(artifact_dir)
    run_models = run_dir / versioning.LATEST_DIRNAME
    outputs["run_dir"] = run_dir
    logger.info("Run %s: %s", run_id, run_dir)

    primary_metric_name = cfg.training.primary_metric
    selection_extra = _selection_extra(prep.selection)
    for name, mr in result.results.items():
        sub_extra: dict[str, Any] = {
            "run_id": run_id,
            "best_iteration": mr.model.best_iteration,
            "fold_best_iterations": mr.fold_best_iterations,
            "best_params": mr.params,
            "training_mode": cfg.training.final_fit_strategy,
            "categorical_encoding": (
                cfg.models[name].categorical_encoding if name in cfg.models else None
            ),
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
        sub_path = run_models / f"{name}.joblib"
        save_artifact(sub_path, prep.preprocessor, mr.model, sub_metadata)
        logger.info("Saved sub-artifact (%s): %s", name, sub_path)

    best_result = result.best
    versioning.write_run_info(run_dir, {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "best_model": best_result.name,
        "primary_metric": primary_metric_name,
        "models": {
            name: {
                "test_metrics": mr.test_metrics,
                "cv_metrics": mr.cv_metrics,
                "best_iteration": mr.model.best_iteration,
            }
            for name, mr in result.results.items()
        },
        "feature_columns": prep.feature_columns,
        "selected_features": prep.selected_features,
        "categorical_levels": prep.preprocessor.category_grouping_summary(),
        "categorical_columns": prep.categorical_columns,
        "data": {
            "train": _file_info(cfg.train_data_path, len(prep.y_train)),
            "test": _file_info(cfg.test_data_path, len(prep.y_test)),
        },
        "config": asdict(cfg),
        **versioning.environment_info(),
    })
    sub_dir = versioning.refresh_latest_models(artifact_dir, run_dir)
    outputs["sub_artifacts"] = sub_dir

    # test 예측 export --------------------------------------------------
    # 외부 분석용 (BI / 추가 진단) — best 모델의 holdout 예측을 키와 함께 저장.
    # 스키마: <id_columns> + score + prediction + <target_column>.
    # 운영 스코어링 출력과 동일한 컬럼 규약을 유지하되 actual 추가.
    threshold = cfg.scoring.threshold
    keep_id_cols = list(prep.test_ids.columns)
    pred_df = pd.DataFrame(index=prep.test_ids.index)
    for col in keep_id_cols:
        pred_df[col] = prep.test_ids[col].values
    pred_df["score"] = best_result.test_proba
    pred_df["prediction"] = (best_result.test_proba >= threshold).astype(int)
    pred_df[cfg.target_column] = prep.y_test.values
    predictions_path = artifact_dir.parent / "predictions" / "test_predictions.parquet"
    for path in (predictions_path, run_dir / "predictions" / "test_predictions.parquet"):
        path.parent.mkdir(parents=True, exist_ok=True)
        pred_df.to_parquet(path, index=False)
    outputs["test_predictions"] = predictions_path
    logger.info(
        "Wrote test predictions: %s (rows=%d, id_cols=%s)",
        predictions_path, len(pred_df), keep_id_cols or "(none)",
    )

    # 리포트 ------------------------------------------------------------
    # (PDF 변환 실패는 ReportBuilder 안에서 WARNING 처리되어 학습을 중단하지 않는다)
    logger.info("Step 4/4: report generation")
    _remove_stale_sub_reports(cfg, result)
    report_paths = ReportBuilder(cfg).build(result, selection=prep.selection)
    for src, dst in (
        ("html", "report_html"),
        ("pdf", "report_pdf"),
        ("feature_importance_csv", "feature_importance_csv"),
        ("feature_selection_csv", "feature_selection_csv"),
    ):
        if src in report_paths:
            outputs[dst] = report_paths[src]
    report_dir = Path(cfg.reporting.output_dir)
    if report_dir.exists():
        shutil.copytree(report_dir, run_dir / "reports",
                        ignore=shutil.ignore_patterns("_previous"))

    # 배포 ---------------------------------------------------------------
    best_selection = (
        "user_override" if cfg.training.best_model is not None else "auto_by_metric"
    )
    if cfg.versioning.auto_deploy:
        outputs["artifact"] = versioning.deploy(
            cfg, run_models / f"{best_result.name}.joblib",
            action="train", best_selection=best_selection,
        )
    else:
        logger.info(
            "versioning.auto_deploy is false — production model unchanged. Deploy with: "
            "python -m auto_ml.cli.set_best --config <yaml> --run %s --model %s",
            run_id, best_result.name,
        )

    versioning.prune_runs(artifact_dir, cfg.versioning.keep_last_runs)
    logger.info(
        "Best model: %s, %s=%.4f",
        result.best_model_name,
        result.primary_metric,
        best_result.test_metrics[result.primary_metric],
    )
    return outputs


def _file_info(path: str, n_rows: int) -> dict[str, Any]:
    """run_info 용 데이터 파일 정보 (경로, 사용 행 수, 수정 시각)."""
    p = Path(path)
    info: dict[str, Any] = {"path": str(path), "rows": n_rows}
    if p.exists():
        info["modified_at"] = datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).isoformat()
        info["size_bytes"] = p.stat().st_size
    return info
