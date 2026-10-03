"""추론(스코어링) 3단계: 전처리 → 예측 → 후처리.

    1) ``preprocess``  — artifact 로드, 입력 Parquet 로드 + 검증,
                         학습 시 fit 된 전처리 transform + 변수 선택
    2) ``predict``     — 모델 ``predict_proba``
    3) ``postprocess`` — 임계값 적용, id 컬럼 결합, 결과 Parquet 저장

``run_scoring()`` 은 세 함수를 메모리에서 연달아 호출하고,
``inference/*.py`` 스크립트는 단계 사이 산출물을 work_dir 에 저장해 따로 실행한다.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from auto_ml.config import AutoMLConfig
from auto_ml.scoring.scorer import Scorer, build_score_frame
from auto_ml.utils.io import ARTIFACT_FILENAME, summarize_dataframe, warn_degenerate_columns
from auto_ml.utils.logger import get_logger

logger = get_logger("stages.inference")

# work_dir 내 단계 산출물 이름
PREPROCESS_OUTPUT = "inference_preprocessed"
PREDICT_OUTPUT = "inference_predicted"


@dataclass
class InferencePreprocessed:
    """전처리 단계 산출물 — 예측 단계 입력."""

    artifact_path: Path
    X_for_model: pd.DataFrame      # 전처리 + 변수 선택 후 모델 입력
    ids: pd.DataFrame              # 결과에 보존할 id 컬럼 (없으면 컬럼 0개)


@dataclass
class InferencePredicted:
    """예측 단계 산출물 — 후처리 단계 입력."""

    artifact_path: Path
    model_name: str
    ids: pd.DataFrame
    proba: np.ndarray


def artifact_path_of(cfg: AutoMLConfig) -> Path:
    """설정의 ``artifact_dir`` 에서 best artifact 경로를 결정한다."""
    path = Path(cfg.artifact_dir) / ARTIFACT_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"Artifact not found at {path}. Run training first (train/run_all.py)."
        )
    return path


# ----------------------------------------------------------------------
# 1) 전처리
# ----------------------------------------------------------------------
def preprocess(
    cfg: AutoMLConfig,
    scorer: Scorer | None = None,
) -> InferencePreprocessed:
    """artifact 의 전처리기로 입력 Parquet 을 모델 입력 행렬로 변환한다.

    Args:
        cfg: AutoMLConfig.
        scorer: 이미 로드한 Scorer. None 이면 ``artifact_dir/best.joblib`` 을 로드.
    """
    artifact_path = artifact_path_of(cfg)
    input_path = Path(cfg.scoring.input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Scoring input not found: {input_path}")

    if scorer is None:
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

    # id_columns 우선순위: scoring.id_columns (override) > top-level config.id_columns
    # 둘 다 비어 있으면 scorer 가 artifact metadata 의 학습 시점 id_columns 로 fallback.
    id_cols = cfg.scoring.id_columns or cfg.id_columns or None
    X_for_model, ids = scorer.prepare(df, id_columns=id_cols)
    logger.info(
        "Preprocessed input: rows=%d, model_features=%d",
        len(X_for_model), X_for_model.shape[1],
    )
    return InferencePreprocessed(
        artifact_path=artifact_path, X_for_model=X_for_model, ids=ids,
    )


# ----------------------------------------------------------------------
# 2) 예측
# ----------------------------------------------------------------------
def predict(
    prep: InferencePreprocessed,
    scorer: Scorer | None = None,
) -> InferencePredicted:
    """전처리된 입력의 양성 확률을 계산한다."""
    if scorer is None:
        logger.info("Loading artifact: %s", prep.artifact_path)
        scorer = Scorer.from_artifact(prep.artifact_path)
    proba = scorer.predict(prep.X_for_model)
    logger.info("Predicted %d rows (model=%s)", len(proba), scorer.metadata.model_name)
    return InferencePredicted(
        artifact_path=prep.artifact_path,
        model_name=scorer.metadata.model_name,
        ids=prep.ids,
        proba=proba,
    )


# ----------------------------------------------------------------------
# 3) 후처리
# ----------------------------------------------------------------------
def postprocess(cfg: AutoMLConfig, pred: InferencePredicted) -> Path:
    """임계값 적용 → ``id + score + prediction`` 결과 Parquet 저장.

    Returns:
        결과 Parquet 경로.
    """
    threshold = cfg.scoring.threshold
    out = build_score_frame(pred.ids, pred.proba, threshold)
    logger.info(
        "Scored %d rows (model=%s, threshold=%.3f)",
        len(out), pred.model_name, threshold,
    )

    output_path = Path(cfg.scoring.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)
    logger.info("Wrote scored output: %s (rows=%d)", output_path, len(out))
    return output_path
