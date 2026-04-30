"""학습 산출물(artifact) 저장 / 로드 유틸리티.

artifact 번들 = 단일 joblib 파일 1개에 다음을 함께 저장한다:
    - preprocessor : 학습 시 fit 된 PreprocessingPipeline
    - model        : 학습 시 fit 된 BaseModel 구현체 (LGBM/XGB/CatBoost 중 1개)
    - metadata     : 스키마, 학습 시점, feature/target 명, 평가 지표 등 운영 정보

이렇게 한 번들로 묶으면 폐쇄 환경 이관이 단순해진다 — joblib 파일과
스코어링용 config YAML 두 개만 옮기면 즉시 운영이 가능하다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib


@dataclass
class ArtifactMetadata:
    """artifact 와 함께 저장되는 운영 메타데이터.

    Attributes:
        target_column: 학습에 사용한 타깃 컬럼 이름.
        feature_columns: 학습 시 전처리에 사용한 feature 컬럼 순서
                         (스코어링 입력 검증과 preprocessor.transform 입력에 사용).
        selected_features: 변수 선택 후 모델이 실제로 학습한 컬럼 부분집합.
                           변수 선택을 끄면 ``feature_columns`` 와 동일하다.
                           스코어링 시 preprocessor 통과 후 모델 입력으로 좁힐 때 사용.
        categorical_columns: 범주형으로 취급된 컬럼.
        id_columns: 학습에서는 제외했지만 스코어링 결과에 보존할 컬럼.
        model_name: best 로 선정된 모델 이름 (lgbm | xgb | catboost).
        primary_metric: 모델 선택 기준 지표 이름.
        metric_value: 그 지표의 holdout 점수.
        trained_at: 학습 완료 시각 (UTC ISO).
        library_version: 라이브러리 버전 — 추후 호환성 확인용.
        extra: 자유 메타데이터 (각 모델의 best_iteration 등).
    """

    target_column: str
    feature_columns: list[str]
    categorical_columns: list[str]
    id_columns: list[str]
    model_name: str
    primary_metric: str
    metric_value: float
    selected_features: list[str] = field(default_factory=list)
    trained_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    library_version: str = "0.1.0"
    extra: dict[str, Any] = field(default_factory=dict)


def save_artifact(
    path: str | Path,
    preprocessor: Any,
    model: Any,
    metadata: ArtifactMetadata,
) -> Path:
    """preprocessor + model + metadata 를 단일 joblib 파일로 저장한다.

    Args:
        path: 저장 경로 (``.joblib`` 확장자 권장).
        preprocessor: fit 된 ``PreprocessingPipeline``.
        model: fit 된 ``BaseModel`` 구현체.
        metadata: 운영 메타데이터.

    Returns:
        실제 저장된 ``Path``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"preprocessor": preprocessor, "model": model, "metadata": metadata}
    # compress=3 은 속도/용량 균형이 가장 무난한 값
    joblib.dump(payload, path, compress=3)
    return path


def load_artifact(path: str | Path) -> dict[str, Any]:
    """저장된 artifact 번들을 로드한다.

    Args:
        path: ``save_artifact`` 로 저장된 파일 경로.

    Returns:
        ``{"preprocessor", "model", "metadata"}`` 키를 갖는 dict.
    """
    return joblib.load(Path(path))
