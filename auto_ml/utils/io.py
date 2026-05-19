"""학습 산출물(artifact) 저장 / 로드 유틸리티.

artifact 번들 = 단일 파일에 다음을 함께 저장한다:
    - preprocessor : 학습 시 fit 된 PreprocessingPipeline
    - model        : 학습 시 fit 된 BaseModel 구현체 (LGBM/XGB/CatBoost 중 1개)
    - metadata     : 스키마, 학습 시점, feature/target 명, 평가 지표 등 운영 정보

cloudpickle + gzip 으로 직렬화하므로 auto_ml 패키지가 설치되지 않은 환경에서도
클래스 정의가 파일 안에 내장되어 있어 표준 pickle 로 로드할 수 있다.
구버전 joblib 형식(.joblib) 도 하위 호환으로 로드 가능.

이렇게 한 번들로 묶으면 폐쇄 환경 이관이 단순해진다 — 파일과
스코어링용 config YAML 두 개만 옮기면 즉시 운영이 가능하다.

부가로 학습/스코어링 진입점에서 데이터 로드 직후 한 줄 요약을 찍어주기 위한
``summarize_dataframe`` 도 제공한다.
"""
from __future__ import annotations

import gzip
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import cloudpickle
import joblib
import pandas as pd


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
    """preprocessor + model + metadata 를 단일 파일로 저장한다.

    cloudpickle + gzip 으로 직렬화하여 auto_ml 패키지가 없는 환경에서도
    표준 ``pickle.load`` 로 로드할 수 있다.

    Args:
        path: 저장 경로 (``.joblib`` 확장자 권장, 형식은 무관).
        preprocessor: fit 된 ``PreprocessingPipeline``.
        model: fit 된 ``BaseModel`` 구현체.
        metadata: 운영 메타데이터.

    Returns:
        실제 저장된 ``Path``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"preprocessor": preprocessor, "model": model, "metadata": metadata}
    with gzip.open(path, "wb") as f:
        cloudpickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_artifact(path: str | Path) -> dict[str, Any]:
    """저장된 artifact 번들을 로드한다.

    cloudpickle+gzip 형식(신규) 과 구버전 joblib 형식을 모두 지원한다.

    Args:
        path: ``save_artifact`` 로 저장된 파일 경로.

    Returns:
        ``{"preprocessor", "model", "metadata"}`` 키를 갖는 dict.
    """
    p = Path(path)
    # gzip 매직 바이트(0x1f 0x8b) 로 형식 판별
    with open(p, "rb") as f:
        magic = f.read(2)
    if magic == b"\x1f\x8b":
        with gzip.open(p, "rb") as f:
            return pickle.load(f)
    # 구버전 joblib 형식 하위 호환
    return joblib.load(p)


def find_degenerate_columns(
    df: pd.DataFrame,
    columns: list[str] | None = None,
) -> dict[str, list[str]]:
    """모델 입력으로 의미 없는 (degenerate) 컬럼을 찾아 분류해 반환한다.

    범주:
        - ``all_null``: 모든 값이 NaN — 학습/스코어링 신호가 전혀 없다.
        - ``all_zero``: NaN 이 아닌 값이 모두 0 — 보통 데이터 파이프라인 오류 또는
          의미 없는 카운터.
        - ``constant_nonzero``: NaN 제외 unique 값이 1개이며 0 이 아닌 상수.
          모델에 정보를 주지 않는다.

    Args:
        df: 검사할 DataFrame.
        columns: 검사 대상 컬럼. None 이면 모든 컬럼.

    Returns:
        ``{"all_null": [...], "all_zero": [...], "constant_nonzero": [...]}``.
        세 키 모두 항상 존재하며 비어 있을 수 있다.
    """
    cols = columns if columns is not None else list(df.columns)
    out: dict[str, list[str]] = {"all_null": [], "all_zero": [], "constant_nonzero": []}
    for col in cols:
        if col not in df.columns:
            continue
        s = df[col]
        if s.isna().all():
            out["all_null"].append(col)
            continue
        non_na = s.dropna()
        # 수치형이면 numeric 비교가 가능. 범주형/문자열은 0 비교가 안 되므로 skip.
        if pd.api.types.is_numeric_dtype(s):
            if (non_na == 0).all():
                out["all_zero"].append(col)
                continue
        # NaN 제외 unique 가 1개면 상수 (위에서 0 케이스는 이미 처리됨)
        if non_na.nunique() == 1:
            out["constant_nonzero"].append(col)
    return out


def warn_degenerate_columns(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    *,
    logger: Any = None,
    context: str = "data",
) -> dict[str, list[str]]:
    """``find_degenerate_columns`` 결과를 ``logger.warning`` 으로 띄운다.

    Args:
        df: 검사 대상.
        columns: 검사 대상 컬럼. None 이면 전체.
        logger: 표준 logging.Logger. None 이면 경고를 띄우지 않고 결과만 반환.
        context: 로그 메시지의 prefix 문자열 (예: "train", "test", "score_input").

    Returns:
        ``find_degenerate_columns`` 와 동일한 dict.
    """
    found = find_degenerate_columns(df, columns)
    if logger is not None:
        if found["all_null"]:
            logger.warning(
                "[%s] All-NaN columns (%d): %s — 컬럼 자체가 빈 값. features.csv 의 'used'"
                " 를 false 로 두거나 데이터 파이프라인 점검 필요.",
                context, len(found["all_null"]), found["all_null"],
            )
        if found["all_zero"]:
            logger.warning(
                "[%s] All-zero columns (%d): %s — NaN 제외 모든 값이 0. 카운터 누락"
                " 또는 ETL 오류일 가능성.",
                context, len(found["all_zero"]), found["all_zero"],
            )
        if found["constant_nonzero"]:
            logger.warning(
                "[%s] Constant columns (%d): %s — 단일 값으로 채워져 모델에 정보를"
                " 주지 못함. features.csv 에서 제거 권장.",
                context, len(found["constant_nonzero"]), found["constant_nonzero"],
            )
    return found


def summarize_dataframe(
    df: pd.DataFrame,
    target_column: str | None = None,
) -> str:
    """학습/스코어링 진입점에서 데이터 로드 직후 찍을 한 줄 요약을 만든다.

    Args:
        df: 요약할 DataFrame.
        target_column: 지정 시 클래스 비율을 함께 표시. 컬럼이 없으면 무시.

    Returns:
        ``"N rows × M cols | target '<col>': k/N positive (P%) | missing X.XX%"`` 형식의
        한 줄 문자열. target 이 없거나 컬럼이 누락되면 해당 절을 생략한다.

    예:
        Train: 24000 rows × 25 cols | target 'default': 5331/24000 positive (22.21%) | missing 0.00%
    """
    n_rows, n_cols = df.shape
    parts = [f"{n_rows:,} rows × {n_cols} cols"]

    if target_column is not None and target_column in df.columns:
        y = df[target_column]
        n_total = len(y)
        n_pos = int((y == 1).sum())
        rate = (n_pos / n_total * 100.0) if n_total > 0 else 0.0
        parts.append(f"target '{target_column}': {n_pos:,}/{n_total:,} positive ({rate:.2f}%)")

    # 전체 cell 중 NaN 비율 — 부분 결측 규모 감 잡기용 (컬럼별 상세는 전처리 로그에서)
    n_cells = n_rows * n_cols
    if n_cells > 0:
        miss_rate = float(df.isna().sum().sum()) / n_cells * 100.0
        parts.append(f"missing {miss_rate:.2f}%")

    return " | ".join(parts)
