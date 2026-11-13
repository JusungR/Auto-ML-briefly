"""범주형 값 정규화 유틸리티.

범주형 컬럼은 모델마다 인코딩 방식이 달라(문자열 변환 / pandas category /
LabelEncoder) 같은 값이 다른 범주로 갈라지기 쉽다. 대표적으로:

    - 학습 데이터에 결측이 있으면 정수 코드 컬럼이 float 로 읽혀 ``1.0`` 이
      되고, 결측이 없는 추론 데이터는 int ``1`` 로 들어온다. 문자열로 바꾸면
      ``"1.0"`` 과 ``"1"`` 로 갈라져 추론 데이터 전체가 미학습 범주가 된다.
    - 결측 대체값을 문자열로 넣으면 float ``1.0`` 과 문자열 ``"1.0"`` 이 섞인다.

``normalize_category_series`` 는 모든 값을 같은 규칙의 문자열로 맞춰
이런 불일치를 막는다.

    - 결측(None / NaN / NaT)        → NaN 유지 (결측 대체 단계에서 처리)
    - 정수 또는 정수값 float (1, 1.0) → ``"1"``
    - 그 외 float (1.5)              → ``"1.5"``
    - bool                           → ``"True"`` / ``"False"``
    - 문자열                          → 그대로 (공백·대소문자 변경 없음)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _normalize_scalar(value: object) -> object:
    if value is None:
        return np.nan
    if isinstance(value, str):
        return value
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return np.nan
        if float(value).is_integer():
            return str(int(value))
        return str(float(value))
    try:
        if pd.isna(value):
            return np.nan
    except (TypeError, ValueError):
        pass
    return str(value)


def normalize_category_series(s: pd.Series) -> pd.Series:
    """범주형 Series 를 정규화된 문자열(object dtype) Series 로 변환한다.

    결측은 NaN 으로 남긴다. 인덱스와 이름은 유지한다.
    """
    if isinstance(s.dtype, pd.CategoricalDtype):
        s = s.astype(object)
    if pd.api.types.is_bool_dtype(s):
        return s.map(_normalize_scalar).astype(object)
    if pd.api.types.is_integer_dtype(s):
        # 정수형(int64 / uint64 / nullable Int64) 은 float 을 거치지 않고 그대로 문자열로 바꾼다.
        # float 을 거치면 2^53 보다 큰 코드의 정밀도가 깨져 서로 다른 코드가 합쳐진다.
        out = np.full(len(s), np.nan, dtype=object)
        notna = s.notna().to_numpy()
        if notna.any():
            out[notna] = s[notna].astype(str).to_numpy()
        return pd.Series(out, index=s.index, name=s.name, dtype=object)
    if pd.api.types.is_numeric_dtype(s):
        # 실수형은 벡터 연산으로 처리 (대용량 대비)
        vals = s.to_numpy(dtype=float, na_value=np.nan)
        out = np.full(len(vals), np.nan, dtype=object)
        notna = ~np.isnan(vals)
        integral = notna & (vals == np.floor(vals)) & np.isfinite(vals)
        # int64 범위 안은 벡터 변환, 밖(예: 1e20)은 오버플로를 피해 파이썬 int 로 변환
        # (_normalize_scalar 의 str(int(v)) 와 같은 결과)
        in_range = integral & (np.abs(vals) < 2.0**63)
        if in_range.any():
            out[in_range] = vals[in_range].astype(np.int64).astype(str)
        huge = integral & ~in_range
        if huge.any():
            out[huge] = [str(int(v)) for v in vals[huge]]
        other = notna & ~integral
        if other.any():
            out[other] = [str(float(v)) for v in vals[other]]
        return pd.Series(out, index=s.index, name=s.name, dtype=object)
    return s.map(_normalize_scalar).astype(object)


def normalize_categories(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """``columns`` 중 df 에 있는 컬럼만 정규화한 사본을 반환한다."""
    present = [c for c in columns if c in df.columns]
    if not present:
        return df
    out = df.copy()
    for col in present:
        out[col] = normalize_category_series(out[col])
    return out


# 미학습 범주 비율이 이 값 이상이면 WARNING, 그 미만(1건 이상)이면 INFO 로 남긴다.
UNSEEN_WARN_RATE = 0.01


def log_unseen_categories(
    preprocessor: object,
    X_processed: pd.DataFrame,
    logger: object,
    context: str,
    columns: list[str] | None = None,
) -> dict[str, dict]:
    """전처리 결과의 미학습 범주 비율을 로그로 남기고 통계를 반환한다.

    Args:
        preprocessor: ``unseen_category_stats`` 를 가진 PreprocessingPipeline.
            과거 artifact 처럼 메서드가 없으면 아무것도 하지 않는다.
        X_processed: ``preprocessor.transform`` 결과.
        logger: 로거.
        context: 로그에 표시할 데이터 이름 (예: ``"score_input"``, ``"test"``).
        columns: 진단할 컬럼 (모델 입력 컬럼). None 이면 전체 범주형.
    """
    fn = getattr(preprocessor, "unseen_category_stats", None)
    if fn is None:
        return {}
    stats = fn(X_processed)
    if columns is not None:
        keep = set(columns)
        stats = {c: v for c, v in stats.items() if c in keep}
    for col, st in stats.items():
        level = "warning" if st["rate"] >= UNSEEN_WARN_RATE else "info"
        grouped = f" -> grouped into '{st['grouped_into']}'" if st.get("grouped_into") else ""
        getattr(logger, level)(
            "[%s] unseen categories in '%s': %d / %d rows (%.2f%%), e.g. %s%s",
            context, col, st["n_unseen"], st["n_rows"], 100 * st["rate"],
            st["examples"], grouped,
        )
    return stats
