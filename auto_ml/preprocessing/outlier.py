"""2단계: 수치형 컬럼의 이상치 처리.

학습 시 컬럼별 경계(lower/upper) 를 계산해 저장하고, 스코어링 시
동일 경계를 적용한다 — 학습 분포 기준으로 일관 처리한다.

지원 방식:
    - method = ``iqr``    : Q1 - k*IQR, Q3 + k*IQR (k 는 보통 1.5)
    - method = ``zscore`` : mean ± k*std
    - method = ``none``   : 비활성화

처리 액션:
    - action = ``clip``             : 경계 밖 값을 경계로 잘라낸다 (기본).
    - action = ``null_then_impute`` : 경계 밖 값을 NaN 으로 만든 뒤 학습 시
                                       median 으로 다시 채운다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from auto_ml.preprocessing.base import BaseStep


class OutlierHandler(BaseStep):
    """수치형 컬럼 이상치를 처리하는 전처리 단계.

    Attributes:
        numeric_columns: 적용 대상 컬럼.
        method: ``"iqr" | "zscore" | "none"``.
        iqr_multiplier: IQR 방식의 경계 배수.
        zscore_threshold: Z-score 방식의 임계치.
        action: ``"clip" | "null_then_impute"``.
    """

    def __init__(
        self,
        numeric_columns: list[str],
        method: str = "iqr",
        iqr_multiplier: float = 1.5,
        zscore_threshold: float = 3.0,
        action: str = "clip",
    ) -> None:
        super().__init__()
        self.numeric_columns = list(numeric_columns)
        self.method = method
        self.iqr_multiplier = iqr_multiplier
        self.zscore_threshold = zscore_threshold
        self.action = action

        # 학습 시 컬럼별로 계산되는 (lower, upper) 경계
        self._bounds: dict[str, tuple[float, float]] = {}
        # null_then_impute 액션에서 NaN 을 다시 채울 때 사용할 median
        self._medians: dict[str, float] = {}

    def fit(self, df: pd.DataFrame) -> "OutlierHandler":
        """학습 데이터에서 컬럼별 경계와 median 을 계산한다."""
        if self.method == "none":
            self._fitted = True
            return self

        for col in self.numeric_columns:
            if col not in df.columns:
                continue
            s = df[col].astype(float)
            if self.method == "iqr":
                q1 = s.quantile(0.25)
                q3 = s.quantile(0.75)
                iqr = q3 - q1
                lower = q1 - self.iqr_multiplier * iqr
                upper = q3 + self.iqr_multiplier * iqr
            elif self.method == "zscore":
                mean = s.mean()
                # ddof=0 으로 모집단 표준편차 사용 (대용량에서는 차이 미미)
                std = s.std(ddof=0)
                lower = mean - self.zscore_threshold * std
                upper = mean + self.zscore_threshold * std
            else:
                raise ValueError(f"Unknown outlier method: {self.method}")
            self._bounds[col] = (float(lower), float(upper))
            self._medians[col] = float(s.median())

        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """학습 시 저장된 경계를 사용해 이상치를 처리한다."""
        self._check_fitted()
        if self.method == "none" or not self._bounds:
            return df.copy()

        out = df.copy()
        for col, (lower, upper) in self._bounds.items():
            if col not in out.columns:
                continue
            s = out[col].astype(float)
            if self.action == "clip":
                out[col] = s.clip(lower=lower, upper=upper)
            elif self.action == "null_then_impute":
                # 경계 밖이면 NaN, 그 후 학습 시 median 으로 재충전
                mask = (s < lower) | (s > upper)
                s = s.where(~mask, other=np.nan)
                out[col] = s.fillna(self._medians[col])
            else:
                raise ValueError(f"Unknown outlier action: {self.action}")
        return out
