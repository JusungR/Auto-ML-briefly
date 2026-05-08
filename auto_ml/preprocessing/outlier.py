"""2단계: 수치형 컬럼의 이상치 처리 (백분위수 기반 윈저라이징).

학습 시 컬럼별로 ``(quantile(lower_q), quantile(upper_q))`` 경계를 계산해
저장하고, 스코어링 시 동일 경계를 적용한다 — 학습 분포 기준으로 일관된
처리를 보장한다.

지원 방식:
    - method = ``percentile`` : 학습 분포의 분위수를 경계로 윈저라이징 (기본).
    - method = ``none``       : 비활성화.

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
    """수치형 컬럼 이상치를 백분위수 기반으로 윈저라이징하는 전처리 단계.

    Attributes:
        numeric_columns: 적용 대상 컬럼.
        method: ``"percentile" | "none"``.
        lower_quantile: 윈저라이징 하한 분위수 (0~1).
        upper_quantile: 윈저라이징 상한 분위수 (0~1).
        action: ``"clip" | "null_then_impute"``.
    """

    def __init__(
        self,
        numeric_columns: list[str],
        method: str = "percentile",
        lower_quantile: float = 0.01,
        upper_quantile: float = 0.99,
        action: str = "clip",
    ) -> None:
        super().__init__()
        self.numeric_columns = list(numeric_columns)
        self.method = method
        self.lower_quantile = float(lower_quantile)
        self.upper_quantile = float(upper_quantile)
        self.action = action

        if not (0.0 <= self.lower_quantile < self.upper_quantile <= 1.0):
            raise ValueError(
                f"Invalid quantile bounds: lower_quantile={self.lower_quantile}, "
                f"upper_quantile={self.upper_quantile}. "
                f"Must satisfy 0 <= lower_quantile < upper_quantile <= 1."
            )

        # 학습 시 컬럼별로 계산되는 (lower, upper) 경계
        self._bounds: dict[str, tuple[float, float]] = {}
        # null_then_impute 액션에서 NaN 을 다시 채울 때 사용할 median
        self._medians: dict[str, float] = {}

    def fit(self, df: pd.DataFrame) -> "OutlierHandler":
        """학습 데이터에서 컬럼별 분위수 경계와 median 을 계산한다."""
        if self.method == "none":
            self._fitted = True
            return self
        if self.method != "percentile":
            raise ValueError(f"Unknown outlier method: {self.method}")

        for col in self.numeric_columns:
            if col not in df.columns:
                continue
            s = df[col].astype(float)
            lower = float(s.quantile(self.lower_quantile))
            upper = float(s.quantile(self.upper_quantile))
            # 학습 데이터의 컬럼이 전부 NaN 이면 분위수가 NaN 이 되어
            # clip(NaN, NaN) 이 모든 값을 NaN 으로 만든다. imputer 와 동일하게
            # 명시적으로 실패시켜 운영 가시성을 확보한다.
            if pd.isna(lower) or pd.isna(upper):
                raise ValueError(
                    f"Numeric column '{col}' has no non-null values in the "
                    f"training data; cannot compute percentile bounds. "
                    f"Remove the column from features.csv (set used=false) "
                    f"or set outlier_method to 'none'."
                )
            self._bounds[col] = (lower, upper)
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
