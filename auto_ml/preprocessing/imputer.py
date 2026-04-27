"""1단계: 결측(Null) 처리.

수치형은 median/mean/constant 중 한 가지로 채우고, 범주형은 최빈값 또는
지정 문자열로 채운다. 학습 데이터에서 계산한 채움 값을 저장해 두었다가
스코어링 시 동일하게 적용한다 — 학습/추론 일관성을 보장한다.
"""
from __future__ import annotations

import pandas as pd

from auto_ml.preprocessing.base import BaseStep


class NullImputer(BaseStep):
    """결측치를 채우는 전처리 단계.

    Attributes:
        numeric_columns: 수치형 컬럼 이름 목록.
        categorical_columns: 범주형 컬럼 이름 목록.
        numeric_strategy: ``"median" | "mean" | "constant"``.
        numeric_fill_value: ``numeric_strategy == "constant"`` 일 때만 사용.
        categorical_strategy: ``"most_frequent" | "constant"``.
        categorical_fill_value: ``categorical_strategy == "constant"`` 일 때만 사용.
    """

    def __init__(
        self,
        numeric_columns: list[str],
        categorical_columns: list[str],
        numeric_strategy: str = "median",
        numeric_fill_value: float = 0.0,
        categorical_strategy: str = "most_frequent",
        categorical_fill_value: str = "MISSING",
    ) -> None:
        super().__init__()
        self.numeric_columns = list(numeric_columns)
        self.categorical_columns = list(categorical_columns)
        self.numeric_strategy = numeric_strategy
        self.numeric_fill_value = numeric_fill_value
        self.categorical_strategy = categorical_strategy
        self.categorical_fill_value = categorical_fill_value

        # fit 단계에서 채워질 컬럼별 채움 값. 스코어링 시 그대로 재사용된다.
        self._numeric_fills: dict[str, float] = {}
        self._categorical_fills: dict[str, str] = {}

    def fit(self, df: pd.DataFrame) -> "NullImputer":
        """학습 데이터에서 컬럼별 채움 값을 계산해 저장한다."""
        # ----- 수치형 -----
        for col in self.numeric_columns:
            if col not in df.columns:
                # 설정에는 있으나 실제 데이터에 없는 경우는 조용히 건너뛴다.
                # (선택적 feature 운용 케이스 대응)
                continue
            s = df[col]
            if self.numeric_strategy == "median":
                self._numeric_fills[col] = float(s.median())
            elif self.numeric_strategy == "mean":
                self._numeric_fills[col] = float(s.mean())
            elif self.numeric_strategy == "constant":
                self._numeric_fills[col] = float(self.numeric_fill_value)
            else:
                raise ValueError(
                    f"Unknown numeric null strategy: {self.numeric_strategy}"
                )

        # ----- 범주형 -----
        for col in self.categorical_columns:
            if col not in df.columns:
                continue
            s = df[col]
            if self.categorical_strategy == "most_frequent":
                mode = s.mode(dropna=True)
                # mode 가 비어있는 극단적 케이스(전부 NaN) 는 fallback 사용
                self._categorical_fills[col] = (
                    str(mode.iloc[0]) if not mode.empty else self.categorical_fill_value
                )
            elif self.categorical_strategy == "constant":
                self._categorical_fills[col] = str(self.categorical_fill_value)
            else:
                raise ValueError(
                    f"Unknown categorical null strategy: {self.categorical_strategy}"
                )

        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """저장된 채움 값으로 결측을 채운 새 DataFrame 을 반환한다."""
        self._check_fitted()
        out = df.copy()
        for col, value in self._numeric_fills.items():
            if col in out.columns:
                out[col] = out[col].fillna(value)
        # pandas 2.x 의 fillna 가 결과를 silently downcast 하는 동작이
        # deprecated 됐다. 우리는 명시적으로 object dtype 을 유지하고 싶으므로
        # 새 동작(downcast 안 함) 을 option_context 로 강제해 경고를 제거한다.
        with pd.option_context("future.no_silent_downcasting", True):
            for col, value in self._categorical_fills.items():
                if col in out.columns:
                    # object 로 캐스팅해 categorical dtype 의 미등록 카테고리 오류 회피
                    out[col] = out[col].astype("object").fillna(value)
        return out
