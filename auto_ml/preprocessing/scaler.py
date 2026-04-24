"""3단계: 수치형 컬럼 스케일링.

scikit-learn 의 스케일러를 pandas 입출력으로 감싼 얇은 래퍼다.
부스팅 트리 모델은 사실상 스케일링이 필요 없지만, 향후 선형/신경망
모델로 확장될 가능성을 고려해 옵션으로 제공한다 (기본 ``standard``).
"""
from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from auto_ml.preprocessing.base import BaseStep

# 설정 문자열 → 사이킷런 스케일러 클래스 매핑
_SCALERS = {
    "standard": StandardScaler,   # 평균 0, 분산 1
    "minmax": MinMaxScaler,       # [0, 1] 정규화
    "robust": RobustScaler,       # 중앙값 / IQR 기준 — 이상치에 강건
}


class NumericScaler(BaseStep):
    """수치형 컬럼 스케일링 단계.

    Attributes:
        numeric_columns: 적용 대상 컬럼.
        method: ``"standard" | "minmax" | "robust" | "none"``.
    """

    def __init__(self, numeric_columns: list[str], method: str = "standard") -> None:
        super().__init__()
        self.numeric_columns = list(numeric_columns)
        self.method = method
        self._scaler = None
        # fit 시점에 실제 데이터에 존재한 컬럼 — transform 에서도 동일 셋을 사용
        self._cols_in_fit: list[str] = []

    def fit(self, df: pd.DataFrame) -> "NumericScaler":
        """학습 데이터로 스케일러를 fit 한다."""
        if self.method == "none" or not self.numeric_columns:
            # 스케일링을 끄거나 대상 컬럼이 없으면 무동작 모드로 둔다.
            self._fitted = True
            return self
        if self.method not in _SCALERS:
            raise ValueError(f"Unknown scaling method: {self.method}")

        cols = [c for c in self.numeric_columns if c in df.columns]
        self._cols_in_fit = cols
        self._scaler = _SCALERS[self.method]()
        self._scaler.fit(df[cols].astype(float).values)
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """fit 된 스케일러로 컬럼을 변환한다."""
        self._check_fitted()
        if self._scaler is None:
            return df.copy()
        out = df.copy()
        # transform 시점에 누락된 컬럼은 건너뛴다 (선택적 feature 운용 대응)
        cols = [c for c in self._cols_in_fit if c in out.columns]
        out[cols] = self._scaler.transform(out[cols].astype(float).values)
        return out
