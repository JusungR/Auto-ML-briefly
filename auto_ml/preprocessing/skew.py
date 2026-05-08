"""2.5단계: 강하게 skew 된 수치형 컬럼의 분포 변환.

학습 데이터에서 컬럼별 skew 를 측정해 ``|skew| > threshold`` 인 컬럼만
자동으로 선택하고, 학습 시 추정된 파라미터(분위수 테이블 등) 를 저장해
스코어링 시 동일 변환을 적용한다 — 학습/추론 일관성을 보장한다.

지원 방식:
    - method = ``signed_log1p``    : sign(x) * log1p(|x|). 음수도 단조 보존.
                                      무상태 (학습 파라미터 없음). (기본)
    - method = ``quantile_normal`` : sklearn QuantileTransformer 로 학습 분포의
                                      분위수를 표준정규로 매핑. 학습 시 분위수
                                      테이블을 저장.
    - method = ``none``            : 비활성.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer

from auto_ml.preprocessing.base import BaseStep

_VALID_METHODS = ("signed_log1p", "quantile_normal", "none")


class SkewTransformer(BaseStep):
    """강한 skew 컬럼만 자동 선택해 분포를 대칭화하는 전처리 단계.

    Attributes:
        numeric_columns: 후보 컬럼.
        method: ``"signed_log1p" | "quantile_normal" | "none"``.
        threshold: ``abs(skew) > threshold`` 인 컬럼만 변환 대상.
    """

    def __init__(
        self,
        numeric_columns: list[str],
        method: str = "signed_log1p",
        threshold: float = 1.0,
    ) -> None:
        super().__init__()
        self.numeric_columns = list(numeric_columns)
        self.method = method
        self.threshold = float(threshold)

        # fit 시점에 결정되는 변환 대상 컬럼 — 스코어링 시에도 같은 셋만 변환
        self._selected_columns: list[str] = []
        # 진단/보고용 학습-시-skew (모든 후보 컬럼 기록)
        self._train_skews: dict[str, float] = {}
        # quantile_normal 일 때만 사용되는 sklearn transformer
        self._qt: QuantileTransformer | None = None

    def fit(self, df: pd.DataFrame) -> "SkewTransformer":
        """학습 데이터에서 변환 대상 컬럼을 선정하고 필요한 통계를 학습한다."""
        if self.method == "none":
            self._fitted = True
            return self
        if self.method not in _VALID_METHODS:
            raise ValueError(
                f"Unknown skew method: {self.method}. "
                f"Choose from {_VALID_METHODS}."
            )

        selected: list[str] = []
        for col in self.numeric_columns:
            if col not in df.columns:
                continue
            s = df[col].astype(float)
            sk = s.skew()
            # 상수 컬럼은 skew 가 NaN → 자연스럽게 제외
            self._train_skews[col] = float(sk) if pd.notna(sk) else float("nan")
            if pd.notna(sk) and abs(sk) > self.threshold:
                selected.append(col)
        self._selected_columns = selected

        if self.method == "quantile_normal" and selected:
            # sklearn 기본 n_quantiles=1000, subsample=10_000. random_state 는
            # subsample 추출 재현성에만 영향.
            self._qt = QuantileTransformer(
                output_distribution="normal", random_state=42,
            )
            self._qt.fit(df[selected].astype(float).values)

        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """학습 시 선택된 컬럼만 동일 규칙으로 변환한다."""
        self._check_fitted()
        if self.method == "none" or not self._selected_columns:
            return df.copy()

        out = df.copy()
        # 스코어링에서 일부 컬럼이 누락된 경우 건너뜀 (선택적 feature 운용 대응)
        present = [c for c in self._selected_columns if c in out.columns]
        if not present:
            return out

        if self.method == "signed_log1p":
            for col in present:
                x = out[col].astype(float).values
                out[col] = np.sign(x) * np.log1p(np.abs(x))
            return out

        # quantile_normal: 학습 시 fit 된 _qt 를 그대로 사용.
        # _qt 는 fit 때의 컬럼 순서/개수에 맞춰져 있으므로 누락 컬럼이 있으면
        # transform 이 깨진다. 본 라이브러리의 운영 가정상 학습/스코어링 컬럼
        # 셋은 features.csv 로 동일하게 강제되므로 통상 문제되지 않는다.
        if present != self._selected_columns:
            raise RuntimeError(
                "quantile_normal transform requires the same selected columns "
                f"as fit; missing: "
                f"{set(self._selected_columns) - set(present)}"
            )
        out[present] = self._qt.transform(out[present].astype(float).values)
        return out
