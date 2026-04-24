"""전처리 단계가 공통으로 따르는 인터페이스.

scikit-learn 의 ``fit / transform`` 패턴을 그대로 따르되, 입출력은 항상
pandas ``DataFrame`` 으로 고정한다 — 컬럼명을 잃지 않아 디버깅과
스코어링 시 검증이 쉽다.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class BaseStep(ABC):
    """fit / transform 인터페이스를 갖는 stateful 전처리 단계.

    상속 클래스는 ``fit`` 안에서 학습 데이터를 보고 통계량을 저장하고,
    ``transform`` 에서는 저장된 통계량을 읽어 입력을 변환한다.
    학습 단계와 스코어링 단계의 변환 규칙이 동일해야 하므로 ``transform``
    은 무상태가 아니라 fit 에서 학습된 상태에 의존한다.
    """

    def __init__(self) -> None:
        # fit 호출 여부를 기록해 잘못된 호출 순서를 가드한다.
        self._fitted: bool = False

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> "BaseStep":
        """학습 데이터에서 변환에 필요한 통계량을 계산한다."""

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """fit 에서 학습된 통계량을 사용해 입력을 변환한다."""

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """fit 후 동일 데이터에 transform 을 적용해 반환한다."""
        return self.fit(df).transform(df)

    def _check_fitted(self) -> None:
        """transform 이 fit 이전에 호출되는 실수를 방지한다."""
        if not self._fitted:
            raise RuntimeError(f"{type(self).__name__} must be fit before transform.")
