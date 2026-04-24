"""전처리 파이프라인: 결측 → 이상치 → 스케일링 순서로 실행.

본 파이프라인은 학습/스코어링 양 단계에서 동일하게 사용되며, 학습 시
저장된 통계량을 스코어링 시 그대로 재사용해 일관성을 보장한다.

범주형 컬럼은 본 파이프라인에서 결측 채움까지만 수행한다. 인코딩은
모델 래퍼 내부에서 처리한다 — LightGBM/CatBoost 는 native 처리,
XGBoost 는 LabelEncoder 적용. 이렇게 분리하면 단일 전처리 결과를
3개 모델이 공통으로 받아 사용할 수 있다.
"""
from __future__ import annotations

import pandas as pd

from auto_ml.config import PreprocessingConfig
from auto_ml.preprocessing.imputer import NullImputer
from auto_ml.preprocessing.outlier import OutlierHandler
from auto_ml.preprocessing.scaler import NumericScaler


class PreprocessingPipeline:
    """3단계(결측 → 이상치 → 스케일링) 전처리를 묶어 실행하는 컨테이너.

    Attributes:
        numeric_columns: 수치형 컬럼.
        categorical_columns: 범주형 컬럼.
        config: 단계별 옵션 모음.
        imputer / outlier / scaler: 단계별 인스턴스. 디버깅·테스트 시 직접 접근 가능.
    """

    def __init__(
        self,
        numeric_columns: list[str],
        categorical_columns: list[str],
        config: PreprocessingConfig,
    ) -> None:
        self.numeric_columns = list(numeric_columns)
        self.categorical_columns = list(categorical_columns)
        self.config = config

        # 단계별 인스턴스 생성. 순서 자체는 fit/transform 안에서 강제된다.
        self.imputer = NullImputer(
            numeric_columns=self.numeric_columns,
            categorical_columns=self.categorical_columns,
            numeric_strategy=config.numeric_null_strategy,
            numeric_fill_value=config.numeric_null_fill_value,
            categorical_strategy=config.categorical_null_strategy,
            categorical_fill_value=config.categorical_null_fill_value,
        )
        self.outlier = OutlierHandler(
            numeric_columns=self.numeric_columns,
            method=config.outlier_method,
            iqr_multiplier=config.outlier_iqr_multiplier,
            zscore_threshold=config.outlier_zscore_threshold,
            action=config.outlier_action,
        )
        self.scaler = NumericScaler(
            numeric_columns=self.numeric_columns,
            method=config.scaling_method,
        )
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> "PreprocessingPipeline":
        """각 단계를 순서대로 fit 한다.

        Note:
            이상치 단계는 결측이 채워진 데이터를, 스케일러는 이상치가
            처리된 데이터를 학습해야 통계가 왜곡되지 않는다. 그래서
            ``fit_transform`` 으로 순차 적용한다.
        """
        step1 = self.imputer.fit_transform(df)
        step2 = self.outlier.fit_transform(step1)
        self.scaler.fit(step2)
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """학습 시 저장된 상태로 동일 순서의 변환을 수행한다."""
        if not self._fitted:
            raise RuntimeError("PreprocessingPipeline must be fit before transform.")
        x = self.imputer.transform(df)
        x = self.outlier.transform(x)
        x = self.scaler.transform(x)
        return x

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """fit 직후 동일 데이터에 transform 을 적용해 반환한다."""
        return self.fit(df).transform(df)
