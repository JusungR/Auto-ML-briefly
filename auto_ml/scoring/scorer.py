"""저장된 artifact 를 사용해 입력 DataFrame 을 스코어링한다.

학습 단계에서 저장한 ``preprocessor + model + metadata`` 를 모두 로드하여
스코어링 전 과정(전처리 → 예측 → 임계값 적용) 을 동일하게 재현한다.

이 모듈은 I/O 에 의존하지 않는다 → 단위 테스트가 쉽고, runner.py 가
Parquet I/O 를 담당한다.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from auto_ml.utils.io import load_artifact
from auto_ml.utils.logger import get_logger
from auto_ml.utils.validation import validate_schema

logger = get_logger("scorer")


class Scorer:
    """artifact 한 개로 입력 DataFrame 을 점수화하는 클래스.

    사용 예:
        >>> scorer = Scorer.from_artifact("artifacts/models/best.joblib")
        >>> df_out = scorer.score(df_in, threshold=0.5, id_columns=["user_id"])
    """

    def __init__(self, preprocessor: Any, model: Any, metadata: Any) -> None:
        self.preprocessor = preprocessor
        self.model = model
        self.metadata = metadata

    # ------------------------------------------------------------------
    @classmethod
    def from_artifact(cls, path: str | Path) -> "Scorer":
        """artifact 번들 파일을 로드해 Scorer 를 생성한다."""
        bundle = load_artifact(path)
        return cls(
            preprocessor=bundle["preprocessor"],
            model=bundle["model"],
            metadata=bundle["metadata"],
        )

    # ------------------------------------------------------------------
    def score(
        self,
        df: pd.DataFrame,
        threshold: float = 0.5,
        id_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """입력 DataFrame 에 점수와 0/1 예측을 붙여 반환한다.

        Args:
            df: 스코어링 대상. 학습 시 사용한 모든 feature 를 포함해야 한다.
            threshold: ``predict_proba`` 결과를 0/1 로 변환할 임계값.
            id_columns: 결과에 보존할 식별자 컬럼. None 이면 metadata 의 값을 사용.

        Returns:
            ``id_columns + ['score', 'prediction']`` 컬럼을 갖는 DataFrame.
        """
        feature_cols = list(self.metadata.feature_columns)
        validate_schema(df, feature_cols)

        # 전처리 fit 시점과 동일한 컬럼 순서로 정렬해 일관성 확보
        X = df[feature_cols]
        X_processed = self.preprocessor.transform(X)
        proba = self.model.predict_proba(X_processed)
        prediction = (proba >= threshold).astype(int)

        ids = id_columns if id_columns is not None else list(self.metadata.id_columns)
        keep_id_cols = [c for c in ids if c in df.columns]

        out = pd.DataFrame(index=df.index)
        for col in keep_id_cols:
            out[col] = df[col].values
        out["score"] = proba
        out["prediction"] = prediction
        logger.info(
            "Scored %d rows (model=%s, threshold=%.3f)",
            len(out), self.metadata.model_name, threshold,
        )
        return out
