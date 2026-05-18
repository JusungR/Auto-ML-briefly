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

        # 전처리 전 별도 단계: 학습 시 봤던 컬럼이 통째로 누락된 경우
        # 학습 데이터의 median(수치) / 최빈값(범주) 으로 컬럼 전체를 채워 넣는다.
        df_filled, filled_cols = self.preprocessor.fill_missing_columns(df)
        if filled_cols:
            logger.warning(
                "Input is missing %d feature column(s); filled with training defaults: %s",
                len(filled_cols), filled_cols,
            )

        validate_schema(df_filled, feature_cols)

        # 전처리 fit 시점과 동일한 컬럼 순서로 정렬해 일관성 확보
        X = df_filled[feature_cols]
        X_processed = self.preprocessor.transform(X)

        # 학습 시 변수 선택을 거쳤다면 모델 입력은 그 부분집합이어야 한다.
        # 옛 artifact (selected_features 가 비어있는 경우) 와의 호환을 위해 fallback.
        selected = list(getattr(self.metadata, "selected_features", []) or feature_cols)
        X_for_model = X_processed[selected]

        proba = self.model.predict_proba(X_for_model)
        prediction = (proba >= threshold).astype(int)

        # ids 결정 — 호출자가 명시한 값이 비어 있지 않으면 그대로 사용,
        # 아니면(None 또는 빈 리스트) metadata 의 학습 시점 id_columns 로 fallback.
        # 빈 리스트가 'IDs 의도적 제외' 가 아니라 단순 미설정인 경우가 대부분이므로
        # falsy 면 fallback 시키는 게 운영 실수를 가장 잘 막는다.
        ids = list(id_columns) if id_columns else list(self.metadata.id_columns)
        keep_id_cols = [c for c in ids if c in df.columns]
        # 설정된 id 중 입력에 누락된 컬럼이 있으면 운영 진단을 위해 WARN.
        missing_ids = [c for c in ids if c not in df.columns]
        if missing_ids:
            logger.warning(
                "Configured id_columns missing from input: %s (kept: %s)",
                missing_ids, keep_id_cols or "(none)",
            )

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
