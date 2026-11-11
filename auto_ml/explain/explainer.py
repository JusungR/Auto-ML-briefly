"""artifact 를 사용해 건별·변수별 SHAP 기여도를 계산한다.

``Scorer`` 와 동일한 입력 변환 흐름(``fill_missing_columns`` → schema 검증 →
``preprocessor.transform`` → ``selected_features`` 축소) 을 거치고 마지막에
모델의 ``shap_values`` 를 호출한다. native API 는 raw-margin (logit) 도메인의
SHAP 을 반환하므로 본 모듈에서 probability 도메인으로 정규화한다.

정규화 공식 (scale-and-shift):
    p_base = sigmoid(base_raw)
    p_pred = sigmoid(base_raw + sum(feature_raw))
    delta  = p_pred - p_base
    shap_prob[i] = feature_raw[i] / sum(feature_raw) * delta   (sum != 0)

이렇게 하면 모든 행에서 ``base_value + sum(shap_*) == score`` 가 정확히 성립.
(변환 구현: ``auto_ml.utils.shap_domain.margin_shap_to_probability``)

앙상블은 예측이 서브모델 확률의 가중 평균이므로 위 변환을 서브모델마다 적용한 뒤
가중 평균한다 (``EnsembleModel.shap_values_proba``). 그래야 ``score`` 가 실제 예측
(``Scorer`` 결과) 과 일치한다.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from auto_ml.utils.categorical import log_unseen_categories
from auto_ml.utils.shap_domain import margin_shap_to_probability
from auto_ml.utils.io import load_artifact
from auto_ml.utils.logger import get_logger
from auto_ml.utils.validation import validate_schema

logger = get_logger("explainer")




class Explainer:
    """artifact 한 개로 입력 DataFrame 에서 SHAP wide 표를 만든다.

    사용 예:
        >>> exp = Explainer.from_artifact("artifacts/models/best.joblib")
        >>> df_out = exp.explain(df_in, id_columns=["user_id"])
    """

    def __init__(self, preprocessor: Any, model: Any, metadata: Any) -> None:
        self.preprocessor = preprocessor
        self.model = model
        self.metadata = metadata

    @classmethod
    def from_artifact(cls, path: str | Path) -> "Explainer":
        bundle = load_artifact(path)
        return cls(
            preprocessor=bundle["preprocessor"],
            model=bundle["model"],
            metadata=bundle["metadata"],
        )

    # ------------------------------------------------------------------
    def explain(
        self,
        df: pd.DataFrame,
        id_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """입력 DataFrame 에 건별·변수별 SHAP 을 붙여 wide 포맷으로 반환한다.

        Args:
            df: 학습 시 사용한 feature 컬럼을 포함하는 DataFrame.
            id_columns: 결과에 보존할 식별자. None / 빈 리스트면 metadata fallback.

        Returns:
            ``<id_columns> + shap_<feature>... + base_value + score`` 컬럼을
            갖는 DataFrame. ``base_value + sum(shap_*) == score`` (확률 도메인).
        """
        feature_cols = list(self.metadata.feature_columns)

        df_filled, filled_cols = self.preprocessor.fill_missing_columns(df)
        if filled_cols:
            logger.warning(
                "Input is missing %d feature column(s); filled with training defaults: %s",
                len(filled_cols), filled_cols,
            )
        validate_schema(df_filled, feature_cols)

        X = df_filled[feature_cols]
        X_processed = self.preprocessor.transform(X)
        selected = list(getattr(self.metadata, "selected_features", []) or feature_cols)
        X_for_model = X_processed[selected]
        log_unseen_categories(
            self.preprocessor, X_processed, logger, context="explain_input", columns=selected,
        )

        if hasattr(self.model, "shap_values_proba"):
            # 앙상블: 서브모델별 확률 도메인 SHAP 의 가중 평균
            shap_prob, p_base = self.model.shap_values_proba(X_for_model)
            shap_prob = np.asarray(shap_prob, dtype=float)
            p_base = np.asarray(p_base, dtype=float)
            p_pred = p_base + shap_prob.sum(axis=1)
            n_cols = shap_prob.shape[1] + 1
        else:
            # 단일 모델: raw-margin SHAP — shape (n, n_features + 1), 마지막 열 = base value
            shap_raw = np.asarray(self.model.shap_values(X_for_model))
            n_cols = shap_raw.shape[1]
            shap_prob, p_base, p_pred = margin_shap_to_probability(shap_raw)
        if n_cols != len(selected) + 1:
            raise RuntimeError(
                f"shap_values returned {n_cols} columns, expected "
                f"{len(selected) + 1} (= {len(selected)} features + base value)."
            )

        # id_columns 우선순위 — Scorer.score 와 동일
        ids = list(id_columns) if id_columns else list(self.metadata.id_columns)
        keep_id_cols = [c for c in ids if c in df.columns]
        missing_ids = [c for c in ids if c not in df.columns]
        if missing_ids:
            logger.warning(
                "Configured id_columns missing from input: %s (kept: %s)",
                missing_ids, keep_id_cols or "(none)",
            )

        out = pd.DataFrame(index=df.index)
        for col in keep_id_cols:
            out[col] = df[col].values
        for i, feat in enumerate(selected):
            out[f"shap_{feat}"] = shap_prob[:, i]
        out["base_value"] = p_base
        out["score"] = p_pred
        logger.info(
            "Explained %d rows over %d features (model=%s)",
            len(out), len(selected), self.metadata.model_name,
        )
        return out
