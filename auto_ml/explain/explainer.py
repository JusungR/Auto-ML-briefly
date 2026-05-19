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
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from auto_ml.utils.io import load_artifact
from auto_ml.utils.logger import get_logger
from auto_ml.utils.validation import validate_schema

logger = get_logger("explainer")


def _sigmoid(z: np.ndarray) -> np.ndarray:
    # 큰 음수에서의 overflow 방지를 위해 분기 — np.exp(|z|) > 1e308 회피
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[~pos])
    out[~pos] = e / (1.0 + e)
    return out


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

        # raw-margin SHAP — shape (n, n_features + 1), 마지막 열 = base value
        shap_raw = np.asarray(self.model.shap_values(X_for_model))
        if shap_raw.shape[1] != len(selected) + 1:
            raise RuntimeError(
                f"shap_values returned {shap_raw.shape[1]} columns, expected "
                f"{len(selected) + 1} (= {len(selected)} features + base value)."
            )
        feature_raw = shap_raw[:, :-1]
        base_raw = shap_raw[:, -1]

        # probability 도메인으로 변환 (가법성 보존)
        p_base = _sigmoid(base_raw)
        p_pred = _sigmoid(base_raw + feature_raw.sum(axis=1))
        delta = p_pred - p_base
        sum_raw = feature_raw.sum(axis=1)
        # 0 나눗셈 회피 — sum_raw==0 인 행은 SHAP 도 모두 0
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(sum_raw != 0, delta / sum_raw, 0.0)
        shap_prob = feature_raw * ratio[:, None]

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
