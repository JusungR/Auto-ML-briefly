"""Stability Selection (Meinshausen & Bühlmann, 2010) 구현.

핵심 절차:
    1) 학습 데이터에서 stratified 부분표본을 ``n_subsamples`` 번 추출
       (subsample_ratio 만큼의 행을 클래스 비율 유지하며 뽑는다)
    2) 각 부분표본에서 ``base_estimator`` 로 변수 선택을 수행
       - lasso  : L1 로지스틱 회귀 → non-zero coef
       - lgbm   : LightGBM gain 중요도 상위 ``lgbm_top_k`` 개
    3) 변수별 선택 빈도 (selection probability) 를 누적
    4) ``threshold`` 이상 변수만 채택. 임계값 이상이 ``min_selected`` 개
       미만이면 frequency 상위 N 개로 fallback 하여 모델 입력이 비는 것을 방지.

전처리 단계 이후를 가정하고 동작한다 — 수치형은 이미 스케일된 상태,
범주형은 object/문자열 그대로다. Lasso 분기는 범주형을 frequency
encoding 으로 1회 변환해 사용한다 (subsample 마다 새로 인코딩하지 않는다 —
bias 와 비용을 줄이기 위함).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from auto_ml.config import FeatureSelectionConfig
from auto_ml.utils.logger import get_logger

logger = get_logger("feature_selection")


@dataclass
class SelectionResult:
    """변수 선택 결과 묶음.

    Attributes:
        selected_features: 최종 채택 변수 (입력 컬럼 순서를 유지).
        frequencies: 모든 입력 변수의 선택 빈도 (0.0 ~ 1.0).
        threshold: 적용된 채택 임계값.
        n_subsamples: 실제 사용된 부분표본 수.
        base_estimator: 사용한 base 선택기 이름.
        fallback_used: ``min_selected`` fallback 이 발동했는지.
    """

    selected_features: list[str]
    frequencies: dict[str, float]
    threshold: float
    n_subsamples: int
    base_estimator: str
    fallback_used: bool = False
    extra: dict = field(default_factory=dict)


class StabilitySelector:
    """Stability Selection 으로 변수 선택을 수행한다.

    사용 예:
        >>> selector = StabilitySelector(cfg.feature_selection,
        ...                              numeric_columns, categorical_columns)
        >>> res = selector.fit_select(X_train_processed, y_train)
        >>> res.selected_features
        ['age', 'fee', 'gender']
    """

    def __init__(
        self,
        config: FeatureSelectionConfig,
        numeric_columns: list[str],
        categorical_columns: list[str],
    ) -> None:
        self.config = config
        self.numeric_columns = list(numeric_columns)
        self.categorical_columns = list(categorical_columns)

    # ------------------------------------------------------------------
    def fit_select(self, X: pd.DataFrame, y: pd.Series) -> SelectionResult:
        """Stability Selection 을 수행해 ``SelectionResult`` 를 반환한다."""
        cfg = self.config
        all_features = list(X.columns)
        if not all_features:
            raise ValueError("StabilitySelector requires at least one feature column.")

        # base_estimator 분기에 따라 입력 표현(X_repr) 을 미리 준비한다 —
        # 부분표본마다 동일한 인코딩을 재사용해 비용을 줄인다.
        if cfg.base_estimator == "lasso":
            X_repr = self._encode_for_lasso(X)
            select_fn = self._lasso_select
        elif cfg.base_estimator == "lgbm":
            X_repr = self._prepare_for_lgbm(X)
            select_fn = self._lgbm_select
        else:
            raise ValueError(
                f"Unknown base_estimator: {cfg.base_estimator}. Use 'lasso' or 'lgbm'."
            )

        from sklearn.model_selection import StratifiedShuffleSplit

        splitter = StratifiedShuffleSplit(
            n_splits=cfg.n_subsamples,
            train_size=cfg.subsample_ratio,
            random_state=cfg.random_state,
        )

        counts = {f: 0 for f in all_features}
        n_done = 0
        y_arr = np.asarray(y)
        for sub_idx, _ in splitter.split(np.zeros(len(X_repr)), y_arr):
            try:
                selected = select_fn(X_repr.iloc[sub_idx], y.iloc[sub_idx])
            except Exception as exc:  # 부분표본 단위 실패는 건너뛰고 계속한다
                logger.warning(
                    "Stability selection subsample %d failed: %s", n_done, exc,
                )
                continue
            for f in selected:
                counts[f] = counts.get(f, 0) + 1
            n_done += 1

        if n_done == 0:
            raise RuntimeError(
                "All stability selection subsamples failed. Check base_estimator settings."
            )

        frequencies = {f: counts[f] / n_done for f in all_features}
        selected = [f for f in all_features if frequencies[f] >= cfg.threshold]

        fallback_used = False
        if len(selected) < cfg.min_selected:
            ranked = sorted(all_features, key=lambda f: frequencies[f], reverse=True)
            selected = ranked[: max(cfg.min_selected, 1)]
            fallback_used = True
            logger.warning(
                "Stability selection threshold %.2f kept fewer than min_selected=%d; "
                "falling back to top %d by frequency.",
                cfg.threshold, cfg.min_selected, len(selected),
            )

        return SelectionResult(
            selected_features=selected,
            frequencies=frequencies,
            threshold=cfg.threshold,
            n_subsamples=n_done,
            base_estimator=cfg.base_estimator,
            fallback_used=fallback_used,
        )

    # ------------------------------------------------------------------
    def _encode_for_lasso(self, X: pd.DataFrame) -> pd.DataFrame:
        """Lasso 입력용 인코딩.

        - 수치형: NaN→0 으로 안전화 (전처리 단계에서 이미 채워졌을 것이지만 방어).
        - 범주형: full X 기준 frequency encoding (값 → 등장 비율).
        """
        out = pd.DataFrame(index=X.index)
        for col in X.columns:
            if col in self.categorical_columns:
                vc = X[col].value_counts(normalize=True, dropna=False)
                out[col] = X[col].map(vc).fillna(0.0).astype(float)
            else:
                out[col] = pd.to_numeric(X[col], errors="coerce").fillna(0.0).astype(float)
        return out

    def _lasso_select(self, X_sub: pd.DataFrame, y_sub: pd.Series) -> list[str]:
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression(
            solver="liblinear",
            C=self.config.lasso_C,
            l1_ratio=1.0,
            max_iter=200,
            random_state=self.config.random_state,
        )
        clf.fit(X_sub.values, y_sub.values)
        coefs = clf.coef_[0]
        return [f for f, c in zip(X_sub.columns, coefs) if abs(c) > 1e-8]

    # ------------------------------------------------------------------
    def _prepare_for_lgbm(self, X: pd.DataFrame) -> pd.DataFrame:
        """LightGBM 입력 준비. 범주형 컬럼은 pandas ``category`` dtype 으로 변환."""
        out = X.copy()
        for col in self.categorical_columns:
            if col in out.columns:
                out[col] = out[col].astype("category")
        return out

    def _lgbm_select(self, X_sub: pd.DataFrame, y_sub: pd.Series) -> list[str]:
        from lightgbm import LGBMClassifier

        cat_idx = [
            i for i, c in enumerate(X_sub.columns) if c in self.categorical_columns
        ]
        model = LGBMClassifier(
            n_estimators=self.config.lgbm_n_estimators,
            learning_rate=self.config.lgbm_learning_rate,
            random_state=self.config.random_state,
            verbose=-1,
        )
        model.fit(
            X_sub, y_sub,
            categorical_feature=cat_idx if cat_idx else "auto",
        )
        importance = model.feature_importances_
        if importance.sum() == 0:
            return []
        order = np.argsort(-importance)
        # 상위 K + 양수 importance 만 선택 (0 importance 는 의미 없음)
        top_k = self.config.lgbm_top_k
        chosen = [
            X_sub.columns[i] for i in order[:top_k] if importance[i] > 0
        ]
        return chosen
