"""전처리 파이프라인: 결측 → 이상치 → skew 변환 → 스케일링 순서로 실행.

본 파이프라인은 학습/스코어링 양 단계에서 동일하게 사용되며, 학습 시
저장된 통계량을 스코어링 시 그대로 재사용해 일관성을 보장한다.

범주형 컬럼은 본 파이프라인에서 값 정규화(``utils.categorical``) 와
결측 채움까지만 수행한다. 정규화는 ``1`` / ``1.0`` / ``"1.0"`` 처럼 같은
범주가 타입 차이로 갈라지는 것을 막기 위해 모든 값을 같은 규칙의 문자열로
맞춘다. ``categorical_levels`` 가 있으면 결측 대체 직후 변수별로 남길 값만 남기고
나머지를 ``categorical_other_label`` 로 묶는다 (``CategoryGrouper``). 인코딩은 모델 래퍼 내부에서 처리한다 — LightGBM/CatBoost/XGBoost 는
native 처리, ElasticNet 은 One-hot. 이렇게 분리하면 단일 전처리 결과를
모든 모델이 공통으로 받아 사용할 수 있다.

학습 시 본 범주 목록을 보관해 추론 입력의 미학습 범주 비율을 진단한다
(``unseen_category_stats``).
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from auto_ml.config import PreprocessingConfig
from auto_ml.preprocessing.category_grouper import CategoryGrouper
from auto_ml.preprocessing.imputer import NullImputer
from auto_ml.preprocessing.outlier import OutlierHandler
from auto_ml.preprocessing.scaler import NumericScaler
from auto_ml.preprocessing.skew import SkewTransformer
from auto_ml.utils.categorical import normalize_categories


class PreprocessingPipeline:
    """4단계(결측 → 이상치 → skew → 스케일링) 전처리를 묶어 실행하는 컨테이너.

    Attributes:
        numeric_columns: 수치형 컬럼.
        categorical_columns: 범주형 컬럼.
        config: 단계별 옵션 모음.
        imputer / outlier / skew / scaler: 단계별 인스턴스.
            디버깅·테스트 시 직접 접근 가능.
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
        # 범주값 정리 (선택). 대상은 사용 중인 범주형 변수여야 한다.
        levels = dict(getattr(config, "categorical_levels", None) or {})
        bad = [c for c in levels if c not in self.categorical_columns]
        if bad:
            raise ValueError(
                f"preprocessing.categorical_levels: {bad} are not categorical features in use "
                f"(categorical features: {self.categorical_columns})"
            )
        self._grouper: CategoryGrouper | None = (
            CategoryGrouper(levels, getattr(config, "categorical_other_label", "__other__"))
            if levels else None
        )
        self.outlier = OutlierHandler(
            numeric_columns=self.numeric_columns,
            method=config.outlier_method,
            lower_quantile=config.outlier_lower_quantile,
            upper_quantile=config.outlier_upper_quantile,
            action=config.outlier_action,
        )
        self.skew = SkewTransformer(
            numeric_columns=self.numeric_columns,
            method=config.skew_method,
            threshold=config.skew_threshold,
        )
        self.scaler = NumericScaler(
            numeric_columns=self.numeric_columns,
            method=config.scaling_method,
        )
        # 스코어링 입력에서 컬럼 자체가 누락된 경우 채워 넣을 기본값.
        # 결측 처리 전략(median/mean/constant) 과는 별개로 항상 수치형은 median,
        # 범주형은 최빈값을 학습 시점에 산출해 보관한다.
        self._numeric_defaults: dict[str, float] = {}
        self._categorical_defaults: dict[str, Any] = {}
        # 범주형 값 정규화 여부. 이 속성이 없는 과거 artifact 는 정규화 없이
        # 학습됐으므로 getattr(..., False) 로 기존 동작을 유지한다.
        self._normalize_categories = True
        # 학습 데이터에서 본 범주 (정규화·결측 대체 후). 미학습 범주 진단용.
        self._categorical_levels: dict[str, frozenset] = {}
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> "PreprocessingPipeline":
        """각 단계를 순서대로 fit 한다.

        Note:
            각 단계는 이전 단계의 출력을 학습해야 통계가 왜곡되지 않는다
            (예: skew 측정은 이상치가 클립된 데이터에서, 스케일러는 skew
            변환된 데이터에서). 그래서 ``fit_transform`` 으로 순차 적용한다.
        """
        df = self._normalize(df)
        self._fit_column_defaults(df)
        step1 = self.imputer.fit_transform(df)
        if self._grouper is not None:
            step1 = self._grouper.fit_transform(step1)
        self._categorical_levels = {
            col: frozenset(step1[col].dropna().unique())
            for col in self.categorical_columns
            if col in step1.columns
        }
        step2 = self.outlier.fit_transform(step1)
        step3 = self.skew.fit_transform(step2)
        self.scaler.fit(step3)
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """학습 시 저장된 상태로 동일 순서의 변환을 수행한다."""
        if not self._fitted:
            raise RuntimeError("PreprocessingPipeline must be fit before transform.")
        x = self._normalize(df)
        x = self.imputer.transform(x)
        grouper = getattr(self, "_grouper", None)  # 과거 artifact 에는 속성이 없다
        grouped_unseen: dict[str, dict[str, Any]] = {}
        if grouper is not None:
            x = grouper.transform(x)
            grouped_unseen = dict(grouper.last_unseen_)
        x = self.outlier.transform(x)
        x = self.skew.transform(x)
        x = self.scaler.transform(x)
        # 정리 대상 컬럼의 미학습 값은 정리 후 결과로는 알 수 없으므로, 이 결과 객체에 대한
        # unseen_category_stats 호출에서 함께 돌려준다 (다른 DataFrame 에는 쓰지 않음).
        self._grouped_unseen = (id(x), grouped_unseen)
        return x

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """fit 직후 동일 데이터에 transform 을 적용해 반환한다."""
        return self.fit(df).transform(df)

    # ------------------------------------------------------------------
    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        if not getattr(self, "_normalize_categories", False):
            return df
        return normalize_categories(df, self.categorical_columns)

    # ------------------------------------------------------------------
    def category_grouping_summary(self) -> dict[str, dict[str, Any]]:
        """범주값 정리 요약 (컬럼별 남긴 값·묶인 값 수·묶인 행 비율). 정리가 없으면 빈 dict."""
        grouper = getattr(self, "_grouper", None)
        return dict(grouper.summary_) if grouper is not None else {}

    # ------------------------------------------------------------------
    def unseen_category_stats(
        self, df: pd.DataFrame,
    ) -> dict[str, dict[str, Any]]:
        """``transform`` 결과에서 학습 때 없던 범주의 비율을 컬럼별로 반환한다.

        Args:
            df: ``transform`` 을 거친 DataFrame.

        Returns:
            ``{col: {"n_unseen", "n_rows", "rate", "examples"}}`` — 미학습 범주가
            1건 이상인 컬럼만 포함. 범주 목록을 보관하지 않은 과거 artifact 는 빈 dict.
        """
        levels = getattr(self, "_categorical_levels", None) or {}
        stats: dict[str, dict[str, Any]] = {}
        n_rows = len(df)
        if n_rows == 0:
            return stats
        out_id, grouped = getattr(self, "_grouped_unseen", (None, {}))
        if out_id == id(df):
            stats.update(grouped)
            levels = {c: v for c, v in levels.items() if c not in grouped}
        for col, seen in levels.items():
            if col not in df.columns:
                continue
            s = df[col]
            mask = s.notna() & ~s.isin(seen)
            n_unseen = int(mask.sum())
            if n_unseen == 0:
                continue
            stats[col] = {
                "n_unseen": n_unseen,
                "n_rows": n_rows,
                "rate": n_unseen / n_rows,
                "examples": list(s[mask].value_counts().index[:5]),
            }
        return stats

    # ------------------------------------------------------------------
    def fill_missing_columns(
        self, df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, list[str]]:
        """입력 DataFrame 에서 학습 시 본 컬럼이 누락됐다면 기본값으로 채워 반환.

        스코어링 단계 ``transform`` 호출 전에 별도로 부르도록 설계되었다.
        결측 셀(부분 NaN) 은 후속 ``imputer.transform`` 에서 처리된다 —
        이 메서드는 어디까지나 컬럼 자체가 없는 케이스만 다룬다.

        Returns:
            (새 DataFrame, 채워 넣은 컬럼명 목록).
        """
        # 과거 artifact 와의 호환을 위해 getattr 로 방어
        numeric_defaults = getattr(self, "_numeric_defaults", {})
        categorical_defaults = getattr(self, "_categorical_defaults", {})

        out = df.copy()
        filled: list[str] = []
        for col, value in numeric_defaults.items():
            if col not in out.columns:
                out[col] = value
                filled.append(col)
        for col, value in categorical_defaults.items():
            if col not in out.columns:
                out[col] = value
                filled.append(col)
        return out, filled

    # ------------------------------------------------------------------
    def _fit_column_defaults(self, df: pd.DataFrame) -> None:
        """학습 데이터에서 컬럼별 누락-대비 기본값(median / 최빈값) 을 산출."""
        self._numeric_defaults = {}
        for col in self.numeric_columns:
            if col not in df.columns:
                continue
            med = df[col].median()
            # 컬럼 전체가 NaN 인 극단 케이스는 설정의 constant fallback 사용
            self._numeric_defaults[col] = (
                float(med) if pd.notna(med) else float(self.config.numeric_null_fill_value)
            )

        self._categorical_defaults = {}
        for col in self.categorical_columns:
            if col not in df.columns:
                continue
            mode = df[col].mode(dropna=True)
            self._categorical_defaults[col] = (
                mode.iloc[0] if not mode.empty else self.config.categorical_null_fill_value
            )
