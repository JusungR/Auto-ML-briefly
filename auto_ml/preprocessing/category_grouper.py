"""범주값 정리: 변수별로 남길 범주값만 남기고 나머지는 하나로 묶는다.

실무에서는 범주형 변수 중 특정 값만 의미가 있고, 나머지는 건수가 적거나 의미가
없는 경우가 많다. ``preprocessing.categorical_levels`` 로 변수마다 규칙을 정한다.

    region:  {keep: ["서울", "부산"]}   # 지정한 값만 남김
    channel: {min_count: 50}            # 학습 데이터에서 50 건 이상인 값만 남김

남기지 않은 값(추론 때 처음 보는 값 포함) 은 ``other_label`` (기본 ``"__other__"``)
로 바뀐다. 남길 값 목록은 학습 때 정해 저장하므로 추론에도 같은 규칙이 적용된다.

입력은 범주 정규화(``utils.categorical``) 와 결측 대체가 끝난 값이다. ``keep`` 에 적은
값도 같은 규칙으로 정규화해 비교하므로 YAML 에 ``1`` 로 적어도 ``"1"`` 과 맞는다.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from auto_ml.utils.categorical import _normalize_scalar
from auto_ml.utils.logger import get_logger

logger = get_logger("preprocessing.category_grouper")


class CategoryGrouper:
    """변수별 범주값 정리기.

    Attributes:
        rules: ``{컬럼: {"keep": [...]} | {"min_count": N}}``.
        other_label: 남기지 않은 값을 바꿀 이름.
        kept: 학습 후 컬럼별로 남긴 값 집합.
        summary_: 학습 데이터 기준 정리 요약 (로그·run_info 용).
    """

    def __init__(self, rules: dict[str, dict[str, Any]], other_label: str = "__other__") -> None:
        self.rules = {col: dict(rule) for col, rule in rules.items()}
        self.other_label = other_label
        self.kept: dict[str, frozenset] = {}
        self.seen: dict[str, frozenset] = {}  # 학습 데이터에 있던 값 (미학습 값 진단용)
        self.last_unseen_: dict[str, dict[str, Any]] = {}  # 직전 transform 의 미학습 값 통계
        self.summary_: dict[str, dict[str, Any]] = {}

    @property
    def columns(self) -> list[str]:
        return list(self.rules)

    def fit(self, df: pd.DataFrame) -> "CategoryGrouper":
        self.kept = {}
        self.seen = {}
        self.summary_ = {}
        for col, rule in self.rules.items():
            if col not in df.columns:
                raise ValueError(f"categorical_levels: column '{col}' is not in the data")
            counts = df[col].value_counts(dropna=True)
            if self.other_label in counts.index:
                raise ValueError(
                    f"categorical_levels: '{col}' already contains the value "
                    f"'{self.other_label}'. Set preprocessing.categorical_other_label to "
                    "a value that does not occur in the data."
                )
            if "keep" in rule:
                keep = {_normalize_scalar(v) for v in rule["keep"]}
                keep = {v for v in keep if isinstance(v, str)}  # 결측(None/NaN) 은 제외
                absent = sorted(keep - set(counts.index))
                if keep and not (keep & set(counts.index)):
                    raise ValueError(
                        f"categorical_levels: none of the keep values {sorted(keep)} of '{col}' "
                        f"occur in the training data (values include: {list(counts.index[:10])}). "
                        "Codes with leading zeros must be quoted in YAML — keep: [01] is read "
                        "as the number 1; write keep: [\"01\"]."
                    )
                if absent:
                    logger.warning(
                        "categorical_levels: values %s of '%s' do not occur in the training "
                        "data (typo?). Training values include: %s",
                        absent, col, list(counts.index[:10]),
                    )
                kept = frozenset(keep)
            else:
                kept = frozenset(counts.index[counts >= int(rule["min_count"])])
            self.kept[col] = kept
            self.seen[col] = frozenset(counts.index)

            n_rows = int(counts.sum())
            other_rows = int(counts[~counts.index.isin(kept)].sum())
            self.summary_[col] = {
                "rule": rule,
                "kept": sorted(kept & set(counts.index)),
                "n_grouped_values": int((~counts.index.isin(kept)).sum()),
                "other_rate": (other_rows / n_rows) if n_rows else 0.0,
            }
            if not (kept & set(counts.index)):
                logger.warning(
                    "categorical_levels: no value of '%s' is kept — the column becomes a "
                    "single value '%s'", col, self.other_label,
                )
            logger.info(
                "categorical_levels '%s': kept %d value(s) %s, grouped %d value(s) "
                "(%.1f%% of rows) into '%s'",
                col, len(self.summary_[col]["kept"]), self.summary_[col]["kept"][:10],
                self.summary_[col]["n_grouped_values"], 100 * self.summary_[col]["other_rate"],
                self.other_label,
            )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.kept:
            return df
        out = df.copy()
        self.last_unseen_ = {}
        for col, kept in self.kept.items():
            if col not in out.columns:
                continue
            s = out[col]
            stats = self._unseen(col, s)
            if stats:
                self.last_unseen_[col] = stats
            mask = s.notna() & ~s.isin(kept)
            if mask.any():
                out[col] = s.where(~mask, self.other_label)
        return out

    def _unseen(self, col: str, s: pd.Series) -> dict[str, Any] | None:
        """학습 때 없던 값의 통계 — 정리 후에는 other_label 에 섞여 구분되지 않으므로 정리 전에 센다.

        반환 형식은 ``PreprocessingPipeline.unseen_category_stats`` 와 같다 (그쪽에서 합쳐
        ``[test]`` / ``[score_input]`` 표시와 함께 로그로 남긴다).
        """
        seen = self.seen.get(col)
        if seen is None or len(s) == 0:
            return None
        mask = s.notna() & ~s.isin(seen)
        n = int(mask.sum())
        if n == 0:
            return None
        return {
            "n_unseen": n,
            "n_rows": len(s),
            "rate": n / len(s),
            "examples": list(s[mask].value_counts().index[:5]),
            "grouped_into": self.other_label,
        }

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)
