"""``_encode_for_lasso`` 의 PerformanceWarning 제거 및 동작 보존 회귀 테스트."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from auto_ml.config import FeatureSelectionConfig
from auto_ml.feature_selection.stability import StabilitySelector


def _make_wide_df(n_numeric: int = 380, n_cat: int = 20, n_rows: int = 200):
    """실무 규모(400 컬럼) 시뮬레이션. numeric 다수 + categorical 소수."""
    rng = np.random.default_rng(0)
    cols = {f"n{i}": rng.normal(size=n_rows) for i in range(n_numeric)}
    for i in range(n_cat):
        cols[f"c{i}"] = rng.choice(["a", "b", "c", "d"], size=n_rows)
    X = pd.DataFrame(cols)
    return X, [f"c{i}" for i in range(n_cat)]


def test_encode_for_lasso_no_fragmentation_warning():
    """400 컬럼 입력에서 PerformanceWarning 이 발생하지 않아야 한다."""
    X, cat_cols = _make_wide_df()
    cfg = FeatureSelectionConfig(enabled=True, base_estimator="lasso")
    selector = StabilitySelector(
        config=cfg,
        numeric_columns=[c for c in X.columns if c not in cat_cols],
        categorical_columns=cat_cols,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = selector._encode_for_lasso(X)

    perf = [w for w in caught if "fragmented" in str(w.message).lower()]
    assert not perf, (
        f"PerformanceWarning 이 여전히 발생: {[str(w.message) for w in perf]}"
    )
    # 출력 shape 보존
    assert out.shape == X.shape
    assert list(out.columns) == list(X.columns)


def test_encode_for_lasso_equivalent_to_old_pattern():
    """concat 패턴이 컬럼별 할당 패턴과 정확히 동일한 결과를 내는지."""
    X, cat_cols = _make_wide_df(n_numeric=10, n_cat=3, n_rows=100)
    cfg = FeatureSelectionConfig(enabled=True, base_estimator="lasso")
    selector = StabilitySelector(
        config=cfg,
        numeric_columns=[c for c in X.columns if c not in cat_cols],
        categorical_columns=cat_cols,
    )
    new_out = selector._encode_for_lasso(X)

    # 옛 패턴을 인라인으로 재현
    cat_set = set(cat_cols)
    old_out = pd.DataFrame(index=X.index)
    for col in X.columns:
        if col in cat_set:
            vc = X[col].value_counts(normalize=True, dropna=False)
            old_out[col] = X[col].map(vc).fillna(0.0).astype(float)
        else:
            old_out[col] = (
                pd.to_numeric(X[col], errors="coerce").fillna(0.0).astype(float)
            )

    # 값/컬럼/dtype 모두 동일해야 함
    pd.testing.assert_frame_equal(new_out, old_out)


def test_encode_for_lasso_empty_dataframe_safe():
    """비어 있는 X 도 안전하게 처리."""
    X = pd.DataFrame(index=range(5))
    cfg = FeatureSelectionConfig(enabled=True, base_estimator="lasso")
    selector = StabilitySelector(config=cfg, numeric_columns=[], categorical_columns=[])
    out = selector._encode_for_lasso(X)
    assert out.shape == (5, 0)
    assert list(out.index) == list(X.index)
