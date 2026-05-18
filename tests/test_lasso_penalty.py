"""Stability Selection 의 lasso base estimator 가 진짜 L1 로 작동하는지 검증.

이전 버그: penalty 인자 누락으로 sklearn 기본값 'l2' 가 적용되고 동시에
l1_ratio=1.0 이 넘어가 elasticnet 외에는 의미 없는 인자라는 UserWarning 이
매 부분표본·매 fit 마다 출력됐다. 또 실제 학습은 L2 라 'lasso' 가 sparse
선택을 하지 못했다.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from auto_ml.config import FeatureSelectionConfig
from auto_ml.feature_selection.stability import StabilitySelector


@pytest.fixture
def noisy_binary_data() -> tuple[pd.DataFrame, pd.Series]:
    """양수 신호 2개 + noise 8개 — L1 이면 noise 가 0 으로 떨어져야 한다."""
    rng = np.random.default_rng(0)
    n = 300
    X = pd.DataFrame(
        {
            "signal_a": rng.normal(size=n),
            "signal_b": rng.normal(size=n),
            **{f"noise_{i}": rng.normal(size=n) for i in range(8)},
        }
    )
    logits = 1.5 * X["signal_a"].to_numpy() - 1.2 * X["signal_b"].to_numpy()
    p = 1.0 / (1.0 + np.exp(-logits))
    y = pd.Series((rng.uniform(size=n) < p).astype(int))
    return X, y


def test_lasso_select_no_l1_ratio_elasticnet_warning(noisy_binary_data):
    """사용자 환경(sklearn 1.3~1.7) 에서 흔히 보이던 ``l1_ratio is only used when
    penalty is 'elasticnet'`` 경고가 더 이상 나오지 않아야 한다.

    sklearn 1.8 부터는 별도의 ``penalty deprecated`` 경고가 등장하지만 이는 본
    fix 의 범위가 아니라 (장기적으로 sklearn 새 API 로 마이그레이션 필요)
    구체 문구 매칭으로 본 회귀만 확인한다.
    """
    X, y = noisy_binary_data
    cfg = FeatureSelectionConfig(
        enabled=True, base_estimator="lasso",
        n_subsamples=5, subsample_ratio=0.7, threshold=0.5,
        lasso_C=0.1, random_state=0, min_selected=1,
    )
    selector = StabilitySelector(
        config=cfg, numeric_columns=list(X.columns), categorical_columns=[],
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        selector.fit_select(X, y)

    # 사용자가 신고한 정확한 메시지 패턴만 검사
    bad = [
        w for w in caught
        if "l1_ratio" in str(w.message) and "only used" in str(w.message)
        and "elasticnet" in str(w.message)
    ]
    assert not bad, (
        "구버전 sklearn 의 l1_ratio-elasticnet 경고가 여전히 발생: "
        f"{[str(w.message) for w in bad]}"
    )


def test_lasso_produces_sparse_selection(noisy_binary_data):
    """진짜 L1 면 충분히 작은 C (강한 규제) 에서 noise 가 0 으로 떨어진다."""
    X, y = noisy_binary_data
    # 강한 규제 (C 작음) → sparse
    cfg = FeatureSelectionConfig(
        enabled=True, base_estimator="lasso",
        n_subsamples=30, subsample_ratio=0.7, threshold=0.5,
        lasso_C=0.05, random_state=0, min_selected=1,
    )
    selector = StabilitySelector(
        config=cfg, numeric_columns=list(X.columns), categorical_columns=[],
    )
    result = selector.fit_select(X, y)

    # signal_* 의 frequency 가 noise_* 보다 명확히 높아야 한다 (sparse 효과의 핵심 증거)
    signal_freq = max(
        result.frequencies["signal_a"], result.frequencies["signal_b"]
    )
    noise_freq_max = max(
        result.frequencies[k] for k in result.frequencies if k.startswith("noise_")
    )
    assert signal_freq > noise_freq_max, (
        f"L1 이 sparse 하게 동작하지 않음: signal_max={signal_freq:.3f}, "
        f"noise_max={noise_freq_max:.3f}"
    )
