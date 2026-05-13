"""테스트 공통 fixture."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def tiny_binary_data() -> tuple[pd.DataFrame, pd.Series]:
    """분리 가능한 작은 합성 이진 데이터.

    수치형 2개 + 범주형 1개. 300 행. 양성 비율 ~50%. 모든 모델이 AUC > 0.70
    정도로 학습 가능한 정도의 신호를 가진다.
    """
    rng = np.random.default_rng(0)
    n = 300
    X = pd.DataFrame(
        {
            "num1": rng.normal(size=n),
            "num2": rng.normal(size=n),
            "cat1": rng.choice(["a", "b", "c"], size=n),
        }
    )
    logits = (
        0.9 * X["num1"].to_numpy()
        - 0.6 * X["num2"].to_numpy()
        + (X["cat1"] == "a").astype(float).to_numpy() * 0.8
    )
    p = 1.0 / (1.0 + np.exp(-logits))
    y = pd.Series((rng.uniform(size=n) < p).astype(int))
    return X, y
