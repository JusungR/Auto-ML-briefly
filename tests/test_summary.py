"""``summarize_dataframe`` 단위 테스트."""
from __future__ import annotations

import numpy as np
import pandas as pd

from auto_ml.utils.io import summarize_dataframe


def test_summary_without_target():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [None, "x", "y"]})
    s = summarize_dataframe(df)
    assert "3 rows × 2 cols" in s
    assert "target" not in s  # target 절 누락
    # missing 1/6 = 16.67%
    assert "missing 16.67%" in s


def test_summary_with_target():
    df = pd.DataFrame(
        {
            "f1": [1.0, 2.0, np.nan, 4.0],
            "default": [0, 1, 1, 0],
        }
    )
    s = summarize_dataframe(df, target_column="default")
    assert "4 rows × 2 cols" in s
    assert "target 'default': 2/4 positive (50.00%)" in s
    # 1 NaN out of 8 cells = 12.5%
    assert "missing 12.50%" in s


def test_summary_target_column_missing_is_silent():
    """target_column 이 df 에 없으면 target 절을 조용히 생략."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    s = summarize_dataframe(df, target_column="not_there")
    assert "3 rows × 1 cols" in s
    assert "target" not in s


def test_summary_thousands_separator_for_large():
    df = pd.DataFrame({"x": np.zeros(12345, dtype=int)})
    s = summarize_dataframe(df)
    assert "12,345 rows" in s
