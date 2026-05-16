"""Degenerate 컬럼 탐지 / 경고 단위 테스트."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from auto_ml.utils.io import find_degenerate_columns, warn_degenerate_columns


@pytest.fixture
def df_with_degenerates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "good_num": [1.0, 2.0, 3.0, 4.0],
            "good_cat": ["a", "b", "a", "c"],
            "all_null": [np.nan, np.nan, np.nan, np.nan],
            "all_zero": [0.0, 0.0, 0.0, 0.0],
            "zero_with_nan": [0.0, np.nan, 0.0, np.nan],   # 모든 non-NaN 이 0
            "constant_string": ["X", "X", "X", "X"],
            "constant_num": [5.0, 5.0, 5.0, 5.0],
        }
    )


def test_find_categorizes_correctly(df_with_degenerates):
    found = find_degenerate_columns(df_with_degenerates)
    assert set(found["all_null"]) == {"all_null"}
    assert set(found["all_zero"]) == {"all_zero", "zero_with_nan"}
    assert set(found["constant_nonzero"]) == {"constant_string", "constant_num"}
    # good 컬럼은 어디에도 없음
    all_flagged = set(found["all_null"]) | set(found["all_zero"]) | set(found["constant_nonzero"])
    assert "good_num" not in all_flagged
    assert "good_cat" not in all_flagged


def test_find_respects_columns_filter(df_with_degenerates):
    # 일부 컬럼만 검사
    found = find_degenerate_columns(df_with_degenerates, columns=["good_num", "all_null"])
    assert found["all_null"] == ["all_null"]
    # all_zero 등은 검사 범위 밖
    assert found["all_zero"] == []
    assert found["constant_nonzero"] == []


def test_find_missing_column_silently_skipped(df_with_degenerates):
    """columns 에 존재하지 않는 이름이 있어도 조용히 skip 한다."""
    found = find_degenerate_columns(
        df_with_degenerates, columns=["all_null", "not_in_df"]
    )
    assert found["all_null"] == ["all_null"]


def test_warn_emits_warnings(df_with_degenerates, caplog):
    caplog.set_level(logging.WARNING)
    logger = logging.getLogger("test_warn")
    out = warn_degenerate_columns(
        df_with_degenerates, logger=logger, context="train",
    )
    # 결과 동일
    assert "all_null" in out["all_null"]
    # 경고 메시지에 각 범주가 포함되어야 함
    msgs = " ".join(rec.message for rec in caplog.records)
    assert "All-NaN columns" in msgs
    assert "All-zero columns" in msgs
    assert "Constant columns" in msgs
    assert "train" in msgs  # context prefix


def test_warn_no_logger_returns_silently(df_with_degenerates, caplog):
    caplog.set_level(logging.WARNING)
    out = warn_degenerate_columns(df_with_degenerates, logger=None)
    assert out["all_null"] == ["all_null"]
    # 로거가 없으므로 경고 없음
    assert len(caplog.records) == 0


def test_warn_clean_dataframe_no_warning(caplog):
    """모든 컬럼이 정상이면 경고가 나지 않는다."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    caplog.set_level(logging.WARNING)
    logger = logging.getLogger("test_warn_clean")
    out = warn_degenerate_columns(df, logger=logger)
    assert all(len(v) == 0 for v in out.values())
    assert len(caplog.records) == 0
