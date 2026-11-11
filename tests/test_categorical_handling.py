"""범주형 처리 회귀 테스트.

규약:
- 범주형 값은 전처리에서 같은 규칙의 문자열로 정규화된다 (1 / 1.0 / "1" → "1").
  학습 데이터에 결측이 있어 float 로 읽히고 추론 데이터가 int 로 들어와도,
  모든 모델이 같은 예측을 낸다.
- 결측 대체값과 원래 값의 타입이 섞이지 않는다.
- XGBoost 는 native 범주형으로 학습하며, 추론 입력의 범주 구성과 무관하게
  같은 매핑을 쓰고, 미학습 범주는 결측으로 처리한다.
- 학습 때 없던 범주 비율은 추론 시 로그로 남는다.
- 정규화 이전에 학습된 artifact 는 기존 방식으로 동작한다.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from auto_ml.config import PreprocessingConfig
from auto_ml.models.catboost_model import CatBoostModel
from auto_ml.models.elasticnet_model import ElasticNetModel
from auto_ml.models.lgbm_model import LGBMModel
from auto_ml.models.xgb_model import XGBModel
from auto_ml.preprocessing import PreprocessingPipeline
from auto_ml.scoring.scorer import Scorer
from auto_ml.utils.categorical import normalize_category_series
from auto_ml.utils.io import ArtifactMetadata

ROOT = Path(__file__).resolve().parents[1]


def test_normalize_category_series_rules():
    s = pd.Series([1.0, "1.0", "a", None, 2, True, 1.5, np.nan], dtype=object)
    assert list(normalize_category_series(s).fillna("<NA>")) == [
        "1", "1.0", "a", "<NA>", "2", "True", "1.5", "<NA>",
    ]
    assert list(normalize_category_series(pd.Series([1.0, 2.0, np.nan])).fillna("<NA>")) == [
        "1", "2", "<NA>",
    ]
    assert list(normalize_category_series(pd.Series([1, 2], dtype="Int64"))) == ["1", "2"]
    assert list(
        normalize_category_series(pd.Series(["a", None]).astype("category")).fillna("<NA>")
    ) == ["a", "<NA>"]


def _numeric_code_data(n: int = 800):
    """정수 코드 범주형 + 결측 — 학습 컬럼이 float 로 읽히는 상황."""
    rng = np.random.default_rng(0)
    code = rng.choice([1, 2, 3], size=n).astype(float)
    x = rng.normal(size=n)
    y = pd.Series(((code == 3) | (x > 1.2)).astype(int))
    code[rng.random(n) < 0.05] = np.nan
    return pd.DataFrame({"x": x, "code": code}), y


def test_imputed_values_share_one_type():
    X, _ = _numeric_code_data()
    pp = PreprocessingPipeline(["x"], ["code"], PreprocessingConfig())
    out = pp.fit_transform(X)
    assert set(out["code"]) == {"1", "2", "3"}


@pytest.mark.parametrize(
    "model",
    [
        XGBModel(params={"n_estimators": 40}, categorical_columns=["code"]),
        LGBMModel(params={"n_estimators": 40, "verbose": -1}, categorical_columns=["code"]),
        CatBoostModel(
            params={"iterations": 40, "verbose": False, "allow_writing_files": False},
            categorical_columns=["code"],
        ),
        ElasticNetModel(params={}, categorical_columns=["code"]),
    ],
    ids=["xgb", "lgbm", "catboost", "elasticnet"],
)
def test_int_float_str_inference_inputs_agree(model):
    X, y = _numeric_code_data()
    pp = PreprocessingPipeline(["x"], ["code"], PreprocessingConfig())
    model.fit(pp.fit_transform(X), y)

    as_float = pd.DataFrame({"x": [0.0] * 3, "code": [1.0, 2.0, 3.0]})
    as_int = pd.DataFrame({"x": [0.0] * 3, "code": [1, 2, 3]})
    as_str = pd.DataFrame({"x": [0.0] * 3, "code": ["1", "2", "3"]})
    p_float = model.predict_proba(pp.transform(as_float))
    np.testing.assert_allclose(model.predict_proba(pp.transform(as_int)), p_float)
    np.testing.assert_allclose(model.predict_proba(pp.transform(as_str)), p_float)
    # 범주 정보가 실제로 쓰였는지 (code=3 이 양성 신호)
    assert p_float[2] > p_float[0] + 0.3


def test_xgb_native_categorical_mapping(tiny_binary_data):
    X, y = tiny_binary_data
    m = XGBModel(params={"n_estimators": 40}, categorical_columns=["cat1"]).fit(X, y)
    assert m._categories == {"cat1": ["a", "b", "c"]}
    assert m.model.get_params()["enable_categorical"] is True

    # 추론 입력의 범주 구성이 달라도 같은 범주는 같은 예측
    alone = pd.DataFrame({"num1": [0.0], "num2": [0.0], "cat1": ["c"]})
    mixed = pd.DataFrame({"num1": [0.0, 0.0], "num2": [0.0, 0.0], "cat1": ["a", "c"]})
    assert m.predict_proba(alone)[0] == pytest.approx(m.predict_proba(mixed)[1])

    # 미학습 범주는 결측과 같게 처리
    unseen = pd.DataFrame({"num1": [0.0, 0.0], "num2": [0.0, 0.0], "cat1": ["zzz", None]})
    p = m.predict_proba(unseen)
    assert p[0] == pytest.approx(p[1])


def test_unseen_category_stats_and_scorer_log(caplog):
    X, y = _numeric_code_data()
    pp = PreprocessingPipeline(["x"], ["code"], PreprocessingConfig())
    X_p = pp.fit_transform(X)
    model = XGBModel(params={"n_estimators": 20}, categorical_columns=["code"]).fit(X_p, y)

    q = pd.DataFrame({"x": [0.0] * 4, "code": [1, 7, 7, np.nan]})
    stats = pp.unseen_category_stats(pp.transform(q))
    assert stats["code"]["n_unseen"] == 2
    assert stats["code"]["examples"] == ["7"]

    meta = ArtifactMetadata(
        target_column="target", feature_columns=["x", "code"], categorical_columns=["code"],
        id_columns=[], model_name="xgb", primary_metric="roc_auc", metric_value=0.0,
        selected_features=["x", "code"],
    )
    with caplog.at_level(logging.INFO, logger="auto_ml"):
        Scorer(pp, model, meta).score(q)
    msgs = [r for r in caplog.records if "unseen categories in 'code'" in r.getMessage()]
    assert msgs and msgs[0].levelno == logging.WARNING  # 50% ≥ 1%


@pytest.mark.parametrize(
    "path",
    ["artifacts/credit/models/models/xgb.joblib", "artifacts/credit/models/best.joblib"],
)
def test_legacy_artifacts_keep_old_behavior(path):
    """정규화·native 범주형 이전에 학습된 artifact 는 기존 경로로 예측한다."""
    artifact = ROOT / path
    score_input = ROOT / "examples/credit/data/score_input.parquet"
    if not artifact.exists() or not score_input.exists():
        pytest.skip("legacy artifact not present")
    scorer = Scorer.from_artifact(artifact)
    assert not hasattr(scorer.preprocessor, "_normalize_categories")
    if isinstance(scorer.model, XGBModel):
        assert getattr(scorer.model, "_categories", None) is None
        assert scorer.model._encoders  # LabelEncoder 경로
    out = scorer.score(pd.read_parquet(score_input).head(200))
    assert out["score"].between(0, 1).all()


def test_normalize_large_integer_codes_are_exact():
    """2^53 을 넘는 정수 코드도 타입(int64 / Int64 / object) 과 무관하게 같은 문자열."""
    big = [9007199254740993, 9007199254740992, 12345678901234567]
    expected = ["9007199254740993", "9007199254740992", "12345678901234567"]
    assert list(normalize_category_series(pd.Series(big, dtype="int64"))) == expected
    assert list(normalize_category_series(pd.Series(big + [None], dtype="Int64")))[:3] == expected
    assert list(normalize_category_series(pd.Series(big + [None], dtype=object)))[:3] == expected
    assert list(normalize_category_series(pd.Series([2**64 - 1], dtype="uint64"))) == [
        "18446744073709551615"
    ]


def test_normalize_huge_float_does_not_overflow():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)  # int64 오버플로 캐스트 경고도 실패로
        out = list(normalize_category_series(pd.Series([1e20, 2e20, 1.0])))
    assert out == ["100000000000000000000", "200000000000000000000", "1"]
