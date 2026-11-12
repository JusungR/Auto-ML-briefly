"""범주값 정리(preprocessing.categorical_levels) 와 모델별 원-핫 분할(categorical_encoding) 테스트.

규약:
- categorical_levels 는 변수별로 ``keep`` 목록 또는 ``min_count`` 로 남길 값을 정하고, 나머지
  (추론 때 처음 보는 값 포함) 는 ``categorical_other_label`` 로 묶는다. 학습 때 정한 목록을
  저장해 추론에도 같게 적용한다. 변수 이름은 바뀌지 않는다.
- categorical_encoding="onehot" 이면 트리 모델이 범주 하나씩("그 값이냐 아니냐") 분할한다.
  LightGBM·XGBoost 는 max_cat_to_onehot, CatBoost 는 one_hot_max_size 를 범주 수에 맞춰 넣고,
  fixed_params 에 직접 쓴 값이 우선한다.
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
import pytest
from test_versioning import _config

from auto_ml.config import AutoMLConfig, ModelConfig, PreprocessingConfig
from auto_ml.explain.explainer import Explainer
from auto_ml.models.catboost_model import CatBoostModel
from auto_ml.models.lgbm_model import LGBMModel
from auto_ml.models.registry import build_model
from auto_ml.models.xgb_model import XGBModel
from auto_ml.pipeline import AutoMLPipeline
from auto_ml.preprocessing.pipeline import PreprocessingPipeline
from auto_ml.scoring.scorer import Scorer


# ───────────────────────────────────────────────────────────────────────
# 설정 검증
# ───────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("rule", [
    {"keep": ["a"], "min_count": 3},   # 둘 다
    {"top": 3},                        # 모르는 키
    {"keep": []},                      # 빈 목록
    {"keep": "a"},                     # 리스트 아님
    {"min_count": 0},
    {"min_count": 2.5},
    "a",                               # mapping 아님
])
def test_invalid_categorical_levels_rule(rule):
    with pytest.raises(ValueError, match="categorical_levels"):
        PreprocessingConfig(categorical_levels={"cat1": rule})


def test_invalid_categorical_encoding():
    with pytest.raises(ValueError, match="categorical_encoding"):
        ModelConfig(categorical_encoding="dummy")


def test_yaml_keys_map_to_config():
    cfg = AutoMLConfig.from_dict({
        "preprocessing": {"categorical_levels": {"cat1": {"keep": ["a"]}},
                          "categorical_other_label": "OTHER"},
        "models": {"lgbm": {"categorical_encoding": "onehot"}},
    })
    assert cfg.preprocessing.categorical_levels == {"cat1": {"keep": ["a"]}}
    assert cfg.preprocessing.categorical_other_label == "OTHER"
    assert cfg.models["lgbm"].categorical_encoding == "onehot"


def test_levels_on_non_categorical_column_is_rejected():
    with pytest.raises(ValueError, match="not categorical features"):
        PreprocessingPipeline(
            ["num1"], ["cat1"],
            PreprocessingConfig(categorical_levels={"num1": {"keep": ["1"]}}),
        )


# ───────────────────────────────────────────────────────────────────────
# 범주값 정리
# ───────────────────────────────────────────────────────────────────────

def _pipe(levels, **kw) -> PreprocessingPipeline:
    return PreprocessingPipeline(
        ["num1"], ["cat", "code"],
        PreprocessingConfig(categorical_levels=levels, **kw),
    )


def _frame(cat, code=None) -> pd.DataFrame:
    n = len(cat)
    return pd.DataFrame({
        "num1": np.arange(n, dtype=float),
        "cat": cat,
        "code": code if code is not None else ["x"] * n,
    })


def test_keep_and_min_count():
    train = _frame(
        ["서울"] * 5 + ["부산"] * 3 + ["대구"] * 2 + ["광주"],
        [1.0] * 6 + [2.0] * 4 + [3.0],                       # 결측 섞인 정수 코드 → float
    )
    pipe = _pipe({"cat": {"keep": ["서울", "부산"]}, "code": {"min_count": 4}})
    out = pipe.fit_transform(train)
    assert set(out["cat"]) == {"서울", "부산", "__other__"}
    assert (out["cat"] == "__other__").sum() == 3
    assert set(out["code"]) == {"1", "2", "__other__"}      # 3 은 1건이라 묶임

    score = pipe.transform(_frame(["서울", "제주", "대구"], [1, 3, 9]))   # int 로 들어와도 같은 값
    assert list(score["cat"]) == ["서울", "__other__", "__other__"]
    assert list(score["code"]) == ["1", "__other__", "__other__"]

    summary = pipe.category_grouping_summary()
    assert summary["cat"]["kept"] == ["부산", "서울"]
    assert summary["cat"]["n_grouped_values"] == 2
    assert summary["code"]["other_rate"] == pytest.approx(1 / 11)


def test_keep_matches_numeric_codes_written_in_yaml():
    pipe = _pipe({"cat": {"keep": [1, 2.0]}})               # YAML 에 숫자로 적은 경우
    out = pipe.fit_transform(_frame([1, 2, 3, 1]))
    assert list(out["cat"]) == ["1", "2", "__other__", "1"]


def test_custom_other_label_and_collision():
    pipe = _pipe({"cat": {"keep": ["a"]}}, categorical_other_label="기타")
    assert set(pipe.fit_transform(_frame(["a", "b"]))["cat"]) == {"a", "기타"}
    with pytest.raises(ValueError, match="already contains"):
        _pipe({"cat": {"keep": ["a"]}}).fit(_frame(["a", "__other__"]))


def test_missing_values_are_imputed_then_grouped():
    pipe = _pipe({"cat": {"keep": ["a"]}},
                 categorical_null_strategy="constant", categorical_null_fill_value="MISSING")
    out = pipe.fit_transform(_frame(["a", None, "b", "a"]))
    assert list(out["cat"]) == ["a", "__other__", "__other__", "a"]   # MISSING 도 묶임


def test_typo_in_keep_and_unseen_values_are_logged(caplog):
    caplog.set_level(logging.INFO)
    pipe = _pipe({"cat": {"keep": ["a", "zz"]}})
    pipe.fit(_frame(["a", "b", "a"]))
    assert any("zz" in r.getMessage() and "typo" in r.getMessage() for r in caplog.records)
    caplog.clear()
    pipe.transform(_frame(["a", "new"]))
    assert any("unseen categories in 'cat'" in r.getMessage() for r in caplog.records)


def test_pipeline_without_grouper_attribute_still_works():
    """기능 이전에 저장된 전처리기(_grouper 속성 없음) 호환."""
    pipe = _pipe({})
    pipe.fit(_frame(["a", "b"]))
    del pipe._grouper
    assert list(pipe.transform(_frame(["a", "c"]))["cat"]) == ["a", "c"]
    assert pipe.category_grouping_summary() == {}


# ───────────────────────────────────────────────────────────────────────
# 모델별 원-핫 분할
# ───────────────────────────────────────────────────────────────────────

def _six_level_data(n: int = 3000):
    rng = np.random.default_rng(0)
    c = rng.choice(list("ABCDEF"), n)
    effect = {"A": 1.5, "B": 1.0, "C": -1.0, "D": -1.2, "E": 0.3, "F": 0.0}
    logit = np.vectorize(effect.get)(c)
    y = pd.Series((rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int))
    X = pd.DataFrame({"c": c, "z": rng.normal(size=n)})
    return X, y


@pytest.mark.parametrize("name,key,expected", [
    ("lgbm", "max_cat_to_onehot", 7),       # LightGBM·XGBoost 는 범주 수 + 1 부터 하나씩 분할
    ("xgb", "max_cat_to_onehot", 7),
    ("catboost", "one_hot_max_size", 6),
])
def test_onehot_params(name, key, expected):
    X, _ = _six_level_data(200)
    assert build_model(name, categorical_columns=["c"],
                       categorical_encoding="onehot")._onehot_params(X) == {key: expected}
    assert build_model(name, categorical_columns=["c"])._onehot_params(X) == {}
    user = build_model(name, params={key: 2}, categorical_columns=["c"],
                       categorical_encoding="onehot")
    assert user._onehot_params(X) == {}                      # 사용자 지정이 우선


def test_catboost_onehot_cap():
    X = pd.DataFrame({"c": [f"v{i}" for i in range(300)]})
    m = build_model("catboost", categorical_columns=["c"], categorical_encoding="onehot")
    assert m._onehot_params(X) == {"one_hot_max_size": 255}


def _max_categories_per_split(model) -> int:
    if isinstance(model, LGBMModel):
        t = model.model.booster_.trees_to_dataframe()
        thr = t.loc[t["split_feature"] == "c", "threshold"].astype(str)
        return int(thr.str.count(r"\|\|").max()) + 1
    t = model.model.get_booster().trees_to_dataframe()
    cats = t.loc[t["Feature"] == "c", "Category"]
    return max(len(v) for v in cats if isinstance(v, list))


@pytest.mark.parametrize("cls,params", [
    (LGBMModel, {"n_estimators": 30, "verbose": -1}),
    (XGBModel, {"n_estimators": 30}),
])
def test_onehot_makes_single_category_splits(cls, params):
    X, y = _six_level_data()
    native = cls(params=dict(params), categorical_columns=["c"]).fit(X, y)
    onehot = cls(params=dict(params), categorical_columns=["c"]) \
        .set_categorical_encoding("onehot").fit(X, y)
    assert _max_categories_per_split(native) > 1       # 기본은 여러 범주를 묶어 분할
    assert _max_categories_per_split(onehot) == 1


def _catboost_split_types(model: CatBoostModel, tmp_path) -> set[str]:
    path = tmp_path / "cb.json"
    model.model.save_model(str(path), format="json")
    trees = json.loads(path.read_text())["oblivious_trees"]
    return {s["split_type"] for t in trees for s in (t.get("splits") or [])}


def test_catboost_onehot_uses_one_hot_splits(tmp_path):
    X, y = _six_level_data()
    params = {"iterations": 30, "verbose": False, "allow_writing_files": False}
    native = CatBoostModel(params=dict(params), categorical_columns=["c"]).fit(X, y)
    onehot = CatBoostModel(params=dict(params), categorical_columns=["c"]) \
        .set_categorical_encoding("onehot").fit(X, y)
    assert "OnlineCtr" in _catboost_split_types(native, tmp_path)   # 기본은 범주형 통계(CTR)
    types = _catboost_split_types(onehot, tmp_path)
    assert "OneHotFeature" in types and "OnlineCtr" not in types


# ───────────────────────────────────────────────────────────────────────
# 학습 → 스코어링 → SHAP
# ───────────────────────────────────────────────────────────────────────

def test_end_to_end_with_levels_and_onehot(tmp_path, tiny_binary_data):
    cfg = _config(tmp_path, tiny_binary_data)
    cfg.preprocessing = PreprocessingConfig(categorical_levels={"cat1": {"keep": ["a"]}})
    cfg.models["lgbm"].categorical_encoding = "onehot"
    cfg.models["xgb"].categorical_encoding = "onehot"
    out = AutoMLPipeline(cfg).run()

    info = json.loads((out["run_dir"] / "run_info.json").read_text(encoding="utf-8"))
    assert info["categorical_levels"]["cat1"]["kept"] == ["a"]

    best = out["artifact"]
    scorer = Scorer.from_artifact(best)
    assert scorer.metadata.extra["categorical_encoding"] in ("onehot", None)  # ensemble 은 None
    df = pd.read_parquet(cfg.scoring.input_path)
    df.loc[df.index[:3], "cat1"] = "처음보는값"
    scores = scorer.score(df)["score"].to_numpy()
    assert np.all((scores >= 0) & (scores <= 1))

    explainer = Explainer.from_artifact(best)
    shap_df = explainer.explain(df)
    shap_cols = [c for c in shap_df.columns if c.startswith("shap_")]
    assert "shap_cat1" in shap_cols                        # 변수 단위 유지
    np.testing.assert_allclose(
        shap_df[shap_cols].sum(axis=1) + shap_df["base_value"], shap_df["score"], atol=1e-6,
    )
    np.testing.assert_allclose(shap_df["score"].to_numpy(), scores, atol=1e-6)
