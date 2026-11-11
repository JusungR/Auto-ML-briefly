"""핵심 버그 수정 회귀 테스트.

규약:
- XGBoost 의 SHAP 합(raw margin)은 조기 종료 후에도 실제 예측과 일치한다.
- ``best_iteration`` 은 세 부스팅 모델 모두 "best 지점까지의 트리 개수" 이며,
  조기 종료가 동작하지 않으면 None 이다 (LightGBM 의 0 포함).
- PDF 생성이 실패해도 학습 결과(artifact)와 HTML 리포트는 저장된다.
- 타깃에 결측이 있으면 명확한 메시지로 거부한다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import auto_ml.reporting.report as report_mod
from auto_ml.config import (
    AutoMLConfig,
    FeatureSpec,
    LoggingConfig,
    ModelConfig,
    ReportingConfig,
    TrainingConfig,
    TuningConfig,
)
from auto_ml.models.catboost_model import CatBoostModel
from auto_ml.models.lgbm_model import LGBMModel
from auto_ml.models.xgb_model import XGBModel
from auto_ml.pipeline import AutoMLPipeline
from auto_ml.utils.validation import validate_binary_target


def _split(tiny_binary_data):
    X, y = tiny_binary_data
    return X.iloc[:200], y.iloc[:200], X.iloc[200:], y.iloc[200:]


def test_xgb_shap_matches_prediction_after_early_stopping(tiny_binary_data):
    Xt, yt, Xv, yv = _split(tiny_binary_data)
    m = XGBModel(
        params={"n_estimators": 500, "learning_rate": 0.3}, categorical_columns=["cat1"],
    ).fit(Xt, yt, Xv, yv, early_stopping_rounds=20)
    n_trees = m.model.get_booster().num_boosted_rounds()
    assert m.best_iteration < n_trees  # 조기 종료 이후 트리가 남아 있는 상황

    shap_sum = m.shap_values(Xv).sum(axis=1)
    p = m.predict_proba(Xv)
    margin = np.log(p / (1 - p))
    np.testing.assert_allclose(shap_sum, margin, atol=1e-4)


def test_best_iteration_is_tree_count(tiny_binary_data):
    Xt, yt, Xv, yv = _split(tiny_binary_data)
    xgbm = XGBModel(
        params={"n_estimators": 500, "learning_rate": 0.3}, categorical_columns=["cat1"],
    ).fit(Xt, yt, Xv, yv, early_stopping_rounds=20)
    assert xgbm.best_iteration == xgbm.model.best_iteration + 1

    cat = CatBoostModel(
        params={"iterations": 500, "learning_rate": 0.3, "verbose": False,
                "allow_writing_files": False},
        categorical_columns=["cat1"],
    ).fit(Xt, yt, Xv, yv, early_stopping_rounds=20)
    assert cat.best_iteration == cat.model.tree_count_  # use_best_model 로 잘린 트리 수

    lgb = LGBMModel(
        params={"n_estimators": 500, "learning_rate": 0.3, "verbose": -1},
        categorical_columns=["cat1"],
    ).fit(Xt, yt, Xv, yv, early_stopping_rounds=20)
    assert lgb.best_iteration == lgb.model.booster_.best_iteration >= 1


@pytest.mark.parametrize(
    "model",
    [
        LGBMModel(params={"n_estimators": 20, "verbose": -1}, categorical_columns=["cat1"]),
        XGBModel(params={"n_estimators": 20}, categorical_columns=["cat1"]),
        CatBoostModel(
            params={"iterations": 20, "verbose": False, "allow_writing_files": False},
            categorical_columns=["cat1"],
        ),
    ],
    ids=["lgbm", "xgb", "catboost"],
)
def test_best_iteration_none_without_early_stopping(model, tiny_binary_data):
    X, y = tiny_binary_data
    model.fit(X, y)
    assert model.best_iteration is None


def _pipeline_config(tmp_path, tiny_binary_data, **training) -> AutoMLConfig:
    X, y = tiny_binary_data
    df = X.copy()
    df["target"] = y.values
    df.iloc[:220].to_parquet(tmp_path / "train.parquet")
    df.iloc[220:].to_parquet(tmp_path / "test.parquet")
    return AutoMLConfig(
        train_data_path=str(tmp_path / "train.parquet"),
        test_data_path=str(tmp_path / "test.parquet"),
        target_column="target",
        features=[
            FeatureSpec(name="num1", type="numeric"),
            FeatureSpec(name="num2", type="numeric"),
            FeatureSpec(name="cat1", type="categorical"),
        ],
        artifact_dir=str(tmp_path / "artifacts" / "models"),
        training=TrainingConfig(cv_folds=2, **training),
        tuning=TuningConfig(enabled=False),
        models={
            "lgbm": ModelConfig(fixed_params={"n_estimators": 50, "verbose": -1}),
            "xgb": ModelConfig(fixed_params={"n_estimators": 50}),
        },
        reporting=ReportingConfig(
            output_dir=str(tmp_path / "artifacts" / "reports"),
            generate_html=True,
            generate_pdf=True,
        ),
        logging=LoggingConfig(log_dir=str(tmp_path / "logs"), to_file=False),
    )


def test_iteration_capping_without_early_stopping_keeps_trees(tmp_path, tiny_binary_data):
    """조기 종료를 끈 iteration_capping 에서 LightGBM 이 트리 1개로 줄어들지 않는다."""
    cfg = _pipeline_config(
        tmp_path, tiny_binary_data,
        final_fit_strategy="iteration_capping", early_stopping_rounds=None,
    )
    AutoMLPipeline(cfg).run()
    from auto_ml.utils.io import load_artifact

    lgbm = load_artifact(tmp_path / "artifacts/models/models/lgbm.joblib")["model"]
    assert lgbm.model.get_params()["n_estimators"] == 50


def test_pdf_failure_keeps_artifacts_and_html(tmp_path, tiny_binary_data, monkeypatch, capsys):
    def broken_pdf(*args, **kwargs):
        raise OSError("cannot load library 'libpango-1.0-0'")

    monkeypatch.setattr(report_mod.ReportBuilder, "_html_to_pdf", staticmethod(broken_pdf))
    cfg = _pipeline_config(tmp_path, tiny_binary_data)
    outputs = AutoMLPipeline(cfg).run()

    assert outputs["artifact"].exists()
    assert (tmp_path / "artifacts/models/models/lgbm.joblib").exists()
    assert outputs["report_html"].exists()
    assert "report_pdf" not in outputs
    # run() 이 로거를 재구성(propagate=False)하므로 caplog 대신 stdout 로그를 본다.
    log = capsys.readouterr().out
    assert log.count("PDF report skipped") == 1  # 메인·sub 리포트마다 반복하지 않는다


def test_target_with_missing_values_is_rejected():
    with pytest.raises(ValueError, match="'default' has 2 missing value"):
        validate_binary_target(pd.Series([0, 1, np.nan, None, 1], name="default"))
    validate_binary_target(pd.Series([0, 1], dtype="Int64"))  # 정상 입력은 통과


# ───────────────────────────────────────────────────────────────────────
# 앙상블 SHAP — 확률 도메인 가중 평균 (예측과 합이 일치)
# ───────────────────────────────────────────────────────────────────────

def _fitted_ensemble(tiny_binary_data):
    from auto_ml.models.elasticnet_model import ElasticNetModel
    from auto_ml.models.ensemble_model import EnsembleModel

    X, y = tiny_binary_data
    lgb = LGBMModel(params={"n_estimators": 60, "verbose": -1}, categorical_columns=["cat1"]).fit(X, y)
    en = ElasticNetModel(params={}, categorical_columns=["cat1"]).fit(X, y)
    return EnsembleModel({"lgbm": lgb, "elasticnet": en}, {"lgbm": 0.6, "elasticnet": 0.4}), X


def test_ensemble_probability_shap_sums_to_prediction(tiny_binary_data):
    ens, X = _fitted_ensemble(tiny_binary_data)
    shap_prob, base = ens.shap_values_proba(X)
    assert shap_prob.shape == (len(X), len(ens.feature_columns))
    np.testing.assert_allclose(base + shap_prob.sum(axis=1), ens.predict_proba(X), atol=1e-12)


def test_explainer_ensemble_score_matches_scorer(tiny_binary_data):
    from auto_ml.config import PreprocessingConfig
    from auto_ml.explain.explainer import Explainer
    from auto_ml.preprocessing import PreprocessingPipeline
    from auto_ml.scoring.scorer import Scorer
    from auto_ml.utils.io import ArtifactMetadata

    X, y = tiny_binary_data
    pp = PreprocessingPipeline(["num1", "num2"], ["cat1"], PreprocessingConfig())
    Xp = pp.fit_transform(X)
    ens, _ = _fitted_ensemble((Xp, y))
    meta = ArtifactMetadata(
        target_column="target", feature_columns=["num1", "num2", "cat1"],
        categorical_columns=["cat1"], id_columns=[], model_name="ensemble",
        primary_metric="roc_auc", metric_value=0.0, selected_features=["num1", "num2", "cat1"],
    )
    scores = Scorer(pp, ens, meta).score(X)["score"].to_numpy()
    out = Explainer(pp, ens, meta).explain(X)
    shap_cols = [c for c in out.columns if c.startswith("shap_")]
    np.testing.assert_allclose(out["score"].to_numpy(), scores, atol=1e-12)
    np.testing.assert_allclose(
        out["base_value"].to_numpy() + out[shap_cols].sum(axis=1).to_numpy(), scores, atol=1e-12,
    )


# ───────────────────────────────────────────────────────────────────────
# 이전 실행의 sub-artifact / sub-report 정리
# ───────────────────────────────────────────────────────────────────────

def test_stale_sub_artifacts_and_reports_are_archived(tmp_path, tiny_binary_data):
    from auto_ml.cli.set_best import promote

    with_ensemble = {
        "lgbm": ModelConfig(fixed_params={"n_estimators": 30, "verbose": -1}),
        "elasticnet": ModelConfig(enabled=True),
    }
    without = {
        "lgbm": ModelConfig(fixed_params={"n_estimators": 30, "verbose": -1}),
        "xgb": ModelConfig(fixed_params={"n_estimators": 30}),
    }
    cfg = _pipeline_config(tmp_path, tiny_binary_data)
    cfg.reporting.generate_pdf = False
    sub = tmp_path / "artifacts/models/models"
    reports = tmp_path / "artifacts/reports/sub"

    cfg.models = with_ensemble
    AutoMLPipeline(cfg).run()
    assert (sub / "ensemble.joblib").exists()

    cfg.models = without
    AutoMLPipeline(cfg).run()
    assert sorted(p.stem for p in sub.glob("*.joblib")) == ["lgbm", "xgb"]
    archived = sorted(p.name for p in (sub / "_previous").rglob("*.joblib"))
    assert archived == ["elasticnet.joblib", "ensemble.joblib"]  # 삭제가 아니라 보관
    assert not (reports / "ensemble").exists()
    assert (reports / "_previous").exists()

    # 옛 ensemble 은 더 이상 set_best 후보가 아니다
    with pytest.raises(FileNotFoundError, match="Sub-artifact not found"):
        promote(cfg, "ensemble")

    # 같은 설정으로 다시 학습하면 옮길 것이 없다
    n_archives = len(list((sub / "_previous").iterdir()))
    AutoMLPipeline(cfg).run()
    assert len(list((sub / "_previous").iterdir())) == n_archives


# ───────────────────────────────────────────────────────────────────────
# best.joblib 메타데이터 = sub-artifact 메타데이터 + best_selection
# ───────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("strategy", ["iteration_capping", "cv_bagging"])
def test_best_metadata_matches_sub_artifact(tmp_path, tiny_binary_data, strategy):
    from dataclasses import asdict

    from auto_ml.cli.set_best import promote
    from auto_ml.utils.io import load_artifact

    cfg = _pipeline_config(tmp_path, tiny_binary_data, final_fit_strategy=strategy)
    cfg.reporting.generate_pdf = False
    AutoMLPipeline(cfg).run()

    best = load_artifact(tmp_path / "artifacts/models/best.joblib")["metadata"]
    sub = load_artifact(
        tmp_path / f"artifacts/models/models/{best.model_name}.joblib"
    )["metadata"]
    best_d, sub_d = asdict(best), asdict(sub)
    best_extra = best_d.pop("extra")
    sub_extra = sub_d.pop("extra")
    assert best_d == sub_d                       # trained_at 등 공통 필드 동일
    assert best_extra.pop("best_selection") == "auto_by_metric"
    assert best_extra == sub_extra               # 전략별 상세(iteration_cap / cv_bagging) 포함
    assert ("iteration_cap" if strategy == "iteration_capping" else "cv_bagging") in best_extra

    # set_best 로 바꾼 best.joblib 도 같은 항목 + 승격 정보를 갖는다
    other = "xgb" if best.model_name != "xgb" else "lgbm"
    promote(cfg, other)
    promoted = load_artifact(tmp_path / "artifacts/models/best.joblib")["metadata"].extra
    assert promoted["best_selection"] == "promoted"
    assert set(promoted) == set(best_extra) | {"best_selection", "promoted_at", "promoted_from"}


# ───────────────────────────────────────────────────────────────────────
# 앙상블 feature importance — 단위가 다른 서브모델은 비율로 맞춘 뒤 가중합
# ───────────────────────────────────────────────────────────────────────

def test_mixed_ensemble_importance_share_follows_weights(tiny_binary_data):
    ens, _ = _fitted_ensemble(tiny_binary_data)  # lgbm(gain) 0.6 + elasticnet(|coef|) 0.4
    combined = ens.feature_importance()
    assert sum(combined.values()) == pytest.approx(1.0)
    for name, model in ens.base_models.items():
        imp = model.feature_importance()
        total = sum(abs(v) for v in imp.values())
        share = sum(ens.weights[name] * v / total for v in imp.values())
        assert share == pytest.approx(ens.weights[name])  # 단위가 아니라 가중치대로 기여


def test_same_kind_ensemble_importance_keeps_raw_scale(tiny_binary_data):
    from auto_ml.models.ensemble_model import EnsembleModel

    X, y = tiny_binary_data
    folds = [
        LGBMModel(params={"n_estimators": 30, "verbose": -1, "seed": s},
                  categorical_columns=["cat1"]).fit(X.iloc[s * 50:], y.iloc[s * 50:])
        for s in range(3)
    ]
    ens = EnsembleModel.from_cv_folds("lgbm", folds)
    expected: dict[str, float] = {}
    for m in folds:
        for feat, val in m.feature_importance().items():
            expected[feat] = expected.get(feat, 0.0) + val / 3
    assert ens.feature_importance() == pytest.approx(expected)  # cv_bagging 출력은 이전과 동일
