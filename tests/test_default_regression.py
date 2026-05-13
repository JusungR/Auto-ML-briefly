"""기본 (logloss) 경로 회귀 테스트.

``loss`` 를 명시하지 않으면 기존 동작이 그대로 유지되는지 확인한다.
이 테스트가 통과해야 폐쇄망 운영 환경에 안전하게 배포 가능하다.
"""
from __future__ import annotations

import numpy as np
import pytest

from auto_ml.config import ModelConfig, load_config
from auto_ml.models.registry import build_model


def test_model_config_default_loss():
    """ModelConfig 의 기본 loss 는 logloss."""
    cfg = ModelConfig()
    assert cfg.loss == "logloss"


def test_model_config_invalid_loss_raises():
    with pytest.raises(ValueError, match="must be 'logloss' or 'focal'"):
        ModelConfig(loss="unknown")


def test_example_yaml_defaults_to_logloss(tmp_path, monkeypatch):
    """configs/example.yaml 의 모든 모델이 기본 logloss 인지 확인.

    (운영 회귀 안전망 — example.yaml 을 실수로 focal 로 만들지 않도록.)
    """
    # features_csv 의 상대경로 해석을 위해 working dir 이동
    monkeypatch.chdir("/home/user/Auto-ML-briefly")
    cfg = load_config("configs/example.yaml")
    for name, mc in cfg.models.items():
        assert mc.loss == "logloss", f"{name} should default to logloss"


@pytest.mark.parametrize("name", ["lgbm", "xgb", "catboost"])
def test_default_path_unchanged(name, tiny_binary_data):
    """``loss`` 미지정 시 _focal_active=False, native objective/loss_function 유지."""
    X, y = tiny_binary_data
    n_iter_param = "iterations" if name == "catboost" else "n_estimators"
    model = build_model(
        name=name,
        params={n_iter_param: 30},
        categorical_columns=["cat1"],
        random_state=0,
        # loss 인자 생략 → 기본 logloss
    )
    model.fit(X.iloc[:240], y.iloc[:240], X.iloc[240:], y.iloc[240:])

    # focal 분기를 타지 않아야 함
    assert model._focal_active is False

    # 예측이 정상 [0,1] 확률
    proba = model.predict_proba(X.iloc[240:])
    assert proba.shape == (60,)
    assert np.all(np.isfinite(proba))
    assert np.all((proba >= 0.0) & (proba <= 1.0))


def test_resolved_params_keeps_native_objective_strings():
    """logloss 경로의 resolved params 에 native 문자열이 잔존해야 한다."""
    from auto_ml.models.lgbm_model import LGBMModel
    from auto_ml.models.xgb_model import XGBModel
    from auto_ml.models.catboost_model import CatBoostModel

    lgbm = LGBMModel(params={}, categorical_columns=[], random_state=0)
    p = lgbm._resolved_params()
    assert p["objective"] == "binary"

    xgbm = XGBModel(params={}, categorical_columns=[], random_state=0)
    p = xgbm._resolved_params()
    assert p["objective"] == "binary:logistic"

    catm = CatBoostModel(params={}, categorical_columns=[], random_state=0)
    p = catm._resolved_params()
    assert p["loss_function"] == "Logloss"


def test_focal_resolved_params_pop_focal_keys_and_inject_callable():
    """focal 경로의 resolved params 에 focal_gamma/focal_alpha 가 남으면 안 된다."""
    from auto_ml.models.lgbm_model import LGBMModel

    lgbm = LGBMModel(
        params={"focal_gamma": 3.0, "focal_alpha": 0.3},
        categorical_columns=[],
        random_state=0,
        loss="focal",
    )
    p = lgbm._resolved_params()
    # 예약 키는 pop 되어야 함 (라이브러리 생성자에 전달되면 경고 발생)
    assert "focal_gamma" not in p
    assert "focal_alpha" not in p
    # objective 는 callable 로 덮어써져야 함
    assert callable(p["objective"])
