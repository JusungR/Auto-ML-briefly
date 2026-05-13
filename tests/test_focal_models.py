"""Focal Loss end-to-end 모델 학습 테스트.

3개 모델 (lgbm / xgb / catboost) 에 대해 ``loss="focal"`` 로 학습 후
``predict_proba`` 가 [0,1] 유한값을 내고 AUC 가 합리적 수준 (> 0.70) 인지
검증한다. CatBoost 부호 실수, XGB base_score 누락, LGBM raw vs proba 처리
실수 등을 한 번에 잡는다.
"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from auto_ml.models.registry import build_model


@pytest.mark.parametrize("name", ["lgbm", "xgb", "catboost"])
def test_focal_end_to_end(name, tiny_binary_data):
    X, y = tiny_binary_data
    n_iter_param = "iterations" if name == "catboost" else "n_estimators"
    model = build_model(
        name=name,
        params={
            "focal_gamma": 2.0,
            "focal_alpha": 0.25,
            n_iter_param: 100,
        },
        categorical_columns=["cat1"],
        random_state=0,
        loss="focal",
    )

    X_tr, X_va = X.iloc[:240], X.iloc[240:]
    y_tr, y_va = y.iloc[:240], y.iloc[240:]
    model.fit(X_tr, y_tr, X_va, y_va, early_stopping_rounds=10)

    proba = model.predict_proba(X_va)

    assert proba.shape == (60,)
    assert np.all(np.isfinite(proba))
    assert np.all((proba >= 0.0) & (proba <= 1.0))
    # 합성 데이터는 신호가 충분 → focal 도 의미 있는 분리 학습 가능
    auc = roc_auc_score(y_va, proba)
    assert auc > 0.70, f"{name} focal AUC too low: {auc:.3f}"

    # focal 플래그가 켜졌는지 확인
    assert model._focal_active is True


@pytest.mark.parametrize("name", ["lgbm", "xgb", "catboost"])
def test_focal_picklable(name, tiny_binary_data, tmp_path):
    """학습된 focal 모델이 joblib 으로 round-trip 되는지 — 운영 스코어링 경로 검증."""
    import joblib

    X, y = tiny_binary_data
    n_iter_param = "iterations" if name == "catboost" else "n_estimators"
    model = build_model(
        name=name,
        params={n_iter_param: 30},
        categorical_columns=["cat1"],
        random_state=0,
        loss="focal",
    )
    model.fit(X.iloc[:240], y.iloc[:240], X.iloc[240:], y.iloc[240:])

    path = tmp_path / f"{name}.joblib"
    joblib.dump(model, path)
    restored = joblib.load(path)

    p_orig = model.predict_proba(X.iloc[240:])
    p_restored = restored.predict_proba(X.iloc[240:])
    np.testing.assert_allclose(p_orig, p_restored, rtol=1e-6)
