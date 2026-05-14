"""오버핏 점검 (train vs holdout) 리포트 데이터 회귀 테스트."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from auto_ml.config import AutoMLConfig, FeatureSpec, ModelConfig, TrainingConfig, TuningConfig
from auto_ml.models.trainer import Trainer


@pytest.fixture
def quick_config():
    """튜닝 비활성, fold 2 로 매우 빠른 trainer 설정."""
    return AutoMLConfig(
        features=[
            FeatureSpec(name="num1", type="numeric"),
            FeatureSpec(name="num2", type="numeric"),
            FeatureSpec(name="cat1", type="categorical"),
        ],
        training=TrainingConfig(cv_folds=2, random_state=0, early_stopping_rounds=10),
        tuning=TuningConfig(enabled=False),
        models={
            "lgbm": ModelConfig(
                fixed_params={"n_estimators": 30, "verbose": -1},
                search_space={},
            ),
        },
    )


def test_train_metrics_populated(tiny_binary_data, quick_config):
    """ModelResult.train_metrics / train_proba 가 채워지는지 확인."""
    X, y = tiny_binary_data
    X_tr, X_te = X.iloc[:240], X.iloc[240:]
    y_tr, y_te = y.iloc[:240], y.iloc[240:]

    trainer = Trainer(quick_config)
    result = trainer.train(X_tr, y_tr, X_te, y_te)
    mr = result.results["lgbm"]

    # 새 필드들이 정상 채워졌는지
    assert mr.train_proba.shape == (240,)
    assert np.all(np.isfinite(mr.train_proba))
    assert np.all((mr.train_proba >= 0.0) & (mr.train_proba <= 1.0))

    assert mr.train_metrics.keys() == mr.test_metrics.keys()
    assert "roc_auc" in mr.train_metrics
    # train_metrics 값이 유효한 범위 ([0,1] for AUC)
    assert 0.0 <= mr.train_metrics["roc_auc"] <= 1.0
    # 절대 비교는 하지 않는다 — 작은 n_estimators + test 기반 early stopping 조합에서
    # train < test 가 나올 수 있다. overfit 의 방향성은 도메인/데이터에 따라 다르며
    # 리포트의 Δ 컬럼이 사용자에게 보여주는 신호 자체이지 단위 테스트 단언 대상이 아니다.


def test_training_result_has_train_y(tiny_binary_data, quick_config):
    X, y = tiny_binary_data
    trainer = Trainer(quick_config)
    result = trainer.train(X.iloc[:240], y.iloc[:240], X.iloc[240:], y.iloc[240:])
    # 리포트의 overfit 색칠/차트가 train_y 를 참조하므로 보존되어야 함
    assert result.train_y.shape == (240,)
    np.testing.assert_array_equal(result.train_y, y.iloc[:240].to_numpy())


def test_report_overfit_rows_built(tiny_binary_data, quick_config):
    """ReportBuilder 가 overfit_rows context 를 빠뜨리지 않는지 — 템플릿 렌더 회귀."""
    from auto_ml.reporting.report import ReportBuilder

    X, y = tiny_binary_data
    trainer = Trainer(quick_config)
    result = trainer.train(X.iloc[:240], y.iloc[:240], X.iloc[240:], y.iloc[240:])

    quick_config.reporting.generate_pdf = False  # weasyprint 의존성 회피
    builder = ReportBuilder(quick_config)
    html = builder._render_html(result, selection=None)

    # 새 섹션 헤더가 렌더링되어야 함
    assert "오버핏 점검 (Train vs Holdout)" in html
    # gap 셀 클래스가 최소 하나는 출력되어야 함
    assert "gap-low" in html or "gap-mid" in html or "gap-high" in html
    # 후속 섹션 번호가 +1 되었는지 (4번 = 곡선 비교)
    assert ">4</span> 곡선 비교" in html
