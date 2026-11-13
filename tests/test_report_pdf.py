"""PDF 리포트 회귀 테스트 — 글자 깨짐(폰트) 과 양식 잘림(페이지 폭 초과) 방지.

규약:
- 차트는 한글 변수명을 한글 폰트로 그린다 (matplotlib 기본 DejaVu Sans 에는 한글이 없음).
- 리포트 템플릿에는 컬러 이모지를 쓰지 않는다 (서버에 이모지 폰트가 없으면 빈 네모).
- 모델 5개·긴 한글/영문 변수명이어도 PDF 의 어떤 글자도 페이지 본문 폭을 넘지 않는다.
- reporting.font_path 를 주면 그 폰트를 @font-face 로 맨 앞에 쓴다.
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from auto_ml.config import (
    AutoMLConfig,
    EnsembleConfig,
    FeatureSpec,
    LoggingConfig,
    ModelConfig,
    ReportingConfig,
    ScoringConfig,
    TrainingConfig,
    TuningConfig,
)
from auto_ml.reporting import fonts, plots
from auto_ml.reporting.report import _layout_overflow

TEMPLATES = Path(__file__).resolve().parents[1] / "auto_ml" / "reporting" / "templates"


def test_templates_have_no_color_emoji():
    emoji = re.compile("[\U0001F300-\U0001FAFF☀-⛿✀-➿]")
    allowed = {"✓"}  # ✓ — DejaVu Sans 에 있는 기호
    for path in TEMPLATES.iterdir():
        bad = {ch for ch in emoji.findall(path.read_text(encoding="utf-8")) if ch not in allowed}
        assert not bad, f"{path.name}: {bad}"


def test_chart_renders_korean_without_missing_glyphs():
    if fonts.chart_font_family()[0] == "DejaVu Sans":
        pytest.skip("no Korean font installed on this machine")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        plots.feature_importance_plot({"고객_최근12개월_평균사용금액": 0.6, "가입경과월": 0.4})
    assert not [w for w in caught if "Glyph" in str(w.message)]


def test_font_path_adds_font_face(tmp_path):
    font = tmp_path / "my.ttf"
    font.write_bytes(b"\0")
    try:
        fonts.configure(str(font))
        css = fonts.font_face_css()
        assert "@font-face" in css and font.resolve().as_uri() in css
        assert fonts.korean_font_available() is True
        fonts.configure(str(tmp_path / "missing.ttf"))          # 없는 경로 → 설치 폰트 사용
        assert fonts.font_face_css() == ""
    finally:
        fonts.configure(None)


def test_overflow_detector_finds_wide_table():
    weasyprint = pytest.importorskip("weasyprint")
    cells = "".join(f"<td>column_value_{i:02d}</td>" for i in range(40))
    html = f"<html><body><table><tr>{cells}</tr></table><p>ok</p></body></html>"
    assert _layout_overflow(weasyprint.HTML(string=html).render())
    assert not _layout_overflow(weasyprint.HTML(string="<p>짧은 문장</p>").render())


def test_pdf_fits_page_with_many_models_and_long_names(tmp_path):
    """모델 5개(오버핏 표 16열)·공백 없는 긴 변수명에서도 페이지 폭을 넘지 않는다."""
    weasyprint = pytest.importorskip("weasyprint")
    from auto_ml.pipeline import AutoMLPipeline

    rng = np.random.default_rng(0)
    n = 400
    names = ["고객_최근12개월_평균_카드사용금액_원화기준",
             "very_long_feature_name_without_any_spaces_for_overflow_check_0123456789",
             "가입후_경과개월수"]
    X = pd.DataFrame({c: rng.normal(size=n) for c in names})
    X["지역_구분코드"] = rng.choice(["서울특별시", "부산광역시", "기타_지방자치단체"], n)
    y = (rng.random(n) < 1 / (1 + np.exp(-X[names[0]]))).astype(int)
    df = X.assign(target=y)
    df.iloc[:300].to_parquet(tmp_path / "train.parquet")
    df.iloc[300:].to_parquet(tmp_path / "test.parquet")
    df.iloc[300:].drop(columns="target").to_parquet(tmp_path / "score.parquet")
    cfg = AutoMLConfig(
        train_data_path=str(tmp_path / "train.parquet"),
        test_data_path=str(tmp_path / "test.parquet"),
        target_column="target",
        features=[FeatureSpec(name=c, type="numeric") for c in names]
        + [FeatureSpec(name="지역_구분코드", type="categorical")],
        artifact_dir=str(tmp_path / "art" / "models"),
        training=TrainingConfig(cv_folds=2),
        tuning=TuningConfig(enabled=False),
        models={
            "lgbm": ModelConfig(fixed_params={"n_estimators": 20, "verbose": -1}),
            "xgb": ModelConfig(fixed_params={"n_estimators": 20}),
            "catboost": ModelConfig(fixed_params={
                "iterations": 20, "verbose": False, "allow_writing_files": False}),
            "elasticnet": ModelConfig(enabled=True),
        },
        ensemble=EnsembleConfig(enabled=True),
        reporting=ReportingConfig(
            output_dir=str(tmp_path / "art" / "reports"), generate_pdf=False,
            title="아주 긴 한글 제목이 들어간 이진 분류 모델 성능 분석 보고서 — 잘림 점검",
        ),
        scoring=ScoringConfig(input_path=str(tmp_path / "score.parquet"),
                              output_path=str(tmp_path / "art" / "scores.parquet")),
        logging=LoggingConfig(log_dir=str(tmp_path / "logs"), to_file=False),
    )
    AutoMLPipeline(cfg).run()

    reports = Path(cfg.reporting.output_dir)
    htmls = [reports / "report.html", *sorted((reports / "sub").glob("*/report.html"))]
    assert len(htmls) == 5                     # 메인 + 후보 4개
    for html in htmls:
        doc = weasyprint.HTML(filename=str(html)).render()
        assert not _layout_overflow(doc), html.relative_to(reports)
