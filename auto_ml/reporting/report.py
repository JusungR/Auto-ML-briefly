"""학습 결과를 HTML / PDF 리포트로 묶어내는 모듈.

설계 의도:
    - HTML 과 PDF 는 동일한 Jinja2 템플릿에서 만들어진다 → 두 포맷의
      내용이 항상 일치한다.
    - PDF 변환은 ``WeasyPrint`` 를 사용한다 (system fonts, 외부 네트워크
      불필요). 폐쇄망에서는 wheelhouse 로 함께 배포한다.
    - 차트는 base64 임베드 → 단일 파일로 자기완결.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from jinja2 import Environment, FileSystemLoader, select_autoescape

from auto_ml import __version__
from auto_ml.config import AutoMLConfig
from auto_ml.models.trainer import TrainingResult
from auto_ml.reporting import plots  # 동일 패키지 모듈 — 순환 위험 없음
from auto_ml.reporting.metrics import confusion
from auto_ml.utils.logger import get_logger

logger = get_logger("report")

# 리포트 표에 노출할 지표 순서
METRIC_NAMES = ("roc_auc", "pr_auc", "accuracy", "precision", "recall", "f1", "ks")
TOP_FEATURES = 20


class ReportBuilder:
    """``TrainingResult`` 를 HTML / PDF 로 변환한다."""

    def __init__(self, config: AutoMLConfig) -> None:
        self.config = config
        templates_dir = Path(__file__).parent / "templates"
        self.env = Environment(
            loader=FileSystemLoader(str(templates_dir)),
            autoescape=select_autoescape(enabled_extensions=("html",)),
        )

    def build(self, result: TrainingResult) -> dict[str, Path]:
        """리포트를 생성하고 산출 경로를 dict 로 반환한다."""
        out_dir = Path(self.config.reporting.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        html_str = self._render_html(result)

        outputs: dict[str, Path] = {}
        if self.config.reporting.generate_html:
            html_path = out_dir / "report.html"
            html_path.write_text(html_str, encoding="utf-8")
            outputs["html"] = html_path
            logger.info("HTML report written: %s", html_path)

        if self.config.reporting.generate_pdf:
            pdf_path = out_dir / "report.pdf"
            self._html_to_pdf(html_str, pdf_path)
            outputs["pdf"] = pdf_path
            logger.info("PDF report written: %s", pdf_path)

        return outputs

    # ------------------------------------------------------------------
    def _render_html(self, result: TrainingResult) -> str:
        """Jinja2 템플릿에 컨텍스트를 채워 HTML 문자열을 만든다."""
        # ----- 모델 비교 표 데이터 -----
        comparison_rows = []
        cv_rows = []
        tuning_rows = []
        for name, mr in result.results.items():
            best_iters = [bi for bi in mr.fold_best_iterations if bi is not None]
            avg_iter = int(np.mean(best_iters)) if best_iters else None
            comparison_rows.append({
                "name": name,
                "metrics": mr.test_metrics,
                "best_iter_avg": avg_iter,
            })
            cv_rows.append({"name": name, "metrics": mr.cv_metrics})
            tuning_rows.append({
                "name": name,
                "tuned": mr.tuning is not None,
                "n_trials": mr.tuning.n_trials if mr.tuning else 0,
                "best_value": mr.tuning.best_value if mr.tuning else None,
                "params": mr.params,
            })

        # ----- 차트 (ROC / PR / Importance / 분포) -----
        roc_curves = {n: (result.test_y, mr.test_proba) for n, mr in result.results.items()}
        roc_chart = plots.roc_curve_plot(roc_curves)
        pr_chart = plots.pr_curve_plot(roc_curves)

        best = result.best
        importance_chart = plots.feature_importance_plot(
            best.feature_importance, top_n=TOP_FEATURES
        )
        proba_by_label = {
            0: best.test_proba[result.test_y == 0],
            1: best.test_proba[result.test_y == 1],
        }
        score_dist_chart = plots.score_distribution_plot(proba_by_label)

        # ----- Confusion / 설정 요약 -----
        cm = confusion(result.test_y, best.test_proba, threshold=0.5)
        config_summary = self._summarize_config()

        template = self.env.get_template("report.html.j2")
        return template.render(
            title=self.config.reporting.title,
            generated_at=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
            best_model=result.best_model_name,
            primary_metric=result.primary_metric,
            best_score=best.test_metrics[result.primary_metric],
            n_train=len(result.results[next(iter(result.results))].oof_proba),
            n_test=len(result.test_y),
            n_features=len(result.feature_columns),
            metric_names=METRIC_NAMES,
            comparison_rows=comparison_rows,
            cv_rows=cv_rows,
            tuning_rows=tuning_rows,
            roc_chart=roc_chart,
            pr_chart=pr_chart,
            importance_chart=importance_chart,
            score_dist_chart=score_dist_chart,
            top_features=TOP_FEATURES,
            confusion=cm.tolist(),
            config_summary=config_summary,
            library_version=__version__,
        )

    def _summarize_config(self) -> dict[str, Any]:
        """리포트에 노출할 핵심 설정만 추려 dict 로 만든다."""
        cfg = self.config
        pp = cfg.preprocessing
        tr = cfg.training
        tu = cfg.tuning
        return {
            "target_column": cfg.target_column,
            "categorical_columns": ", ".join(cfg.categorical_columns) or "(none)",
            "id_columns": ", ".join(cfg.id_columns) or "(none)",
            "preprocessing.numeric_null_strategy": pp.numeric_null_strategy,
            "preprocessing.categorical_null_strategy": pp.categorical_null_strategy,
            "preprocessing.outlier_method": pp.outlier_method,
            "preprocessing.outlier_action": pp.outlier_action,
            "preprocessing.scaling_method": pp.scaling_method,
            "training.cv_folds": tr.cv_folds,
            "training.early_stopping_rounds": tr.early_stopping_rounds,
            "training.primary_metric": tr.primary_metric,
            "training.random_state": tr.random_state,
            "tuning.enabled": tu.enabled,
            "tuning.n_trials": tu.n_trials,
            "tuning.cv_folds": tu.cv_folds,
            "tuning.timeout": tu.timeout,
        }

    @staticmethod
    def _html_to_pdf(html_str: str, pdf_path: Path) -> None:
        """동일 HTML 을 PDF 로 변환한다 (WeasyPrint).

        WeasyPrint 의존성이 무거우므로 import 는 함수 내부에서 수행한다.
        """
        from weasyprint import HTML  # type: ignore

        HTML(string=html_str).write_pdf(str(pdf_path))
