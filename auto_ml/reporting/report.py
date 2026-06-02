"""학습 결과를 HTML / PDF 리포트로 묶어내는 모듈.

설계 의도:
    - HTML 과 PDF 는 동일한 Jinja2 템플릿에서 만들어진다 → 두 포맷의
      내용이 항상 일치한다.
    - PDF 변환은 ``WeasyPrint`` 를 사용한다 (system fonts, 외부 네트워크
      불필요). 폐쇄망에서는 wheelhouse 로 함께 배포한다.
    - 차트는 base64 임베드 → 단일 파일로 자기완결.
"""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from jinja2 import Environment, FileSystemLoader, select_autoescape

from auto_ml import __version__
from auto_ml.config import AutoMLConfig
from auto_ml.feature_selection import SelectionResult
from auto_ml.models.trainer import TrainingResult
from auto_ml.reporting import plots  # 동일 패키지 모듈 — 순환 위험 없음
from auto_ml.reporting.metrics import confusion
from auto_ml.utils.logger import get_logger

logger = get_logger("report")

# 리포트 표에 노출할 지표 순서
METRIC_NAMES = ("roc_auc", "ks", "lift", "accuracy", "precision", "recall", "f1", "pr_auc")


class ReportBuilder:
    """``TrainingResult`` 를 HTML / PDF 로 변환한다."""

    def __init__(self, config: AutoMLConfig) -> None:
        self.config = config
        templates_dir = Path(__file__).parent / "templates"
        self.env = Environment(
            loader=FileSystemLoader(str(templates_dir)),
            autoescape=select_autoescape(enabled_extensions=("html",)),
        )

    def build(
        self,
        result: TrainingResult,
        selection: SelectionResult | None = None,
    ) -> dict[str, Path]:
        """리포트를 생성하고 산출 경로를 dict 로 반환한다.

        리포트 표/차트는 ``reporting.top_*_features`` 설정만큼만 노출하고,
        CSV (``feature_importance.csv``, ``feature_selection.csv``) 는 항상
        전체 변수를 저장한다 — 외부 분석/감사용.

        Args:
            result: 학습 결과 (모델별 metrics, best 모델 등).
            selection: 변수 선택을 사용했다면 결과 객체. None 이면 해당 섹션은 생략.
        """
        out_dir = Path(self.config.reporting.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # 전체(sorted) 데이터를 한 번만 준비 — HTML 표/차트와 CSV 가 공유한다.
        importance_full = self._prepare_importance(result.best.feature_importance)
        selection_full = (
            self._prepare_selection(selection) if selection is not None else None
        )

        html_str = self._render_html(result, selection, importance_full, selection_full)

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

        # CSV — 표/차트가 잘린 경우에도 외부에서 전체 분포를 확인할 수 있도록 항상 저장.
        imp_csv = out_dir / "feature_importance.csv"
        self._write_csv(
            imp_csv, importance_full,
            ["feature", "importance_gain", "importance_relative"],
        )
        outputs["feature_importance_csv"] = imp_csv
        logger.info(
            "Feature importance CSV written: %s (rows=%d)", imp_csv, len(importance_full),
        )

        if selection_full is not None:
            sel_csv = out_dir / "feature_selection.csv"
            self._write_csv(
                sel_csv, selection_full,
                ["feature", "frequency", "selected"],
            )
            outputs["feature_selection_csv"] = sel_csv
            logger.info(
                "Feature selection CSV written: %s (rows=%d)", sel_csv, len(selection_full),
            )

        # 비-best 모형마다 sub-report 생성 (best 는 메인 리포트가 곧 그 것이므로 제외).
        sub_paths: dict[str, dict[str, Path]] = {}
        for name, mr in result.results.items():
            if name == result.best_model_name:
                continue
            sub_paths[name] = self._build_sub_report(
                name=name, mr=mr, result=result, selection=selection, out_dir=out_dir,
            )
        if sub_paths:
            outputs["sub_reports"] = sub_paths  # type: ignore[assignment]

        return outputs

    # ------------------------------------------------------------------
    @staticmethod
    def _prepare_importance(fi: dict[str, float]) -> list[dict[str, Any]]:
        """feature importance 를 sum=1 의 relative 로 정규화하고 내림차순 정렬.

        Returns:
            ``[{feature, importance_gain, importance_relative}, ...]`` 전체 변수.
            합이 0 이거나 음수 합산이 되는 비정상 케이스는 정규화하지 않고
            relative 를 0 으로 둔다 (분모 0 회피).
        """
        total = float(sum(abs(v) for v in fi.values()))
        # 모든 값이 0 인 비정상 케이스 보호 — relative 를 0 으로 두고 진행
        norm = total if total > 0 else 0.0
        sorted_items = sorted(fi.items(), key=lambda kv: kv[1], reverse=True)
        rows: list[dict[str, Any]] = []
        for name, val in sorted_items:
            rel = (float(val) / norm) if norm > 0 else 0.0
            rows.append({
                "feature": name,
                "importance_gain": float(val),
                "importance_relative": rel,
            })
        return rows

    @staticmethod
    def _prepare_selection(selection: SelectionResult) -> list[dict[str, Any]]:
        """선택 빈도 sorted 내림차순 + 채택 여부 플래그. 전체 변수 반환."""
        selected_set = set(selection.selected_features)
        sorted_freq = sorted(
            selection.frequencies.items(), key=lambda kv: kv[1], reverse=True,
        )
        return [
            {
                "feature": name,
                "frequency": float(freq),
                "selected": name in selected_set,
            }
            for name, freq in sorted_freq
        ]

    @staticmethod
    def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
        """utf-8-sig 로 CSV 저장 (Excel 한글 호환). 헤더 + 모든 rows."""
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(columns)
            for r in rows:
                w.writerow([r.get(c, "") for c in columns])

    # ------------------------------------------------------------------
    def _render_html(
        self,
        result: TrainingResult,
        selection: SelectionResult | None = None,
        importance_full: list[dict[str, Any]] | None = None,
        selection_full: list[dict[str, Any]] | None = None,
    ) -> str:
        """Jinja2 템플릿에 컨텍스트를 채워 HTML 문자열을 만든다.

        Args:
            importance_full / selection_full: ``build()`` 가 전달하는 정규화/정렬
                완료된 전체 데이터. None 이면 본 메소드 내에서 다시 계산한다
                (단위 테스트용 짧은 경로).
        """
        # ----- 모델 비교 표 데이터 -----
        comparison_rows = []
        cv_rows = []
        overfit_rows = []
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
            # Overfit 점검: in-sample train fit 과 test 의 metric 별 gap(train-test).
            # 양수 gap 이 클수록 모델이 학습 데이터를 외운 정도가 크다는 신호.
            gaps = {
                m: mr.train_metrics[m] - mr.test_metrics[m] for m in METRIC_NAMES
            }
            overfit_rows.append({
                "name": name,
                "train_metrics": mr.train_metrics,
                "test_metrics": mr.test_metrics,
                "gaps": gaps,
            })
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

        # Feature importance: 합 1 의 relative 로 정규화 (gain 절대값은 모델/데이터 의존이라 비교가 어려움).
        # build() 가 이미 만들어 둔 전체 데이터를 받고, 없으면 짧은 경로로 다시 만든다.
        if importance_full is None:
            importance_full = self._prepare_importance(best.feature_importance)
        top_imp_n = self.config.reporting.top_importance_features
        importance_rows = (
            importance_full if top_imp_n is None else importance_full[:top_imp_n]
        )
        # 차트도 동일한 top-N 으로. relative 값을 그대로 막대 길이에 사용.
        importance_chart = plots.feature_importance_plot(
            {r["feature"]: r["importance_relative"] for r in importance_rows},
            top_n=None,  # 이미 위에서 잘랐으니 plots 쪽에서는 다시 자르지 않는다
            xlabel="Importance (relative, sums to 1)",
        )
        
        proba_by_label = {
            0: best.test_proba[result.test_y == 0],
            1: best.test_proba[result.test_y == 1],
        }
        score_dist_chart = plots.score_distribution_plot(proba_by_label)

        # ----- 10-분위 분석 (test set, best 모델) -----
        decile_rows, decile_chart = plots.decile_analysis(result.test_y, best.test_proba)

        # ----- 변수 선택 결과 (활성화된 경우만) -----
        selection_ctx = self._build_selection_context(selection, selection_full)

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
            overfit_rows=overfit_rows,
            tuning_rows=tuning_rows,
            roc_chart=roc_chart,
            pr_chart=pr_chart,
            importance_chart=importance_chart,
            importance_rows=importance_rows,
            score_dist_chart=score_dist_chart,
            decile_chart=decile_chart,
            decile_rows=decile_rows,
            top_features=len(importance_rows),
            selection=selection_ctx,
            confusion=cm.tolist(),
            config_summary=config_summary,
            library_version=__version__,
        )

    def _build_selection_context(
        self,
        selection: SelectionResult | None,
        selection_full: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any] | None:
        """변수 선택 결과를 템플릿이 바로 쓰도록 가공한다 (None 이면 None 반환).

        Args:
            selection: 변수 선택 결과 객체.
            selection_full: ``build()`` 가 미리 준비한 정렬 완료 전체 데이터.
                None 이면 본 메소드 내에서 다시 계산.
        """
        if selection is None:
            return None
        if selection_full is None:
            selection_full = self._prepare_selection(selection)

        # 표/차트 노출 개수: 설정 기반. None 이면 모든 변수.
        top_n = self.config.reporting.top_selection_features
        rows = selection_full if top_n is None else selection_full[:top_n]
        chart = plots.feature_selection_plot(
            selection.frequencies,
            threshold=selection.threshold,
            top_n=top_n,
        )
        return {
            "base_estimator": selection.base_estimator,
            "n_subsamples": selection.n_subsamples,
            "threshold": selection.threshold,
            "n_total": len(selection.frequencies),
            "n_selected": len(selection.selected_features),
            "fallback_used": selection.fallback_used,
            "top_n": len(rows),
            "chart": chart,
            "rows": rows,
        }

    def _summarize_config(self) -> dict[str, Any]:
        """리포트에 노출할 핵심 설정만 추려 dict 로 만든다."""
        cfg = self.config
        pp = cfg.preprocessing
        tr = cfg.training
        tu = cfg.tuning
        summary = {
            "target_column": cfg.target_column,
            "categorical_columns": ", ".join(cfg.categorical_columns) or "(none)",
            "id_columns": ", ".join(cfg.id_columns) or "(none)",
            "preprocessing.numeric_null_strategy": pp.numeric_null_strategy,
            "preprocessing.categorical_null_strategy": pp.categorical_null_strategy,
            "preprocessing.outlier_method": pp.outlier_method,
            "preprocessing.outlier_action": pp.outlier_action,
            "preprocessing.skew_method": pp.skew_method,
            "preprocessing.skew_threshold": pp.skew_threshold,
            "preprocessing.scaling_method": pp.scaling_method,
            "training.cv_folds": tr.cv_folds,
            "training.early_stopping_rounds": tr.early_stopping_rounds,
            "training.primary_metric": tr.primary_metric,
            "training.random_state": tr.random_state,
            "training.final_fit_strategy": tr.final_fit_strategy,
            "tuning.enabled": tu.enabled,
            "tuning.n_trials": tu.n_trials,
            "tuning.cv_folds": tu.cv_folds,
            "tuning.timeout": tu.timeout,
        }
        # iteration_capping 일 때만 관련 키를 노출해 불필요한 잡음을 줄인다.
        if tr.final_fit_strategy == "iteration_capping":
            summary["training.iteration_cap_aggregation"] = tr.iteration_cap_aggregation
            summary["training.iteration_cap_headroom"] = tr.iteration_cap_headroom
        return summary

    @staticmethod
    def _html_to_pdf(html_str: str, pdf_path: Path) -> None:
        """동일 HTML 을 PDF 로 변환한다 (WeasyPrint).

        WeasyPrint 의존성이 무거우므로 import 는 함수 내부에서 수행한다.
        """
        from weasyprint import HTML  # type: ignore

        HTML(string=html_str).write_pdf(str(pdf_path))

    # ------------------------------------------------------------------
    def _build_sub_report(
        self,
        name: str,
        mr: Any,
        result: TrainingResult,
        selection: SelectionResult | None,
        out_dir: Path,
    ) -> dict[str, Path]:
        """비-best 모형 1개에 대한 sub-report (HTML/PDF + importance CSV) 생성.

        Args:
            name: 모형 이름.
            mr: 해당 모형의 ``ModelResult``.
            result: 전체 학습 결과 (best 비교에 사용).
            selection: 변수 선택 결과 (모형 무관). 활성화 시 sub-report 에 같이 노출.
            out_dir: 메인 리포트의 ``output_dir``. sub 는 ``<out_dir>/sub/<name>/`` 에 저장.

        Returns:
            ``{"html": ..., "pdf": ..., "feature_importance_csv": ...}`` (생성된 것만).
        """
        sub_dir = out_dir / "sub" / name
        sub_dir.mkdir(parents=True, exist_ok=True)

        importance_full = self._prepare_importance(mr.feature_importance)
        top_n = self.config.reporting.top_importance_features
        importance_rows = (
            importance_full if top_n is None else importance_full[:top_n]
        )

        # plots 는 main 과 동일 헬퍼 사용.
        importance_chart = plots.feature_importance_plot(
            {r["feature"]: r["importance_relative"] for r in importance_rows},
            top_n=None,
            xlabel="Importance (relative, sums to 1)",
        )
        proba_by_label = {
            0: mr.test_proba[result.test_y == 0],
            1: mr.test_proba[result.test_y == 1],
        }
        score_dist_chart = plots.score_distribution_plot(proba_by_label)
        decile_rows, decile_chart = plots.decile_analysis(result.test_y, mr.test_proba)
        cm = confusion(result.test_y, mr.test_proba, threshold=0.5)

        # 변수 선택 컨텍스트 (메인과 동일 — 모형 무관).
        selection_full = (
            self._prepare_selection(selection) if selection is not None else None
        )
        selection_ctx = self._build_selection_context(selection, selection_full)

        best = result.best
        best_iters = [bi for bi in mr.fold_best_iterations if bi is not None]
        avg_best_iter = int(np.mean(best_iters)) if best_iters else None

        template = self.env.get_template("sub_report.html.j2")
        html_str = template.render(
            title=self.config.reporting.title,
            generated_at=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
            model_name=name,
            best_model_name=result.best_model_name,
            primary_metric=result.primary_metric,
            metric_value=mr.test_metrics[result.primary_metric],
            best_score=best.test_metrics[result.primary_metric],
            metric_names=METRIC_NAMES,
            this_test=mr.test_metrics,
            this_train=mr.train_metrics,
            best_test=best.test_metrics,
            best_train=best.train_metrics,
            importance_chart=importance_chart,
            importance_rows=importance_rows,
            top_features=len(importance_rows),
            score_dist_chart=score_dist_chart,
            decile_chart=decile_chart,
            decile_rows=decile_rows,
            confusion=cm.tolist(),
            selection=selection_ctx,
            tuned=mr.tuning is not None,
            n_trials=mr.tuning.n_trials if mr.tuning else 0,
            best_value=mr.tuning.best_value if mr.tuning else None,
            avg_best_iter=avg_best_iter,
            params_repr=repr(mr.params),
            library_version=__version__,
        )

        out: dict[str, Path] = {}
        if self.config.reporting.generate_html:
            html_path = sub_dir / "report.html"
            html_path.write_text(html_str, encoding="utf-8")
            out["html"] = html_path
            logger.info("Sub-report HTML written: %s", html_path)
        if self.config.reporting.generate_pdf:
            pdf_path = sub_dir / "report.pdf"
            self._html_to_pdf(html_str, pdf_path)
            out["pdf"] = pdf_path
            logger.info("Sub-report PDF written: %s", pdf_path)

        imp_csv = sub_dir / "feature_importance.csv"
        self._write_csv(
            imp_csv, importance_full,
            ["feature", "importance_gain", "importance_relative"],
        )
        out["feature_importance_csv"] = imp_csv
        logger.info(
            "Sub-report importance CSV: %s (rows=%d)", imp_csv, len(importance_full),
        )
        return out
