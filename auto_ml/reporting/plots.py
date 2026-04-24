"""리포트용 차트 생성. matplotlib 의 Agg 백엔드를 강제로 사용해
GUI 환경 없이도 동작하도록 한다 (서버/배치/폐쇄망 친화).

차트는 PNG 바이트로 만들어 base64 로 인코딩 → HTML 에 임베드 한다.
이렇게 하면 리포트가 단일 파일(HTML 1개, PDF 1개) 로 자기완결되어
이메일·문서 시스템 첨부가 쉽다.
"""
from __future__ import annotations

import base64
import io
from typing import Iterable

import matplotlib

matplotlib.use("Agg")  # GUI 없이 동작하도록 백엔드 고정. import 직후 1회만 호출.
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402


def _fig_to_base64(fig) -> str:
    """matplotlib Figure 를 base64 PNG 문자열로 변환하고 자원 정리.

    Returns:
        ``data:image/png;base64,...`` 형식의 문자열 (HTML <img src> 에 그대로 사용).
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def roc_curve_plot(curves: dict[str, tuple[np.ndarray, np.ndarray]], title: str = "ROC Curve") -> str:
    """모델별 ROC 곡선을 한 차트에 그린다.

    Args:
        curves: ``{model_name: (y_true, y_proba)}``.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, (y_true, y_proba) in curves.items():
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        ax.plot(fpr, tpr, label=name)
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    return _fig_to_base64(fig)


def pr_curve_plot(curves: dict[str, tuple[np.ndarray, np.ndarray]], title: str = "Precision-Recall Curve") -> str:
    """모델별 Precision-Recall 곡선을 한 차트에 그린다."""
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, (y_true, y_proba) in curves.items():
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        ax.plot(recall, precision, label=name)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="lower left")
    return _fig_to_base64(fig)


def feature_importance_plot(
    importance: dict[str, float], top_n: int = 20, title: str = "Feature Importance",
) -> str:
    """상위 N 개 feature 의 중요도를 가로 막대로 그린다."""
    items = sorted(importance.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    names = [name for name, _ in items][::-1]
    values = [val for _, val in items][::-1]
    fig, ax = plt.subplots(figsize=(7, max(3, 0.3 * len(names) + 1)))
    ax.barh(names, values)
    ax.set_xlabel("Importance (gain)")
    ax.set_title(title)
    return _fig_to_base64(fig)


def score_distribution_plot(
    proba_by_label: dict[int, np.ndarray], title: str = "Score Distribution",
) -> str:
    """라벨별 점수 분포를 히스토그램으로 그린다 (분리도 시각화)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(0, 1, 41)
    for label, scores in proba_by_label.items():
        ax.hist(scores, bins=bins, alpha=0.5, label=f"y={label}", density=True)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    return _fig_to_base64(fig)
