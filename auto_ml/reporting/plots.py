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


def decile_analysis(
    y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10,
) -> tuple[list[dict], str]:
    """10-분위(기본) 누적 분포·타겟/비타겟 캡처율·lift 표와 차트를 생성한다.

    점수 내림차순으로 정렬한 뒤 표본을 동일 크기로 ``n_bins`` 등분하여
    각 분위의 통계와 누적 통계를 계산한다.
    """
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba).astype(float)
    n = len(y_true)
    order = np.argsort(-y_proba)  # 확률 내림차순
    y_sorted = y_true[order]
    p_sorted = y_proba[order]

    pos_total = int(y_sorted.sum())
    neg_total = n - pos_total
    pos_total_safe = max(pos_total, 1)
    neg_total_safe = max(neg_total, 1)

    chunks_y = np.array_split(y_sorted, n_bins)
    chunks_p = np.array_split(p_sorted, n_bins)

    rows: list[dict] = []
    cum_count = 0
    cum_pos = 0
    cum_neg = 0
    for i, (yc, pc) in enumerate(zip(chunks_y, chunks_p), start=1):
        cnt = int(len(yc))
        if cnt == 0:
            continue
        pos = int(yc.sum())
        neg = cnt - pos
        cum_count += cnt
        cum_pos += pos
        cum_neg += neg
        cum_count_pct = cum_count / n
        cum_target_rate = cum_pos / pos_total_safe
        cum_nontarget_rate = cum_neg / neg_total_safe
        # 누적 lift = (누적 양성률) / (누적 표본률) — i=1 일 때 top-decile lift
        lift = (cum_pos / cum_count) / (pos_total / n) if pos_total > 0 else 0.0
        rows.append({
            "decile": i,
            "score_min": float(pc.min()),
            "score_max": float(pc.max()),
            "count": cnt,
            "target": pos,
            "non_target": neg,
            "cum_count_pct": float(cum_count_pct),
            "cum_target_rate": float(cum_target_rate),
            "cum_nontarget_rate": float(cum_nontarget_rate),
            "lift": float(lift),
        })

    chart = _decile_chart(rows)
    return rows, chart


def _decile_chart(rows: list[dict], title: str = "Decile Analysis (Test)") -> str:
    """10-분위 누적 타겟/비타겟 캡처율을 gains chart 형태로 그린다.

    좌측 y축: 누적 캡처율(타겟/비타겟). 대각선은 무작위 모델 기준.
    우측 y축: 분위별 lift (회색 막대).
    """
    deciles = [r["decile"] for r in rows]
    cum_target = [r["cum_target_rate"] for r in rows]
    cum_nontarget = [r["cum_nontarget_rate"] for r in rows]
    cum_count_pct = [r["cum_count_pct"] for r in rows]
    lifts = [r["lift"] for r in rows]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax2 = ax.twinx()

    # lift 막대 (우측 축)
    ax2.bar(deciles, lifts, color="#cccccc", alpha=0.5, label="cumulative lift")
    ax2.set_ylabel("Cumulative Lift")
    ax2.set_ylim(bottom=0)

    # 누적 캡처율 라인 (좌측 축)
    ax.plot(deciles, cum_target, marker="o", color="#1f77b4", label="cum target rate")
    ax.plot(deciles, cum_nontarget, marker="s", color="#ff7f0e", label="cum non-target rate")
    ax.plot(deciles, cum_count_pct, linestyle="--", color="grey", linewidth=1, label="random baseline")
    ax.set_xlabel("Decile (1 = top 10% by score)")
    ax.set_ylabel("Cumulative Rate")
    ax.set_xticks(deciles)
    ax.set_ylim(0, 1.05)
    ax.set_title(title)

    # 두 축의 범례를 함께 표시
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower right", fontsize=9)
    return _fig_to_base64(fig)


def feature_selection_plot(
    frequencies: dict[str, float],
    threshold: float,
    top_n: int = 20,
    title: str = "Stability Selection Frequency",
) -> str:
    """Stability Selection 변수별 선택 빈도(top N) 를 가로 막대로 그린다.

    Args:
        frequencies: ``{feature_name: selection_probability}`` (0.0~1.0).
        threshold: 채택 임계값 — 가로 점선으로 표시된다.
        top_n: 표시할 상위 변수 개수.
    """
    items = sorted(frequencies.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    names = [n for n, _ in items][::-1]   # 위쪽이 가장 높은 빈도가 되도록 역순
    values = [v for _, v in items][::-1]
    # 채택/미채택을 색으로 구분 — 시각적으로 임계값 위/아래를 즉시 구별
    colors = ["#1f77b4" if v >= threshold else "#cccccc" for v in values]

    fig, ax = plt.subplots(figsize=(7, max(3, 0.3 * len(names) + 1)))
    ax.barh(names, values, color=colors)
    ax.axvline(threshold, color="#d62728", linestyle="--", linewidth=1,
               label=f"threshold = {threshold:.2f}")
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Selection Probability")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=9)
    return _fig_to_base64(fig)
