"""이진 분류용 평가 지표 계산.

trainer 와 reporting 모두 본 모듈을 공유해 같은 정의로 같은 점수를 사용한다.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# 지원 지표 목록 — primary_metric 검증에 활용
SUPPORTED_METRICS = ("roc_auc", "pr_auc", "accuracy", "precision", "recall", "f1", "ks")


def ks_statistic(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """KS 통계량(양/음 누적 분포의 최대 차이) 을 계산한다.

    리스크/스코어카드 영역에서 모델 분리력을 평가하는 표준 지표 중 하나다.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    order = np.argsort(-y_proba)  # 확률 내림차순 정렬
    y_sorted = y_true[order]
    pos_total = max(y_sorted.sum(), 1)
    neg_total = max(len(y_sorted) - pos_total, 1)
    cum_pos = np.cumsum(y_sorted) / pos_total
    cum_neg = np.cumsum(1 - y_sorted) / neg_total
    return float(np.max(np.abs(cum_pos - cum_neg)))


def compute_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """주요 지표를 한 번에 계산해 dict 로 반환한다.

    Args:
        y_true: 실제 0/1 라벨.
        y_proba: 양성 확률.
        threshold: precision/recall/f1/accuracy 계산용 임계값.

    Returns:
        ``SUPPORTED_METRICS`` 키를 갖는 dict.
    """
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "ks": ks_statistic(y_true, y_proba),
    }


def score_metric(metric_name: str, y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """단일 지표 값을 반환한다 (best 모델 선정 등에 사용)."""
    if metric_name not in SUPPORTED_METRICS:
        raise ValueError(
            f"Unsupported metric '{metric_name}'. Supported: {SUPPORTED_METRICS}"
        )
    return compute_metrics(y_true, y_proba)[metric_name]


def confusion(y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """``[[TN, FP], [FN, TP]]`` 모양의 혼동행렬을 반환한다."""
    y_pred = (y_proba >= threshold).astype(int)
    return confusion_matrix(y_true, y_pred, labels=[0, 1])
