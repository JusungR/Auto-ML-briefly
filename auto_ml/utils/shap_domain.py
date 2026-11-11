"""raw-margin(logit) SHAP 을 확률 도메인으로 바꾸는 공용 변환.

부스팅/선형 모델의 native SHAP 은 logit 도메인이다. 다음 scale-and-shift 로
확률 도메인으로 바꾸면 행마다 ``base + sum(shap) == sigmoid(margin)`` 이 정확히
성립한다.

    p_base = sigmoid(base_raw)
    p_pred = sigmoid(base_raw + sum(feature_raw))
    shap_prob[i] = feature_raw[i] / sum(feature_raw) * (p_pred - p_base)   (sum != 0)

Explainer(단일 모델) 와 EnsembleModel(서브모델별 변환 후 가중 평균) 이 함께 쓴다.
"""
from __future__ import annotations

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    """overflow 안전한 sigmoid (np.exp 인자를 항상 0 이하로 유지)."""
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def margin_shap_to_probability(
    shap_raw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(n, n_features + 1)`` raw SHAP (마지막 열 = base) 을 확률 도메인으로 바꾼다.

    Returns:
        ``(shap_prob (n, n_features), p_base (n,), p_pred (n,))``.
        ``p_base + shap_prob.sum(axis=1) == p_pred`` 가 행마다 성립한다.
    """
    shap_raw = np.asarray(shap_raw, dtype=float)
    feature_raw = shap_raw[:, :-1]
    base_raw = shap_raw[:, -1]
    sum_raw = feature_raw.sum(axis=1)

    p_base = sigmoid(base_raw)
    p_pred = sigmoid(base_raw + sum_raw)
    delta = p_pred - p_base
    # 0 나눗셈 회피 — sum_raw == 0 인 행은 delta 도 0 이므로 SHAP 도 모두 0
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(sum_raw != 0, delta / sum_raw, 0.0)
    return feature_raw * ratio[:, None], p_base, p_pred
