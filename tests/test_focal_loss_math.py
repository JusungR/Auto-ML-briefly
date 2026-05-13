"""Focal Loss 수학 단위 테스트.

검증 대상:
- ``focal_grad_hess`` 의 shape, 유한성, ``hess > 0``.
- 부호: 잘못 분류된 양성/음성에서 grad 가 올바른 방향.
- ``gamma=0`` 일 때 alpha-weighted BCE 로 환원.
- ``_sigmoid`` 의 안정성 (큰 절댓값 입력).
"""
from __future__ import annotations

import numpy as np
import pytest

from auto_ml.models.losses import _sigmoid, focal_grad_hess


class TestSigmoid:
    def test_basic_values(self):
        z = np.array([-100.0, -1.0, 0.0, 1.0, 100.0])
        s = _sigmoid(z)
        # 양 끝단도 유한, [0, 1]
        assert np.all(np.isfinite(s))
        assert np.all((s >= 0.0) & (s <= 1.0))
        # 중간값 검증
        assert s[2] == pytest.approx(0.5, abs=1e-12)
        assert s[3] == pytest.approx(1.0 / (1.0 + np.exp(-1.0)), abs=1e-12)

    def test_no_overflow_on_extreme_inputs(self):
        # exp(710) 은 overflow 지만 _sigmoid 는 분기로 회피
        z = np.array([-1000.0, 1000.0])
        s = _sigmoid(z)
        assert np.all(np.isfinite(s))
        assert s[0] == pytest.approx(0.0, abs=1e-10)
        assert s[1] == pytest.approx(1.0, abs=1e-10)


class TestFocalGradHess:
    def test_shape_and_finiteness(self):
        y = np.array([0, 0, 1, 1])
        z = np.array([-2.0, 1.5, -1.0, 2.0])
        g, h = focal_grad_hess(y, z, gamma=2.0, alpha=0.25)
        assert g.shape == z.shape
        assert h.shape == z.shape
        assert np.all(np.isfinite(g))
        assert np.all(np.isfinite(h))

    def test_hess_positive(self):
        # 다양한 입력에 대해 hess > 0 (lower-clip 동작)
        y = np.array([0, 1, 0, 1, 0, 1])
        z = np.array([-10.0, -5.0, 0.0, 0.0, 5.0, 10.0])
        _, h = focal_grad_hess(y, z, gamma=2.0, alpha=0.25)
        assert np.all(h > 0)

    def test_grad_sign(self):
        """잘못 분류된 샘플의 grad 방향이 올바른지 확인."""
        y = np.array([0, 0, 1, 1])
        z = np.array([-2.0, 1.5, -1.0, 2.0])
        g, _ = focal_grad_hess(y, z, gamma=2.0, alpha=0.25)
        # y=1, z=-1: 양성을 음성으로 잘못 분류 → z 를 올려야 함 → grad < 0
        assert g[2] < 0
        # y=0, z=1.5: 음성을 양성으로 잘못 분류 → z 를 내려야 함 → grad > 0
        assert g[1] > 0
        # y=0, z=-2: 정확히 분류 → grad 크기가 매우 작음
        assert abs(g[0]) < abs(g[1])
        # y=1, z=2: 정확히 분류 → grad 크기가 매우 작음
        assert abs(g[3]) < abs(g[2])

    def test_gamma_zero_reduces_to_alpha_bce(self):
        """gamma=0 이면 focal = alpha 가중 BCE → grad = a_t * (p - y)."""
        y = np.array([0, 1, 0, 1])
        z = np.array([0.5, -0.3, 1.2, -0.8])
        # alpha=0.5 → a_t 가 양/음 모두 0.5 (균등) → 단순 0.5 * (p - y)
        g, _ = focal_grad_hess(y, z, gamma=0.0, alpha=0.5)
        p = _sigmoid(z)
        expected = 0.5 * (p - y)
        np.testing.assert_allclose(g, expected, atol=1e-7)

    def test_alpha_weights_positive_class(self):
        """alpha 가 크면 양성 클래스의 grad 가 같은 |z| 에서 더 크다."""
        # 같은 |z| 의 정답 분류 양/음 한 쌍
        y = np.array([0, 1])
        z = np.array([-1.0, 1.0])  # 둘 다 p_t = sigmoid(1) ~ 0.73
        g_balanced, _ = focal_grad_hess(y, z, gamma=2.0, alpha=0.5)
        # alpha=0.5 면 양/음 가중이 같으므로 |grad| 가 같아야 함 (대칭성)
        assert abs(g_balanced[0]) == pytest.approx(abs(g_balanced[1]), rel=1e-10)

        g_unbalanced, _ = focal_grad_hess(y, z, gamma=2.0, alpha=0.25)
        # alpha=0.25 → a_t(음성)=0.75, a_t(양성)=0.25 → 음성 grad 가 더 큼
        assert abs(g_unbalanced[0]) > abs(g_unbalanced[1])

    def test_focal_modulator_damps_easy_examples(self):
        """잘 맞춘 (easy) 샘플의 grad 가 어려운 샘플보다 훨씬 작다."""
        y = np.array([1, 1])
        z = np.array([3.0, 0.0])  # easy / hard
        g, _ = focal_grad_hess(y, z, gamma=2.0, alpha=0.25)
        # easy 의 |grad| < hard 의 |grad|, 그리고 비율이 10배 이상이어야
        # (1-p_t)^2 감쇠로 충분히 차이가 나야 함
        assert abs(g[0]) < abs(g[1])
        assert abs(g[1]) / abs(g[0]) > 10.0
