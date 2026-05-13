"""커스텀 손실 함수 모음 — 이진 분류용.

본 모듈은 LGBM / XGBoost / CatBoost 의 custom objective 인터페이스에 끼울 수
있는 손실 함수와 그 도함수를 제공한다. 현재는 Focal Loss (Lin et al., 2017,
ICCV) 만 지원한다.

설계 메모:
    - 모든 함수/클래스는 모듈 최상위에 정의된다 — joblib (pickle) 로 학습된
      모델을 저장한 뒤 별도 프로세스(``auto-ml-score``) 에서 불러올 때
      참조가 풀려야 하기 때문이다. lambda / nested closure 는 사용하지 않는다.
    - numpy 만 사용한다. 폐쇄망 호환을 위해 추가 의존성을 만들지 않는다.
    - 부호 규약: ``focal_grad_hess`` 는 표준 미적분 정의대로 ``dL/dz``,
      ``d^2L/dz^2`` 를 반환한다. CatBoost 어댑터(``CatBoostFocalObjective``)
      는 라이브러리 규약에 맞춰 부호를 뒤집어 반환한다.

수식 (이진):

    p = sigmoid(z),  p_t = p if y=1 else 1-p,  a_t = alpha if y=1 else 1-alpha
    L = -a_t * (1 - p_t)^gamma * log(p_t)

기본값은 원 논문 권장값 (gamma=2.0, alpha=0.25) 를 따른다.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

# 로그/나눗셈 안정화용 작은 상수. 너무 작으면 (1e-12 이하) double 정밀도에서
# 노이즈가 폭증하고, 너무 크면 (1e-4 이상) 양 끝단의 손실 신호가 약해진다.
_EPS = 1e-7


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """오버플로 안전한 sigmoid.

    ``z >= 0`` 과 ``z < 0`` 으로 분기해 양쪽 모두 ``exp`` 의 인자가 0 이하가
    되도록 한다 (overflow 회피).
    """
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    pos = z >= 0
    # 양수 영역: 1 / (1 + exp(-z))
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    # 음수 영역: exp(z) / (1 + exp(z))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def focal_grad_hess(
    y_true: np.ndarray,
    z: np.ndarray,
    gamma: float = 2.0,
    alpha: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    """이진 Focal Loss 의 raw margin ``z`` 에 대한 1차/2차 도함수.

    Args:
        y_true: shape (n,), 값은 0 또는 1 (bool/int/float 모두 허용).
        z: shape (n,), raw margin (sigmoid 전 logit).
        gamma: easy-example 감쇠 지수. 기본 2.0.
        alpha: 양성 클래스 가중. 기본 0.25.

    Returns:
        (grad, hess) — 둘 다 shape (n,) float64. ``hess`` 는 ``_EPS`` 로 lower-clip
        되어 항상 양수다 (부스터의 leaf split 계산이 0/0 으로 발산하지 않도록).
    """
    y = np.asarray(y_true, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)

    p = _sigmoid(z)
    # log(0), 1/0 방지. 양 끝단에서만 발동한다.
    p = np.clip(p, _EPS, 1.0 - _EPS)

    # p_t = 정답 클래스의 확률, a_t = 정답 클래스의 가중
    pt = np.where(y == 1.0, p, 1.0 - p)
    at = np.where(y == 1.0, alpha, 1.0 - alpha)
    one_minus_pt = np.clip(1.0 - pt, _EPS, 1.0)

    # (1 - p_t)^gamma — easy example 감쇠 항
    mod = np.power(one_minus_pt, gamma)

    # d(p_t)/dz = (2y-1) * p*(1-p)  (체인 룰)
    sign = 2.0 * y - 1.0
    dpt_dz = sign * p * (1.0 - p)

    # dL/dz = -a_t * d/dz[(1-p_t)^gamma * log(p_t)]
    #       = -a_t * [-gamma*(1-p_t)^(gamma-1)*log(p_t) + (1-p_t)^gamma / p_t] * dp_t/dz
    #       = a_t * (1-p_t)^gamma * dp_t/dz * [ gamma*log(p_t)/(1-p_t) - 1/p_t ]
    log_pt = np.log(pt)
    bracket = gamma * log_pt / one_minus_pt - 1.0 / pt
    grad = at * mod * dpt_dz * bracket

    # 2차 도함수는 닫힌 형태가 복잡해 수치적으로 불안정해지기 쉽다. GBDT 커뮤니티에서
    # 널리 쓰이는 양의 근사식을 사용한다:
    #   hess ≈ a_t * (1-p_t)^gamma * p*(1-p) * (1 + gamma*(1-p_t))
    # 이는 본래 hess 의 dominant 항이며, p_t -> 0 / 1 양 끝에서도 양수 / 유한값을 보장한다.
    hess = at * mod * p * (1.0 - p) * (1.0 + gamma * one_minus_pt)
    hess = np.maximum(hess, _EPS)
    return grad, hess


# ---------------------------------------------------------------------------
# 라이브러리별 어댑터 — 모두 모듈 최상위 클래스 (pickle 가능)
#
# 왜 functools.partial 이 아니라 callable 클래스인가:
#   LightGBM 은 custom objective 의 인자 개수를 ``inspect.signature`` 로 세서
#   2/3/4 개에 따라 다른 호출 분기를 탄다. ``partial(f, gamma=g, alpha=a)`` 에
#   ``inspect.signature`` 를 적용하면 *바인딩되지 않은* 인자 (gamma, alpha) 가
#   여전히 보여 4-인자 분기로 빠지고 충돌 (multiple values for 'gamma') 이 난다.
#   callable 클래스 인스턴스의 signature 는 __call__ 의 self 제외 인자만 보여
#   2-인자로 깔끔하게 잡힌다. CatBoost 어댑터도 동일한 패턴으로 통일한다.
# ---------------------------------------------------------------------------


class _BinaryFocalObjective:
    """LGBM / XGBoost sklearn API 공용 callable objective.

    호출 시그니처: ``__call__(y_true, y_pred) -> (grad, hess)``.

    LightGBM / XGBoost 모두 callable objective 일 때 ``y_pred`` 로 raw margin
    (sigmoid 전 logit) 을 전달한다.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25) -> None:
        self.gamma = float(gamma)
        self.alpha = float(alpha)

    def __call__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        return focal_grad_hess(y_true, y_pred, gamma=self.gamma, alpha=self.alpha)


def lgb_focal_objective(gamma: float = 2.0, alpha: float = 0.25) -> _BinaryFocalObjective:
    """LightGBM 용 focal objective 인스턴스 생성 헬퍼."""
    return _BinaryFocalObjective(gamma=gamma, alpha=alpha)


def xgb_focal_objective(gamma: float = 2.0, alpha: float = 0.25) -> _BinaryFocalObjective:
    """XGBoost 용 focal objective 인스턴스 생성 헬퍼."""
    return _BinaryFocalObjective(gamma=gamma, alpha=alpha)


class CatBoostFocalObjective:
    """CatBoost 의 user-defined loss 인터페이스.

    CatBoost 는 ``loss_function`` 에 ``calc_ders_range`` 메소드를 가진 객체를
    받는다. 반환은 ``[(der1_i, der2_i), ...]`` 리스트이며, CatBoost 의 부호 규약상
    ``der1`` 은 *negative gradient* 로, ``der2`` 는 *negative second derivative* 로
    해석된다 — 따라서 표준 정의의 ``(grad, hess)`` 에서 부호를 뒤집어 반환해야 한다.

    검증: 표준 logloss 의 ``(p - y, p(1-p))`` 를 그대로 반환하면 학습이 발산하지만,
    ``(-(p - y), -p(1-p))`` 를 반환하면 AUC ≈ 0.99 로 정상 수렴한다 (실측).
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25) -> None:
        self.gamma = float(gamma)
        self.alpha = float(alpha)

    def calc_ders_range(
        self,
        approxes: Iterable[float],
        targets: Iterable[float],
        weights: Iterable[float] | None,
    ):
        z = np.asarray(approxes, dtype=np.float64)
        y = np.asarray(targets, dtype=np.float64)
        grad, hess = focal_grad_hess(y, z, gamma=self.gamma, alpha=self.alpha)
        # CatBoost 부호 규약: -grad, -hess
        neg_grad = -grad
        neg_hess = -hess
        if weights is not None:
            w = np.asarray(weights, dtype=np.float64)
            neg_grad = neg_grad * w
            neg_hess = neg_hess * w
        # tuple of (der1, der2) 리스트로 반환 — CatBoost 의 공식 시그니처
        return list(zip(neg_grad.tolist(), neg_hess.tolist()))
