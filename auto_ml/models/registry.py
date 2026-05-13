"""모델 이름 → 클래스 매핑 팩토리.

설정 YAML 의 ``models:`` 섹션 키를 그대로 모델 식별자로 사용한다.
새로운 모델을 추가할 때는 본 파일에 등록만 하면 trainer / scorer
가 자동으로 인식한다.
"""
from __future__ import annotations

from typing import Any

from auto_ml.models.base import BaseModel
from auto_ml.models.catboost_model import CatBoostModel
from auto_ml.models.lgbm_model import LGBMModel
from auto_ml.models.xgb_model import XGBModel

# 등록된 모델 이름 → 클래스
_REGISTRY: dict[str, type[BaseModel]] = {
    "lgbm": LGBMModel,
    "xgb": XGBModel,
    "catboost": CatBoostModel,
}


def available_models() -> list[str]:
    """등록된 모델 이름 목록을 반환한다."""
    return sorted(_REGISTRY.keys())


def build_model(
    name: str,
    params: dict[str, Any] | None = None,
    categorical_columns: list[str] | None = None,
    random_state: int = 42,
    loss: str = "logloss",
) -> BaseModel:
    """모델 이름으로 인스턴스를 생성한다.

    Args:
        name: ``available_models()`` 중 하나.
        params: 모델 라이브러리 원본 하이퍼파라미터.
        categorical_columns: 범주형 컬럼.
        random_state: 시드.
        loss: 손실 함수 식별자 ("logloss" | "focal"). 기본 logloss.

    Returns:
        BaseModel 인스턴스 (아직 fit 되지 않은 상태).

    Raises:
        KeyError: 등록되지 않은 이름이 들어올 경우.
    """
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown model '{name}'. Available: {available_models()}"
        )
    cls = _REGISTRY[name]
    return cls(
        params=params,
        categorical_columns=categorical_columns,
        random_state=random_state,
        loss=loss,
    )
