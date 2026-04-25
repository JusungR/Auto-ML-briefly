"""YAML 으로 받은 search space 명세를 Optuna trial 의 ``suggest_*`` 호출로 변환.

지원하는 ``type`` 과 추가 키:
    - float       : low, high, [log: bool]
    - int         : low, high, [log: bool], [step: int]
    - categorical : choices: list

사용 예 (YAML):
    learning_rate:
      type: float
      low: 0.01
      high: 0.3
      log: true
    num_leaves:
      type: int
      low: 15
      high: 127
"""
from __future__ import annotations

from typing import Any

import optuna

# 허용되는 type 토큰 — validate / sample 양쪽에서 공유
_VALID_TYPES = {"float", "int", "categorical"}


def validate_search_space(space: dict[str, dict[str, Any]]) -> None:
    """search_space dict 의 형식을 사전 검증한다.

    설정 파일 작성 실수를 학습 시작 전에 잡아내기 위함.

    Raises:
        ValueError: 알 수 없는 type 이거나 필수 키가 누락된 경우.
    """
    for name, spec in space.items():
        if not isinstance(spec, dict):
            raise ValueError(f"search_space[{name}] must be a dict, got {type(spec)}")
        t = spec.get("type")
        if t not in _VALID_TYPES:
            raise ValueError(
                f"search_space[{name}].type must be one of {_VALID_TYPES}, got {t!r}"
            )
        if t in {"float", "int"}:
            if "low" not in spec or "high" not in spec:
                raise ValueError(
                    f"search_space[{name}] requires 'low' and 'high' for type={t}"
                )
        elif t == "categorical":
            if not spec.get("choices"):
                raise ValueError(
                    f"search_space[{name}] requires non-empty 'choices' for categorical"
                )


def sample_params(
    trial: "optuna.Trial",
    space: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """trial 객체에 ``suggest_*`` 를 호출해 하나의 파라미터 셋을 샘플링한다.

    Args:
        trial: Optuna trial.
        space: validate_search_space 를 통과한 search space dict.

    Returns:
        ``{param_name: sampled_value}`` 형태의 dict.
    """
    params: dict[str, Any] = {}
    for name, spec in space.items():
        t = spec["type"]
        if t == "float":
            params[name] = trial.suggest_float(
                name,
                float(spec["low"]),
                float(spec["high"]),
                log=bool(spec.get("log", False)),
            )
        elif t == "int":
            params[name] = trial.suggest_int(
                name,
                int(spec["low"]),
                int(spec["high"]),
                step=int(spec.get("step", 1)),
                log=bool(spec.get("log", False)),
            )
        elif t == "categorical":
            params[name] = trial.suggest_categorical(name, list(spec["choices"]))
        else:  # pragma: no cover — validate_search_space 에서 이미 차단
            raise ValueError(f"Unknown search space type: {t}")
    return params
