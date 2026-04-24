"""표준 라이브러리만 사용하는 최소한의 로깅 설정.

폐쇄 환경에서 외부 로그 수집기(예: ELK, Datadog) 의존성을 제거하기 위해
stdout 으로만 출력한다. 운영 환경에서는 파이프 리다이렉션이나 OS 레벨
로그 수집(rsyslog 등) 으로 충분히 수집 가능하다.
"""
from __future__ import annotations

import logging
import sys

# 모듈 단위 플래그: 핸들러 중복 등록을 막기 위함
_CONFIGURED = False


def _configure_root() -> None:
    """``auto_ml`` 루트 로거를 1회만 설정한다."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root = logging.getLogger("auto_ml")
    root.setLevel(logging.INFO)
    root.handlers.clear()
    root.addHandler(handler)
    # 부모 로거(루트) 로 전파하지 않아야 다른 핸들러와 중복 출력되지 않는다.
    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """``auto_ml.<name>`` 네임스페이스의 로거를 반환한다.

    Args:
        name: 모듈 식별자 (예: ``"trainer"``).

    Returns:
        설정이 완료된 ``logging.Logger`` 인스턴스.
    """
    _configure_root()
    return logging.getLogger(f"auto_ml.{name}")
