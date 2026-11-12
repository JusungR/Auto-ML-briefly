"""CPU 스레드 수 설정.

LightGBM·XGBoost·CatBoost 는 스레드 수를 지정하지 않으면 프로세스마다 전체 코어를 쓴다.
같은 서버에서 학습·스코어링이 동시에 돌면 스레드 합이 코어 수를 크게 넘고, OpenMP
스레드들이 서로를 기다리며 수십 배 느려질 수 있다. 설정의 ``n_jobs`` 를 각 라이브러리
인자로 바꾸는 규칙을 한 곳에 모은다.
"""
from __future__ import annotations

import os


def available_cpus() -> int:
    """이 프로세스가 쓸 수 있는 CPU 수 (CPU affinity 반영, 없으면 os.cpu_count)."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):  # Windows / macOS
        return max(1, os.cpu_count() or 1)


def resolve_n_jobs(n_jobs: int | None) -> int | None:
    """설정값을 실제 스레드 수로 바꾼다.

    - ``None`` → None (라이브러리 기본 = 전체 코어, 현행 동작 유지)
    - 양수 → 그대로 (가용 CPU 수를 넘으면 가용 CPU 수로 줄임)
    - 음수 → sklearn 규칙: ``-1`` = 전체, ``-2`` = 전체 - 1, … (최소 1)
    - 0 → ValueError
    """
    if n_jobs is None:
        return None
    n = int(n_jobs)
    if n == 0:
        raise ValueError("n_jobs must be a positive or negative integer (or null), not 0")
    cpus = available_cpus()
    if n < 0:
        return max(1, cpus + 1 + n)
    return min(n, cpus)
