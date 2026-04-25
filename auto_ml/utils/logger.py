"""작업 로그 관리.

콘솔(stdout) 출력 + 실행마다 별도 파일 기록을 동시에 지원한다.
폐쇄망 운영에서도 실행 이력을 사후에 추적할 수 있도록 stdlib ``logging``
만으로 구성한다 (외부 로그 수집 도구에 의존하지 않음).

사용 패턴:
    1) 모듈 상단에서 ``logger = get_logger("trainer")`` 로 로거 획득
       - import 시점에 stdout 부트스트랩 핸들러 1개가 자동 등록되므로
         설정이 늦어도 메시지가 유실되지 않는다.
    2) CLI / 파이프라인 진입점에서 ``setup_logging(...)`` 1회 호출
       - 핸들러를 재구성하여 파일 출력을 추가한다.
       - stage(`"train"` / `"score"`) 별로 파일명이 분리된다.
"""
from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

# 본 라이브러리의 모든 로거가 공유하는 루트 네임스페이스
_BASE = "auto_ml"
_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"

# 부트스트랩(stdout) 핸들러 중복 등록을 막기 위한 모듈 단위 플래그
_BOOTSTRAPPED = False


def _bootstrap() -> None:
    """``setup_logging`` 호출 전이라도 ``get_logger`` 가 동작하도록 stdout 핸들러 1개를 등록한다.

    setup_logging 이 다시 호출되면 핸들러가 모두 교체되므로 부트스트랩은
    1회만 의미가 있다.
    """
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return
    root = logging.getLogger(_BASE)
    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
        root.addHandler(handler)
    root.setLevel(logging.INFO)
    # 부모(루트) 로 전파되면 다른 라이브러리 핸들러와 중복 출력될 수 있다.
    root.propagate = False
    _BOOTSTRAPPED = True


def get_logger(name: str) -> logging.Logger:
    """``auto_ml.<name>`` 네임스페이스의 로거를 반환한다.

    Args:
        name: 모듈 식별자 (예: ``"trainer"``).

    Returns:
        설정이 완료된 ``logging.Logger`` 인스턴스.
    """
    _bootstrap()
    return logging.getLogger(f"{_BASE}.{name}")


def setup_logging(
    log_dir: str | Path = "./artifacts/logs",
    level: str = "INFO",
    to_stdout: bool = True,
    to_file: bool = True,
    stage: str = "run",
) -> Path | None:
    """루트 로거를 재구성한다 (stdout + 선택적 파일 핸들러).

    실행 시점 타임스탬프 기반 파일 이름을 사용해 같은 디렉토리에서 여러 실행
    이력이 충돌 없이 누적되도록 한다. 파일은 매 실행 새로 만든다 (rotation 없음 —
    실행 단위가 곧 단위 로그).

    Args:
        log_dir: 파일 로그 저장 디렉토리.
        level: 루트 로깅 레벨 (DEBUG / INFO / WARNING / ERROR).
        to_stdout: 콘솔 출력 여부.
        to_file: 파일 출력 여부.
        stage: 파일명 prefix. 예) ``"train"``, ``"score"``.

    Returns:
        파일 로그 경로 (to_file=False 면 None).
    """
    root = logging.getLogger(_BASE)
    # 부트스트랩 또는 이전 실행에서 남은 핸들러를 모두 제거 후 재구성
    for h in list(root.handlers):
        root.removeHandler(h)
        # FileHandler 는 파일을 쥐고 있으므로 명시적으로 닫아준다.
        try:
            h.close()
        except Exception:  # pragma: no cover - 닫기 실패는 치명적이지 않음
            pass

    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.propagate = False

    formatter = logging.Formatter(_FORMAT, datefmt=_DATEFMT)

    if to_stdout:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(formatter)
        root.addHandler(sh)

    log_file: Path | None = None
    if to_file:
        log_dir_path = Path(log_dir)
        log_dir_path.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir_path / f"{stage}_{ts}.log"
        fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        fh.setFormatter(formatter)
        root.addHandler(fh)

    global _BOOTSTRAPPED
    _BOOTSTRAPPED = True
    return log_file
