"""리포트 폰트 — PDF·차트에서 한글이 깨지지 않게 한다.

- PDF(WeasyPrint) 는 서버에 설치된 폰트(fontconfig) 로 글자를 그린다. 한글 폰트가 없으면
  한글이 빈 네모로 나온다. CSS 글꼴 목록은 대표적인 한글 폰트를 차례로 지정하고,
  ``reporting.font_path`` 로 폰트 파일을 직접 줄 수도 있다 (폐쇄망에서 폰트 파일만 복사).
- matplotlib 차트는 기본 폰트(DejaVu Sans) 에 한글이 없어 변수명이 한글이면 깨진다.
  설치된 한글 폰트를 찾아 등록하고 차트 글꼴로 쓴다.
"""
from __future__ import annotations

import functools
import shutil
import subprocess
from pathlib import Path

from auto_ml.utils.logger import get_logger

logger = get_logger("reporting.fonts")

# 리포트 CSS 에 쓰는 글꼴 이름. reporting.font_path 를 주면 이 이름으로 등록한다.
CUSTOM_FONT_FAMILY = "AutoMLReportFont"

# 한글을 포함한 sans 폰트 — 앞에 있을수록 우선 (CSS 글꼴 목록·matplotlib 공통)
KOREAN_FONT_FAMILIES = [
    "Noto Sans CJK KR", "Noto Sans KR", "NanumGothic", "NanumBarunGothic",
    "Malgun Gothic", "Apple SD Gothic Neo", "AppleGothic",
]
# fontconfig 가 없을 때(Windows·macOS) 찾아볼 폰트 파일
_COMMON_FONT_FILES = [
    "C:/Windows/Fonts/malgun.ttf",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    "/Library/Fonts/AppleGothic.ttf",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
]

_font_path: str | None = None


def configure(font_path: str | None) -> None:
    """``reporting.font_path`` 를 반영한다 (ReportBuilder 생성 시 호출)."""
    global _font_path
    path = str(font_path) if font_path else None
    if path and not Path(path).is_file():
        logger.warning("reporting.font_path not found: %s — using installed fonts", path)
        path = None
    if path != _font_path:
        _font_path = path
        chart_font_family.cache_clear()


def font_face_css() -> str:
    """``reporting.font_path`` 가 있으면 그 폰트를 맨 앞 글꼴로 쓰는 CSS (없으면 빈 문자열)."""
    if not _font_path:
        return ""
    uri = Path(_font_path).resolve().as_uri()
    return (
        f'@font-face {{ font-family: "{CUSTOM_FONT_FAMILY}"; src: url("{uri}"); }}\n'
        f'@font-face {{ font-family: "{CUSTOM_FONT_FAMILY}"; font-weight: bold; src: url("{uri}"); }}\n'
        f'body {{ font-family: "{CUSTOM_FONT_FAMILY}", Inter, '
        + ", ".join(f'"{f}"' for f in KOREAN_FONT_FAMILIES)
        + ', "DejaVu Sans", sans-serif; }\n'
    )


def _fontconfig_korean_files() -> list[str]:
    """fontconfig 에 등록된 한글 지원 폰트 파일 (sans 우선, serif·mono 는 뒤로)."""
    if shutil.which("fc-list") is None:
        return []
    try:
        out = subprocess.run(
            ["fc-list", ":lang=ko", "file"],
            capture_output=True, text=True, timeout=10, check=False,
        ).stdout
    except Exception:  # noqa: BLE001
        return []
    files = sorted({line.split(":")[0].strip() for line in out.splitlines() if line.strip()})

    def rank(path: str) -> tuple[int, str]:
        name = Path(path).name.lower()
        penalty = ("serif" in name) * 2 + ("mono" in name) + ("bold" in name)
        return penalty, name

    return sorted(files, key=rank)


def korean_font_available() -> bool | None:
    """PDF 에 쓸 한글 폰트가 있는지. 확인할 수 없으면 None (fontconfig 없음)."""
    if _font_path:
        return True
    if shutil.which("fc-list") is None:
        return None
    return bool(_fontconfig_korean_files())


@functools.lru_cache(maxsize=1)
def chart_font_family() -> list[str]:
    """matplotlib 에 쓸 글꼴 목록. 한글 폰트를 찾으면 등록해 맨 앞에 둔다."""
    from matplotlib import font_manager

    candidates = ([_font_path] if _font_path else []) + _fontconfig_korean_files() + [
        p for p in _COMMON_FONT_FILES if Path(p).is_file()
    ]
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for fam in KOREAN_FONT_FAMILIES:          # 이미 matplotlib 가 아는 한글 폰트
        if fam in installed and not _font_path:
            return [fam, "DejaVu Sans"]
    for path in candidates:
        try:
            font_manager.fontManager.addfont(path)
            name = font_manager.FontProperties(fname=path).get_name()
            return [name, "DejaVu Sans"]
        except Exception as exc:  # noqa: BLE001 — 읽을 수 없는 파일은 건너뛴다
            logger.debug("Skipping font %s: %s", path, exc)
            continue
    logger.warning(
        "No Korean font found for charts — Korean feature names may render as boxes. "
        "Install a Korean font (e.g. fonts-noto-cjk, fonts-nanum) or set reporting.font_path."
    )
    return ["DejaVu Sans"]
