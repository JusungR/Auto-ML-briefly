"""타이타닉 데이터 준비 스크립트.

표준 타이타닉 CSV 를 가져와 80/20 으로 분할해 auto_ml 파이프라인이 요구하는
Parquet 형태로 저장한다.

데이터 소스 우선순위:
    1) 환경변수 ``TITANIC_CSV`` 가 가리키는 로컬 CSV
    2) GitHub: datasciencedojo/datasets/master/titanic.csv (네트워크 1회)
    3) sklearn ``fetch_openml('titanic', version=1)`` (캐시 사용)

폐쇄망 운영에서는 1) 경로로 사전 배포한 CSV 를 사용한다.

산출:
    data/titanic/train.parquet        — 학습용 (target 포함)
    data/titanic/test.parquet         — 테스트용 (target 포함)
    data/titanic/score_input.parquet  — 스코어링용 (target 제외, id 포함)
"""
from __future__ import annotations

import io
import os
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# 본 라이브러리에서 학습/스코어링에 사용할 컬럼만 추려 보관한다.
# (Cabin / Ticket / Name 등은 결측·고차원 카디널리티 때문에 제외)
KEEP_COLUMNS = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
TARGET_COLUMN = "survived"
ID_COLUMN = "passenger_id"

OUT_DIR = Path("./data/titanic")
RANDOM_STATE = 42
GITHUB_URL = (
    "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
)


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """원본 컬럼명이 PascalCase / snake_case 어느 쪽이든 통일된 스키마로 변환."""
    df = df.rename(columns={c: c.lower() for c in df.columns})
    # GitHub 본은 'pclass' 와 같이 이미 소문자지만, OpenML 본은 동일 — 그대로 사용 가능.
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"input CSV missing target column '{TARGET_COLUMN}'")
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)
    return df


def _from_local_csv(path: str | os.PathLike) -> pd.DataFrame:
    """사용자가 사전에 배포한 로컬 CSV 를 사용한다 (폐쇄망용)."""
    df = pd.read_csv(path)
    return _normalize(df)


def _from_github() -> pd.DataFrame:
    """GitHub 공개 저장소에서 표준 타이타닉 CSV 를 받는다."""
    req = Request(GITHUB_URL, headers={"User-Agent": "auto-ml-titanic-fetch/1.0"})
    with urlopen(req, timeout=15) as resp:
        raw = resp.read()
    return _normalize(pd.read_csv(io.BytesIO(raw)))


def _from_openml() -> pd.DataFrame:
    """fallback: sklearn fetch_openml (네트워크 또는 캐시)."""
    from sklearn.datasets import fetch_openml

    bunch = fetch_openml("titanic", version=1, as_frame=True)
    return _normalize(bunch.frame.copy())


def _load() -> pd.DataFrame:
    """우선순위에 따라 타이타닉 데이터를 로드한다."""
    local = os.environ.get("TITANIC_CSV")
    if local:
        print(f"Loading Titanic from local CSV: {local}")
        return _from_local_csv(local)
    try:
        print(f"Loading Titanic from GitHub: {GITHUB_URL}")
        return _from_github()
    except Exception as e:
        print(f"  GitHub fetch failed ({e!r}); falling back to fetch_openml")
        return _from_openml()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = _load()
    # 사용 컬럼 + target 만 남기고 식별자 부여
    missing = [c for c in KEEP_COLUMNS + [TARGET_COLUMN] if c not in df.columns]
    if missing:
        raise ValueError(f"source data missing columns: {missing}")
    df = df[KEEP_COLUMNS + [TARGET_COLUMN]].copy()
    df.insert(0, ID_COLUMN, np.arange(len(df), dtype=np.int64))

    print(f"Loaded {len(df)} rows, columns={list(df.columns)}")
    print("Missing per column:")
    print(df.isna().sum().to_string())

    df_train, df_test = train_test_split(
        df,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=df[TARGET_COLUMN],
    )

    df_train.to_parquet(OUT_DIR / "train.parquet", index=False)
    df_test.to_parquet(OUT_DIR / "test.parquet", index=False)
    df_test.drop(columns=[TARGET_COLUMN]).to_parquet(
        OUT_DIR / "score_input.parquet", index=False
    )

    print(f"Wrote {OUT_DIR}/train.parquet         (n={len(df_train)})")
    print(f"Wrote {OUT_DIR}/test.parquet          (n={len(df_test)})")
    print(f"Wrote {OUT_DIR}/score_input.parquet   (n={len(df_test)})")


if __name__ == "__main__":
    main()
