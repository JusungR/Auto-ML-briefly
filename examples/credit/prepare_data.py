"""UCI 'Default of Credit Card Clients' (Yeh & Lien, 2009) 데이터 준비.

30,000행 × 23 features 의 이진 분류 데이터셋. Titanic 보다 약 34배 큰 규모로
파이프라인의 skew 변환·튜닝 안정성을 검증하기 좋다.

데이터 소스 우선순위:
    1) 환경변수 ``CREDIT_CARD_CSV`` 가 가리키는 로컬 CSV (폐쇄망)
    2) GitHub mirror (네트워크 1회)

산출:
    examples/credit/data/train.parquet        — 학습용 (target 포함)
    examples/credit/data/test.parquet         — 테스트용 (target 포함)
    examples/credit/data/score_input.parquet  — 스코어링용 (target 제외, id 포함)
"""
from __future__ import annotations

import io
import os
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd
from sklearn.model_selection import train_test_split

# 학습/스코어링에 사용할 컬럼 (UCI 원본 명명을 유지)
NUMERIC_FEATURES = [
    "LIMIT_BAL", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
]
CATEGORICAL_FEATURES = ["SEX", "EDUCATION", "MARRIAGE"]
ORIG_TARGET = "default.payment.next.month"
TARGET_COLUMN = "default"
ID_COLUMN = "client_id"

# 예시는 self-contained — 데이터도 examples/credit/data 하위에 둔다.
OUT_DIR = Path(__file__).resolve().parent / "data"
RANDOM_STATE = 42
GITHUB_URL = (
    "https://raw.githubusercontent.com/thomasXwang/UCI-Credit-card-defaults/"
    "master/UCI_Credit_Card.csv"
)


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    if ORIG_TARGET not in df.columns:
        raise ValueError(f"input CSV missing target column '{ORIG_TARGET}'")
    df = df.rename(columns={ORIG_TARGET: TARGET_COLUMN, "ID": ID_COLUMN})
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)
    # 범주형은 문자열로 강제 — auto_ml 이 카테고리 컬럼을 model wrapper 수준에서
    # native/LabelEncoder 로 처리하므로 일관된 dtype 이 필요하다.
    for c in CATEGORICAL_FEATURES:
        if c in df.columns:
            df[c] = df[c].astype(int).astype(str)
    return df


def _from_local_csv(path: str | os.PathLike) -> pd.DataFrame:
    return _normalize(pd.read_csv(path))


def _from_github() -> pd.DataFrame:
    req = Request(GITHUB_URL, headers={"User-Agent": "auto-ml-credit-fetch/1.0"})
    with urlopen(req, timeout=30) as resp:
        raw = resp.read()
    return _normalize(pd.read_csv(io.BytesIO(raw)))


def _load() -> pd.DataFrame:
    local = os.environ.get("CREDIT_CARD_CSV")
    if local:
        print(f"Loading credit data from local CSV: {local}")
        return _from_local_csv(local)
    print(f"Loading credit data from GitHub: {GITHUB_URL}")
    return _from_github()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = _load()
    keep = [ID_COLUMN] + NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET_COLUMN]
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise ValueError(f"source data missing columns: {missing}")
    df = df[keep].copy()

    print(f"Loaded {len(df)} rows, {df.shape[1]} columns")
    print(f"Target balance: {df[TARGET_COLUMN].value_counts(normalize=True).round(4).to_dict()}")
    print("Top skewed numeric columns (|skew| sorted):")
    sk = df[NUMERIC_FEATURES].skew().abs().sort_values(ascending=False)
    print(sk.head(10).round(2).to_string())

    df_train, df_test = train_test_split(
        df, test_size=0.2, random_state=RANDOM_STATE, stratify=df[TARGET_COLUMN],
    )
    df_train.to_parquet(OUT_DIR / "train.parquet", index=False)
    df_test.to_parquet(OUT_DIR / "test.parquet", index=False)
    df_test.drop(columns=[TARGET_COLUMN]).to_parquet(
        OUT_DIR / "score_input.parquet", index=False,
    )
    print(
        f"Wrote {OUT_DIR}/train.parquet         (n={len(df_train)})\n"
        f"Wrote {OUT_DIR}/test.parquet          (n={len(df_test)})\n"
        f"Wrote {OUT_DIR}/score_input.parquet   (n={len(df_test)})"
    )


if __name__ == "__main__":
    main()
