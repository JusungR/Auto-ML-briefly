"""더미 학습 / 테스트 / 스코어링 데이터를 Parquet 으로 생성하는 헬퍼.

폐쇄망 이전 단계의 로컬 검증용 — 실제 운영 데이터 대신 사용한다.
configs/example.yaml 의 features 정의와 컬럼 명이 일치하도록 만든다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification


def _make_frame(n: int, rng: np.random.Generator, seed: int) -> pd.DataFrame:
    """동일 분포의 n 행짜리 DataFrame 을 생성한다."""
    X, y = make_classification(
        n_samples=n,
        n_features=10,
        n_informative=6,
        n_redundant=2,
        weights=[0.7, 0.3],
        random_state=seed,
    )
    df = pd.DataFrame(X, columns=[f"num_{i}" for i in range(X.shape[1])])
    df["gender"] = rng.choice(["M", "F"], size=n)
    # None 을 섞어 결측 처리 단계 동작도 검증
    df["device"] = rng.choice(["ios", "android", "web", None], size=n, p=[0.4, 0.4, 0.15, 0.05])
    df["user_id"] = np.arange(n)
    df["target"] = y.astype(int)
    return df


def main() -> None:
    out_dir = Path("./data")
    out_dir.mkdir(parents=True, exist_ok=True)

    rng_train = np.random.default_rng(42)
    rng_test = np.random.default_rng(43)
    rng_score = np.random.default_rng(44)

    df_train = _make_frame(n=4000, rng=rng_train, seed=42)
    df_test = _make_frame(n=1000, rng=rng_test, seed=43)
    df_score = _make_frame(n=1000, rng=rng_score, seed=44).drop(columns=["target"])

    df_train.to_parquet(out_dir / "train.parquet", index=False)
    df_test.to_parquet(out_dir / "test.parquet", index=False)
    df_score.to_parquet(out_dir / "score_input.parquet", index=False)
    print("Wrote ./data/train.parquet, ./data/test.parquet, ./data/score_input.parquet")


if __name__ == "__main__":
    main()
