"""더미 학습/스코어링 데이터를 Parquet 으로 생성하는 헬퍼.

폐쇄망 이전 단계의 로컬 검증용 — 실제 운영 데이터 대신 사용한다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification


def main() -> None:
    rng = np.random.default_rng(42)
    n = 5000

    X, y = make_classification(
        n_samples=n,
        n_features=10,
        n_informative=6,
        n_redundant=2,
        weights=[0.7, 0.3],
        random_state=42,
    )
    df = pd.DataFrame(X, columns=[f"num_{i}" for i in range(X.shape[1])])
    # 범주형 / 식별자 컬럼 추가
    df["gender"] = rng.choice(["M", "F"], size=n)
    df["device"] = rng.choice(["ios", "android", "web", None], size=n, p=[0.4, 0.4, 0.15, 0.05])
    df["user_id"] = np.arange(n)
    df["target"] = y.astype(int)

    out_dir = Path("./data")
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "train.parquet", index=False)
    # 스코어링 입력은 target 을 제외
    df.drop(columns=["target"]).iloc[:1000].to_parquet(out_dir / "score_input.parquet", index=False)
    print("Wrote ./data/train.parquet and ./data/score_input.parquet")


if __name__ == "__main__":
    main()
