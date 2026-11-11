"""더미 학습 / 테스트 / 스코어링 데이터를 Parquet 으로 생성하는 헬퍼.

폐쇄망 이전 단계의 로컬 검증용 — 실제 운영 데이터 대신 사용한다.
configs/example.yaml 의 features 정의와 컬럼 명이 일치하도록 만든다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification


def _make_frame(n: int, seed: int = 42) -> pd.DataFrame:
    """n 행짜리 DataFrame 을 한 번에 생성한다.

    train / test / score 는 이 결과를 나눠 쓴다. make_classification 은
    random_state 가 다르면 생성 규칙(어떤 변수가 어떤 방향으로 타깃과 관련되는지)
    자체가 달라지므로, 파일별로 따로 생성하면 학습과 평가의 분포가 어긋난다.
    """
    rng = np.random.default_rng(seed)
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
    df["user_id"] = np.arange(n)          # 파일 간에도 겹치지 않는 식별자
    df["target"] = y.astype(int)
    return df


def main() -> None:
    out_dir = Path("./data")
    out_dir.mkdir(parents=True, exist_ok=True)

    n_train, n_test, n_score = 4000, 1000, 1000
    df = _make_frame(n=n_train + n_test + n_score)
    df_train = df.iloc[:n_train]
    df_test = df.iloc[n_train:n_train + n_test]
    df_score = df.iloc[n_train + n_test:].drop(columns=["target"])

    df_train.to_parquet(out_dir / "train.parquet", index=False)
    df_test.to_parquet(out_dir / "test.parquet", index=False)
    df_score.to_parquet(out_dir / "score_input.parquet", index=False)
    print("Wrote ./data/train.parquet, ./data/test.parquet, ./data/score_input.parquet")


if __name__ == "__main__":
    main()
