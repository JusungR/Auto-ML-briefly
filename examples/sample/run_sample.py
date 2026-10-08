"""실행 샘플 — 데이터 생성부터 학습 3단계, 추론 3단계까지 순서대로 실행한다.

    python examples/sample/run_sample.py

어느 디렉토리에서 실행해도 되며, 내부적으로 저장소 루트에서 다음 명령을 차례로 호출한다
(각 명령은 터미널에서 직접 실행해도 결과가 같다)::

    python train/preprocess.py      --config examples/sample/config.yaml
    python train/train.py           --config examples/sample/config.yaml
    python train/postprocess.py     --config examples/sample/config.yaml
    python inference/preprocess.py  --config examples/sample/config.yaml
    python inference/predict.py     --config examples/sample/config.yaml
    python inference/postprocess.py --config examples/sample/config.yaml

산출물은 ``artifacts/sample/`` 아래에 생긴다.

옵션:
    --skip-data   이미 만든 examples/sample/data/*.parquet 를 그대로 사용
    --only train | inference
                  학습 또는 추론 단계만 실행 (inference 는 학습 산출물이 있어야 한다)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

ROOT = Path(__file__).resolve().parents[2]
SAMPLE_DIR = Path(__file__).resolve().parent
DATA_DIR = SAMPLE_DIR / "data"
CONFIG = "examples/sample/config.yaml"          # 저장소 루트 기준

TRAIN_STEPS = [
    ("학습 1/3 전처리", "train/preprocess.py"),
    ("학습 2/3 학습", "train/train.py"),
    ("학습 3/3 후처리", "train/postprocess.py"),
]
INFERENCE_STEPS = [
    ("추론 1/3 전처리", "inference/preprocess.py"),
    ("추론 2/3 예측", "inference/predict.py"),
    ("추론 3/3 후처리", "inference/postprocess.py"),
]


# ----------------------------------------------------------------------
# 0) 샘플 데이터 생성
# ----------------------------------------------------------------------
def _make_frame(n: int, seed: int = 0) -> pd.DataFrame:
    """고객 이탈 형태의 합성 이진분류 데이터 (양성 비율 약 25%).

    train / test / score 가 같은 분포를 갖도록 한 번에 생성한 뒤 나눠 쓴다
    (make_classification 은 random_state 가 다르면 생성 규칙 자체가 달라진다).
    """
    rng = np.random.default_rng(seed)
    X, y = make_classification(
        n_samples=n,
        n_features=6,
        n_informative=4,
        n_redundant=1,
        weights=[0.75, 0.25],
        random_state=seed,
    )
    df = pd.DataFrame(X, columns=[f"num_{i}" for i in range(6)])
    # 오른쪽 꼬리가 긴 금액 컬럼 (skew 변환 대상) — 양성일수록 약간 크다
    df["amount"] = rng.lognormal(mean=3.0 + 0.4 * y, sigma=1.0)
    df["region"] = rng.choice(["north", "south", "east", "west"], size=n)
    # 결측이 섞인 범주형 (결측 대체 대상)
    df["channel"] = rng.choice(["app", "web", "store", None], size=n, p=[0.45, 0.35, 0.15, 0.05])
    # features.csv 에서 used=false — 학습에 쓰이지 않는 컬럼 예시
    df["memo_code"] = rng.choice(["A", "B"], size=n)
    # 수치형 결측
    df.loc[rng.random(n) < 0.03, "num_0"] = np.nan
    df["customer_id"] = np.arange(100_000, 100_000 + n)
    df["target"] = y.astype(int)
    return df


def make_data() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = _make_frame(n=2800)
    train = df.iloc[:2000]
    test = df.iloc[2000:2500]
    score_input = df.iloc[2500:].drop(columns=["target"])
    train.to_parquet(DATA_DIR / "train.parquet", index=False)
    test.to_parquet(DATA_DIR / "test.parquet", index=False)
    score_input.to_parquet(DATA_DIR / "score_input.parquet", index=False)
    print(
        f"  train={len(train)} rows, test={len(test)} rows, "
        f"score_input={len(score_input)} rows -> {DATA_DIR.relative_to(ROOT)}/"
    )


# ----------------------------------------------------------------------
# 1~6) 단계 스크립트 실행
# ----------------------------------------------------------------------
def run_step(title: str, script: str) -> None:
    cmd = [sys.executable, script, "--config", CONFIG]
    print(f"\n=== {title}: python {script} --config {CONFIG}")
    started = time.perf_counter()
    proc = subprocess.run(cmd, cwd=ROOT)
    if proc.returncode != 0:
        sys.exit(f"[실패] {script} (exit={proc.returncode}) — 위 로그를 확인하세요.")
    print(f"=== {title} 완료 ({time.perf_counter() - started:.1f}s)")


def show_results(ran_train: bool, ran_inference: bool) -> None:
    out = ROOT / "artifacts" / "sample"
    print("\n" + "=" * 60)
    print("샘플 실행 완료 — 주요 산출물")
    print("=" * 60)
    paths = []
    if ran_train:
        paths += [
            out / "models" / "best.joblib",
            out / "reports" / "report.html",
            out / "predictions" / "test_predictions.parquet",
            out / "work" / "train",
        ]
    if ran_inference:
        paths += [out / "scores" / "scores.parquet", out / "work" / "inference"]
    paths.append(out / "logs")
    for p in paths:
        mark = "OK " if p.exists() else "-- "
        print(f"  {mark}{p.relative_to(ROOT)}")

    if ran_inference:
        scores = pd.read_parquet(out / "scores" / "scores.parquet")
        print(f"\nscores.parquet 상위 5행 (전체 {len(scores)}행):")
        print(scores.head().to_string(index=False))
        print(f"\nprediction=1 비율: {scores['prediction'].mean():.1%}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-ML 실행 샘플")
    parser.add_argument("--skip-data", action="store_true", help="샘플 데이터 재생성 생략")
    parser.add_argument("--only", choices=["train", "inference"], default=None)
    args = parser.parse_args()

    if not args.skip_data and args.only != "inference":
        print("=== 0) 샘플 데이터 생성")
        make_data()

    run_train = args.only in (None, "train")
    run_inference = args.only in (None, "inference")
    if run_train:
        for title, script in TRAIN_STEPS:
            run_step(title, script)
    if run_inference:
        for title, script in INFERENCE_STEPS:
            run_step(title, script)
    show_results(run_train, run_inference)


if __name__ == "__main__":
    main()
