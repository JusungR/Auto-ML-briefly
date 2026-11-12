"""[학습 2/3] 학습.

전처리 산출물을 읽어 모델 학습 (튜닝 → CV → 최종 fit) + best 선정.
결과는 work_dir/train_result.pkl.gz 로 저장된다.

    python train/train.py --config configs/example.yaml
"""
import sys
from pathlib import Path

# 설치 없이 실행: 저장소 루트를 import 경로에 추가한다.
# (스크립트 디렉토리 대신 루트를 넣어 preprocess.py 등 파일명이 모듈을 가리지 않게 한다.)
sys.path[0] = str(Path(__file__).resolve().parents[1])

from auto_ml.stages import load_stage, save_stage  # noqa: E402
from auto_ml.stages import train as stages  # noqa: E402
from auto_ml.stages.cli import parse_stage_args, start_stage_logging  # noqa: E402


def main(argv: list[str] | None = None) -> Path:
    config, work_dir = parse_stage_args("[train 2/3] train", "train", argv)
    start_stage_logging(config, "train_train")
    prep = load_stage(work_dir, stages.PREPROCESS_OUTPUT)
    result = stages.train(config, prep)
    path = save_stage(result, work_dir, stages.TRAIN_OUTPUT)
    print(f"trained: {path} (best={result.best_model_name})")
    return path


if __name__ == "__main__":
    main()
