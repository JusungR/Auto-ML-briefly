"""[추론 2/3] 예측.

전처리 산출물을 읽어 best 모델로 양성 확률(score) 을 계산한다.
결과는 work_dir/inference_predicted.pkl.gz 로 저장된다.

    python inference/predict.py --config configs/example.yaml
"""
import sys
from pathlib import Path

# 설치 없이 실행: 저장소 루트를 import 경로에 추가한다.
# (스크립트 디렉토리 대신 루트를 넣어 preprocess.py 등 파일명이 모듈을 가리지 않게 한다.)
sys.path[0] = str(Path(__file__).resolve().parents[1])

from auto_ml.stages import inference as stages  # noqa: E402
from auto_ml.stages import load_stage, save_stage  # noqa: E402
from auto_ml.stages.cli import parse_stage_args, start_stage_logging  # noqa: E402


def main(argv: list[str] | None = None) -> Path:
    config, work_dir = parse_stage_args("[inference 2/3] predict", "inference", argv)
    start_stage_logging(config, "inference_predict")
    prep = load_stage(work_dir, stages.PREPROCESS_OUTPUT)
    pred = stages.predict(prep)
    path = save_stage(pred, work_dir, stages.PREDICT_OUTPUT)
    print(f"predicted: {path}")
    return path


if __name__ == "__main__":
    main()
