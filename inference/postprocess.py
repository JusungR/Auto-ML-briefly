"""[추론 3/3] 후처리.

예측 산출물에 scoring.threshold 를 적용해 0/1 prediction 을 만들고
id 컬럼과 결합해 scoring.output_path 에 Parquet 으로 저장한다.

    python inference/postprocess.py --config configs/example.yaml
"""
import sys
from pathlib import Path

# 설치 없이 실행: 저장소 루트를 import 경로에 추가한다.
# (스크립트 디렉토리 대신 루트를 넣어 preprocess.py 등 파일명이 모듈을 가리지 않게 한다.)
sys.path[0] = str(Path(__file__).resolve().parents[1])

from auto_ml.stages import inference as stages  # noqa: E402
from auto_ml.stages import load_stage  # noqa: E402
from auto_ml.stages.cli import parse_stage_args, start_stage_logging  # noqa: E402


def main(argv: list[str] | None = None) -> Path:
    config, work_dir = parse_stage_args("[inference 3/3] postprocess", "inference", argv)
    start_stage_logging(config, "inference_postprocess")
    pred = load_stage(work_dir, stages.PREDICT_OUTPUT)
    path = stages.postprocess(config, pred)
    print(f"scored output: {path}")
    return path


if __name__ == "__main__":
    main()
