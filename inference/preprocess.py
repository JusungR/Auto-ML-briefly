"""[추론 1/3] 전처리.

artifact_dir/best.joblib 의 전처리기로 scoring.input_path 를 변환하고
변수 선택 결과에 맞춰 모델 입력을 만든다.
결과는 work_dir/inference_preprocessed.pkl.gz 로 저장된다.

    python inference/preprocess.py --config configs/example.yaml
"""
import sys
from pathlib import Path

# 설치 없이 실행: 저장소 루트를 import 경로에 추가한다.
# (스크립트 디렉토리 대신 루트를 넣어 preprocess.py 등 파일명이 모듈을 가리지 않게 한다.)
sys.path[0] = str(Path(__file__).resolve().parents[1])

from auto_ml.stages import inference as stages  # noqa: E402
from auto_ml.stages import save_stage  # noqa: E402
from auto_ml.stages.cli import parse_stage_args, start_stage_logging  # noqa: E402


def main(argv: list[str] | None = None) -> Path:
    config, work_dir = parse_stage_args("[inference 1/3] preprocess", "inference", argv)
    start_stage_logging(config, "inference_preprocess")
    prep = stages.preprocess(config)
    path = save_stage(prep, work_dir, stages.PREPROCESS_OUTPUT)
    print(f"preprocessed: {path}")
    return path


if __name__ == "__main__":
    main()
