"""[학습 3/3] 후처리.

전처리·학습 산출물을 읽어 HTML/PDF 리포트 생성, artifact 저장
(artifact_dir/best.joblib + models/<name>.joblib), test 예측 export.

    python train/postprocess.py --config configs/example.yaml
"""
import sys
from pathlib import Path

# 설치 없이 실행: 저장소 루트를 import 경로에 추가한다.
# (스크립트 디렉토리 대신 루트를 넣어 preprocess.py 등 파일명이 모듈을 가리지 않게 한다.)
sys.path[0] = str(Path(__file__).resolve().parents[1])

from auto_ml.stages import load_stage  # noqa: E402
from auto_ml.stages import train as stages  # noqa: E402
from auto_ml.stages.cli import parse_stage_args, start_stage_logging  # noqa: E402


def main(argv: list[str] | None = None) -> dict[str, Path]:
    config, work_dir = parse_stage_args("[train 3/3] postprocess", "train", argv)
    start_stage_logging(config, "train_postprocess")
    prep = load_stage(work_dir, stages.PREPROCESS_OUTPUT)
    result = load_stage(work_dir, stages.TRAIN_OUTPUT)
    outputs = stages.postprocess(config, prep, result)
    for key, path in outputs.items():
        print(f"{key}: {path}")
    return outputs


if __name__ == "__main__":
    main()
