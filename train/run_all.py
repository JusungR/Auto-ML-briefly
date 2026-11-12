"""[학습 전체] 전처리 → 학습 → 후처리 를 한 번에 실행한다.

단계별 산출물을 work_dir 에 저장하므로, 중간에 실패하면 해당 단계 스크립트만
다시 실행할 수 있다.

    python train/run_all.py --config configs/example.yaml
"""
import sys
from pathlib import Path

# 설치 없이 실행: 저장소 루트를 import 경로에 추가한다.
# (스크립트 디렉토리 대신 루트를 넣어 preprocess.py 등 파일명이 모듈을 가리지 않게 한다.)
sys.path[0] = str(Path(__file__).resolve().parents[1])

from auto_ml.stages import save_stage  # noqa: E402
from auto_ml.stages import train as stages  # noqa: E402
from auto_ml.stages.cli import parse_stage_args, start_stage_logging  # noqa: E402


def main(argv: list[str] | None = None) -> dict[str, Path]:
    config, work_dir = parse_stage_args("[train] preprocess -> train -> postprocess", "train", argv)
    start_stage_logging(config, "train")
    prep = stages.preprocess(config)
    save_stage(prep, work_dir, stages.PREPROCESS_OUTPUT)
    result = stages.train(config, prep)
    save_stage(result, work_dir, stages.TRAIN_OUTPUT)
    outputs = stages.postprocess(config, prep, result)
    for key, path in outputs.items():
        print(f"{key}: {path}")
    return outputs


if __name__ == "__main__":
    main()
