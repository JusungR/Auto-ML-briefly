"""[추론 전체] 전처리 → 예측 → 후처리 를 한 번에 실행한다.

cron / Airflow 등 배치 스케줄러에서 이 파일을 호출하면 된다.

    python inference/run_all.py --config configs/example.yaml
"""
import sys
from pathlib import Path

# 설치 없이 실행: 저장소 루트를 import 경로에 추가한다.
# (스크립트 디렉토리 대신 루트를 넣어 preprocess.py 등 파일명이 모듈을 가리지 않게 한다.)
sys.path[0] = str(Path(__file__).resolve().parents[1])

from auto_ml.scoring.scorer import Scorer  # noqa: E402
from auto_ml.stages import inference as stages  # noqa: E402
from auto_ml.stages import save_stage  # noqa: E402
from auto_ml.stages.cli import parse_stage_args, start_stage_logging  # noqa: E402


def main(argv: list[str] | None = None) -> Path:
    config, work_dir = parse_stage_args(
        "[inference] preprocess -> predict -> postprocess", "inference", argv,
    )
    start_stage_logging(config, "inference")
    # artifact 를 한 번만 로드해 세 단계가 공유한다.
    scorer = Scorer.from_artifact(stages.artifact_path_of(config))
    prep = stages.preprocess(config, scorer=scorer)
    save_stage(prep, work_dir, stages.PREPROCESS_OUTPUT)
    pred = stages.predict(prep, scorer=scorer)
    save_stage(pred, work_dir, stages.PREDICT_OUTPUT)
    path = stages.postprocess(config, pred)
    print(f"scored output: {path}")
    return path


if __name__ == "__main__":
    main()
