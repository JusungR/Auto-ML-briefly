"""단계(stage) 단위 실행 함수.

학습 / 추론을 각각 ``전처리 → 학습(예측) → 후처리`` 3단계로 나눈다.
각 단계 함수는 메모리 상에서 동작하고, 단계 사이 산출물은
``save_stage`` / ``load_stage`` 로 작업 디렉토리(work_dir) 에 저장·로드한다.
``train/*.py``, ``inference/*.py`` 스크립트가 이 함수들을 단계별로 호출한다.
"""
from auto_ml.stages.common import default_work_dir, load_stage, save_stage

__all__ = ["default_work_dir", "load_stage", "save_stage"]
