"""변수 선택(Feature Selection) 모듈.

전처리(scaling) 단계 이후, 모델 학습 전에 적용되는 단계.
현재는 Stability Selection 한 가지를 제공한다.
"""
from auto_ml.feature_selection.stability import (
    SelectionResult,
    StabilitySelector,
)

__all__ = ["SelectionResult", "StabilitySelector"]
