"""Auto-ML library for binary classification."""
# __version__ 은 다른 import 보다 먼저 정의해야 한다 — reporting.report 가
# import 체인 도중에 본 값을 참조하므로, 순서가 뒤바뀌면 부분 초기화 상태에서
# 순환 import 오류가 난다.
__version__ = "0.1.0"

from auto_ml.config import AutoMLConfig, load_config  # noqa: E402
from auto_ml.pipeline import AutoMLPipeline  # noqa: E402

__all__ = ["AutoMLConfig", "load_config", "AutoMLPipeline", "__version__"]
