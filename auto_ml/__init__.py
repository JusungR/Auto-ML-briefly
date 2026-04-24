"""Auto-ML library for binary classification."""
from auto_ml.config import AutoMLConfig, load_config
from auto_ml.pipeline import AutoMLPipeline

__all__ = ["AutoMLConfig", "load_config", "AutoMLPipeline"]
__version__ = "0.1.0"
