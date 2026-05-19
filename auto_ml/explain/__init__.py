"""SHAP 기반 건별·변수별 기여도 해석.

학습된 artifact 를 로드해 각 입력 행에 대해 변수 단위 SHAP 값을 계산한다.
출력은 wide 포맷 — 한 행 = 한 건, 컬럼은
``<id_columns> + shap_<feature>... + base_value + score``.

도메인은 probability — raw-margin SHAP 을 sigmoid 변환 후 비율 유지하며
정규화하여 ``base_value + sum(shap_*) == score`` 가 보장된다.
"""
from auto_ml.explain.explainer import Explainer

__all__ = ["Explainer"]
