"""리포트 패키지: 학습 결과를 HTML / PDF 단일 파일로 산출한다.

순환 import 회피: ``ReportBuilder`` 는 모델 학습 결과 dataclass 를 import 하고
trainer 는 ``reporting.metrics`` 의 ``compute_metrics`` 를 import 하므로 본
패키지 ``__init__`` 에서 ``ReportBuilder`` 를 즉시 가져오지 않는다 — 사용처에서
``from auto_ml.reporting.report import ReportBuilder`` 로 직접 import 한다.
"""

__all__: list[str] = []
