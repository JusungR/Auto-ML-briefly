"""Auto-ML 파이프라인 설정 스키마.

운영팀이 코드 수정 없이 YAML 한 파일만으로 동작을 바꿀 수 있도록
모든 런타임 옵션을 dataclass 형태로 명시한다.

폐쇄 환경 친화 설계:
    - 외부 네트워크 호출이 없는 순수 dict/YAML 입력만 허용한다.
    - 비밀값을 외부 시스템에서 가져오지 않는다.
    - 누락된 항목은 기본값으로 채워져 운영 안정성을 확보한다.

사용 예시:
    >>> from auto_ml.config import load_config
    >>> cfg = load_config("configs/example.yaml")
    >>> cfg.training.cv_folds
    5
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class PreprocessingConfig:
    """전처리 단계 옵션.

    전처리는 항상 다음 순서로 실행된다:
        1) 결측 처리 → 2) 이상치 처리 → 3) 스케일링
    """

    # 1) 결측 처리 (Null Handling)
    numeric_null_strategy: str = "median"          # median | mean | constant
    numeric_null_fill_value: float = 0.0           # numeric_null_strategy == "constant" 일 때만 사용
    categorical_null_strategy: str = "most_frequent"  # most_frequent | constant
    categorical_null_fill_value: str = "MISSING"   # categorical_null_strategy == "constant" 일 때만 사용

    # 2) 이상치 처리 (Outlier Handling) — 수치형에만 적용
    outlier_method: str = "iqr"                    # iqr | zscore | none
    outlier_iqr_multiplier: float = 1.5            # IQR 방식의 경계 배수 (보수적: 3.0)
    outlier_zscore_threshold: float = 3.0          # Z-score 방식의 임계치
    outlier_action: str = "clip"                   # clip | null_then_impute

    # 3) 스케일링 (Scaling) — 수치형에만 적용
    scaling_method: str = "standard"               # standard | minmax | robust | none


@dataclass
class ModelConfig:
    """단일 모델 활성화/하이퍼파라미터 설정.

    Attributes:
        enabled: False 로 두면 학습에서 제외된다.
        params: 모델 라이브러리 원래 파라미터 그대로 전달된다.
                (예: LGBM 의 ``num_leaves`` 등)
    """

    enabled: bool = True
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingConfig:
    """학습 공통 옵션.

    Attributes:
        test_size: holdout 비율. CV 외에 최종 평가용 hold-out 분리에 사용.
        cv_folds: StratifiedKFold 분할 수.
        random_state: 재현성을 위한 시드.
        early_stopping_rounds: 부스팅 모델 조기종료 라운드 수.
        primary_metric: 최적 모델을 선택할 때 사용하는 지표.
                        지원: roc_auc | pr_auc | f1 | accuracy
    """

    test_size: float = 0.2
    cv_folds: int = 5
    random_state: int = 42
    early_stopping_rounds: int = 50
    primary_metric: str = "roc_auc"


@dataclass
class ReportingConfig:
    """리포트 산출 옵션. HTML 과 PDF 두 가지를 동일 내용으로 생성한다."""

    output_dir: str = "./artifacts/reports"
    generate_html: bool = True
    generate_pdf: bool = True
    title: str = "Auto-ML Binary Classification Report"


@dataclass
class ScoringConfig:
    """주기적 배치 스코어링 옵션.

    Attributes:
        input_path: 스코어링 대상 Parquet 경로.
        output_path: 결과 Parquet 저장 경로.
        id_columns: 결과에 그대로 보존할 식별자 컬럼.
        threshold: ``predict_proba`` 결과를 0/1 로 변환할 임계값.
    """

    input_path: str = "./data/score_input.parquet"
    output_path: str = "./artifacts/scores/scores.parquet"
    id_columns: list[str] = field(default_factory=list)
    threshold: float = 0.5


@dataclass
class AutoMLConfig:
    """파이프라인 전체 설정.

    하위 dataclass 들을 묶는 최상위 컨테이너. ``load_config()`` 가
    YAML 을 읽어 본 클래스 인스턴스로 변환한다.
    """

    # 데이터 관련
    train_data_path: str = "./data/train.parquet"
    target_column: str = "target"
    feature_columns: list[str] | None = None       # None 이면 target / id 제외 전부 사용
    id_columns: list[str] = field(default_factory=list)
    categorical_columns: list[str] = field(default_factory=list)

    # 산출물 저장 위치
    artifact_dir: str = "./artifacts/models"

    # 하위 설정
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    models: dict[str, ModelConfig] = field(
        default_factory=lambda: {
            "lgbm": ModelConfig(),
            "xgb": ModelConfig(),
            "catboost": ModelConfig(),
        }
    )
    reporting: ReportingConfig = field(default_factory=ReportingConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AutoMLConfig":
        """YAML 파싱 결과(dict) 를 AutoMLConfig 로 변환한다.

        하위 섹션은 각 dataclass 로 한번 더 변환하여 타입 안전성을 확보한다.
        """
        data = dict(data)
        if "preprocessing" in data:
            data["preprocessing"] = PreprocessingConfig(**data["preprocessing"])
        if "training" in data:
            data["training"] = TrainingConfig(**data["training"])
        if "reporting" in data:
            data["reporting"] = ReportingConfig(**data["reporting"])
        if "scoring" in data:
            data["scoring"] = ScoringConfig(**data["scoring"])
        if "models" in data:
            data["models"] = {
                name: ModelConfig(**cfg) for name, cfg in data["models"].items()
            }
        return cls(**data)


def load_config(path: str | Path) -> AutoMLConfig:
    """YAML 파일을 읽어 AutoMLConfig 로 반환한다.

    Args:
        path: 설정 YAML 경로.

    Returns:
        AutoMLConfig 인스턴스. 빈 파일이면 모든 기본값으로 채워진다.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return AutoMLConfig.from_dict(raw or {})
