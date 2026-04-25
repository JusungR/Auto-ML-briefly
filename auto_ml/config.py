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

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class FeatureSpec:
    """학습/스코어링에 사용할 단일 컬럼의 정의.

    Attributes:
        name: Parquet 컬럼 이름.
        type: 컬럼 종류 (내부 표기). 다음 중 하나여야 한다 — 다른 값은 ValueError.
            - ``"numeric"``     : 연속형 (int / float). 결측·이상치·스케일링이 적용됨.
            - ``"categorical"`` : 범주형 (문자열 / 코드). 인코딩은 모델 래퍼에서 처리.

    Note:
        사용자 입력 CSV 에서는 ``continuous`` / ``category`` 로 표기하고,
        ``load_features_csv`` 가 위 내부 표기로 변환한다.
    """

    name: str
    type: str

    # 명세 외 값이 들어오면 학습 시작 전에 즉시 실패하도록 검증
    def __post_init__(self) -> None:
        if self.type not in {"numeric", "categorical"}:
            raise ValueError(
                f"FeatureSpec.type must be 'numeric' or 'categorical', got {self.type!r} "
                f"(feature='{self.name}')"
            )


# CSV 입력 표기 ↔ 내부 표기 매핑. 사용자 입력은 CSV 표기를 따른다.
_CSV_TYPE_TO_INTERNAL = {
    "continuous": "numeric",
    "category": "categorical",
}
_TRUTHY = {"true", "1", "yes", "y", "t"}
_FALSY = {"false", "0", "no", "n", "f", ""}


def _to_bool(value: Any, *, field_name: str, row_idx: int) -> bool:
    """CSV 의 used 컬럼 값을 bool 로 파싱한다.

    허용 값 (대소문자 무시): true/false, 1/0, yes/no, y/n, t/f, (빈 문자열 = false).
    """
    s = str(value).strip().lower()
    if s in _TRUTHY:
        return True
    if s in _FALSY:
        return False
    raise ValueError(
        f"features CSV row {row_idx}: '{field_name}' must be true/false, got {value!r}"
    )


def load_features_csv(path: str | Path) -> list[FeatureSpec]:
    """features 정의 CSV 를 읽어 ``used == true`` 인 컬럼들의 ``FeatureSpec`` 리스트를 반환한다.

    CSV 형식 (필수 컬럼: ``name``, ``type``, ``used``):

        name,type,used
        age,continuous,true
        gender,category,true
        income,continuous,false

    Args:
        path: CSV 경로. CWD 기준 상대경로 또는 절대경로.

    Returns:
        used == true 인 행만 ``FeatureSpec`` 으로 변환한 리스트
        (CSV 의 행 순서를 유지).

    Raises:
        FileNotFoundError: CSV 가 존재하지 않을 때.
        ValueError: 필수 컬럼 누락 / 알 수 없는 type / 잘못된 used 값 / 결과가 비어있는 경우.
    """
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"features CSV not found: {csv_path}")

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"features CSV {csv_path} is empty")
        required = {"name", "type", "used"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"features CSV {csv_path} missing columns: {sorted(missing)}"
            )

        features: list[FeatureSpec] = []
        seen: set[str] = set()
        for i, row in enumerate(reader, start=2):  # 1행은 헤더
            name = (row.get("name") or "").strip()
            if not name:
                # 빈 줄(공백 행) 은 조용히 건너뛴다.
                continue
            if name in seen:
                raise ValueError(
                    f"features CSV row {i}: duplicate name '{name}'"
                )
            seen.add(name)

            if not _to_bool(row.get("used"), field_name="used", row_idx=i):
                continue

            type_str = (row.get("type") or "").strip().lower()
            if type_str not in _CSV_TYPE_TO_INTERNAL:
                raise ValueError(
                    f"features CSV row {i} ('{name}'): type must be one of "
                    f"{sorted(_CSV_TYPE_TO_INTERNAL.keys())}, got {row.get('type')!r}"
                )
            features.append(
                FeatureSpec(name=name, type=_CSV_TYPE_TO_INTERNAL[type_str])
            )

    if not features:
        raise ValueError(
            f"features CSV {csv_path} contains no rows with used=true"
        )
    return features


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
    """단일 모델 활성화 / 하이퍼파라미터 설정.

    Attributes:
        enabled: False 로 두면 학습에서 제외된다.
        fixed_params: 탐색하지 않고 항상 고정으로 적용할 파라미터.
                       (예: ``objective``, ``metric`` 등 도메인 상 고정값)
        search_space: 베이지안 최적화 탐색 공간 (Optuna 가 샘플링).
                       비어 있으면 해당 모델은 튜닝을 생략하고
                       ``fixed_params`` 만으로 학습한다.

    ``search_space`` 형식 (key 는 모델 라이브러리 원본 파라미터 이름):

        learning_rate:
          type: float        # float | int | categorical
          low: 0.01
          high: 0.3
          log: true          # 선택 (기본 false). 로그 스케일 탐색 여부
        num_leaves:
          type: int
          low: 15
          high: 127
        boosting_type:
          type: categorical
          choices: [gbdt, dart]
    """

    enabled: bool = True
    fixed_params: dict[str, Any] = field(default_factory=dict)
    search_space: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class TrainingConfig:
    """학습 공통 옵션.

    Attributes:
        cv_folds: 최종 OOF 평가용 StratifiedKFold 분할 수.
        random_state: 재현성을 위한 시드.
        early_stopping_rounds: 부스팅 모델 조기종료 라운드 수.
        primary_metric: 모델 선택 / 튜닝 목적함수에 사용하는 지표.
                        지원: roc_auc | pr_auc | f1 | accuracy | precision | recall | ks
    """

    cv_folds: int = 5
    random_state: int = 42
    early_stopping_rounds: int = 50
    primary_metric: str = "roc_auc"


@dataclass
class TuningConfig:
    """베이지안 하이퍼파라미터 최적화 (Optuna) 옵션.

    Attributes:
        enabled: False 면 모든 모델에 대해 튜닝을 생략한다.
                 (모델별로 ``search_space`` 가 비어 있으면 자동으로 생략)
        n_trials: 모델당 시도 횟수.
        timeout: 모델당 최대 탐색 시간(초). None 이면 ``n_trials`` 까지 진행.
        cv_folds: 튜닝 단계의 KFold 분할 수. 보통 최종 CV 보다 작게 잡아 속도를 확보한다.
        random_state: TPE 샘플러 시드.
    """

    enabled: bool = True
    n_trials: int = 30
    timeout: int | None = None
    cv_folds: int = 3
    random_state: int = 42


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
class LoggingConfig:
    """작업 로그 옵션.

    Attributes:
        log_dir: 파일 로그 저장 디렉토리.
        level: 로깅 레벨 (DEBUG / INFO / WARNING / ERROR).
        to_stdout: 콘솔(stdout) 출력 여부.
        to_file: 파일 출력 여부. 비활성하면 본 stage 의 파일은 만들어지지 않는다.
    """

    log_dir: str = "./artifacts/logs"
    level: str = "INFO"
    to_stdout: bool = True
    to_file: bool = True


@dataclass
class AutoMLConfig:
    """파이프라인 전체 설정.

    하위 dataclass 들을 묶는 최상위 컨테이너. ``load_config()`` 가
    YAML 을 읽어 본 클래스 인스턴스로 변환한다.
    """

    # 데이터 관련 — 학습/평가 데이터셋을 별도 파일로 받는다
    train_data_path: str = "./data/train.parquet"
    test_data_path: str = "./data/test.parquet"
    target_column: str = "target"
    # 학습/스코어링에 사용할 컬럼 정의를 담은 CSV 경로.
    # CSV 컬럼: name, type(continuous|category), used(true|false).
    # used == true 인 행만 실제 학습/스코어링 feature 로 사용된다.
    features_csv: str = "./configs/features.csv"
    # ``load_config`` 가 features_csv 를 읽어 자동으로 채운다.
    # 코드에서 직접 AutoMLConfig 를 만들 때는 본 필드에 list[FeatureSpec] 를 넣어도 된다.
    features: list[FeatureSpec] = field(default_factory=list)
    # 결과(스코어링 출력) 에 식별자로 보존할 컬럼. 학습/예측에는 사용되지 않는다.
    id_columns: list[str] = field(default_factory=list)

    # ----- 파생 헬퍼 -----------------------------------------------------
    @property
    def feature_columns(self) -> list[str]:
        """``features`` 의 이름 순서대로 반환."""
        return [f.name for f in self.features]

    @property
    def numeric_columns(self) -> list[str]:
        return [f.name for f in self.features if f.type == "numeric"]

    @property
    def categorical_columns(self) -> list[str]:
        return [f.name for f in self.features if f.type == "categorical"]

    # 산출물 저장 위치
    artifact_dir: str = "./artifacts/models"

    # 하위 설정
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    tuning: TuningConfig = field(default_factory=TuningConfig)
    models: dict[str, ModelConfig] = field(
        default_factory=lambda: {
            "lgbm": ModelConfig(),
            "xgb": ModelConfig(),
            "catboost": ModelConfig(),
        }
    )
    reporting: ReportingConfig = field(default_factory=ReportingConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AutoMLConfig":
        """YAML 파싱 결과(dict) 를 AutoMLConfig 로 변환한다.

        하위 섹션은 각 dataclass 로 한번 더 변환하여 타입 안전성을 확보한다.
        """
        data = dict(data)
        if "features" in data:
            # YAML 의 dict 또는 dataclass 인스턴스 모두 허용
            data["features"] = [
                f if isinstance(f, FeatureSpec) else FeatureSpec(**f)
                for f in data["features"]
            ]
        if "preprocessing" in data:
            data["preprocessing"] = PreprocessingConfig(**data["preprocessing"])
        if "training" in data:
            data["training"] = TrainingConfig(**data["training"])
        if "tuning" in data:
            data["tuning"] = TuningConfig(**data["tuning"])
        if "reporting" in data:
            data["reporting"] = ReportingConfig(**data["reporting"])
        if "scoring" in data:
            data["scoring"] = ScoringConfig(**data["scoring"])
        if "logging" in data:
            data["logging"] = LoggingConfig(**data["logging"])
        if "models" in data:
            data["models"] = {
                name: ModelConfig(**cfg) for name, cfg in data["models"].items()
            }
        return cls(**data)


def load_config(path: str | Path) -> AutoMLConfig:
    """YAML 설정 + features CSV 를 함께 읽어 AutoMLConfig 를 반환한다.

    동작 순서:
        1) YAML 을 파싱해 AutoMLConfig 를 만든다.
        2) ``features_csv`` 에 명시된 CSV 를 읽어 ``used == true`` 인 컬럼들로
           ``cfg.features`` 를 덮어쓴다 (YAML 에서 inline 으로 features 를
           지정한 경우라도 CSV 가 우선).

    Args:
        path: 설정 YAML 경로.

    Returns:
        features 까지 채워진 AutoMLConfig 인스턴스.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cfg = AutoMLConfig.from_dict(raw or {})
    # CSV 경로가 명시된 경우 항상 CSV 가 우선 — 운영 시 CSV 한 파일만 관리하면 된다.
    if cfg.features_csv:
        cfg.features = load_features_csv(cfg.features_csv)
    return cfg
