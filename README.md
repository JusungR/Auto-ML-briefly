# auto_ml

이진 분류(0/1) 문제를 자동으로 학습·스코어링하는 사내 라이브러리.
폐쇄 환경 이관 / 운영을 전제로 모듈화·설정 주도(YAML)로 설계되었다.

## 구성

```
auto_ml/
├── config.py           설정 dataclass + YAML 로더
├── pipeline.py         학습 파이프라인 (auto-ml-train)
├── preprocessing/      1) 결측 → 2) 이상치 → 3) 스케일링
├── models/             LGBM / XGBoost / CatBoost 래퍼 + Trainer
├── tuning/             Optuna 베이지안 하이퍼파라미터 최적화
├── reporting/          HTML + PDF 리포트 (동일 내용)
├── scoring/            배치 스코어링 (auto-ml-score)
└── utils/              io / logger / validation
```

## 4단계 파이프라인

1. **전처리** — 결측 → 이상치 → 스케일링 순서 고정. 학습 시 통계량을
   저장해 스코어링 시 동일 적용.
2. **모형 적합** — LGBM / XGBoost / CatBoost 3종에 대해 (a) Optuna TPE 로
   하이퍼파라미터 베이지안 최적화 → (b) StratifiedKFold OOF 평가 →
   (c) 학습 전체로 최종 fit → (d) 테스트 데이터로 평가. primary_metric
   (기본 ROC-AUC) 기준으로 best 모델을 선정한다.
3. **보고서 산출** — HTML / PDF 동일 내용 (Jinja2 + WeasyPrint). 모델 비교표,
   튜닝 결과, ROC / PR 곡선, feature importance, score 분포, confusion matrix 포함.
4. **주기 스코어링** — 단일 artifact (preprocessor + model + metadata)
   파일 1개 + 설정 YAML 1개로 운영 가능. cron 등에서 `auto-ml-score` 호출.

## 입력 데이터

- 학습 / 평가 데이터를 **별도 Parquet 파일** 로 받는다 (`train_data_path`,
  `test_data_path`). 내부에서 임의 분할하지 않는다.
- 학습/스코어링에 사용할 컬럼은 **CSV 파일** 로 명시한다.
  YAML 에서 `features_csv` 에 경로를 지정하면 `load_config` 가 자동으로 읽어
  `cfg.features` 를 채운다.

CSV 형식 (필수 컬럼: `name`, `type`, `used`):

```csv
name,type,used
age,continuous,true
gender,category,true
income,continuous,true
internal_score,continuous,false
```

- `type`
  - `continuous` — 연속형 (int / float). 결측·이상치·스케일링 적용.
  - `category`   — 범주형 (문자열 / 코드). 모델 래퍼에서 자체 인코딩 처리.
- `used`
  - `true` 인 행만 학습/스코어링에 사용된다 (`false`/공백 → 제외).
  - 운영 중 컬럼을 켜고 끌 때 코드 변경 없이 본 CSV 만 수정하면 된다.

```yaml
features_csv: ./configs/features.csv
```

## 하이퍼파라미터 최적화

각 모델 블록에 `fixed_params` (고정) 와 `search_space` (탐색 범위) 를 둔다.
`tuning.enabled: true` 이고 모델별 `search_space` 가 비어있지 않으면 Optuna
TPE 가 KFold OOF `primary_metric` 을 최대화하는 파라미터를 찾는다.

```yaml
tuning:
  enabled: true
  n_trials: 30
  cv_folds: 3
  random_state: 42

models:
  lgbm:
    enabled: true
    fixed_params: { objective: binary, metric: auc, verbose: -1, n_estimators: 2000 }
    search_space:
      learning_rate: { type: float, low: 0.01, high: 0.3, log: true }
      num_leaves:    { type: int,   low: 15,   high: 255 }
      reg_lambda:    { type: float, low: 1.0e-8, high: 10.0, log: true }
```

`search_space` 항목 형식:

- `type: float`       — `low`, `high`, `log` (선택, 기본 false)
- `type: int`         — `low`, `high`, `log` (선택), `step` (선택, 기본 1)
- `type: categorical` — `choices: [...]`

## 설치

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

WeasyPrint 는 시스템 폰트와 일부 라이브러리(libpango 등) 가 필요하다.
폐쇄망에서는 wheelhouse + 시스템 패키지를 함께 배포한다.

## 사용

```bash
# 1) 더미 데이터 생성 (검증용)
python examples/make_dummy_data.py

# 2) 학습 — 산출물:
#    artifacts/models/best.joblib
#    artifacts/reports/report.html, report.pdf
auto-ml-train --config configs/example.yaml

# 3) 스코어링 — 산출물:
#    artifacts/scores/scores.parquet  (id_columns + score + prediction)
auto-ml-score --config configs/example.yaml
```

코드에서 직접 호출하는 예시는 `examples/run_train.py`, `examples/run_score.py` 참고.

## 폐쇄망 이관

다음 4가지만 옮기면 된다:

1. 패키지 wheel (`pip wheel -r requirements.txt -w wheelhouse/`)
2. 본 repo 자체 (또는 `pip install -e .` 으로 wheel 생성)
3. 학습 산출물 `artifacts/models/best.joblib`
4. 스코어링 설정 YAML

## 설정

전체 옵션은 `configs/example.yaml` 참고. 모든 키는
`auto_ml/config.py` 의 dataclass 와 1:1 대응한다.

## 작업 로그

실행마다 stage(`train` / `score`) 별 별도 로그 파일이 자동 생성된다.

```
artifacts/logs/
├── train_20260425_143052.log
└── score_20260425_180001.log
```

설정 (`configs/example.yaml` 의 `logging` 섹션):

```yaml
logging:
  log_dir: ./artifacts/logs
  level: INFO          # DEBUG | INFO | WARNING | ERROR
  to_stdout: true
  to_file: true
```

콘솔 출력과 파일 출력은 동시에 활성화 가능하며, `to_file: false` 로 두면
파일은 만들지 않는다. 각 로그는 시작/종료 마커, 단계별 진행, 모델 튜닝 결과,
산출물 경로, 총 소요 시간을 포함한다.

## 운영 메모

- 타깃은 0/1 만 허용 (`utils/validation.py` 가 검증).
- best 선정은 테스트 데이터 점수 기준.
- 학습 시 사용한 features 정의가 artifact 메타데이터에 저장되어
  스코어링 시 동일 컬럼 셋·동일 전처리·동일 모델로 처리된다.
- 스코어링 결과 컬럼: `<id_columns> + score + prediction`.
