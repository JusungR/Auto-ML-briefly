# auto_ml 사용 설명서

이 문서는 두 파트로 구성됩니다.

- **Part I. 튜토리얼 (§1–5)** — 처음 사용하는 분이 설치부터 첫 실행까지 따라할 수 있는 단계별 안내.
- **Part II. 레퍼런스 (§6–18)** — 설정 파일의 모든 섹션을 모듈별로 정리. 각 섹션은 **역할 → 옵션 표 → YAML 예시 → 동작 메모** 4단으로 반복되어, 필요한 부분만 찾아 읽도록 설계되었습니다.

> **새 기능 — 최종 모델 학습 전략 (`final_fit_strategy`)**: 테스트셋 누설 없이 오버핏(Train·Test Δ)을
> 줄이는 `iteration_capping` / `cv_bagging` 옵션이 추가되었습니다. → [§11 최종 모델 학습 전략](#최종-모델-학습-전략-final_fit_strategy)

---

## 목차

**Part I. 튜토리얼**

1. [라이브러리 소개](#1-라이브러리-소개)
2. [전체 흐름](#2-전체-흐름)
3. [사전 준비](#3-사전-준비)
4. [설치](#4-설치)
5. [빠른 시작](#5-빠른-시작)

**Part II. 설정 레퍼런스**

6. [설정 파일 구조와 YAML 문법](#6-설정-파일-구조와-yaml-문법)
7. [데이터 입력 (top-level)](#7-데이터-입력-top-level)
8. [features.csv — 사용할 컬럼 정의](#8-featurescsv--사용할-컬럼-정의)
9. [전처리 `preprocessing`](#9-전처리-preprocessing)
10. [변수 선택 `feature_selection`](#10-변수-선택-feature_selection)
11. [학습 `training`](#11-학습-training)
    - [↳ 최종 모델 학습 전략 — iteration_capping / cv_bagging](#최종-모델-학습-전략-final_fit_strategy)
12. [튜닝 `tuning`](#12-튜닝-tuning)
13. [모델 `models`](#13-모델-models)
14. [앙상블 `ensemble`](#14-앙상블-ensemble)
15. [리포트 `reporting`](#15-리포트-reporting)
16. [스코어링 `scoring`](#16-스코어링-scoring)
17. [SHAP 해석 `explain`](#17-shap-해석-explain)
18. [로깅 `logging`](#18-로깅-logging)

**부록**

- [A. CLI 명령어 모음](#a-cli-명령어-모음)
- [B. 산출물 안내](#b-산출물-안내)
- [C. 트러블슈팅 FAQ](#c-트러블슈팅-faq)
- [D. 폐쇄망 이관](#d-폐쇄망-이관)
- [E. 용어 사전](#e-용어-사전)

---

# Part I. 튜토리얼

## 1. 라이브러리 소개

`auto_ml` 은 **이진 분류(0/1)** 문제를 자동으로 학습·스코어링하는 사내 라이브러리입니다.
이탈/비이탈, 사기/정상, 구매/비구매 같은 "예/아니오" 답을 맞히는 모든 문제에 쓸 수 있습니다.

ML 코드를 직접 작성할 필요는 없습니다. **YAML 설정 한 개**와 **컬럼 정의 CSV 한 개**만
준비하면 라이브러리가 다음을 수행합니다.

- 결측치 채우기, 이상치 윈저라이징, skew 변환, 스케일링
- (선택) Stability Selection 으로 변수 자동 선별
- LightGBM / XGBoost / CatBoost / ElasticNet 네 가지 모델 학습
- Optuna 로 하이퍼파라미터 베이지안 탐색
- 모델들을 가중 평균 앙상블로 자동 결합
- 테스트 점수 기준 best 모델 선정
- HTML / PDF 리포트 생성
- 학습 결과 한 덩어리(`best.joblib`)로 저장

학습이 끝나면 **`auto-ml-score`** 한 줄로 새 데이터에 점수를 매길 수 있습니다(cron 등 배치 운영용).

---

## 2. 전체 흐름

```
[학습 Parquet]   [테스트 Parquet]
       │               │
       ▼               ▼
       └── auto-ml-train ──► best.joblib  + 리포트(HTML/PDF) + 로그
                                  │
                       ┌──────────┤
                       ▼          ▼
[입력 Parquet] ► auto-ml-score   auto-ml-explain
                       │                │
                       ▼                ▼
              scores.parquet      shap.parquet
          (id + score + prediction)  (id + shap_<feature>... + base_value + score)
```

핵심 명령어는 세 개입니다.

| 명령어 | 언제 쓰나 | 결과물 |
|---|---|---|
| `auto-ml-train` | 모델을 만들 때 (보통 주 1회) | `best.joblib`, HTML/PDF 리포트, 로그 |
| `auto-ml-score` | 새 데이터에 점수를 매길 때 (반복) | `scores.parquet` |
| `auto-ml-explain` | 건별 변수 기여도가 필요할 때 | `shap.parquet` |

보조 명령어:
- `auto-ml-set-best` — 학습 후 best 모델을 다른 후보로 교체(재학습 불필요).

---

## 3. 사전 준비

### 필요한 것

- **Python 3.10 이상, 3.14 미만** (`python --version` 으로 확인)
- **pip** (보통 Python 과 함께 설치)
- **Git** (소스 코드 clone 용)
- (SHAP 해석을 쓰려면) **shap** — `requirements.txt` 에 포함되어 있어 `pip install -r requirements.txt` 로 자동 설치됩니다. ElasticNet 모델의 SHAP 산출 시 필요합니다.
- (PDF 리포트를 만들려면) **WeasyPrint 시스템 의존성**
  - macOS: `brew install pango libffi`
  - Ubuntu/Debian: `sudo apt-get install libpango-1.0-0 libpangoft2-1.0-0`
  - 미설치 시 학습/스코어링은 정상 동작하고 HTML 리포트만 생성됩니다.

### 디스크 / 메모리

- 더미 예제 기준 디스크 **500MB** 면 충분.
- 학습 데이터가 수백만 행이면 메모리 **8GB 이상** 권장.

---

## 4. 설치

터미널에서 한 줄씩 차례대로 실행합니다.

```bash
# 1) 프로젝트 폴더로 이동
cd auto-ml-briefly

# 2) 가상환경 생성 (시스템 Python 을 더럽히지 않도록)
python -m venv .venv

# 3) 가상환경 활성화
source .venv/bin/activate          # macOS / Linux
.venv\Scripts\Activate.ps1         # Windows (PowerShell)

# 4) 의존성 설치
pip install -r requirements.txt

# 5) auto_ml 자체를 editable 모드로 설치 (CLI 명령 활성화)
pip install -e .
```

설치를 확인합니다.

```bash
auto-ml-train --help
auto-ml-score --help
```

도움말이 나오면 성공입니다. `command not found` 가 보이면 가상환경 활성화 여부를 다시 확인하세요(프롬프트 앞에 `(.venv)` 가 보여야 합니다).

---

## 5. 빠른 시작

진짜 데이터를 준비하기 전에, 더미 데이터로 동작 여부를 한 번 확인합니다.

```bash
# 1) 더미 데이터 생성 — data/ 에 train/test/score_input parquet 생성
python examples/make_dummy_data.py

# 2) 학습
auto-ml-train --config configs/example.yaml

# 3) 스코어링
auto-ml-score --config configs/example.yaml

# 4) SHAP 해석 (건별 변수 기여도)
auto-ml-explain --config configs/example.yaml
```

산출물은 다음 구조로 만들어집니다.

```
artifacts/
├── models/
│   ├── best.joblib              ← 전처리기 + 모델 + 메타데이터 한 덩어리
│   └── models/                  ← 후보 모델별 sub-artifact
│       ├── lgbm.joblib
│       ├── xgb.joblib
│       ├── catboost.joblib
│       └── ensemble.joblib      ← (앙상블 활성 시)
├── predictions/
│   └── test_predictions.parquet ← 학습 시 holdout 예측 (id + score + prediction + target)
├── reports/
│   ├── report.html              ← 브라우저로 열기
│   ├── report.pdf               ← 동일 내용 PDF
│   ├── feature_importance.csv   ← 전체 변수 중요도
│   ├── feature_selection.csv    ← (변수 선택 활성 시) 선택 빈도
│   └── sub/                     ← 비-best 모형별 상세 리포트
│       └── <model>/report.html, feature_importance.csv …
├── scores/
│   └── scores.parquet           ← id_columns + score + prediction
├── explanations/
│   └── shap.parquet             ← id_columns + shap_<feature>... + base_value + score
└── logs/
    ├── train_YYYYMMDD_HHMMSS.log
    ├── score_YYYYMMDD_HHMMSS.log
    └── explain_YYYYMMDD_HHMMSS.log
```

`artifacts/reports/report.html` 을 브라우저로 열어 모델 비교표, ROC 곡선, feature importance
등이 표시되면 설치가 정상입니다. 다음 단계는 본인 데이터로 같은 흐름을 돌리는 것입니다 —
필요한 설정은 Part II 에서 섹션별로 찾아보면 됩니다.

자기 데이터로 돌릴 때 보통 손대는 곳은 세 군데입니다.

1. `configs/example.yaml` 을 복사해 데이터 경로·타깃 컬럼 이름만 수정 → [§7 데이터 입력](#7-데이터-입력-top-level)
2. `configs/features.csv` 에 사용할 컬럼을 적기 → [§8 features.csv](#8-featurescsv--사용할-컬럼-정의)
3. 필요에 따라 모델·튜닝·앙상블 설정 조정 → [§13 모델](#13-모델-models), [§12 튜닝](#12-튜닝-tuning), [§14 앙상블](#14-앙상블-ensemble)

> **더 알아보기 — 실전 예시와 Python API**
>
> - `examples/credit/prepare_data.py` — 신용카드 데이터를 Parquet 으로 변환하는 전처리 예시
> - `examples/titanic/prepare_data.py` — Titanic 데이터셋 전처리 예시
> - `examples/run_train.py` — CLI 대신 Python 코드에서 학습 파이프라인을 직접 호출하는 예시
> - `examples/run_score.py` — Python 코드에서 배치 스코어링을 호출하는 예시

---

# Part II. 설정 레퍼런스

## 6. 설정 파일 구조와 YAML 문법

### 6.1 파일 구조

설정 YAML 은 다음 최상위 키들로 구성됩니다. **모든 키는 선택**이며, 누락 시 기본값이
적용됩니다(코드의 `auto_ml/config.py` dataclass 와 1:1 대응).

```yaml
# ----- top-level (데이터 입력) -----
train_data_path: ./data/train.parquet     # §7
test_data_path:  ./data/test.parquet
target_column:   target
features_csv:    ./configs/features.csv
id_columns: [user_id]
artifact_dir:    ./artifacts/models

# ----- 하위 섹션 -----
preprocessing:     {...}    # §9
feature_selection: {...}    # §10
training:          {...}    # §11
tuning:            {...}    # §12
models:            {...}    # §13
ensemble:          {...}    # §14
reporting:         {...}    # §15
scoring:           {...}    # §16
explain:           {...}    # §17
logging:           {...}    # §18
```

전체 옵션을 한눈에 보고 싶다면 `configs/example.yaml` 을 참고하세요 — 모든 기본값이 주석과
함께 노출되어 있어 그대로 복사·수정해서 쓸 수 있습니다.

### 6.2 YAML 문법 빠른 참고

```yaml
# 1) 스칼라
key: value             # 문자열·숫자·boolean
flag: true             # boolean: true/false
n: 5                   # int
ratio: 0.5             # float
maybe: null            # 명시적 null

# 2) 리스트 (블록 / 인라인)
items:
  - a
  - b
items_inline: [a, b]   # 동일

# 3) 매핑 (중첩)
training:
  cv_folds: 5
  primary_metric: roc_auc

# 4) 인라인 dict — search_space 처럼 한 줄에 여러 키
learning_rate: { type: float, low: 0.01, high: 0.3, log: true }

# 5) 들여쓰기는 항상 공백 2칸 (탭 금지)
# 6) 주석은 #
# 7) 문자열 따옴표는 생략 가능하나 특수문자 포함 시 권장
title: "Credit Default Auto-ML Report"
```

YAML 작성 시 흔히 겪는 함정:

- **탭 들여쓰기 금지** — 모든 들여쓰기는 공백. 에디터 설정에서 "Insert spaces" 켜두기.
- **불리언 표기** — `True`/`False` 도 동작하지만, 표준은 소문자 `true`/`false`.
- **`null` 과 빈 문자열** — `key: null` (값 없음) 과 `key: ""` (빈 문자열) 은 다릅니다.
- **상대경로 기준** — 본 라이브러리의 모든 경로는 **현재 작업 디렉토리(CWD)** 기준입니다.
  보통 프로젝트 루트에서 실행하세요.

---

## 7. 데이터 입력 (top-level)

### 역할

학습/테스트 데이터의 위치, 타깃 컬럼 이름, 결과에 보존할 식별자를 지정합니다. 본 라이브러리는
학습 데이터를 **자동으로 train/test 분할하지 않습니다** — 사용자가 별도 Parquet 두 개로
준비해야 합니다.

### 옵션

| 키 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `train_data_path` | str | `./data/train.parquet` | 학습 데이터 Parquet 경로 |
| `test_data_path`  | str | `./data/test.parquet`  | 테스트 데이터 Parquet 경로 (별도 파일) |
| `target_column`   | str | `target` | 타깃 컬럼 이름. **값은 0/1 만 허용** (다른 값 섞이면 `ValueError`) |
| `features_csv`    | str | `./configs/features.csv` | 사용할 컬럼 정의 CSV 경로. 빈 문자열 `""` 이면 CSV 무시(코드에서 직접 `features` 지정) |
| `id_columns`      | list[str] | `[]` | 결과(스코어링/SHAP) 에 그대로 보존할 식별자 컬럼 목록 |
| `artifact_dir`    | str | `./artifacts/models` | `best.joblib` 과 sub-artifact 가 저장될 디렉토리 |

### YAML 예시

```yaml
train_data_path: ./data/my_train.parquet
test_data_path:  ./data/my_test.parquet
target_column:   churn
features_csv:    ./configs/my_features.csv
id_columns:
  - user_id
artifact_dir:    ./artifacts/models
```

### 동작 메모

- **타깃은 0/1 만 허용** — `utils/validation.py` 가 검증합니다. 결측·문자열·2 이상의 정수 등이
  섞여 있으면 즉시 실패합니다.
- **컬럼 일관성** — 학습/테스트/스코어링 입력에 같은 컬럼 셋이 있어야 합니다. 학습 시 사용된
  features 정의는 artifact 메타데이터에 박혀 스코어링 시 검증됩니다(빠진 컬럼은 명시적 에러).
- **`id_columns`** 는 top-level 에서 한 번만 적어두면 `scoring.id_columns` /
  `explain.id_columns` 가 비어 있을 때 자동 fallback 됩니다.

---

## 8. features.csv — 사용할 컬럼 정의

### 역할

데이터의 어떤 컬럼을 어떤 타입으로 학습에 쓸지 명시하는 명단입니다. **운영 중 변수를 켜고 끌
때 코드를 고치지 않고 이 CSV 만 수정**하면 됩니다.

### CSV 형식

| 컬럼 | 의미 | 허용 값 |
|---|---|---|
| `name` | Parquet 의 컬럼명과 정확히 일치 | (Parquet 컬럼명) |
| `type` | 컬럼 종류 | `continuous` (수치형) / `category` (범주형) |
| `used` | 학습/스코어링 사용 여부 | `true` / `false` (대소문자 무시, `1`/`0`, `yes`/`no` 도 허용) |

### 예시

```csv
name,type,used
age,continuous,true
gender,category,true
income,continuous,true
internal_score,continuous,false
```

### 동작 메모

- **타깃 컬럼·id 컬럼은 적지 않습니다.** 각각 `target_column` / `id_columns` 로 따로 지정합니다.
- `used: false` 인 행은 학습/스코어링에 포함되지 않습니다 (CSV 에 남겨두기만 해도 무방).
- 빈 행은 조용히 건너뜁니다. 같은 `name` 이 두 번 나오면 `ValueError`.
- 결과적으로 `used == true` 인 행이 하나도 없으면 `ValueError` (학습할 컬럼이 없음).

---

## 9. 전처리 `preprocessing`

### 역할

전처리는 **항상** 다음 순서로 적용됩니다.

```
(1) 결측 처리  →  (2) 이상치 처리  →  (2.5) skew 변환  →  (3) 스케일링
```

학습 시 산출된 통계량(중앙값, 분위수 경계, 분위수 테이블, 스케일러 fit 결과)은 모두
`best.joblib` 에 저장되어 스코어링 시 **그대로 재적용**됩니다 — 학습/스코어링 일관성이
자동 보장됩니다.

### 옵션

| 키 | 타입 | 기본값 | 허용 값 | 설명 |
|---|---|---|---|---|
| `numeric_null_strategy` | str | `median` | `median` / `mean` / `constant` | 수치형 결측 채움 전략 |
| `numeric_null_fill_value` | float | `0.0` | — | `constant` 일 때 채움 값 |
| `categorical_null_strategy` | str | `most_frequent` | `most_frequent` / `constant` | 범주형 결측 채움 전략 |
| `categorical_null_fill_value` | str | `MISSING` | — | `constant` 일 때 채움 값 |
| `outlier_method` | str | `percentile` | `percentile` / `none` | 이상치 처리 방식 |
| `outlier_lower_quantile` | float | `0.01` | `[0, 1)` | 윈저라이징 하한 분위수 |
| `outlier_upper_quantile` | float | `0.99` | `(0, 1]` | 윈저라이징 상한 분위수 |
| `outlier_action` | str | `clip` | `clip` / `null_then_impute` | 경계 밖 값 처리 방식 |
| `skew_method` | str | `signed_log1p` | `signed_log1p` / `quantile_normal` / `none` | skew 변환 방식 |
| `skew_threshold` | float | `1.0` | — | `|skew| > threshold` 인 컬럼만 변환 |
| `scaling_method` | str | `standard` | `standard` / `minmax` / `robust` / `none` | 스케일링 방식 |

### YAML 예시

```yaml
preprocessing:
  numeric_null_strategy: median               # median | mean | constant
  numeric_null_fill_value: 0.0
  categorical_null_strategy: most_frequent    # most_frequent | constant
  categorical_null_fill_value: MISSING

  outlier_method: percentile                  # percentile | none
  outlier_lower_quantile: 0.01
  outlier_upper_quantile: 0.99
  outlier_action: clip                        # clip | null_then_impute

  skew_method: signed_log1p                   # signed_log1p | quantile_normal | none
  skew_threshold: 1.0

  scaling_method: standard                    # standard | minmax | robust | none
```

### 동작 메모

**결측 처리**
- 전부-NaN 가드: 학습 데이터에서 어떤 수치형 컬럼이 전부 NaN 이면 `median`/`mean` 계산이
  불가능해 silent 전파를 막기 위해 `ValueError` 로 실패합니다. `features.csv` 에서 해당
  컬럼을 `used: false` 로 두거나 `constant` 전략으로 전환하세요.
- 스코어링 시 컬럼 자체가 통째로 누락된 경우에도 학습 시점의 기본값(수치형 median, 범주형
  최빈값)으로 자동 채워집니다 — 일시 누락에 대한 안전망.

**이상치 처리**
- `clip` (기본): 경계 밖 값을 경계로 잘라냅니다.
- `null_then_impute`: 경계 밖을 NaN 으로 바꾼 뒤 median 으로 다시 채웁니다.
- `lower < upper` 위반 시 `ValueError`.

**Skew 변환**
- 변환 대상은 학습 시 한 번 결정됩니다 (`|skew| > skew_threshold` 인 컬럼). 미선택 컬럼은
  스코어링에서도 통과하여 분포 일관성이 유지됩니다.
- `signed_log1p`: `sign(x) * log1p(|x|)`. 무상태(추가 저장 없음), 음수 보존.
- `quantile_normal`: 학습 분포의 분위수를 표준정규로 매핑. 이상치/skew 모두에 강건하지만
  분위수 테이블이 artifact 에 저장됩니다(크기 약간 증가).
- 부스팅 트리만 쓰면 효과가 작으므로 `none` 으로 꺼도 무방합니다.

**스케일링**
- 부스팅 트리는 스케일링에 본질적으로 무관합니다. 옵션을 두는 이유는 선형/신경망 모델
  확장을 대비함이며, 트리만 쓴다면 `none` 으로 둬도 됩니다.

---

## 10. 변수 선택 `feature_selection`

### 역할

학습 데이터의 부분표본을 여러 번 뽑아 각 부분표본에서 변수를 골라 보고, **자주 선택되는
변수만** 최종 채택합니다 (Meinshausen & Bühlmann, 2010 의 Stability Selection). 단일 fit
한 번의 우연성을 제거해 noise 변수가 채택되는 위험을 줄입니다.

기본값은 `enabled: false` — 변수가 적으면 (예: < 30개) 부스팅 모델 내장 selection 으로
충분하므로 굳이 켜지 않아도 됩니다.

### 옵션

| 키 | 타입 | 기본값 | 허용 값 | 설명 |
|---|---|---|---|---|
| `enabled` | bool | `false` | — | 사용 여부 |
| `base_estimator` | str | `lasso` | `lasso` / `lgbm` | 부분표본 단위 선택기 |
| `n_subsamples` | int | `200` | — | 부분표본 추출 횟수 |
| `subsample_ratio` | float | `0.5` | `(0, 1)` | 부분표본 크기 비율 |
| `threshold` | float | `0.6` | `[0, 1]` | 채택 임계값 (선택 빈도) |
| `random_state` | int | `42` | — | 재현성 시드 |
| `min_selected` | int | `1` | — | 임계값 미달 시 빈도 상위 N 개 fallback |
| `lasso_C` | float | `0.1` | — | (lasso 전용) L1 규제 강도 |
| `lgbm_top_k` | int | `30` | — | (lgbm 전용) 부분표본당 gain 상위 K |
| `lgbm_n_estimators` | int | `100` | — | (lgbm 전용) 부분표본 학습용 트리 수 |
| `lgbm_learning_rate` | float | `0.1` | — | (lgbm 전용) 부분표본 학습률 |

### YAML 예시

```yaml
feature_selection:
  enabled: true
  base_estimator: lasso
  n_subsamples: 200
  subsample_ratio: 0.5
  threshold: 0.6
  random_state: 42
  min_selected: 3

  # base_estimator: lasso 일 때만 사용
  lasso_C: 0.1

  # base_estimator: lgbm 일 때만 사용
  lgbm_top_k: 30
  lgbm_n_estimators: 100
  lgbm_learning_rate: 0.1
```

### 동작 메모

**`base_estimator` 선택 기준 비교**

| 항목 | `lasso` (L1 로지스틱) | `lgbm` (LightGBM gain) |
|---|---|---|
| 선택 기준 | non-zero coefficient | gain 상위 K (>0 만) |
| 잘 잡는 신호 | 선형 효과 | 비선형 · 상호작용 |
| 범주형 처리 | full-X frequency encoding (사전 1회) | pandas `category` dtype native |
| 부분표본당 속도 | 빠름 | 상대적으로 느림 |

**설정 가이드**
- `n_subsamples` — 표준 100~500. 작은 데이터면 50~100 도 충분, 큰 데이터·고차원이면 200+.
- `subsample_ratio` — 원 논문 권고 0.5. 너무 크면 모든 부분표본이 비슷해져 안정화 안 됨,
  너무 작으면 fit 자체가 흔들림.
- `threshold` — 0.6 보수적, 0.8 매우 보수적. false-positive 통제 강도의 trade-off.
- `min_selected` — 채택이 부족할 때 발동하는 안전망. 평소엔 발동되지 않아야 하고, 발동 시
  WARN 로그가 남습니다.
- `lasso_C` — 낮을수록 규제 강함(더 sparse). 0.1 시작, 변수가 너무 많이 살면 0.01, 너무
  잘려나가면 1.0.
- `lgbm_top_k` — 전체 feature 의 30~50% 정도가 적당. 너무 크면 threshold 통과 변수가 폭증.

**산출물**
- 채택된 변수 목록·선택 빈도·fallback 발동 여부가 `SelectionResult` 로 `best.joblib`
  메타데이터에 저장됩니다.
- 스코어링 시 동일 컬럼 셋만 모델로 흘러갑니다 — 일관성 자동 보장.
- HTML/PDF 리포트에 변수별 빈도 막대와 채택/제외 라벨이 표기됩니다.

---

## 11. 학습 `training`

### 역할

CV 폴드 수, 조기 종료 라운드, 모델 선택 지표, **[최종 모델 학습 전략](#최종-모델-학습-전략-final_fit_strategy)**
을 결정합니다. 파이프라인은 각 모델에 대해 (1) 튜닝 → (2) StratifiedKFold OOF 평가 → (3) 최종 fit →
(4) 테스트 평가의 순서로 동작하며, best 모델은 테스트 점수 기준으로 자동 선정됩니다.

### 옵션

| 키 | 타입 | 기본값 | 허용 값 | 설명 |
|---|---|---|---|---|
| `cv_folds` | int | `5` | — | 최종 OOF 평가용 StratifiedKFold 분할 수 |
| `random_state` | int | `42` | — | 폴드 분할 등 재현성 시드 |
| `early_stopping_rounds` | int | `50` | — | 부스팅 모델 조기 종료 라운드 수 |
| `primary_metric` | str | `roc_auc` | `roc_auc` / `pr_auc` / `ks` / `f1` / `accuracy` / `precision` / `recall` / `lift` | 모델 선택·튜닝 목적함수 |
| `best_model` | str / null | `null` | `null` / `lgbm` / `xgb` / `catboost` / `elasticnet` / `ensemble` | best 강제 지정 (null = 자동) |
| `final_fit_strategy` | str | `early_stop_on_test` | `early_stop_on_test` / `iteration_capping` / `cv_bagging` | 최종 모델 학습 방식 |
| `iteration_cap_aggregation` | str | `mean` | `mean` / `median` | (capping 전용) fold best_iter 집계 방식 |
| `iteration_cap_headroom` | float | `1.0` | `> 0` | (capping 전용) 집계값 × headroom 후 ceil |

`primary_metric` 가이드:

| 지표 | 추천 도메인 |
|---|---|
| `roc_auc` | 일반 (기본) |
| `pr_auc` | positive 비율이 매우 낮은 불균형 데이터 |
| `ks` | 신용평점 / 리스크 모델링 |
| `f1` / `precision` / `recall` | 임계값 0.5 기준 평가 |
| `lift` | 상위 분위 lift |

### YAML 예시 (기본 옵션)

```yaml
training:
  cv_folds: 5
  random_state: 42
  early_stopping_rounds: 50
  primary_metric: roc_auc
  best_model: null
```

> 최종 모델 학습 전략(`final_fit_strategy` / `iteration_cap_*`)은 옵션이 많아 아래
> [최종 모델 학습 전략](#최종-모델-학습-전략-final_fit_strategy) 서브섹션에서 따로 다룹니다.

### 동작 메모

**`early_stopping_rounds`** 는 부스팅 세 모델에 통합 적용되지만 내부 구현이 다릅니다
(LGBM callbacks, XGB 생성자 인자, CatBoost 학습 인자). OOF fold 학습에는 항상 적용되지만,
**최종 fit 의 적용 여부는 `final_fit_strategy` 에 좌우**됩니다 (아래 서브섹션 참고).

**`best_model` 강제 지정**
- `lgbm` / `xgb` / `catboost` / `elasticnet` / `ensemble` 중 하나로 지정 가능. 알 수 없는
  이름이면 `ValueError`.
- 학습 후 변경하려면 `auto-ml-set-best --model <name>` CLI 사용 (재학습 불필요).

### 최종 모델 학습 전략 (`final_fit_strategy`)

`final_fit_strategy` 는 OOF 평가가 끝난 뒤 **운영에 쓸 최종 모델을 어떻게 만들지**를 결정합니다.
세 가지 중 하나를 고르며, 기본값은 현행 동작과 동일한 `early_stop_on_test` 입니다 — **지정하지
않으면 동작이 바뀌지 않습니다.**

**왜 필요한가**

기본 `early_stop_on_test` 는 최종 모델을 학습할 때 **테스트셋을 early-stopping 검증셋으로
사용**합니다. 편리하지만 최종 트리 수(`best_iter`)와 best 모델 선정이 테스트셋에 맞춰지므로,
리포트 §3 "오버핏 점검" 의 `Δ = Train − Test` 가 실제보다 **작게(낙관적으로)** 나옵니다 — 즉
테스트 점수가 약간 부풀려집니다. 아래 두 전략은 학습 단계에서 테스트셋을 전혀 보지 않아 Δ 가
정직해집니다.

**세 전략 비교**

| 전략 | 동작 | 테스트셋 노출 | 언제 쓰나 |
|---|---|---|---|
| `early_stop_on_test` (기본) | 학습 전체로 fit, 테스트셋을 조기종료 검증셋으로 사용 | O | 현행 호환 / 단일 데이터셋 일관성 우선 |
| `iteration_capping` | CV fold best_iter 집계로 트리 수 고정, early stop 없이 train 전체로 재학습 | X | 오버핏 Δ 가 큰데 정직한 트리 수로 재학습하고 싶을 때 |
| `cv_bagging` | 최종 재학습 생략, CV 의 K fold 모델을 균등 가중 앙상블로 사용 | X | 분산을 줄이고 안정적인 모델을 원할 때 (재학습 비용도 절약) |

**관련 옵션** (모두 `training:` 아래):

| 키 | 적용 전략 | 설명 |
|---|---|---|
| `final_fit_strategy` | 공통 | `early_stop_on_test`(기본) / `iteration_capping` / `cv_bagging` |
| `iteration_cap_aggregation` | iteration_capping | fold best_iter 집계 방식. `mean`(기본) / `median` |
| `iteration_cap_headroom` | iteration_capping | 집계값에 곱하는 여유 배수. `capped = ceil(agg × headroom)`. 기본 1.0 |

**YAML 예시**

오버핏 Δ 가 큰 경우 — 정직한 트리 수로 재학습:

```yaml
training:
  cv_folds: 5
  primary_metric: roc_auc
  final_fit_strategy: iteration_capping
  iteration_cap_aggregation: mean     # 각 fold best_iter 의 평균 (median 도 가능)
  iteration_cap_headroom: 1.0         # 1.0 = 평균 그대로. 1.05~1.1 로 올리면 약간 더 학습
```

안정적인 앙상블 — CV fold K 개 모델을 균등 평균:

```yaml
training:
  final_fit_strategy: cv_bagging
```

**동작 메모**

- 두 신규 전략은 학습 단계에서 테스트셋을 보지 않으므로 리포트 §3 의 Δ 가 정직해집니다
  (기본 전략보다 약간 커 보일 수 있는데, 편향이 제거된 정상 수치입니다).
- **ElasticNet 은 `iteration_capping` 에서 자동 면제** (부스팅이 아니라 고정할 트리 수가
  없음). 일반 refit 으로 동작하며 INFO 로그가 남습니다.
- **`cv_bagging` 과 `ensemble.enabled: true` 를 함께 켜면** 외부 앙상블이 자동 비활성화되고
  WARN 로그가 남습니다 (각 모델이 이미 fold 평균 = bagging 이므로 중복 방지).
- 사용된 전략은 `<artifact_dir>/models/<name>.joblib` 메타데이터의 `extra.training_mode` 에
  기록됩니다.

---

## 12. 튜닝 `tuning`

### 역할

Optuna TPE 샘플러로 각 모델의 `search_space` 를 탐색해 `primary_metric` KFold OOF 평균을
최대화하는 하이퍼파라미터를 찾습니다. 본 단계는 **테스트 데이터를 절대 사용하지 않습니다**
(정보 누설 방지).

### 옵션

| 키 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `enabled` | bool | `true` | False 면 모든 모델 튜닝 생략 |
| `n_trials` | int | `30` | 모델당 시도 횟수 |
| `timeout` | int / null | `null` | 모델당 wall-clock 상한(초). null = 무제한 |
| `cv_folds` | int | `3` | 튜닝 단계 KFold 분할 수 (보통 `training.cv_folds` 보다 작게) |
| `random_state` | int | `42` | TPE 샘플러 시드 |

### YAML 예시

```yaml
tuning:
  enabled: true
  n_trials: 30
  timeout: 1800        # 모델당 30분 상한
  cv_folds: 3
  random_state: 42
```

### 동작 메모

- 모델별 `search_space` 가 비어 있으면 그 모델은 `fixed_params` 만으로 학습하고 튜닝을
  생략합니다.
- 튜닝 단계의 `early_stopping_rounds` 는 `training.early_stopping_rounds` 를 그대로 사용합니다.
- **빠른 1차 검증**: `n_trials: 10`, `cv_folds: 2`. **운영 권장**: `n_trials: 30~50`,
  `cv_folds: 3`, 필요시 `timeout` 으로 wall-clock 보호.

---

## 13. 모델 `models`

### 역할

활성화할 모델과 각 모델의 하이퍼파라미터 / 탐색 공간 / 손실 함수를 지정합니다. 모든 모델은
동일 인터페이스(`fit / predict_proba / feature_importance / shap_values`)를 따르므로 학습/
스코어링/SHAP 경로가 모델 종류에 무관합니다.

### 13.1 공통 구조

`models.<name>` 의 키 (모든 모델 공통):

| 키 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `enabled` | bool | `true` (ElasticNet 만 `false`) | 학습에 포함 여부 |
| `fixed_params` | dict | `{}` | 항상 적용되는 라이브러리 원본 파라미터 |
| `search_space` | dict | `{}` | Optuna 탐색 공간. 비어 있으면 튜닝 생략 |
| `loss` | str | `logloss` | `logloss` (기본) / `focal` (Focal Loss) |

`search_space` 항목 형식:

```yaml
# 1) float — low/high 필수, log 선택(기본 false)
learning_rate: { type: float, low: 0.01, high: 0.3, log: true }

# 2) int — low/high 필수, log·step 선택(step 기본 1)
num_leaves:    { type: int,   low: 15,   high: 255 }

# 3) categorical — choices 필수
boosting_type: { type: categorical, choices: [gbdt, dart] }
```

### 13.2 LightGBM `lgbm`

- 범주형 처리: pandas `category` dtype native
- early stopping: `callbacks=[early_stopping(rounds)]`
- Focal Loss: 지원

```yaml
models:
  lgbm:
    enabled: true
    fixed_params:
      objective: binary
      metric: auc
      verbose: -1
      n_estimators: 2000
    search_space:
      learning_rate:    { type: float, low: 0.01,   high: 0.3,  log: true }
      num_leaves:       { type: int,   low: 15,     high: 255 }
      max_depth:        { type: int,   low: 3,      high: 12 }
      min_data_in_leaf: { type: int,   low: 10,     high: 100 }
      feature_fraction: { type: float, low: 0.6,    high: 1.0 }
      bagging_fraction: { type: float, low: 0.6,    high: 1.0 }
      reg_alpha:        { type: float, low: 1.0e-8, high: 10.0, log: true }   # L1
      reg_lambda:       { type: float, low: 1.0e-8, high: 10.0, log: true }   # L2
      min_gain_to_split:{ type: float, low: 0.0,    high: 0.5 }
```

### 13.3 XGBoost `xgb`

- 범주형 처리: 내부 `LabelEncoder` (학습 시 fit → 스코어링 재사용, unseen 은 -1)
- early stopping: 생성자 인자 `early_stopping_rounds` (XGB 2.x sklearn API)
- Focal Loss: 지원

```yaml
models:
  xgb:
    enabled: true
    fixed_params:
      objective: binary:logistic
      eval_metric: auc
      tree_method: hist
      n_estimators: 2000
    search_space:
      learning_rate:    { type: float, low: 0.01,   high: 0.3,  log: true }
      max_depth:        { type: int,   low: 3,      high: 10 }
      subsample:        { type: float, low: 0.6,    high: 1.0 }
      colsample_bytree: { type: float, low: 0.6,    high: 1.0 }
      min_child_weight: { type: float, low: 1.0,    high: 10.0 }
      reg_alpha:        { type: float, low: 1.0e-8, high: 10.0, log: true }   # L1
      reg_lambda:       { type: float, low: 1.0e-8, high: 10.0, log: true }   # L2
      gamma:            { type: float, low: 0.0,    high: 5.0 }
```

### 13.4 CatBoost `catboost`

- 범주형 처리: native (`cat_features` 인덱스 전달)
- early stopping: 학습 인자 `early_stopping_rounds`
- iteration 키: **`iterations`** (다른 모델의 `n_estimators` 와 다름)
- Focal Loss: 지원 (`PythonUserDefinedObjective`)
- L1 정규화: **미지원** (L2 만 사용 — `l2_leaf_reg`)

```yaml
models:
  catboost:
    enabled: true
    fixed_params:
      loss_function: Logloss
      eval_metric: AUC
      iterations: 2000              # ← n_estimators 아님!
      verbose: false
      allow_writing_files: false     # 권장 (작업 디렉토리 오염 방지)
    search_space:
      learning_rate:       { type: float, low: 0.01, high: 0.3, log: true }
      depth:               { type: int,   low: 4,    high: 8 }
      l2_leaf_reg:         { type: float, low: 1.0,  high: 100.0, log: true }
      random_strength:     { type: float, low: 0.0,  high: 10.0 }
      bagging_temperature: { type: float, low: 0.0,  high: 1.0 }
```

### 13.5 ElasticNet `elasticnet`

- sklearn `LogisticRegression(solver='saga')` 기반 L1+L2 혼합 규제 선형 모델
- 범주형 처리: 내부 `OneHotEncoder` (학습 시 fit, `handle_unknown="ignore"`)
- early stopping: **무시** (sklearn 에 해당 개념 없음)
- iteration capping: **자동 면제** (부스팅 아님)
- Focal Loss: **미지원** (`loss: logloss` 만)
- 기본값 `enabled: false` — 필요 시 활성화

```yaml
models:
  elasticnet:
    enabled: true
    fixed_params:
      max_iter: 2000        # 수렴이 안 되면 늘리기
    search_space:
      C:        { type: float, low: 1.0e-3, high: 10.0, log: true }
      l1_ratio: { type: float, low: 0.0,    high: 1.0 }
```

- `C` — 역규제 강도. 낮을수록 규제 강함(더 sparse).
- `l1_ratio` — 0.0 = 순수 Ridge (L2), 1.0 = 순수 Lasso (L1). 중간값은 혼합.

### 13.6 Focal Loss (선택)

클래스 불균형이 큰 데이터(예: positive 비율 < 5%)에서 `loss: focal` 로 전환 가능.
모델별 독립이며, LGBM/XGB/CatBoost 만 지원합니다.

```yaml
models:
  xgb:
    loss: focal                     # logloss (기본) | focal
    fixed_params:
      eval_metric: auc              # rank-기반 metric 필수 (auc, pr_auc)
      tree_method: hist
      n_estimators: 2000
      focal_gamma: 2.0              # easy-example 감쇠 지수 (기본 2.0)
      focal_alpha: 0.25             # 양성 클래스 가중 (기본 0.25)
    search_space:
      # focal 하이퍼파라미터도 탐색 가능:
      # focal_gamma: { type: float, low: 0.5, high: 4.0 }
      # focal_alpha: { type: float, low: 0.1, high: 0.9 }
```

- 호환 metric: **rank-based** (`auc`, `pr_auc`, `AUC`). `binary_logloss`/`Logloss` 등 절대값
  기반은 raw score 에 잘못 적용되어 의미가 깨지므로 사용 금지.
- `focal_gamma` / `focal_alpha` 는 wrapper 가 pop 하므로 라이브러리 원본 인자에 전달되지
  않습니다 (예약 키).

---

## 14. 앙상블 `ensemble`

### 역할

활성화된 개별 모델 학습 완료 후 추가로 앙상블 모델을 생성합니다. best 선정 시 앙상블도 후보에
포함됩니다.

### 옵션

| 키 | 타입 | 기본값 | 허용 값 | 설명 |
|---|---|---|---|---|
| `enabled` | bool | `true` | — | 앙상블 생성 여부 (활성 모델 ≥ 2개 필요) |
| `strategy` | str | `elasticnet_plus_best` | `elasticnet_plus_best` / `weighted_average` | 구성 방식 |
| `elasticnet_weight` | float / null | `null` | `(0, 1)` 또는 null | (전자 전용) elasticnet 가중치, null = 자동 |

### YAML 예시

기본 (ElasticNet + 나머지 best):

```yaml
ensemble:
  enabled: true
  strategy: elasticnet_plus_best
  elasticnet_weight: null          # 점수 비례 자동 계산
  # elasticnet_weight: 0.35         # 0~1 사이 값으로 직접 지정도 가능
```

전체 모델 점수 비례 가중 평균:

```yaml
ensemble:
  enabled: true
  strategy: weighted_average
```

### 동작 메모

**전략별 동작**
- `elasticnet_plus_best` — `elasticnet` 모델을 무조건 포함하고 나머지 활성 모델 중 test
  `primary_metric` 1위 1개와 결합. ElasticNet 이 비활성이면 `ValueError`.
- `weighted_average` — 활성화된 모든 모델을 test `primary_metric` 점수에 비례하는 가중치로
  결합. 가중치는 softmax(점수 / 합) 로 정규화.

**호환성**
- 앙상블의 SHAP 은 서브모델 raw-margin SHAP 의 가중 평균을 반환하므로 `auto-ml-explain` 과
  완전 호환됩니다.
- `cv_bagging` 과 함께 켜면 외부 앙상블은 **자동 비활성화** + WARN (각 모델이 이미 fold 평균).
- 앙상블이 best 로 선정되면 `best.joblib` 이 곧 앙상블입니다. cloudpickle 이 서브모델 전체
  객체 그래프를 직렬화하므로 단일 파일로 스코어링·SHAP 모두 동작합니다.

---

## 15. 리포트 `reporting`

### 역할

HTML / PDF 동일 내용 리포트를 생성합니다(Jinja2 + WeasyPrint). 모델 비교표, CV(OOF) 비교,
**오버핏 점검 (Train vs Test, Δ)**, 튜닝 결과, ROC / PR 곡선, feature importance, score
분포, confusion matrix, (활성 시) 변수 선택 결과 등을 포함합니다.

### 옵션

| 키 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `output_dir` | str | `./artifacts/reports` | 리포트와 부속 CSV 저장 디렉토리 |
| `generate_html` | bool | `true` | HTML 생성 여부 |
| `generate_pdf` | bool | `true` | PDF 생성 여부 (WeasyPrint 필요) |
| `title` | str | `Auto-ML Binary Classification Report` | 리포트 제목 |
| `top_importance_features` | int / null | `30` | feature importance 표·차트에 노출할 상위 N개. null = 전체 |
| `top_selection_features` | int / null | `null` | 변수 선택 결과 노출 상위 N개. null = 전체 |

### YAML 예시

```yaml
reporting:
  output_dir: ./artifacts/credit/reports
  generate_html: true
  generate_pdf: true
  title: Credit Card Default Auto-ML Report
  top_importance_features: 30
  top_selection_features: null
```

### 동작 메모

- WeasyPrint 시스템 의존성이 없으면 PDF 생성에 실패합니다 — `generate_pdf: false` 로 명시적으로
  끄거나 §3 의 시스템 패키지를 설치하세요. PDF 실패해도 HTML 은 정상 생성됩니다.
- 부속 CSV (`feature_importance.csv`, `feature_selection.csv`) 는 **항상 전체 변수**를
  저장합니다 (`top_*` 옵션은 HTML 표시에만 영향).
- best 가 아닌 모형마다 별도 sub-report 가 자동 생성됩니다 (`output_dir/sub/<name>/report.{html,pdf}`).

---

## 16. 스코어링 `scoring`

### 역할

`auto-ml-score` 가 사용하는 옵션. 학습된 `best.joblib` 으로 새 데이터에 점수를 매겨 Parquet
으로 출력합니다.

### 옵션

| 키 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `input_path` | str | `./data/score_input.parquet` | 스코어링 대상 Parquet 경로 |
| `output_path` | str | `./artifacts/scores/scores.parquet` | 결과 저장 경로 |
| `id_columns` | list[str] | `[]` | 결과에 보존할 식별자 (override) |
| `threshold` | float | `0.5` | `predict_proba` → 0/1 변환 임계값 |

### YAML 예시

```yaml
scoring:
  input_path:  ./data/score_input.parquet
  output_path: ./artifacts/scores/scores.parquet
  id_columns:
    - user_id
  threshold: 0.5
```

### 동작 메모

- 결과 컬럼: `<id_columns> + score + prediction`.
- `id_columns` 우선순위: (1) `scoring.id_columns` → (2) top-level `id_columns` → (3) artifact
  메타데이터에 저장된 학습 시점 id_columns. 보통 top-level 에 한 번 적어두면 충분합니다.
- 스코어링 입력은 학습 시점의 features 정의(+ 변수 선택 결과)와 동일 컬럼 셋이어야 합니다.
  컬럼이 빠지면 명시적 에러.
- `threshold` 는 도메인에 따라 조정 (예: false-negative 비용이 크면 0.3 으로 낮춤).

---

## 17. SHAP 해석 `explain`

### 역할

`auto-ml-explain` 이 사용하는 옵션. 학습된 모델이 각 행을 왜 그렇게 예측했는지 **건별·변수별
기여도(SHAP)** 를 산출해 Parquet 으로 저장합니다. LGBM/XGB/CatBoost 는 native API
(`pred_contrib`), ElasticNet 은 `shap.LinearExplainer`, 앙상블은 서브모델 SHAP 의 가중
평균을 사용합니다.

### 옵션

| 키 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `input_path` | str | `./data/score_input.parquet` | 해석 대상 Parquet (보통 스코어링 입력과 동일) |
| `output_path` | str | `./artifacts/explanations/shap.parquet` | 결과 저장 경로 |
| `id_columns` | list[str] | `[]` | 결과 식별자 (override). 스코어링과 동일 우선순위 |

### YAML 예시

```yaml
explain:
  input_path:  ./data/score_input.parquet
  output_path: ./artifacts/explanations/shap.parquet
  id_columns:
    - user_id
```

### 출력 스키마

```
<id_columns> + shap_<feature_1> + shap_<feature_2> + ... + base_value + score
```

- `base_value`: 모델의 평균 예측 확률 (각 행 동일).
- `shap_<feature>`: 해당 변수가 평균 대비 예측을 얼마나 끌어올렸/내렸는지 (**확률 도메인**).
- `score`: 최종 예측 확률 (= `scoring` 의 `score` 와 동일).
- **가법성 보장**: 모든 행에서 `base_value + sum(shap_*) ≈ score` (오차 ≤ 1e-12).

### 동작 메모

native API 는 raw-margin (logit) 도메인 SHAP 을 반환합니다. 본 라이브러리는 다음과 같이
확률 도메인으로 변환하면서 가법성을 보존합니다.

```
p_base = sigmoid(base_raw)
p_pred = sigmoid(base_raw + sum(feature_raw))
shap_prob[i] = feature_raw[i] / sum(feature_raw) * (p_pred - p_base)
```

비율은 raw 도메인 그대로 유지되고, 합산만 `(score - base_value)` 와 정확히 일치하도록
스케일링됩니다.

---

## 18. 로깅 `logging`

### 역할

학습/스코어링/SHAP 실행마다 stage 별 별도 로그 파일을 자동 생성합니다.

### 옵션

| 키 | 타입 | 기본값 | 허용 값 | 설명 |
|---|---|---|---|---|
| `log_dir` | str | `./artifacts/logs` | — | 파일 로그 저장 디렉토리 |
| `level` | str | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` | 로깅 레벨 |
| `to_stdout` | bool | `true` | — | 콘솔 출력 여부 |
| `to_file` | bool | `true` | — | 파일 출력 여부 |

### YAML 예시

```yaml
logging:
  log_dir: ./artifacts/logs
  level: INFO
  to_stdout: true
  to_file: true
```

### 동작 메모

- 파일명은 `train_YYYYMMDD_HHMMSS.log` / `score_YYYYMMDD_HHMMSS.log` /
  `explain_YYYYMMDD_HHMMSS.log` 형식으로 stage 별 분리됩니다.
- 콘솔과 파일은 동시에 활성화 가능. `to_file: false` 면 파일은 만들어지지 않습니다.
- 각 로그에는 시작/종료 마커, 단계별 진행, 모델 튜닝 결과, 산출물 경로, 총 소요 시간이
  포함됩니다 — 문제 추적의 1차 자료.

---

# 부록

## A. CLI 명령어 모음

| 명령 | 진입점 | 용도 |
|---|---|---|
| `auto-ml-train --config <yaml>` | `auto_ml.pipeline:cli_train` | 학습 파이프라인 |
| `auto-ml-score --config <yaml>` | `auto_ml.scoring.runner:cli_score` | 배치 스코어링 |
| `auto-ml-explain --config <yaml>` | `auto_ml.explain.runner:cli_explain` | SHAP 해석 |
| `auto-ml-set-best --config <yaml> --model <name>` | `auto_ml.cli.set_best:cli_set_best` | 재학습 없이 best 교체 |

`auto-ml-set-best` 예시:

```bash
# xgb 후보를 best 로 승격
auto-ml-set-best --config configs/my_config.yaml --model xgb

# 앙상블 후보를 best 로 승격
auto-ml-set-best --config configs/my_config.yaml --model ensemble
```

승격된 best 의 metadata 에는 `promoted_at` / `promoted_from` (이전 best 이름) 이 기록됩니다.

---

## B. 산출물 안내

### B.1 `best.joblib` (학습 산출물)

전처리기 + 모델 + 메타데이터가 하나의 파일로 묶인 cloudpickle+gzip 번들입니다. 스코어링/
SHAP 시 이 파일 하나만 있으면 됩니다.

전체 sub-artifact 구조:

```
<artifact_dir>/
├── best.joblib                  ← 운영용 (scoring/explain 대상)
└── models/
    ├── lgbm.joblib              ← 후보 1
    ├── xgb.joblib               ← 후보 2
    ├── catboost.joblib          ← 후보 3
    ├── elasticnet.joblib        ← 후보 4 (enabled 시)
    └── ensemble.joblib          ← 앙상블 후보 (enabled 시)
```

`best.joblib` 은 위 sub-artifact 중 하나의 복사본입니다. 각 sub-artifact 는 독립 번들이라
그 자체로 `auto-ml-score` / `auto-ml-explain` 호환입니다.

### B.2 HTML / PDF 리포트

`reporting.output_dir/report.html` 을 브라우저로 열면 8개 섹션으로 구성된 리포트를 볼 수 있습니다.

| 섹션 | 내용 |
|---|---|
| **§1 모델 비교 (holdout 기준)** | 전 모델 Test 점수 비교표 + best 하이라이트 |
| **§2 CV (OOF) 비교** | Out-Of-Fold 평가 비교표 |
| **§3 오버핏 점검 (Train vs Holdout)** | Train·Test 점수 Δ 표 — 과적합 진단 |
| **§4 곡선 비교** | ROC Curve + Precision-Recall Curve |
| **§5 Best 모델 상세** | Feature Importance (상위 N), 10-분위 분석, Score Distribution, Confusion Matrix |
| **§6 학습 설정 요약** | 사용된 YAML 설정 핵심 요약 |
| **§7 모델 파라미터** | 각 모델의 최종 파라미터 (Optuna best params 포함) |
| **§8 변수 선택** | (활성 시) Stability Selection 빈도 막대 + 채택/제외 라벨 |

PDF 도 동일 내용입니다.

**부속 파일**
- `feature_importance.csv` — 전체 변수 중요도 (HTML 은 `top_importance_features` 만 표시하지만 CSV 는 항상 전체).
- `feature_selection.csv` — (변수 선택 활성 시) 변수별 선택 빈도. 모형 무관이라 메인 리포트에만 1개.

**비-best 모형 sub-report**

best 가 아닌 모형마다 `reporting.output_dir/sub/<model>/` 에 별도 리포트가 자동 생성됩니다.

```
reports/sub/lgbm/report.html, report.pdf, feature_importance.csv
reports/sub/xgb/...
reports/sub/ensemble/...
```

각 sub-report 는 해당 모형 vs best 점수 비교, feature importance, 10-분위 분석,
score distribution, confusion matrix, 변수 선택 결과를 포함합니다.

### B.3 `scores.parquet` (스코어링 산출물)

```
user_id, score, prediction
1001,    0.87,  1
1002,    0.12,  0
```

- `score` — 0~1 사이 확률값
- `prediction` — `score >= scoring.threshold` 면 1, 아니면 0

### B.4 `test_predictions.parquet` (학습 부산물)

학습 시 best 모델의 holdout 예측을 함께 export 합니다:
`<artifact_dir>/../predictions/test_predictions.parquet`,
스키마 `<id_columns> + score + prediction + <target_column>`. 외부 BI / 추가 진단용
(overfit 검증, 임계값 튜닝 등).

### B.5 `shap.parquet` (SHAP 산출물)

`<id_columns> + shap_<feature>... + base_value + score` wide 포맷. 확률 도메인이며
`base_value + sum(shap_*) ≈ score` 가법성이 보장됩니다. 자세한 스키마는 [§17 SHAP 해석](#17-shap-해석-explain) 참고.

### B.6 로그 파일

`artifacts/logs/` 에 stage 별 별도 로그가 생성됩니다.

```
train_YYYYMMDD_HHMMSS.log    ← 학습 파이프라인
score_YYYYMMDD_HHMMSS.log    ← 배치 스코어링
explain_YYYYMMDD_HHMMSS.log  ← SHAP 해석
```

단계별 진행, 모델 튜닝 결과, 산출물 경로, 총 소요시간이 기록됩니다. 문제 발생 시 가장 먼저
보는 곳입니다.

---

## C. 트러블슈팅 FAQ

**Q1. `auto-ml-train: command not found`**
가상환경 활성화 여부 확인. 프롬프트 앞에 `(.venv)` 가 보여야 합니다. 안 보이면
`source .venv/bin/activate` 다시 실행.

**Q2. `ValueError: target column must contain only 0 and 1`**
타깃 컬럼에 0/1 외 값(NaN, 2, "yes" 등)이 섞임. 데이터 정리 후 재시도.

**Q3. `ValueError: column X is all NaN`**
어떤 수치형 컬럼이 학습 데이터에서 전부 결측. 다음 중 하나로 해결:
- `features.csv` 에서 해당 컬럼 `used: false`
- `preprocessing.numeric_null_strategy: constant` 로 전환
- 데이터 자체 정리

**Q4. PDF 리포트가 안 만들어져요**
WeasyPrint 시스템 의존성 누락. HTML 은 정상 생성됩니다. PDF 가 필요하면 §3 의 시스템 패키지
설치, 아니면 `reporting.generate_pdf: false` 로 끄세요.

**Q5. 학습이 너무 오래 걸립니다**
다음을 줄이세요:

```yaml
tuning:
  n_trials: 10           # 30 → 10
  cv_folds: 2            # 3 → 2
training:
  cv_folds: 3            # 5 → 3
```

또는 `tuning.timeout: 1800` (30분) 으로 wall-clock 상한 지정.

**Q6. 모델 한 개만 쓰고 싶어요**
`models.<name>.enabled: false` 로 나머지를 끄세요. 최소 1개는 켜져 있어야 합니다.

**Q7. 임계값을 0.5 가 아닌 값으로**
`scoring.threshold: 0.3` 처럼 변경 (0.3 이상이면 1로 분류).

**Q8. 스코어링 입력에 컬럼 하나가 없어요**
학습 시 features 정의가 `best.joblib` 메타데이터에 박혀 있어 동일 컬럼 셋이 필요합니다.
빠진 컬럼이 있으면 명시적 에러로 알려줍니다. 일시 누락이면 컬럼만 NaN 으로 채워 다시
실행하세요(라이브러리가 자동으로 학습 시 기본값으로 채웁니다).

**Q9. 오버핏 Δ 가 너무 큽니다**
[§11 최종 모델 학습 전략](#최종-모델-학습-전략-final_fit_strategy)의
`final_fit_strategy: iteration_capping` 또는 `cv_bagging` 으로 전환을 검토하세요.
테스트셋이 학습 신호로 노출되는 편향을 제거합니다.

**Q10. ElasticNet 만 Focal Loss 가 안 됩니다**
지원하지 않는 조합입니다. `models.elasticnet.loss: logloss` (기본값) 로 두세요.

---

## D. 폐쇄망 이관

다음 4가지만 옮기면 됩니다.

1. **패키지 wheel 묶음** — 외부망에서 미리 만들어 둡니다.
   ```bash
   pip wheel -r requirements.txt -w wheelhouse/
   ```
2. **본 저장소 자체** — `git clone` 한 폴더 그대로.
3. **학습 산출물** — `artifacts/models/best.joblib`
4. **스코어링용 설정 YAML**

폐쇄망에서 설치:

```bash
pip install --no-index --find-links=wheelhouse -r requirements.txt
pip install -e .
auto-ml-score --config configs/my_config.yaml
```

PDF 리포트가 필요하면 WeasyPrint 시스템 의존성(libpango 등)도 함께 배포하세요.

---

## E. 용어 사전

| 용어 | 설명 |
|---|---|
| **이진 분류** | 0/1 (참/거짓) 두 가지 중 하나를 맞히는 문제 |
| **타깃 (target)** | 예측 대상 컬럼 (0 또는 1) |
| **피처 (feature)** | 예측의 근거가 되는 입력 변수 |
| **전처리** | 결측 채우기, 이상치 자르기, 스케일링 등 학습 전 데이터 정리 |
| **스케일링** | 변수마다 다른 단위를 비슷한 크기로 맞춤 |
| **이상치** | 분포에서 극단적으로 동떨어진 값 |
| **Skew (왜도)** | 분포가 한쪽으로 치우친 정도 |
| **Stability Selection** | 부분표본을 여러 번 뽑아 자주 선택되는 변수만 추리는 절차 (Meinshausen & Bühlmann, 2010) |
| **Optuna / TPE** | 베이지안 하이퍼파라미터 탐색 라이브러리·알고리즘 |
| **하이퍼파라미터** | 학습 전 사람이 정하는 설정값 (학습률, 트리 깊이 등) |
| **KFold / OOF** | 데이터 K 등분 후 돌아가며 평가. Out-Of-Fold 예측으로 안정적 점수 산출 |
| **Early stopping** | 검증셋 점수가 일정 라운드 개선되지 않으면 학습 조기 종료 |
| **ROC-AUC** | 분류 성능 지표 (0.5=찍기, 1.0=완벽). 본 라이브러리 기본 지표 |
| **PR-AUC** | Precision-Recall 곡선 아래 면적. positive 비율이 매우 낮을 때 권장 |
| **KS statistic** | Kolmogorov-Smirnov 통계. 신용평점 도메인 표준 지표 |
| **Focal Loss** | 어려운 샘플에 학습 신호를 집중시키는 손실 (Lin et al. 2017). 불균형 데이터 대응 |
| **SHAP** | 건별·변수별 예측 기여도. Shapley value 기반 |
| **Artifact** | 학습 결과물(모델 + 전처리기 + 메타데이터) 단일 번들 파일 |
| **Parquet** | 컬럼 기반 효율적 데이터 파일 형식 (CSV 보다 빠르고 작음) |

---

문제가 풀리지 않으면 `artifacts/logs/` 의 최신 로그 파일과 함께 담당자에게 공유해 주세요.
