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
├── reporting/          HTML + PDF 리포트 (동일 내용)
├── scoring/            배치 스코어링 (auto-ml-score)
└── utils/              io / logger / validation
```

## 4단계 파이프라인

1. **전처리** — 결측 → 이상치 → 스케일링 순서 고정. 학습 시 통계량을
   저장해 스코어링 시 동일 적용.
2. **모형 적합** — LGBM, XGBoost, CatBoost 3종을 StratifiedKFold OOF 로
   비교하고, primary_metric (기본 ROC-AUC) 기준으로 best 선정.
3. **보고서 산출** — HTML / PDF 동일 내용 (Jinja2 + WeasyPrint). 모델 비교표,
   ROC / PR 곡선, feature importance, score 분포, confusion matrix 포함.
4. **주기 스코어링** — 단일 artifact (preprocessor + model + metadata)
   파일 1개 + 설정 YAML 1개로 운영 가능. cron 등에서 `auto-ml-score` 호출.

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

# 2) 학습 — 산출물: artifacts/models/best.joblib + artifacts/reports/{report.html, report.pdf}
auto-ml-train --config configs/example.yaml

# 3) 스코어링 — 산출물: artifacts/scores/scores.parquet
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

## 운영 메모

- 타깃은 0/1 만 허용 (`utils/validation.py` 가 검증).
- best 선정은 holdout 점수 기준 (overfitting 방지).
- 스코어링 결과 컬럼: `<id_columns> + score + prediction`.
