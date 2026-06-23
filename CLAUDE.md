# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Install (editable):**
```bash
pip install -e .
```

**Run all tests:**
```bash
pytest
```

**Run a single test file:**
```bash
pytest tests/test_focal_loss_math.py -v
```

**CLI entry points (after install):**
```bash
auto-ml-train   --config configs/example.yaml
auto-ml-score   --config configs/score.yaml
auto-ml-explain --config configs/explain.yaml
auto-ml-set-best --artifact ./artifacts/models --best lgbm
```

**Python API:**
```python
from auto_ml import AutoMLPipeline, load_config
outputs = AutoMLPipeline(load_config("configs/example.yaml")).run()
```

## Architecture

### Pipeline (`auto_ml/pipeline.py`)

`AutoMLPipeline.run()` orchestrates five sequential steps:

1. **Load** — reads two separate Parquet files (`train_data_path`, `test_data_path`); validates schema and binary target.
2. **Preprocess** — `PreprocessingPipeline` applies: null imputation → outlier winsorizing → skew transform → scaling. Fitted on train, `transform()`-only on test.
3. **Feature selection** (optional) — `StabilitySelector` runs Stability Selection (Meinshausen & Bühlmann 2010) using repeated subsampling with L1 logistic or LightGBM base estimators. Controlled by `feature_selection.enabled`.
4. **Train** — `Trainer` (in `auto_ml/models/trainer.py`) loops over enabled models: Optuna hyperparameter tuning → stratified K-fold CV → final fit. Builds an optional ensemble. Selects `best` by `primary_metric` on the test set (or user override).
5. **Report + save** — generates HTML/PDF reports; saves every model as `artifact_dir/models/<name>.joblib` and copies best to `artifact_dir/best.joblib`.

### Config (`auto_ml/config.py`)

`AutoMLConfig` is a top-level dataclass containing nested sub-configs for every pipeline stage. `load_config(path)` reads a YAML file and a companion features CSV.

- **Features CSV** (`features_csv`): columns `name`, `type` (`continuous`/`category`), `used` (`true`/`false`). Only `used == true` rows become features. Internally, `continuous` → `"numeric"` and `category` → `"categorical"`.
- All YAML keys map 1:1 to dataclass fields in `config.py`.

### Models (`auto_ml/models/`)

| File | Role |
|---|---|
| `trainer.py` | Orchestrates tune → CV → final-fit for all enabled models |
| `lgbm_model.py`, `xgb_model.py`, `catboost_model.py`, `elasticnet_model.py` | Per-algorithm wrappers implementing `BaseModel` |
| `ensemble_model.py` | Weighted ensemble; supports `from_results()` (performance-weighted), `elasticnet_plus_best`, and `from_cv_folds()` (uniform CV-bagging) |
| `losses.py` | Focal loss custom objective for LGBM/XGBoost |
| `registry.py` | `build_model(name, ...)` factory |

**`final_fit_strategy`** (in `TrainingConfig`) controls how the final model is fit after CV:
- `early_stop_on_test` (default) — uses the test set as the early-stopping validation set. Leaks test signal into iteration count.
- `iteration_capping` — aggregates CV fold `best_iteration` values (mean/median + headroom multiplier), then fits on all training data with a fixed tree count and no early stopping.
- `cv_bagging` — skips re-fitting entirely; uniformly averages the K fold models.

### Artifacts (`auto_ml/utils/io.py`)

Each `.joblib` artifact bundles three objects: `(preprocessor, model, ArtifactMetadata)`. `ArtifactMetadata` carries `feature_columns`, `selected_features`, `categorical_columns`, `id_columns`, `target_column`, `primary_metric`, and an `extra` dict with training details.

`auto-ml-set-best` can swap `best.joblib` to any sub-artifact without retraining by overwriting the symlink/copy.

### Scoring and Explain

- **`auto_ml/scoring/`** — `Scorer.from_artifact(path)` loads the artifact and calls `predict_proba`; outputs a Parquet with score and optional id columns.
- **`auto_ml/explain/`** — SHAP values via `auto-ml-explain`; output is wide-format Parquet: `<id_columns> + shap_<feature>... + base_value + score`.

### Tests (`tests/`)

All tests share one fixture defined in `conftest.py`: `tiny_binary_data` — 300 rows, 2 numeric + 1 categorical feature, synthetically separable (AUC > 0.70). Most tests instantiate a minimal `AutoMLConfig` directly rather than loading YAML.
