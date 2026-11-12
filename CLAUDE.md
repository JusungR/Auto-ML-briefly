# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

The package is **not installed**. Install only the dependencies and run the `.py` scripts from the repo root (YAML relative paths resolve against the cwd):
```bash
pip install -r requirements.txt
```

**Run all tests** (`pytest.ini` sets `pythonpath = .`):
```bash
pytest
```

**Run a single test file:**
```bash
pytest tests/test_focal_loss_math.py -v
```

**Training — preprocess → train → postprocess:**
```bash
python train/preprocess.py  --config configs/example.yaml
python train/train.py       --config configs/example.yaml
python train/postprocess.py --config configs/example.yaml
python train/run_all.py     --config configs/example.yaml   # all three
```

**Inference — preprocess → predict → postprocess:**
```bash
python inference/preprocess.py  --config configs/example.yaml
python inference/predict.py     --config configs/example.yaml
python inference/postprocess.py --config configs/example.yaml
python inference/run_all.py     --config configs/example.yaml   # all three
```
**End-to-end sample** (generates data, runs all six stage scripts, outputs to `artifacts/sample/`):
```bash
python examples/sample/run_sample.py
```
Stage scripts hand off via `<work_dir>/*.pkl.gz` (default `<artifact_dir parent>/work/{train,inference}`, override with `--work-dir`).

**Other entry points (from repo root):**
```bash
python -m auto_ml.explain.runner --config configs/explain.yaml
python -m auto_ml.cli.set_best --config configs/example.yaml --model lgbm
python -m auto_ml.cli.rollback --config configs/example.yaml
```

**Python API** (repo root on `sys.path`):
```python
from auto_ml import AutoMLPipeline, load_config
outputs = AutoMLPipeline(load_config("configs/example.yaml")).run()
```

## Architecture

### Stages (`auto_ml/stages/`)

`stages/train.py` (`preprocess` → `train` → `postprocess`) and `stages/inference.py` (`preprocess` → `predict` → `postprocess`) hold the step logic. `train/*.py` and `inference/*.py` are thin script wrappers that call one stage each and persist its output with `save_stage`/`load_stage` (`stages/common.py`, cloudpickle+gzip). `AutoMLPipeline.run()` and `run_scoring()` call the same functions in memory.

### Pipeline (`auto_ml/pipeline.py`)

`AutoMLPipeline.run()` runs the three training stages, which cover these steps:

1. **Load** — reads two separate Parquet files (`train_data_path`, `test_data_path`); validates schema and binary target.
2. **Preprocess** — `PreprocessingPipeline` applies: null imputation → outlier winsorizing → skew transform → scaling. Fitted on train, `transform()`-only on test.
3. **Feature selection** (optional) — `StabilitySelector` runs Stability Selection (Meinshausen & Bühlmann 2010) using repeated subsampling with L1 logistic or LightGBM base estimators. Controlled by `feature_selection.enabled`.
4. **Train** — `Trainer` (in `auto_ml/models/trainer.py`) loops over enabled models: Optuna hyperparameter tuning → stratified K-fold CV → final fit. Builds an optional ensemble. Selects `best` by `primary_metric` on the test set (or user override).
5. **Save + report + deploy** — saves every model to `artifact_dir/runs/<run_id>/models/<name>.joblib` (copied to `artifact_dir/models/`), writes test predictions and HTML/PDF reports (copied into the run), then deploys best to `artifact_dir/best.joblib` (see Runs below).

### Config (`auto_ml/config.py`)

`AutoMLConfig` is a top-level dataclass containing nested sub-configs for every pipeline stage. `load_config(path)` reads a YAML file and a companion features CSV.

- **Features CSV** (`features_csv`): columns `name`, `type` (`continuous`/`category`), `used` (`true`/`false`). Only `used == true` rows become features. Internally, `continuous` → `"numeric"` and `category` → `"categorical"`.
- **Categoricals**: `PreprocessingPipeline` normalizes categorical values to strings (`auto_ml/utils/categorical.py`: `1`/`1.0` → `"1"`) before imputation and records training levels for unseen-category logging. Encoding is per model: LGBM `category` dtype, XGB native `enable_categorical` with fixed training categories, CatBoost `cat_features`, ElasticNet one-hot. Artifacts trained before this lack `_normalize_categories` / XGB `_categories` and keep the legacy path (no normalization, XGB LabelEncoder).
- **Category grouping / one-hot splits**: `preprocessing.categorical_levels` (`{col: {keep: [...]}|{min_count: N}}`) runs `CategoryGrouper` right after imputation, mapping non-kept (and unseen) values to `categorical_other_label` (default `__other__`); kept sets are fitted on train and saved in the preprocessor (old artifacts without `_grouper` skip it). `models.<name>.categorical_encoding: onehot` makes tree wrappers set `max_cat_to_onehot = k + 1` (LGBM/XGB) or `one_hot_max_size = k` (CatBoost, ≤255) from the max training cardinality k (`BaseModel._onehot_params`; user `fixed_params` win), giving one-vs-rest splits while features stay variable-level.
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

### Runs, deployments, rollback (`auto_ml/versioning.py`)

Each training writes an immutable snapshot `artifact_dir/runs/<run_id>/` (`models/*.joblib`, `reports/`, `predictions/`, `run_info.json`) and replaces `artifact_dir/models/` wholesale with a copy of the latest run. Every production change goes through `versioning.deploy()` — load + test-score on `scoring.input_path` head(100), atomic `best.joblib` replace (`save_artifact` writes temp + `os.replace`), append to `deployments.jsonl` (`action`: train/promote/rollback/legacy, `previous` ref). `VersioningConfig`: `auto_deploy` (default true), `keep_last_runs` (default 5; current and previous production runs are never pruned). An old `artifact_dir` without `runs/` is migrated into `runs/legacy_*` on the next training.

```bash
python -m auto_ml.cli.set_best --config <yaml> --model xgb                 # from latest run (models/)
python -m auto_ml.cli.set_best --config <yaml> --run <run_id> --model lgbm # from an earlier run
python -m auto_ml.cli.set_best --config <yaml> --list                      # runs + deployment history
python -m auto_ml.cli.rollback --config <yaml>                             # toggle to previous production
```

### Threads (`auto_ml/utils/threads.py`)

`training.n_jobs` / `scoring.n_jobs` (default null = library default = node cores; pod CPU limits are not reflected; negatives follow sklearn; positives capped at available CPUs). Wrappers declare `THREAD_PARAM_NAMES` and inject via `BaseModel._thread_params()` unless the user set one in `fixed_params`; `build_model(..., n_jobs=)` sets it at training. `BaseModel.set_n_jobs()` changes threads on a fitted model (LGBM/XGB `set_params`, CatBoost per-call `thread_count`, ensemble propagates); `Scorer/Explainer.from_artifact(path, n_jobs=)` apply `scoring.n_jobs`. Predictions are unchanged.

### Scoring and Explain

- **`auto_ml/scoring/`** — `Scorer.from_artifact(path)` loads the artifact; `score()` = `prepare()` (preprocess) → `predict()` → `finalize()` (threshold + id columns); outputs a Parquet with score and optional id columns.
- **`auto_ml/explain/`** — SHAP values via `python -m auto_ml.explain.runner`; output is wide-format Parquet: `<id_columns> + shap_<feature>... + base_value + score`. Raw-margin SHAP is converted to probability space by `utils/shap_domain.py`; `EnsembleModel.shap_values_proba` converts per sub-model then weight-averages (ensemble predicts a weighted mean of probabilities), so `score` always equals the Scorer output.

### Tests (`tests/`)

All tests share one fixture defined in `conftest.py`: `tiny_binary_data` — 300 rows, 2 numeric + 1 categorical feature, synthetically separable (AUC > 0.70). Most tests instantiate a minimal `AutoMLConfig` directly rather than loading YAML.
