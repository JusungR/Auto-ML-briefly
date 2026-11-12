"""실행 이력(runs) 과 운영 배포(best.joblib) 관리.

폴더 구조 (``artifact_dir`` 기준)::

    artifact_dir/
    ├── best.joblib            운영 모델. 추론·SHAP 이 읽는 유일한 파일
    ├── deployments.jsonl      배포 이력 (한 줄 = 운영 모델 교체 1회, 추가만 함)
    ├── models/                최신 실행 후보 사본 (set_best 기본 대상)
    └── runs/<run_id>/         실행 스냅샷 — 쓴 뒤 수정하지 않는다
        ├── models/*.joblib
        ├── reports/ , predictions/
        └── run_info.json

운영 모델 교체는 모두 :func:`deploy` 를 거친다 (학습 후 자동 배포, set_best, rollback).
:func:`deploy` 는 교체 전에 대상 모델을 실제로 열어 시험 스코어링을 하고, 실패하면
``best.joblib`` 을 건드리지 않는다.
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import socket
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from auto_ml.config import AutoMLConfig
from auto_ml.utils.io import ARTIFACT_FILENAME, load_artifact, save_artifact
from auto_ml.utils.logger import get_logger

logger = get_logger("versioning")

RUNS_DIRNAME = "runs"
LATEST_DIRNAME = "models"
DEPLOY_LOG = "deployments.jsonl"
RUN_INFO = "run_info.json"
LOCK_FILE = ".lock"
SMOKE_ROWS = 100
# 내용을 읽을 수 없는 잠금 파일(만든 직후 비정상 종료 등) 을 버려진 것으로 볼 최소 경과 시간
UNREADABLE_LOCK_GRACE_SEC = 60


class DeploymentError(RuntimeError):
    """배포 대상 모델을 운영에 올릴 수 없을 때 (로드 실패, 시험 스코어링 실패 등)."""


class ArtifactLockError(RuntimeError):
    """다른 학습·배포 작업이 같은 artifact_dir 을 쓰고 있을 때."""


# ----------------------------------------------------------------------
# 쓰기 작업 잠금
# ----------------------------------------------------------------------
_held_locks: dict[Path, int] = {}  # 이 프로세스가 잡은 잠금 → 중첩 횟수 (재진입 허용)


def _read_lock(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # 다른 계정의 살아 있는 프로세스
        return True
    return True


def _is_stale(path: Path, holder: dict[str, Any]) -> bool:
    """잠금을 잡은 프로세스가 이미 없으면 True. 판단할 수 없으면 False (보수적으로)."""
    if not holder:
        try:
            return time.time() - path.stat().st_mtime > UNREADABLE_LOCK_GRACE_SEC
        except OSError:
            return True  # 그새 사라짐
    if os.name == "nt":
        # Windows 의 os.kill(pid, 0) 은 생존 확인이 아니라 프로세스를 종료시킨다 — 판단하지 않는다.
        return False
    if holder.get("host") != socket.gethostname():
        return False  # 다른 호스트(공유 디스크) 의 pid 는 확인할 수 없다
    try:
        return not _pid_alive(int(holder.get("pid")))
    except (TypeError, ValueError):
        return False


@contextmanager
def artifact_lock(artifact_dir: str | Path, purpose: str) -> Iterator[None]:
    """``artifact_dir`` 에 쓰는 작업(학습·배포) 을 한 번에 하나만 허용한다.

    ``artifact_dir/.lock`` 을 원자적으로(O_EXCL) 만들고 작업이 끝나면 지운다. 이미 있으면
    기다리지 않고 :class:`ArtifactLockError` 로 즉시 실패한다 — 긴 학습을 마친 뒤에야
    충돌을 알게 되는 일이 없도록 학습 시작 시점에 잡는다. 같은 호스트에서 잠금을 잡은
    프로세스가 이미 죽었으면(비정상 종료) WARNING 후 넘겨받는다. 같은 프로세스 안의
    중첩 호출은 허용한다. 추론·SHAP 은 잠그지 않는다 (best.joblib 은 원자적으로 교체됨).
    """
    path = (Path(artifact_dir) / LOCK_FILE).resolve()
    if path in _held_locks:
        _held_locks[path] += 1
        try:
            yield
        finally:
            _held_locks[path] -= 1
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    me = {
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "started_at": _now().isoformat(),
        "purpose": purpose,
    }
    for attempt in range(2):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            holder = _read_lock(path)
            if attempt == 0 and _is_stale(path, holder):
                logger.warning(
                    "Removing stale lock %s left by a process that is no longer running: %s",
                    path, holder or "(unreadable)",
                )
                path.unlink(missing_ok=True)
                continue
            raise ArtifactLockError(
                f"Another job is using {path.parent}: {holder or '(lock file unreadable)'}. "
                "Wait for it to finish. If no such job is running (e.g. it was killed on "
                f"another host), delete {path} and retry."
            ) from None
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(me, f)
        break

    _held_locks[path] = 1
    try:
        yield
    finally:
        del _held_locks[path]
        if _read_lock(path).get("pid") == me["pid"]:  # 넘겨받힌 경우 남의 잠금은 지우지 않는다
            path.unlink(missing_ok=True)


# ----------------------------------------------------------------------
# 경로 / 실행 목록
# ----------------------------------------------------------------------
def _now() -> datetime:
    return datetime.now(timezone.utc)


def runs_root(artifact_dir: str | Path) -> Path:
    return Path(artifact_dir) / RUNS_DIRNAME


def _unique_dir(parent: Path, name: str) -> Path:
    """``parent/name`` 이 이미 있으면 ``name-1``, ``name-2`` … 중 비어 있는 경로를 만든다."""
    path, i = parent / name, 1
    while True:
        try:
            path.mkdir(parents=True, exist_ok=False)
            return path
        except FileExistsError:
            path = parent / f"{name}-{i}"
            i += 1


def new_run_dir(artifact_dir: str | Path) -> tuple[str, Path]:
    """새 실행 폴더 ``runs/<YYYYMMDD_HHMMSS>`` 를 만들고 ``(run_id, 경로)`` 를 반환한다."""
    path = _unique_dir(runs_root(artifact_dir), datetime.now().strftime("%Y%m%d_%H%M%S"))
    return path.name, path


def read_run_info(run_dir: Path) -> dict[str, Any]:
    try:
        return json.loads((run_dir / RUN_INFO).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def write_run_info(run_dir: Path, info: dict[str, Any]) -> Path:
    path = run_dir / RUN_INFO
    path.write_text(
        json.dumps(info, ensure_ascii=False, indent=2, default=str), encoding="utf-8",
    )
    return path


def _created_at(run_dir: Path) -> str:
    created = read_run_info(run_dir).get("created_at")
    if created:
        return str(created)
    return datetime.fromtimestamp(run_dir.stat().st_mtime, timezone.utc).isoformat()


def list_runs(artifact_dir: str | Path) -> list[Path]:
    """실행 폴더를 생성 시각 순(오래된 것 먼저) 으로 반환한다.

    폴더 이름이 아니라 run_info 의 ``created_at`` 으로 정렬한다 — 기존 폴더 이전으로
    만든 ``legacy_*`` 실행은 원래 학습 시각을 가지므로 새 실행보다 앞에 온다.
    """
    root = runs_root(artifact_dir)
    if not root.exists():
        return []
    runs = [d for d in root.iterdir() if d.is_dir() and not d.name.startswith(".")]
    return sorted(runs, key=lambda d: (_created_at(d), d.name))


def latest_run_id(artifact_dir: str | Path) -> str | None:
    runs = list_runs(artifact_dir)
    return runs[-1].name if runs else None


def missing_production_message(artifact_dir: str | Path) -> str:
    """best.joblib 이 없을 때의 안내 — 학습 실행은 있는데 배포가 안 된 경우를 구분한다."""
    path = Path(artifact_dir) / ARTIFACT_FILENAME
    run_id = latest_run_id(artifact_dir)
    if run_id is None:
        return f"Artifact not found at {path}. Run training first (train/run_all.py)."
    info = read_run_info(runs_root(artifact_dir) / run_id)
    model = info.get("best_model") or "<name>"
    return (
        f"Artifact not found at {path}. Training run {run_id} exists but no model is "
        f"deployed (versioning.auto_deploy may be false). Deploy one with: "
        f"python -m auto_ml.cli.set_best --config <yaml> --run {run_id} --model {model}"
    )


def environment_info() -> dict[str, Any]:
    """재현·호환성 판단용 버전 정보. git 은 없거나 실패해도 무시한다 (폐쇄망 대비)."""
    versions: dict[str, Any] = {"python": platform.python_version()}
    for mod in ("numpy", "pandas", "sklearn", "lightgbm", "xgboost", "catboost", "cloudpickle"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001 — 미설치 라이브러리는 기록만 생략
            versions[mod] = None
    commit = None
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            commit = out.stdout.strip() or None
    except Exception:  # noqa: BLE001
        commit = None
    return {"library_versions": versions, "git_commit": commit}


# ----------------------------------------------------------------------
# 배포 이력
# ----------------------------------------------------------------------
def deploy_log_path(artifact_dir: str | Path) -> Path:
    return Path(artifact_dir) / DEPLOY_LOG


def read_deployments(artifact_dir: str | Path) -> list[dict[str, Any]]:
    path = deploy_log_path(artifact_dir)
    if not path.exists():
        return []
    entries: list[dict[str, Any]] = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            entries.append(json.loads(line))
        except ValueError:
            logger.warning("Skipping malformed line %d in %s", i, path)
    return entries


def current_deployment(artifact_dir: str | Path) -> dict[str, Any] | None:
    entries = read_deployments(artifact_dir)
    return entries[-1] if entries else None


def append_deployment(artifact_dir: str | Path, entry: dict[str, Any]) -> None:
    path = deploy_log_path(artifact_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _ref(entry: dict[str, Any] | None) -> dict[str, Any] | None:
    if not entry:
        return None
    return {k: entry.get(k) for k in ("run_id", "model", "source")}


# ----------------------------------------------------------------------
# 호환성 점검 / 배포
# ----------------------------------------------------------------------
def _read_sample(path: Path, n: int):
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    batch = next(pf.iter_batches(batch_size=n), None)
    return pf.schema_arrow.names, (batch.to_pandas() if batch is not None else None)


def check_compatibility(
    bundle: dict[str, Any],
    cfg: AutoMLConfig,
    allow_missing_features: bool = False,
) -> dict[str, Any]:
    """운영에 올리기 전 시험: 현재 스코어링 입력으로 실제 점수를 계산해 본다.

    - **모델이 쓰는 변수**(selected_features) 가 입력에 없으면 :class:`DeploymentError`.
      스코어링은 그 변수를 학습 기본값(평균·최빈값) 으로 모든 행에 채워 예측하므로,
      예측 품질이 떨어져도 WARNING 로그로만 드러난다. ``allow_missing_features=True``
      이면 WARNING 후 진행한다.
    - 변수 선택에서 빠진 변수만 없으면 WARNING (예측에는 영향 없음).
    - 시험 스코어링이 실패하거나 점수가 [0, 1] 밖이면 :class:`DeploymentError`.
    - 스코어링 입력 파일이 없으면 로드 확인만 하고 WARNING.
    """
    from auto_ml.scoring.scorer import Scorer  # 순환 import 회피

    meta = bundle["metadata"]
    result: dict[str, Any] = {"missing_features": [], "scored_rows": 0}
    input_path = Path(cfg.scoring.input_path)
    if not input_path.exists():
        logger.warning(
            "Compatibility check: scoring input %s not found — loaded the model but "
            "skipped test scoring", input_path,
        )
        return result

    columns, sample = _read_sample(input_path, SMOKE_ROWS)
    missing = [c for c in meta.feature_columns if c not in columns]
    result["missing_features"] = missing
    used = set(getattr(meta, "selected_features", None) or meta.feature_columns)
    missing_used = [c for c in missing if c in used]
    if missing_used and not allow_missing_features:
        raise DeploymentError(
            f"Model '{meta.model_name}' uses {len(missing_used)} feature(s) missing from "
            f"{input_path}: {missing_used}. Scoring would fill them with training defaults "
            "for every row. Fix the scoring input, or pass --allow-missing-features to "
            "deploy anyway."
        )
    if missing:
        logger.warning(
            "Compatibility check: %d feature(s) of model '%s' are missing from %s and "
            "would be filled with training defaults: %s (used by the model: %s)",
            len(missing), meta.model_name, input_path, missing, missing_used or "none",
        )
    if sample is None or sample.empty:
        return result
    try:
        scores = Scorer(bundle["preprocessor"], bundle["model"], meta).score(sample)["score"]
    except Exception as exc:
        raise DeploymentError(
            f"Test scoring of model '{meta.model_name}' on {input_path} failed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    values = scores.to_numpy(dtype=float)
    if not np.all(np.isfinite(values)) or values.min() < 0 or values.max() > 1:
        raise DeploymentError(
            f"Test scoring of model '{meta.model_name}' produced invalid scores "
            f"(min={np.nanmin(values)}, max={np.nanmax(values)})"
        )
    result["scored_rows"] = len(values)
    return result


def _relative(path: Path, artifact_dir: Path) -> str:
    try:
        return str(path.resolve().relative_to(artifact_dir.resolve()))
    except ValueError:
        return str(path)


def _run_id_from_path(path: Path, artifact_dir: Path) -> str | None:
    """``artifact_dir/runs/<run_id>/...`` 경로면 run_id, 아니면 None."""
    try:
        rel = path.resolve().relative_to(runs_root(artifact_dir).resolve())
    except ValueError:
        return None
    return rel.parts[0] if rel.parts else None


def deploy(
    cfg: AutoMLConfig,
    source: str | Path,
    action: str,
    best_selection: str,
    allow_missing_features: bool = False,
) -> Path:
    """``source`` 모델을 운영 모델(best.joblib) 로 올리고 배포 이력에 기록한다.

    Args:
        cfg: 설정 (``artifact_dir``, ``scoring.input_path`` 사용).
        source: 올릴 모델 파일 (보통 ``runs/<id>/models/<name>.joblib``).
        action: ``train`` / ``promote`` / ``rollback`` / ``legacy``.
        best_selection: 메타데이터 ``extra.best_selection`` 값.
        allow_missing_features: True 면 모델이 쓰는 변수가 스코어링 입력에 없어도
            WARNING 후 배포한다 (:func:`check_compatibility`).

    Raises:
        DeploymentError: 모델을 열 수 없거나, 모델이 쓰는 변수가 입력에 없거나,
            시험 스코어링이 실패한 경우. 이때 ``best.joblib`` 과 이력은 바뀌지 않는다.
    """
    artifact_dir = Path(cfg.artifact_dir)
    source = Path(source)
    try:
        bundle = load_artifact(source)
    except Exception as exc:
        raise DeploymentError(
            f"Cannot load {source}: {type(exc).__name__}: {exc}. "
            "It may have been saved by an incompatible code or library version."
        ) from exc
    check_compatibility(bundle, cfg, allow_missing_features=allow_missing_features)

    meta = bundle["metadata"]
    previous = current_deployment(artifact_dir)
    extra = dict(meta.extra)
    extra["best_selection"] = best_selection
    extra["deployed_at"] = _now().isoformat()
    if action in ("promote", "rollback"):
        extra["promoted_at"] = extra["deployed_at"]
        extra["promoted_from"] = (previous or {}).get("model")
    # 파일이 실제로 들어 있는 실행(runs/<id>/) 을 우선한다 — 메타데이터의 run_id 는
    # 복사된 파일(legacy 이전 등) 에서 옛 값일 수 있고, 과거 모델에는 아예 없다.
    run_id = _run_id_from_path(source, artifact_dir) or extra.get("run_id")
    extra["run_id"] = run_id
    meta = replace(meta, extra=extra)

    best_path = artifact_dir / ARTIFACT_FILENAME
    save_artifact(best_path, bundle["preprocessor"], bundle["model"], meta)
    entry = {
        "at": extra["deployed_at"],
        "action": action,
        "run_id": run_id,
        "model": meta.model_name,
        "source": _relative(source, artifact_dir),
        "primary_metric": meta.primary_metric,
        "metric": meta.metric_value,
        "previous": _ref(previous),
    }
    append_deployment(artifact_dir, entry)
    logger.info(
        "Deployed %s (run=%s, %s=%.4f, action=%s) -> %s",
        meta.model_name, run_id, meta.primary_metric, meta.metric_value, action, best_path,
    )
    return best_path


# ----------------------------------------------------------------------
# 기존 폴더 이전 / 최신 사본 / 보존 정책
# ----------------------------------------------------------------------
def _legacy_trained_at(path: Path) -> datetime:
    """과거 모델 파일의 학습 시각 — 메타데이터 ``trained_at`` 우선, 없으면 파일 수정 시각.

    파일 수정 시각은 복사·git checkout 으로 바뀌므로 보조 수단으로만 쓴다.
    """
    try:
        trained = load_artifact(path)["metadata"].trained_at
        dt = datetime.fromisoformat(str(trained))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)  # 과거 값은 UTC naive
    except Exception:  # noqa: BLE001
        return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)


def migrate_legacy(cfg: AutoMLConfig) -> str | None:
    """``runs/`` 가 없는 기존 artifact_dir 을 ``runs/legacy_<시각>/`` 실행으로 담는다.

    기존 ``models/*.joblib`` 과 ``best.joblib`` 을 **복사**한다 (best.joblib 은 추론이
    계속 읽어야 하므로 그대로 둔다). 배포 이력이 없으면 현재 운영 모델을 가리키는 첫 줄을
    남겨, 다음 학습 후 ``rollback`` 이 이 모델로 돌아올 수 있게 한다.

    Returns:
        만든 실행 id, 이전할 것이 없으면 None.
    """
    artifact_dir = Path(cfg.artifact_dir)
    if runs_root(artifact_dir).exists():
        return None
    latest = artifact_dir / LATEST_DIRNAME
    subs = sorted(latest.glob("*.joblib")) if latest.exists() else []
    best = artifact_dir / ARTIFACT_FILENAME
    if not subs and not best.exists():
        return None

    trained = _legacy_trained_at(best if best.exists() else subs[-1])
    run_dir = _unique_dir(
        runs_root(artifact_dir), "legacy_" + trained.astimezone().strftime("%Y%m%d_%H%M%S"),
    )
    (run_dir / LATEST_DIRNAME).mkdir()
    for p in subs:
        shutil.copy2(p, run_dir / LATEST_DIRNAME / p.name)
    if best.exists():
        shutil.copy2(best, run_dir / ARTIFACT_FILENAME)
    # 기존 리포트 / test 예측도 함께 담는다 — 이후 학습이 이 위치를 덮어써도 남도록.
    reports = Path(cfg.reporting.output_dir)
    if reports.exists():
        shutil.copytree(reports, run_dir / "reports",
                        ignore=shutil.ignore_patterns("_previous"))
    preds = artifact_dir.parent / "predictions"
    if preds.exists():
        shutil.copytree(preds, run_dir / "predictions")
    write_run_info(run_dir, {
        "run_id": run_dir.name,
        "created_at": trained.isoformat(),
        "legacy": True,
        "note": "Migrated from the artifact_dir layout before run versioning.",
        # 새 실행과 같은 형태(이름 → 정보). 과거 파일에는 지표 기록이 없어 비어 있다.
        "models": {p.stem: {} for p in subs},
    })

    if best.exists() and not read_deployments(artifact_dir):
        try:
            meta = load_artifact(best)["metadata"]
            model, metric, pm = meta.model_name, meta.metric_value, meta.primary_metric
        except Exception:  # noqa: BLE001 — 열리지 않아도 이력은 남긴다
            model, metric, pm = None, None, None
        append_deployment(artifact_dir, {
            "at": _now().isoformat(),
            "action": "legacy",
            "run_id": run_dir.name,
            "model": model,
            "source": _relative(run_dir / ARTIFACT_FILENAME, artifact_dir),
            "primary_metric": pm,
            "metric": metric,
            "previous": None,
        })
    logger.warning(
        "Migrated existing artifacts in %s into run %s (models: %s)",
        artifact_dir, run_dir.name, [p.stem for p in subs] or "(best.joblib only)",
    )
    return run_dir.name


def refresh_latest_models(artifact_dir: str | Path, run_dir: Path) -> Path:
    """``models/`` 를 ``run_dir/models`` 의 사본으로 통째로 교체한다.

    새 사본을 임시 폴더에 다 만든 뒤 rename 두 번으로 바꿔치기한다 — 두 rename 사이의
    아주 짧은 순간에만 ``models/`` 가 없다 (추론은 best.joblib 만 읽으므로 무관).
    """
    artifact_dir = Path(artifact_dir)
    latest = artifact_dir / LATEST_DIRNAME
    tmp = artifact_dir / f".{LATEST_DIRNAME}.new-{os.getpid()}"
    old = artifact_dir / f".{LATEST_DIRNAME}.old-{os.getpid()}"
    for p in (tmp, old):
        if p.exists():
            shutil.rmtree(p)
    shutil.copytree(run_dir / LATEST_DIRNAME, tmp)
    if latest.exists():
        os.replace(latest, old)
    os.replace(tmp, latest)
    if old.exists():
        shutil.rmtree(old)
    return latest


def protected_run_ids(artifact_dir: str | Path) -> set[str]:
    """보존 정책과 무관하게 지우면 안 되는 실행 — 현재 운영 실행과 직전 운영 실행."""
    current = current_deployment(artifact_dir)
    ids: set[str] = set()
    if current:
        for ref in (current, current.get("previous") or {}):
            if ref.get("run_id"):
                ids.add(str(ref["run_id"]))
    return ids


def prune_runs(artifact_dir: str | Path, keep: int) -> list[str]:
    """최근 ``keep`` 개 실행 + 보호 대상만 남기고 나머지 실행 폴더를 삭제한다."""
    runs = list_runs(artifact_dir)
    keep_ids = {d.name for d in runs[-keep:]} | protected_run_ids(artifact_dir)
    removed: list[str] = []
    for d in runs:
        if d.name not in keep_ids:
            shutil.rmtree(d)
            removed.append(d.name)
    if removed:
        logger.info(
            "Removed %d old run(s) beyond keep_last_runs=%d: %s", len(removed), keep, removed,
        )
    return removed
