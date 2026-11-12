"""best 교체 CLI 진입점.

학습 결과 후보 중 하나를 운영 모델(``artifact_dir/best.joblib``) 로 올린다. 재학습 없이
운영 모델을 바꿀 수 있어, 자동 선정된 모형이 운영 중 부적합할 때 빠르게 대응할 수 있다.
교체는 :func:`auto_ml.versioning.deploy` 를 거치므로 시험 스코어링에 실패하면 운영 모델이
바뀌지 않고, 성공하면 배포 이력(``deployments.jsonl``) 에 기록된다.

사용 예::

    # 최신 실행의 후보 중 xgb 로 교체
    python -m auto_ml.cli.set_best --config configs/example.yaml --model xgb
    # 특정 이전 실행의 모델로 교체
    python -m auto_ml.cli.set_best --config configs/example.yaml --run 20261110_093000 --model lgbm
    # 실행 목록과 배포 이력 보기
    python -m auto_ml.cli.set_best --config configs/example.yaml --list
"""
from __future__ import annotations

import argparse
from pathlib import Path

from auto_ml import versioning
from auto_ml.config import AutoMLConfig, load_config
from auto_ml.utils.io import load_artifact
from auto_ml.utils.logger import get_logger, setup_logging

logger = get_logger("set_best")
SUB_ARTIFACT_DIRNAME = versioning.LATEST_DIRNAME


def _candidates(sub_dir: Path) -> list[str]:
    return sorted(p.stem for p in sub_dir.glob("*.joblib")) if sub_dir.exists() else []


def promote(
    config: AutoMLConfig,
    model_name: str,
    run_id: str | None = None,
    allow_missing_features: bool = False,
) -> Path:
    """후보 모델을 운영 모델(best.joblib) 로 올린다.

    Args:
        config: AutoMLConfig.
        model_name: 올릴 모형 이름 (후보 파일 stem).
        run_id: None 이면 최신 실행 후보(``artifact_dir/models/``), 지정하면
            ``artifact_dir/runs/<run_id>/models/`` 에서 찾는다.
        allow_missing_features: 모델이 쓰는 변수가 스코어링 입력에 없어도 배포한다.

    Returns:
        운영 모델 경로 (best.joblib).

    Raises:
        FileNotFoundError: 실행 또는 후보 모델이 없을 때.
        auto_ml.versioning.DeploymentError: 시험 스코어링 실패 등으로 올릴 수 없을 때.
    """
    artifact_dir = Path(config.artifact_dir)
    if run_id is not None:
        run_dir = versioning.runs_root(artifact_dir) / run_id
        if not run_dir.exists():
            available = [d.name for d in versioning.list_runs(artifact_dir)]
            raise FileNotFoundError(
                f"Run not found: {run_dir}. Available runs: {available or '(none)'}"
            )
        sub_dir = run_dir / SUB_ARTIFACT_DIRNAME
    else:
        sub_dir = artifact_dir / SUB_ARTIFACT_DIRNAME

    sub_path = sub_dir / f"{model_name}.joblib"
    if not sub_path.exists():
        raise FileNotFoundError(
            f"Sub-artifact not found: {sub_path}. "
            f"Available sub-artifacts: {_candidates(sub_dir) or '(none — train first)'}"
        )

    source = sub_path
    if run_id is None:
        # models/ 는 최신 실행의 사본이다. 이력에는 원본(실행 스냅샷) 경로를 남겨
        # 보존 정책이 그 실행을 운영 실행으로 인식·보호하게 한다.
        try:
            rid = load_artifact(sub_path)["metadata"].extra.get("run_id")
        except Exception:  # noqa: BLE001 — deploy 가 다시 열어 명확한 에러를 낸다
            rid = None
        snap = versioning.runs_root(artifact_dir) / str(rid) / SUB_ARTIFACT_DIRNAME / sub_path.name
        if rid and snap.exists():
            source = snap

    best_path = versioning.deploy(
        config, source, action="promote", best_selection="promoted",
        allow_missing_features=allow_missing_features,
    )
    versioning.prune_runs(artifact_dir, config.versioning.keep_last_runs)
    return best_path


def describe(config: AutoMLConfig, n_history: int = 10) -> str:
    """실행 목록과 최근 배포 이력을 사람이 읽기 좋은 문자열로 만든다 (``--list``)."""
    artifact_dir = Path(config.artifact_dir)
    current = versioning.current_deployment(artifact_dir) or {}
    prev = current.get("previous") or {}
    lines = [f"Runs in {versioning.runs_root(artifact_dir)} (oldest first):"]
    runs = versioning.list_runs(artifact_dir)
    if not runs:
        lines.append("  (none)")
    for d in runs:
        info = versioning.read_run_info(d)
        flags = []
        if d.name == current.get("run_id"):
            flags.append(f"PRODUCTION: {current.get('model')}")
        if d.name == prev.get("run_id"):
            flags.append(f"previous: {prev.get('model')}")
        best = info.get("best_model")
        models_info = info.get("models")
        # legacy 실행이나 손상된 run_info 에서도 목록 출력이 실패하지 않게 방어한다
        model_info = models_info.get(best) if isinstance(models_info, dict) else None
        metric = ((model_info or {}).get("test_metrics") or {}).get(info.get("primary_metric", ""))
        best_txt = f"best={best} {info.get('primary_metric')}={metric:.4f}" if metric is not None \
            else ("legacy import" if info.get("legacy") else "")
        lines.append(
            f"  {d.name:<24} {str(info.get('created_at', ''))[:19]:<20} "
            f"models={_candidates(d / SUB_ARTIFACT_DIRNAME)} {best_txt}"
            + (f"  [{'; '.join(flags)}]" if flags else "")
        )
    lines.append(f"Recent deployments ({versioning.DEPLOY_LOG}, newest last):")
    existing = {d.name for d in runs}
    history = versioning.read_deployments(artifact_dir)[-n_history:]
    if not history:
        lines.append("  (none)")
    for e in history:
        gone = "" if e.get("run_id") in existing else "  (run deleted)"
        lines.append(
            f"  {str(e.get('at', ''))[:19]}  {e.get('action'):<8} "
            f"run={e.get('run_id')} model={e.get('model')}{gone}"
        )
    return "\n".join(lines)


def cli_set_best() -> None:
    """``python -m auto_ml.cli.set_best`` 진입점."""
    parser = argparse.ArgumentParser(
        description="Deploy a trained candidate model to artifact_dir/best.joblib "
                    "without retraining, or list runs and deployments."
    )
    parser.add_argument("--config", required=True, help="설정 YAML 경로")
    parser.add_argument(
        "--model",
        help="올릴 모형 이름 (후보 파일 stem). 예: lgbm | xgb | catboost | ensemble",
    )
    parser.add_argument(
        "--run", default=None,
        help="이전 실행 id (runs/<id>). 생략하면 최신 실행 후보(models/) 에서 찾는다.",
    )
    parser.add_argument("--list", action="store_true", help="실행 목록과 배포 이력만 출력")
    parser.add_argument(
        "--allow-missing-features", action="store_true",
        help="모델이 쓰는 변수가 scoring.input_path 에 없어도 배포한다 "
             "(그 변수는 학습 기본값으로 채워져 예측된다)",
    )
    args = parser.parse_args()
    if not args.list and not args.model:
        parser.error("--model is required unless --list is given")

    config = load_config(args.config)
    if args.list:
        print(describe(config))
        return
    setup_logging(
        log_dir=config.logging.log_dir,
        level=config.logging.level,
        to_stdout=config.logging.to_stdout,
        to_file=config.logging.to_file,
        stage="set_best",
    )
    promote(
        config, args.model, run_id=args.run,
        allow_missing_features=args.allow_missing_features,
    )


if __name__ == "__main__":
    cli_set_best()
