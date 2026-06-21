"""
training.tracking
=================
MLflow experiment tracking and reproducibility manifest (T2.7).

Every run is logged with:
  - Parameters: feature_set_id, label_col, n_splits, horizon, lgbm_params, seed
  - Metrics: rank_ic_mean, icir, sharpe, max_drawdown, net_ls_spread, ece
  - Artifacts: eval report JSON, backtest summary, robustness report, manifest
  - Tags: data_snapshot_id, feature_set_hash, label_set_hash, git_commit, env

Tracking backend: SQLite (MLflow 3.x file store is deprecated).
The SQLite file lives at ``<store_root>/mlflow/mlflow.db``.

Reproducibility manifest
------------------------
A JSON file capturing every input that would be needed to reproduce a run:
  data_snapshot_id  : hash of the OHLCV Parquet mtimes
  feature_set_id    : from FeatureRegistry (T1.5)
  label_col         : which label was used
  feature_cols_hash : SHA-256 of sorted feature column list
  lgbm_params       : full params dict
  seed              : random seed
  python_version    : sys.version
  package_versions  : key packages
  generated_at      : ISO timestamp
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import datetime as dt
from pathlib import Path

import mlflow
import pandas as pd

from quant_platform.core.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# MLflow setup
# ---------------------------------------------------------------------------

def get_or_create_experiment(
    store_root: Path | str,
    experiment_name: str = "quant_platform",
) -> str:
    """
    Set up MLflow with SQLite backend and return the experiment ID.
    Creates the experiment if it doesn't exist.
    """
    db_path = Path(store_root) / "mlflow" / "mlflow.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    tracking_uri = f"sqlite:///{db_path}"
    mlflow.set_tracking_uri(tracking_uri)

    existing = mlflow.get_experiment_by_name(experiment_name)
    if existing is not None:
        exp_id = existing.experiment_id
    else:
        exp_id = mlflow.create_experiment(experiment_name)
        logger.info("Created MLflow experiment '%s' (id=%s)", experiment_name, exp_id)

    return exp_id


# ---------------------------------------------------------------------------
# Reproducibility manifest
# ---------------------------------------------------------------------------

def make_manifest(
    store_root: Path | str,
    feature_set_id: str,
    feature_cols: list[str],
    label_col: str,
    lgbm_params: dict,
    seed: int,
    extra: dict | None = None,
) -> dict:
    """
    Build a reproducibility manifest dict.
    """
    # Hash of sorted feature column list
    feat_hash = hashlib.sha256(
        json.dumps(sorted(feature_cols)).encode()
    ).hexdigest()[:12]

    # Snapshot of key OHLCV Parquet mtimes
    from quant_platform.store.lake import ohlcv_dir
    ohlcv_d = ohlcv_dir(store_root)
    parquet_files = sorted(ohlcv_d.glob("*.parquet")) if ohlcv_d.exists() else []
    data_hash = hashlib.sha256(
        json.dumps([
            (str(p.name), int(p.stat().st_mtime * 1000))
            for p in parquet_files
        ]).encode()
    ).hexdigest()[:12]

    # Git commit (best-effort)
    try:
        import subprocess
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            cwd=str(Path(store_root).parent),
        ).decode().strip()
    except Exception:
        git_commit = "unavailable"

    # Package versions
    pkg_versions = {}
    for pkg in ("lightgbm", "mlflow", "shap", "sklearn", "pandas", "numpy", "duckdb"):
        try:
            m = __import__(pkg)
            pkg_versions[pkg] = getattr(m, "__version__", "?")
        except ImportError:
            pkg_versions[pkg] = "not installed"

    manifest = {
        "generated_at":       dt.datetime.now().isoformat(timespec="seconds"),
        "data_snapshot_id":   data_hash,
        "feature_set_id":     feature_set_id,
        "feature_cols_hash":  feat_hash,
        "label_col":          label_col,
        "lgbm_params":        lgbm_params,
        "seed":               seed,
        "git_commit":         git_commit,
        "python_version":     sys.version.split()[0],
        "platform":           platform.platform(),
        "package_versions":   pkg_versions,
    }
    if extra:
        manifest.update(extra)
    return manifest


# ---------------------------------------------------------------------------
# Run logger
# ---------------------------------------------------------------------------

class RunLogger:
    """
    Context manager that logs a P2 experiment run to MLflow.

    Usage
    -----
        with RunLogger(store_root, exp_id) as run:
            # ... fit, evaluate, backtest ...
            run.log_metrics(report.summary_dict())
            run.log_artifact_json(manifest, "manifest.json")
    """

    def __init__(
        self,
        store_root: Path | str,
        experiment_id: str,
        run_name: str | None = None,
        tags: dict | None = None,
    ) -> None:
        self.store_root    = Path(store_root)
        self.experiment_id = experiment_id
        self.run_name      = run_name
        self.tags          = tags or {}
        self._run          = None

    def __enter__(self) -> "RunLogger":
        self._run = mlflow.start_run(
            experiment_id=self.experiment_id,
            run_name=self.run_name,
            tags=self.tags,
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        mlflow.end_run(status="FAILED" if exc_type else "FINISHED")

    @property
    def run_id(self) -> str:
        return self._run.info.run_id if self._run else ""

    def log_params(self, params: dict) -> None:
        # MLflow param values must be str; truncate long values
        safe = {k: str(v)[:250] for k, v in params.items()}
        mlflow.log_params(safe)

    def log_metrics(self, metrics: dict) -> None:
        # Only log numeric values
        numeric = {
            k: float(v)
            for k, v in metrics.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
            and not (isinstance(v, float) and (v != v))  # skip NaN
        }
        if numeric:
            mlflow.log_metrics(numeric)

    def log_artifact_json(self, data: dict, filename: str) -> None:
        """Write *data* as JSON to a temp file and log it as an artifact."""
        tmp_path = self.store_root / "mlflow" / "tmp" / filename
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.write_text(json.dumps(data, indent=2, default=str))
        mlflow.log_artifact(str(tmp_path))

    def log_artifact_df(self, df: pd.DataFrame, filename: str) -> None:
        """Write DataFrame as CSV artifact."""
        tmp_path = self.store_root / "mlflow" / "tmp" / filename
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(tmp_path)
        mlflow.log_artifact(str(tmp_path))
