"""
training.reproducibility
========================
Reproducibility hardening (T3.9).

Three layers:
  1. **DVC data versioning**: tracks Parquet files in the lake so any past
     experiment can be reproduced with the exact same data.
  2. **Environment snapshot**: pins all relevant package versions and Python
     version to a JSON file at run time.
  3. **CI metric stability check**: given a previously logged manifest and
     a tolerance, re-runs a lightweight check to confirm the environment
     produces metrics within tolerance.

DVC notes
---------
DVC requires a git repository.  In environments without git, the DVC layer
logs a warning and proceeds (the rest of the platform continues to work).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from quant_platform.core.logging import get_logger

logger = get_logger(__name__)

_ENV_SNAPSHOT_FILE = "env_snapshot.json"


# ---------------------------------------------------------------------------
# Environment snapshot
# ---------------------------------------------------------------------------

def snapshot_environment(store_root: Path | str) -> dict:
    """
    Capture the current environment and write to
    ``<store_root>/env_snapshot.json``.

    Returns the snapshot dict.
    """
    import platform, datetime as dt

    snap: dict = {
        "timestamp":    dt.datetime.now().isoformat(timespec="seconds"),
        "python":       sys.version.split()[0],
        "platform":     platform.platform(),
        "packages":     {},
    }

    for pkg in (
        "lightgbm", "xgboost", "catboost", "optuna",
        "mlflow", "shap", "dvc", "sklearn", "numpy", "pandas",
        "duckdb", "pyarrow", "scipy",
    ):
        try:
            m   = __import__(pkg)
            ver = getattr(m, "__version__", "?")
        except ImportError:
            ver = "not_installed"
        snap["packages"][pkg] = ver

    path = Path(store_root) / _ENV_SNAPSHOT_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snap, indent=2))
    logger.info("Environment snapshot written → %s", path)
    return snap


def load_env_snapshot(store_root: Path | str) -> dict:
    """Load a previously saved environment snapshot."""
    path = Path(store_root) / _ENV_SNAPSHOT_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"Environment snapshot not found at {path}. "
            "Call snapshot_environment() first."
        )
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# DVC versioning
# ---------------------------------------------------------------------------

def dvc_add(file_path: str | Path) -> bool:
    """
    Track *file_path* with DVC (``dvc add``).

    Returns True on success, False if DVC is unavailable or git is absent.
    Never raises — DVC is a best-effort layer.
    """
    try:
        r = subprocess.run(
            ["dvc", "add", str(file_path)],
            capture_output=True, text=True, timeout=30,
        )
        if r.returncode == 0:
            logger.info("DVC tracked: %s", file_path)
            return True
        else:
            logger.warning("dvc add failed for %s: %s", file_path, r.stderr[:200])
            return False
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        logger.warning("DVC unavailable: %s", exc)
        return False


def dvc_add_lake(store_root: Path | str) -> dict[str, bool]:
    """
    Track all silver Parquet files in the lake with DVC.

    Returns a dict mapping file path → success.
    """
    from quant_platform.store.lake import ohlcv_dir, fundamentals_dir

    results: dict[str, bool] = {}
    for directory in (ohlcv_dir(store_root), fundamentals_dir(store_root)):
        if not directory.exists():
            continue
        for pq_file in directory.glob("*.parquet"):
            results[str(pq_file)] = dvc_add(pq_file)

    n_ok   = sum(results.values())
    n_fail = len(results) - n_ok
    logger.info("DVC lake snapshot: %d tracked, %d failed", n_ok, n_fail)
    return results


# ---------------------------------------------------------------------------
# CI metric stability check
# ---------------------------------------------------------------------------

def check_metric_stability(
    manifest_path: str | Path,
    current_metrics: dict,
    tolerance: float = 0.05,
) -> tuple[bool, list[str]]:
    """
    Compare *current_metrics* against the metrics stored in a manifest.

    Used in CI to confirm that a re-run produces metrics within *tolerance*
    (relative difference) of the original logged values.

    Parameters
    ----------
    manifest_path : path to a reproducibility manifest JSON (from T2.7)
    current_metrics : dict of metric_name → current value
    tolerance : max relative difference allowed (default 5%)

    Returns
    -------
    (passed: bool, failures: list[str])
        ``passed`` is True if all metrics are within tolerance.
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        return False, [f"Manifest not found: {manifest_path}"]

    manifest = json.loads(manifest_path.read_text())
    ref_metrics = manifest.get("metrics", {})

    if not ref_metrics:
        logger.warning("Manifest has no 'metrics' key — skipping CI check")
        return True, []

    failures: list[str] = []
    for key, ref_val in ref_metrics.items():
        if key not in current_metrics:
            continue
        cur_val = current_metrics[key]
        if ref_val == 0:
            rel_diff = abs(cur_val)
        else:
            rel_diff = abs(cur_val - ref_val) / abs(ref_val)
        if rel_diff > tolerance:
            failures.append(
                f"{key}: expected {ref_val:.6f}, got {cur_val:.6f}, "
                f"rel_diff={rel_diff:.2%} > tolerance {tolerance:.0%}"
            )

    passed = len(failures) == 0
    if passed:
        logger.info("CI metric stability check PASSED (%d metrics)", len(ref_metrics))
    else:
        logger.warning("CI metric stability check FAILED:\n%s", "\n".join(failures))

    return passed, failures
