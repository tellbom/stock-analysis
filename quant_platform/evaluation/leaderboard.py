"""
evaluation.leaderboard
======================
Automated evaluation report + leaderboard (T3.3).

Every run that calls ``record_run()`` appends its metrics to a Parquet
leaderboard file, keyed by (run_id, model_name, feature_set_id, label_col,
fold_seed).  The leaderboard is sortable and comparable only when fold seeds
match — the ``fold_seed`` column is the guarantee that two entries were
evaluated on identical splits.

Design rules
------------
- Identical folds enforced: fold_seed is derived from (feature_set_id,
  label_col, n_splits, horizon, embargo, seed) so two runs with the same
  config share the same split.
- Metrics stored: rank_ic_mean, icir, quantile_spread, precision_at_10,
  ece, sharpe, net_ls_spread.
- The leaderboard is append-only; no rows are ever deleted.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from quant_platform.evaluation.metrics import EvalReport
from quant_platform.evaluation.backtest import BacktestResult
from quant_platform.core.logging import get_logger

logger = get_logger(__name__)

_LB_PATH = "leaderboard.parquet"


def compute_fold_seed(
    feature_set_id: str,
    label_col:      str,
    n_splits:       int,
    horizon:        int,
    embargo:        int,
    seed:           int,
) -> str:
    """
    8-char hex ID that uniquely identifies a CV configuration.
    Two runs with the same fold_seed were evaluated on identical splits.
    """
    payload = json.dumps({
        "feature_set_id": feature_set_id,
        "label_col":      label_col,
        "n_splits":       n_splits,
        "horizon":        horizon,
        "embargo":        embargo,
        "seed":           seed,
    }, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:8]


def record_run(
    store_root:     Path | str,
    run_id:         str,
    model_name:     str,
    feature_set_id: str,
    label_col:      str,
    fold_seed:      str,
    eval_report:    EvalReport,
    backtest:       BacktestResult | None = None,
    extra:          dict | None = None,
) -> pd.DataFrame:
    """
    Append one run's metrics to the leaderboard Parquet.

    Returns the updated leaderboard DataFrame.
    """
    store_root = Path(store_root)
    lb_path    = store_root / _LB_PATH

    row = {
        "run_id":          run_id,
        "model_name":      model_name,
        "feature_set_id":  feature_set_id,
        "label_col":       label_col,
        "fold_seed":       fold_seed,
        "rank_ic_mean":    eval_report.rank_ic_mean,
        "rank_ic_std":     eval_report.rank_ic_std,
        "icir":            eval_report.icir,
        "quantile_spread": eval_report.quantile_spread,
        "precision_at_10": eval_report.precision_at_k.get(10, float("nan")),
        "ece":             eval_report.ece,
        "n_dates":         eval_report.n_dates,
        "n_predictions":   eval_report.n_predictions,
        "sharpe":          backtest.sharpe            if backtest else float("nan"),
        "net_ls_spread":   backtest.net_long_minus_short if backtest else float("nan"),
        "max_drawdown":    backtest.max_drawdown      if backtest else float("nan"),
    }
    if extra:
        row.update(extra)

    existing = _load_leaderboard(lb_path)
    updated  = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    store_root.mkdir(parents=True, exist_ok=True)
    updated.to_parquet(lb_path, index=False)
    logger.info(
        "Leaderboard updated: %d entries, best ICIR=%.4f (%s)",
        len(updated),
        updated["icir"].max(),
        updated.loc[updated["icir"].idxmax(), "model_name"],
    )
    return updated


def load_leaderboard(store_root: Path | str) -> pd.DataFrame:
    """Load and return the leaderboard, sorted by ICIR descending."""
    path = Path(store_root) / _LB_PATH
    df   = _load_leaderboard(path)
    return df.sort_values("icir", ascending=False).reset_index(drop=True)


def _load_leaderboard(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception as exc:
        logger.error("Cannot read leaderboard at %s: %s", path, exc)
        return pd.DataFrame()


def print_leaderboard(store_root: Path | str) -> None:
    """Print a human-readable leaderboard summary."""
    df = load_leaderboard(store_root)
    if df.empty:
        print("Leaderboard is empty.")
        return
    cols = ["model_name", "label_col", "rank_ic_mean", "icir",
            "quantile_spread", "sharpe", "fold_seed", "run_id"]
    display = df[[c for c in cols if c in df.columns]].copy()
    display["run_id"] = display["run_id"].str[:8] + "…"
    print("\n" + "=" * 70)
    print("LEADERBOARD (sorted by ICIR)")
    print("=" * 70)
    print(display.to_string(index=True))
    print("=" * 70 + "\n")
