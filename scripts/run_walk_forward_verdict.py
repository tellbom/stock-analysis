"""
scripts/run_walk_forward_verdict.py
====================================
T4.1 — Adopt WalkForwardEvaluator (n_windows >= 5) as the PRIMARY offline
verdict, replacing reliance on:
  - the single-date evaluation_config_e_20260626.py script, and
  - the P2 walk-forward run that used n_windows=2 (which pooled-aggregate
    masked a window-1 collapse -- see plan.md cause D).

This module adds NO new evaluation logic. WalkForwardEvaluator already
prints a per-window breakdown (not just the pooled aggregate) in
print_summary(). What T4.1 needed was an adoption point: a single,
reusable entry point that always requests >=5 windows and always reports
per-window IC alongside the aggregate, so nobody can quietly go back to
reading only the pooled number.

Usage
-----
    from scripts.run_walk_forward_verdict import run_primary_verdict

    result = run_primary_verdict(
        panel=panel,                 # feature+label panel, one row per (symbol,date)
        feature_cols=feature_cols,
        label_col="ret_fwd_5d",      # or "ret_fwd_3d" -- see T2.3
        horizon=5,
        model_factory=None,          # None -> default LightGBM pipeline
    )

CLI-style use with a saved panel:
    python scripts/run_walk_forward_verdict.py --panel path/to/panel.parquet \\
        --label-col ret_fwd_5d --horizon 5 --feature-cols-file feats.txt
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Callable

import pandas as pd

from quant_platform.core.logging import get_logger
from quant_platform.evaluation.walk_forward import WalkForwardEvaluator, WalkForwardResult

logger = get_logger(__name__)

MIN_WINDOWS_REQUIRED = 5  # T4.1's hard floor -- do not silently accept fewer


def run_primary_verdict(
    panel: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    horizon: int = 5,
    n_windows: int = 5,
    window_months: int = 12,
    model_factory: Callable | None = None,
    store_root: Path | str | None = None,
    trading_calendar: pd.DataFrame | None = None,
    cost_bps: float = 10.0,
) -> WalkForwardResult:
    """
    Run the walk-forward evaluator as the primary offline verdict.

    Raises a warning (not an error) if fewer than MIN_WINDOWS_REQUIRED
    windows actually complete (e.g. insufficient history) -- the caller
    still gets the result, but is told explicitly not to trust it as a
    T4.1-grade verdict.
    """
    if not re.fullmatch(r"ret_fwd_\d+d", label_col) or label_col != f"ret_fwd_{horizon}d":
        raise ValueError("Primary verdict requires the exact raw ret_fwd_{horizon}d label")
    if trading_calendar is None and store_root is not None:
        calendar_path = Path(store_root) / "calendar/trading_calendar.parquet"
        if calendar_path.exists():
            trading_calendar = pd.read_parquet(calendar_path)
    if n_windows < MIN_WINDOWS_REQUIRED:
        logger.warning(
            "run_primary_verdict: n_windows=%d < %d -- task.md T4.1 requires "
            ">= 5 windows for the pooled aggregate to be trustworthy; this "
            "run raised to %d.", n_windows, MIN_WINDOWS_REQUIRED, MIN_WINDOWS_REQUIRED,
        )
        n_windows = MIN_WINDOWS_REQUIRED

    evaluator = WalkForwardEvaluator(
        n_windows=n_windows, window_months=window_months, horizon=horizon, cost_bps=cost_bps,
    )
    result = evaluator.run(
        panel=panel, feature_cols=feature_cols, label_col=label_col,
        model_factory=model_factory,
        trading_calendar=trading_calendar,
    )
    result.metadata.update({
        "required_windows": MIN_WINDOWS_REQUIRED,
        "requested_windows": n_windows,
        "completed_windows": result.n_windows(),
        "verdict_status": "evaluation_complete_not_promotion_approval"
        if result.n_windows() == n_windows and result.n_windows() >= MIN_WINDOWS_REQUIRED and trading_calendar is not None
        else "preliminary",
    })
    if trading_calendar is None:
        logger.warning("No exchange calendar supplied: observed panel dates are provisional; verdict remains PRELIMINARY")

    if result.n_windows() < MIN_WINDOWS_REQUIRED:
        logger.warning(
            "run_primary_verdict: only %d/%d windows actually completed "
            "(insufficient history) -- treat this verdict as PRELIMINARY, "
            "not the T4.1-grade primary verdict.",
            result.n_windows(), n_windows,
        )

    # Always print the per-window breakdown alongside the aggregate --
    # this is the concrete fix for "pooled aggregate masked a window-1
    # collapse" (plan.md cause D / P2 postmortem).
    result.print_summary()

    if store_root is not None:
        store_root = Path(store_root)
        out_dir = store_root / "evaluation"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"walk_forward_verdict_{label_col}_h{horizon}.csv"
        result.to_dataframe().to_csv(out_path, index=False)
        out_path.with_suffix(".json").write_text(json.dumps(result.metadata, indent=2), encoding="utf-8")
        logger.info("Per-window verdict saved -> %s", out_path)

    return result


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Run the T4.1 primary walk-forward verdict.")
    parser.add_argument("--panel", required=True, help="Path to a feature+label panel Parquet.")
    parser.add_argument("--label-col", required=True)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--n-windows", type=int, default=5)
    parser.add_argument("--feature-cols-file", required=True,
                         help="Text file, one feature column name per line.")
    parser.add_argument("--store-root", default=None)
    parser.add_argument("--calendar", help="Exchange-calendar parquet (otherwise try store-root/calendar/trading_calendar.parquet)")
    parser.add_argument("--window-months", type=int, default=12)
    parser.add_argument("--cost-bps", type=float, default=10.0,
                        help="Legacy diagnostic long-short one-way turnover cost, not long-only round-trip cost")
    args = parser.parse_args()

    panel = pd.read_parquet(args.panel)
    feature_cols = [l.strip() for l in Path(args.feature_cols_file).read_text().splitlines() if l.strip()]

    run_primary_verdict(
        panel, feature_cols, args.label_col,
        horizon=args.horizon, n_windows=args.n_windows,
        store_root=args.store_root,
        window_months=args.window_months, cost_bps=args.cost_bps,
        trading_calendar=pd.read_parquet(args.calendar) if args.calendar else None,
    )


if __name__ == "__main__":
    _cli()
