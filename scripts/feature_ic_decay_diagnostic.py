"""
scripts/feature_ic_decay_diagnostic.py
=======================================
T2.1 — Run the IC-decay diagnostic across existing horizons, and quantify
        the D+3-vs-D+5 reversal pattern across MANY dates (not a single
        cross-section) with a paired t-stat.
T3.2 — Re-run the same diagnostic with `reversal_3d` included, after T3.1
        adds it, to measure its marginal IC / decay half-life.

This is the SAME script used at two points in the roadmap: once before
T3.1 (feature_cols = existing TECHNICAL_SPECS/CROSS_SECTIONAL_SPECS names,
to confirm the D+3>D+5 pattern) and once after (feature_cols including
"reversal_3d", to check its marginal contribution). No new statistics
module was built beyond what compute_feature_ic_report already provides,
plus one small paired-t-test helper for the D+3-vs-D+5 comparison that
compute_feature_ic_report's decay_curve (mean-only) does not expose.

Reused as-is (no changes): evaluation.feature_ic.compute_feature_ic_report.

Usage
-----
    from quant_platform.features.pipeline import FeaturePipeline
    from quant_platform.labels.builder import build_label_panel
    from scripts.feature_ic_decay_diagnostic import run_diagnostic

    feat_panel  = pipeline.build_panel(symbols, feature_set_id)
    label_panel = build_label_panel(store_root, symbols)
    panel = feat_panel.merge(label_panel, on=["symbol", "date"], how="inner")

    run_diagnostic(
        panel,
        feature_cols=[...],          # e.g. TECHNICAL feature names
        label_cols=["ret_fwd_1d", "ret_fwd_3d", "ret_fwd_5d"],  # T2.1 note:
            # pass only labels confirmed present in the panel; do not assume
            # ret_fwd_10d exists without checking panel.columns first.
        store_root=store_root,
    )

NOTE ON DEFAULT_LABEL_COLS drift
---------------------------------
task.md/plan.md were written against a snapshot where
labels.builder.DEFAULT_HORIZONS == [1, 5, 20]. The current codebase's
DEFAULT_HORIZONS is [1, 3, 5, 10, 20] (P4A-04) -- ret_fwd_3d and
ret_fwd_10d are supported horizons now, but whether they have actually been
*generated* for a given
store_root/symbol set is a data-pipeline execution question, not a code
question. This script never assumes either way: it intersects the
caller-requested label_cols with panel.columns and logs what it found.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, spearmanr

from quant_platform.core.logging import get_logger
from quant_platform.evaluation.feature_ic import (
    compute_feature_ic_report, DEFAULT_DECAY_LAGS, FeatureICReport,
)

logger = get_logger(__name__)


def _daily_ic_at_lag(
    panel: pd.DataFrame,
    feature_col: str,
    lag: int,
    min_stocks: int = 5,
) -> pd.Series:
    """
    Per-date cross-sectional Rank IC between *feature_col* and the
    close-price-derived forward return at *lag* trading days, using the
    same T+1 -> T+1+lag convention as labels.builder / feature_ic.
    Returns a Series indexed by date (only dates with >= min_stocks).
    """
    df = panel[["symbol", "date", "close", feature_col]].dropna(subset=[feature_col]).copy()
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    df["_fwd"] = df.groupby("symbol")["close"].transform(
        lambda x: x.shift(-lag) / x.shift(-1) - 1
    )
    sub = df[[feature_col, "_fwd", "date"]].dropna()

    dates, rics = [], []
    for date, grp in sub.groupby("date"):
        if len(grp) < min_stocks:
            continue
        ric, _ = spearmanr(grp[feature_col], grp["_fwd"])
        if not np.isnan(ric):
            dates.append(date)
            rics.append(ric)
    return pd.Series(rics, index=pd.Index(dates, name="date"))


def d3_vs_d5_tstat(panel: pd.DataFrame, feature_col: str, min_stocks: int = 5) -> dict:
    """
    Paired t-test of per-date Rank IC at lag=3 vs lag=5 for one feature,
    across every date both are computable -- this is the "quantify across
    many dates, not a single cross-section" step T2.1 asks for.

    Returns
    -------
    dict with keys: n_dates, mean_ic_d3, mean_ic_d5, mean_diff, tstat, pvalue
    """
    ic_d3 = _daily_ic_at_lag(panel, feature_col, 3, min_stocks)
    ic_d5 = _daily_ic_at_lag(panel, feature_col, 5, min_stocks)
    common = ic_d3.index.intersection(ic_d5.index)
    if len(common) < 2:
        return {"n_dates": len(common), "mean_ic_d3": float("nan"),
                "mean_ic_d5": float("nan"), "mean_diff": float("nan"),
                "tstat": float("nan"), "pvalue": float("nan")}

    a, b = ic_d3.loc[common], ic_d5.loc[common]
    tstat, pvalue = ttest_rel(a, b)
    return {
        "n_dates": len(common),
        "mean_ic_d3": float(a.mean()),
        "mean_ic_d5": float(b.mean()),
        "mean_diff": float((a - b).mean()),
        "tstat": float(tstat),
        "pvalue": float(pvalue),
    }


def run_diagnostic(
    panel: pd.DataFrame,
    feature_cols: list[str],
    label_cols: list[str],
    store_root: Path | str | None = None,
    reversal_feature: str = "reversal_3d",
) -> FeatureICReport:
    """
    Run compute_feature_ic_report() and print the D+3-vs-D+5 t-stat table
    plus (if present) the reversal factor's own decay profile.
    """
    present_labels = [c for c in label_cols if c in panel.columns]
    missing_labels = [c for c in label_cols if c not in panel.columns]
    if missing_labels:
        logger.warning(
            "run_diagnostic: requested label_cols %s not in panel -- "
            "dropping them, not assuming they exist. Present: %s",
            missing_labels, present_labels,
        )

    report = compute_feature_ic_report(
        panel, feature_cols=feature_cols, label_cols=present_labels,
        decay_lags=DEFAULT_DECAY_LAGS, store_root=store_root,
    )
    report.print_summary(top_n=20)

    print("\n" + "=" * 80)
    print("D+3 vs D+5 REVERSAL CHECK (paired t-test across ALL dates)")
    print("=" * 80)
    rows = []
    for feat in feature_cols:
        if feat not in panel.columns:
            continue
        stats = d3_vs_d5_tstat(panel, feat)
        stats["feature"] = feat
        rows.append(stats)
    d3d5_df = pd.DataFrame(rows).sort_values("tstat", key=lambda s: s.abs(), ascending=False)
    print(d3d5_df.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))

    if reversal_feature in panel.columns:
        print("\n" + "=" * 80)
        print(f"T3.2: marginal profile of '{reversal_feature}'")
        print("=" * 80)
        row = next((r for r in report.rows if r.feature == reversal_feature), None)
        if row is not None:
            print(f"  ic_1d={row.ic_1d:+.4f} (t={row.tstat_1d:+.2f})  "
                  f"ic_5d={row.ic_5d:+.4f} (t={row.tstat_5d:+.2f})  "
                  f"decay_halflife={row.decay_halflife}")
            print(f"  decay_curve: {row.decay_curve}")
            expected_short_halflife = (
                not np.isnan(row.decay_halflife) and row.decay_halflife <= 5
            )
            if expected_short_halflife:
                print("  -> half-life <= 5d: consistent with a genuine short-term "
                      "reversal signal, keep.")
            else:
                print("  -> half-life > 5d or undefined: does NOT show the expected "
                      "short reversal profile -- flag as a pruning candidate, "
                      "do not keep by default.")

    return report
