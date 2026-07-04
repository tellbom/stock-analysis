"""
scripts/feature_pruning_diagnostic_table.py
==============================================
T5.1 — Collinearity + t-stat pruning pass.

Both underlying tools already exist and are reused UNMODIFIED:
  - quant_platform.evaluation.feature_ic.FeatureICReport.pruning_candidates()
    (weak-signal axis: |IC_5d| and |t-stat_5d| both below threshold)
  - quant_platform.features.pruning.FeaturePruner
    (collinearity axis: within a >85% Spearman cluster, keep only the
    highest-|IC_5d| member)

T5.1 is glue + presentation over these two existing tools: one ranked
table with IC, t-stat, decay half-life, and a documented keep/prune
DECISION per feature. It does NOT auto-apply any pruning -- per task.md's
explicit constraint ("do not prune casually ... treat each removal as a
measured, logged walk-forward comparison"), this script only produces
CANDIDATES. Actually removing a feature from the active list still
requires a T4.1 walk-forward A/B, done separately and logged.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant_platform.core.logging import get_logger
from quant_platform.evaluation.feature_ic import FeatureICReport, compute_feature_ic_report
from quant_platform.features.pruning import FeaturePruner, PruningResult

logger = get_logger(__name__)


def build_pruning_decision_table(
    ic_report: FeatureICReport,
    pruning_result: PruningResult,
    ic_threshold: float = 0.01,
    tstat_threshold: float = 1.5,
    horizon: str = "5d",
) -> pd.DataFrame:
    """
    Merge the weak-IC candidates (feature_ic.pruning_candidates) and the
    collinearity-pruned set (FeaturePruner.run) into one ranked table.

    Returns
    -------
    pd.DataFrame
        Columns: feature, ic_5d, tstat_5d, decay_halflife, n_dates,
        collinear_with, weak_signal, decision, reason.
        Sorted by |ic_5d| descending (same order as feature_ic_table()).
    """
    table = ic_report.feature_ic_table()
    if table.empty:
        logger.warning("build_pruning_decision_table: empty IC table")
        return table

    weak_candidates = set(
        ic_report.pruning_candidates(
            ic_threshold=ic_threshold, tstat_threshold=tstat_threshold, horizon=horizon,
        )
    )
    collinear_map = {p["feature"]: p for p in pruning_result.pruned}

    rows = []
    for _, row in table.iterrows():
        feat = row["feature"]
        is_weak = feat in weak_candidates
        collinear_info = collinear_map.get(feat)
        is_collinear = collinear_info is not None

        reasons = []
        if is_collinear:
            reasons.append(
                f"collinear with {collinear_info['corr_partner']} "
                f"(|rho|={abs(collinear_info['corr_value']):.3f})"
            )
        if is_weak:
            reasons.append(
                f"weak signal: |IC_{horizon}|<{ic_threshold} and "
                f"|t-stat_{horizon}|<{tstat_threshold}"
            )

        decision = "prune_candidate" if reasons else "keep"
        rows.append({
            "feature":        feat,
            "ic_5d":          row.get("ic_5d"),
            "tstat_5d":       row.get("tstat_5d"),
            "decay_halflife": row.get("decay_halflife"),
            "n_dates":        row.get("n_dates"),
            "collinear_with": collinear_info["corr_partner"] if is_collinear else None,
            "weak_signal":    is_weak,
            "decision":       decision,
            "reason":         "; ".join(reasons) if reasons else "sufficient signal, not collinear",
        })

    return pd.DataFrame(rows)


def run_pruning_pass(
    panel: pd.DataFrame,
    feature_cols: list[str],
    label_cols: list[str],
    store_root: Path | str,
    corr_threshold: float = 0.85,
    ic_threshold: float = 0.01,
    tstat_threshold: float = 1.5,
) -> pd.DataFrame:
    """
    T5.1 entry point: run both existing tools and print the combined
    decision table. Does not modify the feature registry or any Parquet.
    """
    ic_report = compute_feature_ic_report(
        panel, feature_cols=feature_cols, label_cols=label_cols, store_root=store_root,
    )
    pruner = FeaturePruner(store_root=store_root)
    pruning_result = pruner.run(ic_report, primary_label="ret_fwd_5d", corr_threshold=corr_threshold)

    decision_table = build_pruning_decision_table(
        ic_report, pruning_result, ic_threshold=ic_threshold, tstat_threshold=tstat_threshold,
    )

    print("\n" + "=" * 90)
    print("T5.1 — FEATURE PRUNING DECISION TABLE (candidates only, not applied)")
    print("=" * 90)
    if not decision_table.empty:
        print(decision_table.to_string(index=False, float_format=lambda x: f"{x:+.4f}" if isinstance(x, float) else str(x)))
    n_candidates = (decision_table["decision"] == "prune_candidate").sum() if not decision_table.empty else 0
    print(f"\n  {n_candidates}/{len(decision_table)} features flagged as prune candidates.")
    print("  NEXT STEP (not automated here): for each candidate, run a T4.1 walk-forward")
    print("  A/B (with vs without) and log the measured delta before actually removing it.")
    print("=" * 90 + "\n")

    return decision_table
