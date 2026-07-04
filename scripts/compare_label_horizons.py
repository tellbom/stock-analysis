"""
scripts/compare_label_horizons.py
===================================
T2.3 — Compare ret_fwd_3d vs ret_fwd_5d as the model TARGET, using the
T4.1 walk-forward verdict, with the model held fixed (no hyperparameter
tuning) so the label is the only variable that changes between runs.

Depends on: T2.2 (ret_fwd_3d must already be built), T4.1
(run_walk_forward_verdict.run_primary_verdict).

Constraint (task.md): do not tune hyperparameters between the two runs.
This module enforces that by requiring the SAME model_factory callable
be passed for both labels (there is deliberately no separate
"3d_hyperparams" / "5d_hyperparams" argument), and it prints the
model_factory's repr for both runs so the identical-model claim is
auditable rather than assumed.

Verdict rule (task.md, explicit no-assumption rule): if ret_fwd_3d does
NOT beat ret_fwd_5d on the walk-forward aggregate, KEEP ret_fwd_5d.
This module states that rule in code (decide_label) rather than leaving
it as a human judgement call made ad hoc after looking at numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from quant_platform.core.logging import get_logger
from scripts.run_walk_forward_verdict import run_primary_verdict

logger = get_logger(__name__)


@dataclass
class LabelHorizonComparison:
    label_3d: str
    label_5d: str
    rank_ic_3d: float
    rank_ic_5d: float
    icir_3d: float
    icir_5d: float
    sharpe_3d: float
    sharpe_5d: float
    n_windows_3d: int
    n_windows_5d: int
    decision: str          # "ret_fwd_3d" or "ret_fwd_5d"
    reason: str

    def print_summary(self) -> None:
        print("\n" + "=" * 70)
        print("T2.3 — ret_fwd_3d vs ret_fwd_5d (model held fixed)")
        print("=" * 70)
        print(f"  {'metric':<12} {'3d':>10} {'5d':>10}")
        print(f"  {'RankIC':<12} {self.rank_ic_3d:>+10.4f} {self.rank_ic_5d:>+10.4f}")
        print(f"  {'ICIR':<12} {self.icir_3d:>+10.4f} {self.icir_5d:>+10.4f}")
        print(f"  {'Sharpe':<12} {self.sharpe_3d:>+10.4f} {self.sharpe_5d:>+10.4f}")
        print(f"  {'n_windows':<12} {self.n_windows_3d:>10d} {self.n_windows_5d:>10d}")
        print(f"\n  DECISION: keep {self.decision}")
        print(f"  REASON:   {self.reason}")
        print("=" * 70 + "\n")


def decide_label(
    rank_ic_3d: float, rank_ic_5d: float,
    icir_3d: float, icir_5d: float,
) -> tuple[str, str]:
    """
    No-assumption decision rule (task.md T2.3): 3d must clearly beat 5d
    on BOTH the walk-forward aggregate Rank IC and ICIR to be adopted.
    A tie or 3d-worse-on-either-metric keeps 5d, since 5d is the
    platform's existing PRIMARY_LABEL_COL (labels.builder) and changing
    it is the higher-cost move.
    """
    import math
    if math.isnan(rank_ic_3d) or math.isnan(rank_ic_5d):
        return "ret_fwd_5d", "one or both walk-forward runs produced NaN aggregate IC -- insufficient evidence to switch, keep 5d"
    if rank_ic_3d > rank_ic_5d and icir_3d > icir_5d:
        return "ret_fwd_3d", f"3d beats 5d on both Rank IC ({rank_ic_3d:+.4f} > {rank_ic_5d:+.4f}) and ICIR ({icir_3d:+.4f} > {icir_5d:+.4f})"
    return "ret_fwd_5d", (
        f"3d did not clearly beat 5d on both metrics "
        f"(RankIC {rank_ic_3d:+.4f} vs {rank_ic_5d:+.4f}, "
        f"ICIR {icir_3d:+.4f} vs {icir_5d:+.4f}) -- no assumption, keep 5d"
    )


def compare_label_horizons(
    panel: pd.DataFrame,
    feature_cols: list[str],
    model_factory: Callable,
    n_windows: int = 5,
    window_months: int = 12,
    label_3d: str = "ret_fwd_3d",
    label_5d: str = "ret_fwd_5d",
) -> LabelHorizonComparison:
    """
    Run the T4.1 walk-forward verdict once per label, model_factory held
    identical across both runs, and apply the no-assumption decision rule.
    """
    for label, h in [(label_3d, 3), (label_5d, 5)]:
        if label not in panel.columns:
            raise ValueError(
                f"compare_label_horizons: '{label}' not in panel. "
                f"For {label_3d}, run T2.2's append_horizon_labels first."
            )

    logger.info(
        "compare_label_horizons: model_factory=%r held IDENTICAL for both runs "
        "(task.md T2.3 constraint: label is the only variable)", model_factory,
    )

    print(f"\nmodel_factory (identical for both runs): {model_factory!r}")

    print(f"\n--- Walk-forward on {label_3d} (horizon=3) ---")
    result_3d = run_primary_verdict(
        panel, feature_cols, label_3d, horizon=3,
        n_windows=n_windows, window_months=window_months, model_factory=model_factory,
    )

    print(f"\n--- Walk-forward on {label_5d} (horizon=5) ---")
    result_5d = run_primary_verdict(
        panel, feature_cols, label_5d, horizon=5,
        n_windows=n_windows, window_months=window_months, model_factory=model_factory,
    )

    decision, reason = decide_label(
        result_3d.agg_rank_ic_mean, result_5d.agg_rank_ic_mean,
        result_3d.agg_icir, result_5d.agg_icir,
    )

    comparison = LabelHorizonComparison(
        label_3d=label_3d, label_5d=label_5d,
        rank_ic_3d=result_3d.agg_rank_ic_mean, rank_ic_5d=result_5d.agg_rank_ic_mean,
        icir_3d=result_3d.agg_icir, icir_5d=result_5d.agg_icir,
        sharpe_3d=result_3d.agg_sharpe, sharpe_5d=result_5d.agg_sharpe,
        n_windows_3d=result_3d.n_windows(), n_windows_5d=result_5d.n_windows(),
        decision=decision, reason=reason,
    )
    comparison.print_summary()
    return comparison
