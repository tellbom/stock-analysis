"""
evaluation.robustness
=====================
Robustness and null tests (T2.6).

Tests
-----
1. **Label-shuffle null**: retrain on permuted labels → Rank IC must collapse to ~0.
   If it doesn't, the pipeline has a bug (not alpha).

2. **Canary feature**: inject close.shift(-1) → Rank IC must *inflate*.
   Confirms the training harness responds to real signal.

3. **Feature ablation**: drop one feature group at a time and measure IC drop.
   Identifies which feature families carry the most signal.

4. **Subperiod stability**: split the evaluation window into halves and
   compare Rank IC.  Large divergence = regime sensitivity.

P4A-01 fix
----------
The ``embargo`` parameter now defaults to ``None``, which resolves to
``horizon`` inside the function — matching the ``PurgedKFold`` default.
Previously the hard-coded ``embargo=5`` caused ablation re-fits to run
with a different embargo than the baseline (embargo=20 for horizon=20),
making ablation deltas apples-to-oranges and slightly inflating ablation IC.

P4A-02 addition
---------------
``RobustnessReport`` now carries ``subperiod_ic_ratio`` — a stability
index in [0, 1] where 1.0 means both halves are identical in magnitude
and 0.0 means one half is zero.  The alpha verdict uses this to distinguish
"lockbox noise" from "genuine regime sensitivity".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from quant_platform.training.lgbm_model import fit_oof
from quant_platform.evaluation.metrics import evaluate, EvalReport
from quant_platform.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RobustnessReport:
    """Results from all robustness checks."""
    # Null test
    shuffle_rank_ic:  float = float("nan")   # should be ~0
    shuffle_passed:   bool  = False          # True if abs(shuffle_rank_ic) < threshold

    # Canary
    canary_rank_ic:   float = float("nan")   # should be >> baseline IC
    canary_passed:    bool  = False          # True if canary_ic > baseline_ic * 1.5

    # Subperiod
    first_half_ric:    float = float("nan")
    second_half_ric:   float = float("nan")
    subperiod_stable:  bool  = False         # True if both halves share the same sign
    # P4A-02: stability index — min(|h1|,|h2|)/max(|h1|,|h2|) in [0,1].
    # 1.0 = perfectly matched magnitude; 0.0 = one half is zero.
    # NaN when either half is NaN.
    subperiod_ic_ratio: float = float("nan")

    # Ablation
    ablation_results: dict[str, float] = field(default_factory=dict)  # group → ric

    # Baseline (the model being tested)
    baseline_rank_ic: float = float("nan")

    def print_summary(self) -> None:
        print("\n" + "=" * 60)
        print("ROBUSTNESS TESTS")
        print("=" * 60)
        print(f"  Baseline Rank IC:     {self.baseline_rank_ic:+.4f}")
        print(f"  Label-shuffle IC:     {self.shuffle_rank_ic:+.4f}  "
              f"({'PASS ✓' if self.shuffle_passed else 'FAIL ✗'})")
        print(f"  Canary IC:            {self.canary_rank_ic:+.4f}  "
              f"({'PASS ✓' if self.canary_passed else 'FAIL ✗'})")
        print(f"  First-half Rank IC:   {self.first_half_ric:+.4f}")
        print(f"  Second-half Rank IC:  {self.second_half_ric:+.4f}  "
              f"({'stable ✓' if self.subperiod_stable else 'unstable ✗'})")
        if not np.isnan(self.subperiod_ic_ratio):
            interp = _interpret_stability(self.subperiod_ic_ratio, self.subperiod_stable)
            print(f"  Stability ratio:      {self.subperiod_ic_ratio:.3f}  — {interp}")
        if self.ablation_results:
            print("  Feature ablation (IC with group removed):")
            for grp, ic in sorted(self.ablation_results.items()):
                delta = ic - self.baseline_rank_ic
                print(f"    remove {grp:20s}: IC={ic:+.4f}  Δ={delta:+.4f}")
        print("=" * 60 + "\n")

    def summary_dict(self) -> dict:
        return {
            "baseline_rank_ic":    round(self.baseline_rank_ic, 6),
            "shuffle_rank_ic":     round(self.shuffle_rank_ic, 6),
            "shuffle_passed":      self.shuffle_passed,
            "canary_rank_ic":      round(self.canary_rank_ic, 6),
            "canary_passed":       self.canary_passed,
            "first_half_ric":      round(self.first_half_ric, 6),
            "second_half_ric":     round(self.second_half_ric, 6),
            "subperiod_stable":    self.subperiod_stable,
            "subperiod_ic_ratio":  round(self.subperiod_ic_ratio, 4),
        }


def _interpret_stability(ratio: float, same_sign: bool) -> str:
    """Human-readable interpretation of the subperiod stability index."""
    if not same_sign:
        return "signal is REGIME-SENSITIVE (opposite signs — genuine decay likely)"
    if ratio >= 0.7:
        return "signal is REGIME-STABLE (consistent magnitude across halves)"
    if ratio >= 0.4:
        return "signal shows MODERATE regime sensitivity (magnitude varies)"
    return "signal shows HIGH regime sensitivity (magnitude differs strongly)"


def _compute_subperiod_ic_ratio(first: float, second: float) -> float:
    """
    Stability index = min(|a|, |b|) / max(|a|, |b|) in [0, 1].

    Returns NaN when either input is NaN, or max is zero (degenerate).
    Returns 0.0 when one half is near-zero (< 1e-9).
    """
    if np.isnan(first) or np.isnan(second):
        return float("nan")
    a, b = abs(first), abs(second)
    mx = max(a, b)
    if mx < 1e-9:
        return float("nan")  # both near-zero; ratio is undefined
    return float(min(a, b) / mx)


def run_robustness_tests(
    panel: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    baseline_oof: pd.Series,
    n_splits: int = 5,
    horizon: int = 20,
    embargo: int | None = None,      # P4A-01: default None → resolves to horizon
    shuffle_threshold: float = 0.02,
    seed: int = 42,
    feature_groups: dict[str, list[str]] | None = None,
) -> RobustnessReport:
    """
    Run all robustness tests.

    Parameters
    ----------
    panel : pd.DataFrame
        Must have: symbol, date, <feature_cols>, <label_col>.
    baseline_oof : pd.Series
        OOF predictions from the main model (already computed).
    horizon : int
        Label horizon in trading days.  Used to set embargo when embargo is None.
    embargo : int | None
        Embargo gap for all re-fits inside this function.
        **Default is None, which resolves to ``horizon``** — the same default
        used by ``PurgedKFold`` and ``fit_oof``.  This ensures ablation re-fits
        use the same embargo as the baseline, making deltas comparable.
        Pass an explicit integer to override (e.g. embargo=0 to disable).
    feature_groups : dict[str, list[str]] | None
        Groups for ablation test.  If None, ablation is skipped.
        Example: {"technical": ["ma_5","rsi_6",...], "cross_sec": [...]}
    """
    # P4A-01: resolve embargo exactly as PurgedKFold does
    if embargo is None:
        embargo = horizon

    report = RobustnessReport()
    valid = panel[panel[label_col].notna()].copy().reset_index(drop=True)
    dates = pd.to_datetime(valid["date"])

    # Baseline IC
    baseline_eval = evaluate(
        baseline_oof.reindex(valid.index),
        valid[label_col],
        dates,
        label_col=label_col,
    )
    report.baseline_rank_ic = baseline_eval.rank_ic_mean

    # --- 1. Label-shuffle null ---
    logger.info("Robustness: label-shuffle null test")
    shuffled_panel = valid.copy()
    rng = np.random.default_rng(seed)
    shuffled_panel[label_col] = rng.permutation(valid[label_col].values)
    try:
        shuffle_oof = fit_oof(
            shuffled_panel, feature_cols, label_col,
            n_splits=n_splits, horizon=horizon, embargo=embargo,
        )
        shuffle_eval = evaluate(
            shuffle_oof.oof_predictions.reindex(valid.index),
            valid[label_col],  # compare against ORIGINAL label to be rigorous
            dates, label_col="shuffle",
        )
        report.shuffle_rank_ic = shuffle_eval.rank_ic_mean
        report.shuffle_passed  = abs(report.shuffle_rank_ic) < shuffle_threshold
    except Exception as exc:
        logger.warning("Label-shuffle test failed: %s", exc)

    # --- 2. Canary feature ---
    logger.info("Robustness: canary feature test")
    canary_panel = valid.copy().sort_values(["symbol", "date"]).reset_index(drop=True)
    canary_panel["_canary"] = canary_panel.groupby("symbol")[label_col].shift(-1)
    canary_cols = feature_cols + ["_canary"]
    try:
        canary_oof = fit_oof(
            canary_panel, canary_cols, label_col,
            n_splits=n_splits, horizon=horizon, embargo=embargo,
        )
        canary_eval = evaluate(
            canary_oof.oof_predictions.reindex(canary_panel.index),
            canary_panel[label_col],
            pd.to_datetime(canary_panel["date"]),
            label_col="canary",
        )
        report.canary_rank_ic = canary_eval.rank_ic_mean
        # Canary should inflate IC relative to baseline
        report.canary_passed = (
            report.canary_rank_ic > report.baseline_rank_ic * 1.5
            or report.canary_rank_ic > 0.1  # absolute threshold as fallback
        )
    except Exception as exc:
        logger.warning("Canary test failed: %s", exc)

    # --- 3. Subperiod stability ---
    logger.info("Robustness: subperiod stability")
    sorted_dates = sorted(valid["date"].unique())
    mid = sorted_dates[len(sorted_dates) // 2]
    first_half  = valid[pd.to_datetime(valid["date"]) <= pd.to_datetime(mid)]
    second_half = valid[pd.to_datetime(valid["date"]) >  pd.to_datetime(mid)]

    for name, half in [("first", first_half), ("second", second_half)]:
        if len(half) < 50:
            continue
        eval_h = evaluate(
            baseline_oof.reindex(half.index),
            half[label_col],
            pd.to_datetime(half["date"]),
            label_col=f"{name}_half",
        )
        if name == "first":
            report.first_half_ric = eval_h.rank_ic_mean
        else:
            report.second_half_ric = eval_h.rank_ic_mean

    # Stable = both halves have the same sign
    if not np.isnan(report.first_half_ric) and not np.isnan(report.second_half_ric):
        report.subperiod_stable = (
            np.sign(report.first_half_ric) == np.sign(report.second_half_ric)
        )
        # P4A-02: compute stability index
        report.subperiod_ic_ratio = _compute_subperiod_ic_ratio(
            report.first_half_ric, report.second_half_ric
        )
        interp = _interpret_stability(report.subperiod_ic_ratio, report.subperiod_stable)
        logger.info(
            "Subperiod stability: first=%.4f  second=%.4f  ratio=%.3f  → %s",
            report.first_half_ric, report.second_half_ric,
            report.subperiod_ic_ratio, interp,
        )

    # --- 4. Feature ablation ---
    if feature_groups:
        logger.info("Robustness: feature ablation (%d groups)", len(feature_groups))
        for group_name, group_cols in feature_groups.items():
            ablation_cols = [c for c in feature_cols if c not in group_cols]
            if not ablation_cols:
                continue
            try:
                abl_oof = fit_oof(
                    valid, ablation_cols, label_col,
                    n_splits=n_splits, horizon=horizon, embargo=embargo,
                )
                abl_eval = evaluate(
                    abl_oof.oof_predictions.reindex(valid.index),
                    valid[label_col],
                    dates, label_col=f"ablate_{group_name}",
                )
                report.ablation_results[group_name] = abl_eval.rank_ic_mean
            except Exception as exc:
                logger.warning("Ablation for '%s' failed: %s", group_name, exc)

    return report
