"""
evaluation.regime_analysis
==========================
Walk-forward regime analysis and stability report (P4C-05).

Uses the WalkForwardEvaluator from P4A-03 to compute per-window Rank IC
broken down by feature group (technical, valuation, industry, flow, margin,
event).  Maps IC over time to identify:

  (a) Which feature groups are regime-stable vs regime-sensitive.
  (b) Which calendar windows are "hard regimes" (low IC across all groups).
  (c) Whether ensemble performance degrades uniformly or in group-specific
      patterns.

Output
------
  RegimeReport              — structured results dataclass
  per-window IC table       — CSV saved to <store_root>/evaluation/
  hard regime log           — windows where ensemble IC < 0.01 for 2+ consecutive windows
  research ledger entry     — records the regime analysis as a research trial

Usage
-----
    from quant_platform.evaluation.regime_analysis import RegimeAnalyser

    analyser = RegimeAnalyser(store_root=Path("/data/lake"))
    report = analyser.run(
        panel=panel,
        feature_groups={"technical": tech_cols, "valuation": val_cols, ...},
        label_col="ret_fwd_5d",
    )
    report.print_summary()
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from quant_platform.evaluation.walk_forward import WalkForwardEvaluator, WalkForwardResult
from quant_platform.core.logging import get_logger

logger = get_logger(__name__)

_HARD_REGIME_IC_THRESHOLD  = 0.01   # ensemble IC below this = hard window
_HARD_REGIME_CONSEC        = 2      # consecutive hard windows = hard regime


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class RegimeWindowRow:
    """Per-window IC breakdown by feature group."""
    window_id:   int
    test_start:  str
    test_end:    str
    ensemble_ic: float = float("nan")
    # Feature group ICs — keyed by group name
    group_ics:   dict[str, float] = field(default_factory=dict)
    is_hard:     bool = False   # True if ensemble_ic < threshold

    def to_dict(self) -> dict:
        d = {
            "window_id":   self.window_id,
            "test_start":  self.test_start,
            "test_end":    self.test_end,
            "ensemble_ic": round(self.ensemble_ic, 6),
            "is_hard":     self.is_hard,
        }
        for grp, ic in self.group_ics.items():
            d[f"ic_{grp}"] = round(ic, 6)
        return d


@dataclass
class RegimeReport:
    """Full walk-forward regime analysis report."""
    generated_at:     str = ""
    label_col:        str = ""
    n_windows:        int = 0
    horizon:          int = 5

    window_rows:      list[RegimeWindowRow] = field(default_factory=list)
    hard_regime_windows: list[int]          = field(default_factory=list)
    hard_regime_dates:   list[str]          = field(default_factory=list)

    # Aggregate stability per group: fraction of windows with positive IC
    group_stability:  dict[str, float]      = field(default_factory=dict)
    # Ensemble stability: fraction of windows with positive IC
    ensemble_stability: float               = float("nan")

    # Recommendations
    recommendations:  list[str]             = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        """Per-window table as a DataFrame."""
        return pd.DataFrame([r.to_dict() for r in self.window_rows])

    def print_summary(self) -> None:
        print("\n" + "=" * 75)
        print(f"REGIME ANALYSIS  [{self.generated_at}]")
        print(f"  Label: {self.label_col}  |  Horizon: {self.horizon}d  |  Windows: {self.n_windows}")
        print(f"  Ensemble stability: {self.ensemble_stability:.2f} (fraction with positive IC)")
        print()
        # Group stability
        if self.group_stability:
            print("  Feature group stability:")
            for grp, stab in sorted(self.group_stability.items()):
                ic_vals = [r.group_ics.get(grp, np.nan) for r in self.window_rows]
                mean_ic = np.nanmean(ic_vals)
                print(f"    {grp:20s}  stability={stab:.2f}  mean_IC={mean_ic:+.4f}")
        print()
        # Per-window table
        df = self.to_dataframe()
        if not df.empty:
            group_cols = [c for c in df.columns if c.startswith("ic_")]
            show_cols  = ["window_id", "test_start", "test_end", "ensemble_ic", "is_hard"] + group_cols
            print(df[show_cols].to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
        # Hard regimes
        if self.hard_regime_windows:
            print(f"\n  Hard regimes: windows {self.hard_regime_windows}")
            print(f"  ({len(self.hard_regime_dates)} consecutive hard window pairs)")
        # Recommendations
        if self.recommendations:
            print("\n  Recommendations:")
            for rec in self.recommendations:
                print(f"    • {rec}")
        print("=" * 75 + "\n")

    def save_csv(self, store_root: Path | str) -> Path:
        store_root = Path(store_root)
        eval_dir   = store_root / "evaluation"
        eval_dir.mkdir(parents=True, exist_ok=True)
        date_str = dt.date.today().isoformat()
        path = eval_dir / f"regime_analysis_{date_str}.csv"
        df = self.to_dataframe()
        if not df.empty:
            df.to_csv(path, index=False)
            logger.info("Regime analysis saved → %s", path)
        return path


# ---------------------------------------------------------------------------
# Analyser
# ---------------------------------------------------------------------------

class RegimeAnalyser:
    """
    Walk-forward regime analysis.

    Parameters
    ----------
    store_root : Path | str
    n_windows : int
        Number of sequential OOS windows.
    window_months : int
        Length of each test window.
    horizon : int
        Label horizon in trading days.
    """

    def __init__(
        self,
        store_root: Path | str,
        n_windows: int = 5,
        window_months: int = 12,
        horizon: int = 5,
    ) -> None:
        self.store_root   = Path(store_root)
        self.n_windows    = n_windows
        self.window_months = window_months
        self.horizon      = horizon

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        panel: pd.DataFrame,
        feature_groups: dict[str, list[str]],
        label_col: str = "ret_fwd_5d",
        save_csv: bool = True,
        record_to_ledger: bool = True,
    ) -> RegimeReport:
        """
        Run the regime analysis.

        Parameters
        ----------
        panel : pd.DataFrame
            Full enriched feature panel with label columns.
        feature_groups : dict[str, list[str]]
            Mapping of group name → list of feature columns.
            e.g. {"technical": ["rsi_6", "ma_5", ...], "valuation": [...], ...}
            An "ensemble" run using ALL features is always included.
        label_col : str
            Primary label column.
        save_csv : bool
            Save the per-window IC table to CSV.
        record_to_ledger : bool
            Record the regime analysis as a trial in the research ledger.

        Returns
        -------
        RegimeReport
        """
        report = RegimeReport(
            generated_at=dt.datetime.now().isoformat(timespec="seconds"),
            label_col=label_col,
            n_windows=self.n_windows,
            horizon=self.horizon,
        )

        wf = WalkForwardEvaluator(
            n_windows=self.n_windows,
            window_months=self.window_months,
            horizon=self.horizon,
        )

        all_feature_cols = []
        for cols in feature_groups.values():
            all_feature_cols.extend(cols)
        # Deduplicate preserving order
        seen = set()
        all_feature_cols = [c for c in all_feature_cols if not (c in seen or seen.add(c))]

        # Available features in the panel
        avail_all = [c for c in all_feature_cols if c in panel.columns]
        if not avail_all:
            logger.warning("RegimeAnalyser: no feature columns found in panel")
            return report

        # Check label availability
        if label_col not in panel.columns:
            logger.warning("RegimeAnalyser: label %s not in panel", label_col)
            return report

        logger.info(
            "RegimeAnalyser.run: %d windows × %d months, %d features, label=%s",
            self.n_windows, self.window_months, len(avail_all), label_col,
        )

        # --- Ensemble walk-forward ---
        ensemble_result = wf.run(panel, avail_all, label_col)
        logger.info(
            "Ensemble: agg RankIC=%.4f  ICIR=%.4f  IC_stability=%.2f",
            ensemble_result.agg_rank_ic_mean,
            ensemble_result.agg_icir,
            ensemble_result.ic_sign_stability,
        )

        # Ensemble IC by window
        ensemble_ic_by_window: dict[int, float] = {
            w.window_id: w.rank_ic_mean
            for w in ensemble_result.windows
        }

        # --- Per-group walk-forward ---
        group_ic_by_window: dict[str, dict[int, float]] = {}
        for grp_name, grp_cols in feature_groups.items():
            avail_grp = [c for c in grp_cols if c in panel.columns]
            if not avail_grp:
                logger.info("  Group %s: no available columns — skipping", grp_name)
                continue
            logger.info("  Running group '%s' (%d features)...", grp_name, len(avail_grp))
            grp_result = wf.run(panel, avail_grp, label_col)
            group_ic_by_window[grp_name] = {
                w.window_id: w.rank_ic_mean
                for w in grp_result.windows
            }

        # --- Build per-window report rows ---
        all_window_ids = sorted(ensemble_ic_by_window.keys())
        for win_id in all_window_ids:
            ens_ic = ensemble_ic_by_window.get(win_id, float("nan"))
            # Find window dates from ensemble result
            ens_windows = {w.window_id: w for w in ensemble_result.windows}
            win_obj = ens_windows.get(win_id)
            if win_obj is None:
                continue

            row = RegimeWindowRow(
                window_id=win_id,
                test_start=win_obj.test_start,
                test_end=win_obj.test_end,
                ensemble_ic=ens_ic,
                group_ics={
                    grp: group_ic_by_window[grp].get(win_id, float("nan"))
                    for grp in group_ic_by_window
                },
                is_hard=(not np.isnan(ens_ic) and abs(ens_ic) < _HARD_REGIME_IC_THRESHOLD),
            )
            report.window_rows.append(row)

        # --- Identify hard regimes ---
        report.hard_regime_windows = [
            r.window_id for r in report.window_rows if r.is_hard
        ]
        # Consecutive hard windows
        consec_hard = _find_consecutive(report.hard_regime_windows, _HARD_REGIME_CONSEC)
        report.hard_regime_dates = [
            f"{report.window_rows[wid].test_start} – {report.window_rows[wid].test_end}"
            for wid in consec_hard
            if wid < len(report.window_rows)
        ]

        # --- Group stability ---
        for grp_name in group_ic_by_window:
            ics = [r.group_ics.get(grp_name, np.nan) for r in report.window_rows]
            valid = [v for v in ics if not np.isnan(v)]
            if valid:
                report.group_stability[grp_name] = float(
                    sum(1 for v in valid if v > 0) / len(valid)
                )

        # Ensemble stability
        ens_ics = [r.ensemble_ic for r in report.window_rows if not np.isnan(r.ensemble_ic)]
        if ens_ics:
            report.ensemble_stability = float(
                sum(1 for v in ens_ics if v > 0) / len(ens_ics)
            )

        # --- Recommendations ---
        report.recommendations = _generate_recommendations(report)

        # --- Persist ---
        if save_csv:
            report.save_csv(self.store_root)

        if record_to_ledger and ensemble_result.agg_icir:
            self._record_to_ledger(report, ensemble_result)

        return report

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _record_to_ledger(
        self,
        report: RegimeReport,
        ensemble_result: WalkForwardResult,
    ) -> None:
        """Record the regime analysis as a research trial."""
        try:
            from quant_platform.evaluation.research_ledger import ResearchLedger
            ledger = ResearchLedger(self.store_root)
            ledger.record(
                model_name="RegimeAnalysis",
                feature_set_id="regime_analysis",
                label_col=report.label_col,
                fold_seed="wf_regime",
                raw_icir=float(ensemble_result.agg_icir),
                n_dates=ensemble_result.total_independent_periods,
                notes=f"Regime analysis: {report.n_windows} windows × {self.window_months}mo. "
                      f"Hard regimes: {report.hard_regime_windows}. "
                      f"Ensemble stability: {report.ensemble_stability:.2f}.",
            )
        except Exception as exc:
            logger.warning("Could not record regime analysis to ledger: %s", exc)


def _find_consecutive(ids: list[int], min_consec: int) -> list[int]:
    """Return IDs that appear in a run of at least min_consec consecutive IDs."""
    if len(ids) < min_consec:
        return []
    result = []
    i = 0
    while i < len(ids):
        j = i + 1
        while j < len(ids) and ids[j] == ids[j - 1] + 1:
            j += 1
        run_len = j - i
        if run_len >= min_consec:
            result.extend(ids[i:j])
        i = j
    return result


def _generate_recommendations(report: RegimeReport) -> list[str]:
    """Generate human-readable recommendations from the regime report."""
    recs = []

    if report.ensemble_stability < 0.5:
        recs.append(
            f"Ensemble IC is positive in only {report.ensemble_stability:.0%} of windows — "
            "signal is regime-sensitive.  Consider regime-conditional model weighting."
        )
    elif report.ensemble_stability >= 0.8:
        recs.append(
            f"Ensemble IC is positive in {report.ensemble_stability:.0%} of windows — "
            "signal shows good regime stability."
        )

    if report.hard_regime_windows:
        recs.append(
            f"Hard regimes detected: windows {report.hard_regime_windows}. "
            "Investigate whether these correspond to known market events (circuit breakers, "
            "major regulatory changes, COVID shock)."
        )

    unstable_groups = [
        grp for grp, stab in report.group_stability.items() if stab < 0.4
    ]
    stable_groups = [
        grp for grp, stab in report.group_stability.items() if stab >= 0.7
    ]
    if unstable_groups:
        recs.append(
            f"Regime-sensitive feature groups: {unstable_groups}. "
            "Consider downweighting these groups in hard regimes."
        )
    if stable_groups:
        recs.append(
            f"Regime-stable feature groups: {stable_groups}. "
            "These groups provide the most reliable signal across market conditions."
        )

    if not recs:
        recs.append("No strong regime patterns detected.  Continue monitoring.")

    return recs
