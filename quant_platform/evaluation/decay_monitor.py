"""
evaluation.decay_monitor
========================
Decay and drift monitoring (T3.8).

Monitors two independent signals:
  1. **IC decay**: rolling-window Rank IC of the champion model's predictions
     on new data.  A sustained negative trend triggers a retrain flag.
  2. **Feature drift (PSI)**: Population Stability Index measures how much
     the feature distribution has shifted between the training window and a
     recent window.  PSI > 0.2 = significant drift (industry threshold).

Trigger logic
-------------
  retrain_triggered   : rolling IC mean drops below ``ic_floor`` for ``min_bad_windows``
                        consecutive windows.
  retire_triggered    : rolling IC is negative for all of the last ``retire_windows``
                        windows.
  drift_triggered     : any feature has PSI > ``psi_threshold``.

Outputs
-------
  DecayReport — structured result with trigger flags + evidence.
  Written to ``<store_root>/decay_monitor.parquet`` for scheduling.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from quant_platform.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DecayReport:
    """Results of one decay/drift monitoring run."""
    generated_at:        str   = ""
    retrain_triggered:   bool  = False
    retire_triggered:    bool  = False
    drift_triggered:     bool  = False
    rolling_ic_series:   pd.Series = field(default_factory=pd.Series)
    rolling_ic_mean:     float = float("nan")
    rolling_ic_trend:    float = float("nan")  # slope of rolling IC over time
    psi_by_feature:      dict[str, float] = field(default_factory=dict)
    max_psi:             float = float("nan")
    drifted_features:    list[str] = field(default_factory=list)
    recommendation:      str   = ""

    def print_summary(self) -> None:
        print("\n" + "=" * 55)
        print(f"DECAY MONITOR  [{self.generated_at}]")
        print("=" * 55)
        print(f"  Rolling IC mean:  {self.rolling_ic_mean:+.4f}")
        print(f"  Rolling IC trend: {self.rolling_ic_trend:+.6f}")
        print(f"  Max PSI:          {self.max_psi:.4f}")
        if self.drifted_features:
            print(f"  Drifted features: {self.drifted_features}")
        print(f"  Retrain flag:     {self.retrain_triggered}")
        print(f"  Retire flag:      {self.retire_triggered}")
        print(f"  Drift flag:       {self.drift_triggered}")
        print(f"  Recommendation:   {self.recommendation}")
        print("=" * 55 + "\n")

    def summary_dict(self) -> dict:
        return {
            "generated_at":      self.generated_at,
            "retrain_triggered": self.retrain_triggered,
            "retire_triggered":  self.retire_triggered,
            "drift_triggered":   self.drift_triggered,
            "rolling_ic_mean":   round(self.rolling_ic_mean, 6),
            "max_psi":           round(self.max_psi, 4),
            "drifted_features":  self.drifted_features,
            "recommendation":    self.recommendation,
        }


def _compute_psi(
    reference:  pd.Series,
    current:    pd.Series,
    n_bins:     int = 10,
) -> float:
    """
    Population Stability Index between reference and current distributions.
    PSI < 0.1: stable. 0.1–0.2: mild shift. > 0.2: significant drift.
    """
    # Use quantile bins from reference
    bins = np.nanpercentile(reference.dropna(), np.linspace(0, 100, n_bins + 1))
    bins[0]  -= 1e-9
    bins[-1] += 1e-9

    ref_counts = np.histogram(reference.dropna(), bins=bins)[0]
    cur_counts = np.histogram(current.dropna(), bins=bins)[0]

    ref_pct = ref_counts / max(ref_counts.sum(), 1)
    cur_pct = cur_counts / max(cur_counts.sum(), 1)

    # Avoid log(0)
    eps = 1e-9
    psi = np.sum((cur_pct - ref_pct) * np.log((cur_pct + eps) / (ref_pct + eps)))
    return float(psi)


def run_decay_monitor(
    pred_series:        pd.Series,      # champion predictions, index=date
    actual_series:      pd.Series,      # actual forward returns, index=date
    feature_panel_ref:  pd.DataFrame,   # training-window features
    feature_panel_cur:  pd.DataFrame,   # recent-window features
    feature_cols:       list[str],
    window:             int   = 20,     # rolling IC window in dates
    ic_floor:           float = 0.0,    # IC below this = bad window
    min_bad_windows:    int   = 3,      # consecutive bad windows → retrain
    retire_windows:     int   = 5,      # all-negative windows → retire
    psi_threshold:      float = 0.2,
) -> DecayReport:
    """
    Run decay and drift monitoring.

    Parameters
    ----------
    pred_series / actual_series
        Per-date predictions and returns (not per-symbol; aggregate daily Rank IC).
    feature_panel_ref / feature_panel_cur
        Feature DataFrames from the training period and recent period
        respectively, for PSI computation.
    """
    report = DecayReport(
        generated_at=dt.datetime.now().isoformat(timespec="seconds")
    )

    # --- 1. Rolling IC ---
    dates = pred_series.index.sort_values()
    rolling_ic: list[float] = []
    rolling_dates: list = []

    for i in range(window, len(dates) + 1):
        window_dates = dates[i - window:i]
        p = pred_series.reindex(window_dates).dropna()
        a = actual_series.reindex(window_dates).dropna()
        common = p.index.intersection(a.index)
        if len(common) < 5:
            continue
        ic, _ = spearmanr(p.loc[common], a.loc[common])
        rolling_ic.append(float(ic))
        rolling_dates.append(window_dates[-1])

    if rolling_ic:
        report.rolling_ic_series = pd.Series(rolling_ic, index=rolling_dates)
        report.rolling_ic_mean   = float(np.mean(rolling_ic))

        # Trend: linear slope of rolling IC over time
        x = np.arange(len(rolling_ic))
        if len(x) >= 2:
            report.rolling_ic_trend = float(np.polyfit(x, rolling_ic, 1)[0])

        # Retrain trigger: last ``min_bad_windows`` windows all below ic_floor
        if len(rolling_ic) >= min_bad_windows:
            last_n = rolling_ic[-min_bad_windows:]
            report.retrain_triggered = all(v < ic_floor for v in last_n)

        # Retire trigger: all of last ``retire_windows`` are negative
        if len(rolling_ic) >= retire_windows:
            last_r = rolling_ic[-retire_windows:]
            report.retire_triggered = all(v < 0 for v in last_r)

    # --- 2. Feature drift (PSI) ---
    psi_vals: dict[str, float] = {}
    for col in feature_cols:
        if col not in feature_panel_ref.columns or col not in feature_panel_cur.columns:
            continue
        try:
            psi = _compute_psi(feature_panel_ref[col], feature_panel_cur[col])
            psi_vals[col] = psi
        except Exception as exc:
            logger.debug("PSI failed for %s: %s", col, exc)

    if psi_vals:
        report.psi_by_feature  = {k: round(v, 4) for k, v in psi_vals.items()}
        report.max_psi         = float(max(psi_vals.values()))
        report.drifted_features = [k for k, v in psi_vals.items() if v > psi_threshold]
        report.drift_triggered  = bool(report.drifted_features)

    # --- 3. Recommendation ---
    if report.retire_triggered:
        report.recommendation = "RETIRE — model has been consistently negative; retire and retrain from scratch."
    elif report.retrain_triggered and report.drift_triggered:
        report.recommendation = "RETRAIN (URGENT) — IC declining AND feature drift detected."
    elif report.retrain_triggered:
        report.recommendation = "RETRAIN — IC has declined below floor for consecutive windows."
    elif report.drift_triggered:
        report.recommendation = f"MONITOR — feature drift detected ({report.drifted_features}); schedule re-validation."
    else:
        report.recommendation = "OK — no decay or drift triggers fired."

    return report


def save_decay_report(
    report:     DecayReport,
    store_root: Path | str,
) -> None:
    """Append the decay report to a rolling Parquet log."""
    path = Path(store_root) / "decay_monitor.parquet"
    row  = pd.DataFrame([report.summary_dict()])
    if path.exists():
        existing = pd.read_parquet(path)
        updated  = pd.concat([existing, row], ignore_index=True)
    else:
        updated = row
    path.parent.mkdir(parents=True, exist_ok=True)
    updated.to_parquet(path, index=False)
    logger.info(
        "Decay report saved → %s (recommendation: %s)",
        path, report.recommendation,
    )
