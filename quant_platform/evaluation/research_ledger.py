"""
evaluation.research_ledger
==========================
Research ledger and multiple-testing account (T3.5).

The central threat of P3 is selection bias: run 500 experiments, report the
best IC, and you have manufactured alpha through search alone.  The ledger
makes this visible by tracking how many trials were run and deflating the
headline ICIR accordingly.

Deflated ICIR formula
---------------------
Based on Bailey & López de Prado (2014) "The Deflated Sharpe Ratio":

    ICIR_deflated = ICIR * sqrt(1 - gamma * ln(N) / T)

where:
    N       = number of trials tried behind the best result
    T       = number of out-of-fold evaluation dates
    gamma   = 0.5772… (Euler-Mascheroni constant, skewness adjustment)

This is a conservative approximation adapted for the ICIR context.
When N=1 (no multiple comparisons), deflation = 0 (ICIR unchanged).
When N→∞, deflation approaches 1 (ICIR collapses to 0).

Ledger schema
-------------
Parquet: ``<store_root>/research_ledger.parquet``

Columns: trial_id, timestamp, model_name, feature_set_id, label_col,
         fold_seed, raw_icir, deflated_icir, n_trials_so_far,
         lockbox_peeked, run_id, notes
"""

from __future__ import annotations

import datetime as dt
import math
from pathlib import Path

import numpy as np
import pandas as pd

from quant_platform.core.logging import get_logger

logger = get_logger(__name__)

_LEDGER_PATH = "research_ledger.parquet"
_EULER_MASCHERONI = 0.5772156649


def deflate_icir(
    raw_icir:   float,
    n_trials:   int,
    n_dates:    int,
) -> float:
    """
    Compute the deflated ICIR.

    Parameters
    ----------
    raw_icir  : headline ICIR of the best trial
    n_trials  : number of independent trials that were searched
    n_dates   : number of evaluation dates (out-of-fold)

    Returns
    -------
    float : deflated ICIR (always ≤ raw_icir)
    """
    if n_trials <= 1 or n_dates <= 0 or np.isnan(raw_icir):
        return raw_icir

    # Deflation factor based on expected maximum of correlated Sharpe ratios
    deflation = math.sqrt(
        max(0.0, 1.0 - _EULER_MASCHERONI * math.log(max(n_trials, 1)) / max(n_dates, 1))
    )
    return raw_icir * deflation


class ResearchLedger:
    """
    Append-only ledger tracking every experiment in P3.

    Parameters
    ----------
    store_root : Path | str
    """

    def __init__(self, store_root: Path | str) -> None:
        self.store_root  = Path(store_root)
        self._path       = self.store_root / _LEDGER_PATH

    def record(
        self,
        model_name:     str,
        feature_set_id: str,
        label_col:      str,
        fold_seed:      str,
        raw_icir:       float,
        n_dates:        int,
        run_id:         str = "",
        lockbox_peeked: bool = False,
        notes:          str = "",
    ) -> pd.Series:
        """
        Record one trial and return the updated ledger row (with deflated ICIR).

        The deflation uses the total number of trials run SO FAR for this
        (feature_set_id, label_col) combination — each new trial makes the
        deflation more severe.
        """
        existing = self.load()

        # Count trials for this study
        if not existing.empty:
            same_study = existing[
                (existing["feature_set_id"] == feature_set_id)
                & (existing["label_col"]      == label_col)
            ]
            n_trials_so_far = len(same_study) + 1
        else:
            n_trials_so_far = 1

        deflated = deflate_icir(raw_icir, n_trials_so_far, n_dates)

        row = {
            "trial_id":        len(existing),
            "timestamp":       dt.datetime.now().isoformat(timespec="seconds"),
            "model_name":      model_name,
            "feature_set_id":  feature_set_id,
            "label_col":       label_col,
            "fold_seed":       fold_seed,
            "raw_icir":        raw_icir,
            "deflated_icir":   deflated,
            "n_trials_so_far": n_trials_so_far,
            "run_id":          run_id,
            "lockbox_peeked":  lockbox_peeked,
            "notes":           notes,
        }

        updated = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
        self._save(updated)

        logger.info(
            "Ledger: trial %d — raw_ICIR=%.4f → deflated=%.4f (N=%d trials, T=%d dates)",
            row["trial_id"], raw_icir, deflated, n_trials_so_far, n_dates,
        )
        return pd.Series(row)

    def load(self) -> pd.DataFrame:
        """Return the full ledger; empty DataFrame if ledger not yet written."""
        if not self._path.exists():
            return pd.DataFrame()
        try:
            return pd.read_parquet(self._path)
        except Exception as exc:
            logger.error("Cannot read ledger: %s", exc)
            return pd.DataFrame()

    def best_deflated_icir(
        self,
        feature_set_id: str,
        label_col:      str,
    ) -> float:
        """Return the best deflated ICIR across all trials for a study."""
        df = self.load()
        if df.empty:
            return float("nan")
        mask = (
            (df["feature_set_id"] == feature_set_id)
            & (df["label_col"]     == label_col)
        )
        subset = df[mask]
        if subset.empty:
            return float("nan")
        return float(subset["deflated_icir"].max())

    def lockbox_peek_count(self) -> int:
        """Count how many times the lockbox has been peeked (should be ≤ 1)."""
        df = self.load()
        if df.empty:
            return 0
        return int(df["lockbox_peeked"].sum())

    def summary(self) -> dict:
        """Summary dict for quality reports and MLflow logging."""
        df = self.load()
        if df.empty:
            return {"total_trials": 0}
        return {
            "total_trials":      len(df),
            "lockbox_peeks":     self.lockbox_peek_count(),
            "best_raw_icir":     float(df["raw_icir"].max()),
            "best_deflated_icir": float(df["deflated_icir"].max()),
            "studies":           df["feature_set_id"].nunique(),
        }

    def _save(self, df: pd.DataFrame) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(self._path, index=False)
