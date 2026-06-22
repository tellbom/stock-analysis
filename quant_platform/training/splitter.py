"""
training.splitter
=================
Time-aware cross-validation with purging and embargo (T2.1).

Implements the de Prado purged k-fold split:

  1. **Time-based folds**: train < val by date — never random k-fold.
  2. **Purging**: training samples whose label windows overlap the validation
     fold are removed.  For a horizon-h label at row i, the label window is
     [i+1, i+1+h]; if any part of this window falls in the val fold, row i
     is purged from train.
  3. **Embargo**: a gap of ``embargo_days`` trading rows after the val fold
     is also excluded from training, to prevent autocorrelation leakage.

The split operates on a flat panel (symbol, date, features, label) keyed by
integer position.  It yields (train_idx, val_idx) index arrays.

Reference: de Prado, "Advances in Financial Machine Learning", ch. 7.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


class PurgedKFold:
    """
    Purged + embargoed k-fold splitter for a time-indexed panel.

    Parameters
    ----------
    n_splits : int
        Number of folds.  Default 5.
    horizon : int
        Label look-ahead in trading rows (same as the label horizon h).
        Used to determine which training rows to purge.
    embargo : int | None
        Number of rows to exclude after the purge zone (embargo gap).
        Defaults to ``horizon`` — the minimum required so that no training
        sample's label window overlaps the validation fold's label window.
        Setting embargo < horizon leaves (horizon - embargo) days of label
        autocorrelation across the boundary, which inflates apparent IC.
        Pass an explicit integer to override (e.g. embargo=0 to disable).
    """

    def __init__(
        self,
        n_splits: int = 5,
        horizon: int = 20,
        embargo: int | None = None,
    ) -> None:
        self.n_splits  = n_splits
        self.horizon   = horizon
        # Default: embargo must be at least horizon to prevent label-window overlap
        self.embargo   = embargo if embargo is not None else horizon

    def split(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
        groups=None,
    ):
        """
        Yield (train_idx, val_idx) for each fold.

        Parameters
        ----------
        X : pd.DataFrame
            Must have a ``date`` column (or DatetimeIndex).  Rows must be
            sorted by date ascending before calling split().
        """
        n = len(X)
        # Build date array
        if "date" in X.columns:
            dates = pd.to_datetime(X["date"]).values
        elif isinstance(X.index, pd.DatetimeIndex):
            dates = X.index.values
        else:
            # Fall back to integer positions as a proxy for time ordering
            dates = np.arange(n)

        indices = np.arange(n)

        # Split into n_splits folds by position (time-ordered)
        fold_size = n // self.n_splits
        for k in range(self.n_splits):
            # Validation fold: fold k
            val_start = k * fold_size
            val_end   = val_start + fold_size if k < self.n_splits - 1 else n
            val_idx   = indices[val_start:val_end]

            # Training: everything BEFORE the val fold
            # (no future-to-past leakage; we don't use data after val fold)
            train_candidates = indices[:val_start]

            if len(train_candidates) == 0:
                continue

            # Purge: remove rows whose label window [i+1, i+1+horizon]
            # overlaps the validation window [val_start, val_end)
            purge_cutoff = val_start - self.horizon - 1
            train_idx = train_candidates[train_candidates <= purge_cutoff]

            # Embargo: remove the ``embargo`` rows immediately before val_start
            embargo_cutoff = val_start - self.embargo - self.horizon - 1
            train_idx = train_candidates[train_candidates <= embargo_cutoff]

            if len(train_idx) == 0 or len(val_idx) == 0:
                continue

            yield train_idx, val_idx

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits


def make_lockbox_split(
    panel: pd.DataFrame,
    lockbox_months: int = 12,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Carve a lockbox (final test) slice from the panel.

    The lockbox is the most recent ``lockbox_months`` months of data.
    It is returned separately and must NOT be touched during model development.

    Returns
    -------
    (train_val_panel, lockbox_panel)
    """
    panel = panel.copy()
    panel["_date"] = pd.to_datetime(panel["date"])
    max_date   = panel["_date"].max()
    cutoff     = max_date - pd.DateOffset(months=lockbox_months)

    train_val = panel[panel["_date"] <= cutoff].drop(columns=["_date"])
    lockbox   = panel[panel["_date"] >  cutoff].drop(columns=["_date"])

    return train_val.reset_index(drop=True), lockbox.reset_index(drop=True)
