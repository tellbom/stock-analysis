"""
labels.residualiser
===================
Residualised return label builder (P4C-04).

The residualised return is the standard label used by professional factor
desks.  For each date T and horizon h, a cross-sectional OLS regression is
run:

    ret_fwd_{h}d ~ 1 + market_return_h + industry_dummies + log_float_mcap

The OLS residual is the label ``residual_ret_{h}d``.  It represents the
component of return unexplained by:
  - Market beta (market_return_h: CSI300 forward return on same window)
  - Sector beta (industry_code dummies)
  - Size exposure (log_float_mcap)

A model trained on this label predicts *pure idiosyncratic return* —
i.e., return that cannot be captured by a passive factor portfolio.

PIT correctness
---------------
The regression runs *per date* using only the cross-section of stocks
present on that date.  No across-time data is used.  The market return
and industry average returns are computed from the same T+1…T+1+h window
as the stock labels, so the residual labels are correctly PIT-safe.

Dependencies
------------
Requires ``industry_code`` and ``log_float_mcap`` columns in the panel,
which are added by:
  - ``features.industry._join_industry()`` (industry_code)
  - ``features.valuation.build_valuation_features()`` (cs_log_float_mcap
    or float_mcap_yi from which we derive log_float_mcap ourselves)

Usage
-----
    from quant_platform.labels.residualiser import residualise_returns

    label_panel = residualise_returns(
        panel=label_panel_with_industry_and_size,
        horizons=[1, 3, 5, 10, 20],
        market_ret_cols={"5": "index_ret_5d", ...},   # optional
    )
    # Adds: residual_ret_1d, residual_ret_3d, residual_ret_5d,
    #       residual_ret_10d, residual_ret_20d
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from quant_platform.core.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Core residualiser
# ---------------------------------------------------------------------------

def residualise_returns(
    panel: pd.DataFrame,
    horizons: list[int] | None = None,
    industry_col: str = "industry_code",
    mcap_col: str | None = None,
    market_ret_cols: dict[str, str] | None = None,
    min_stocks_per_date: int = 10,
) -> pd.DataFrame:
    """
    Add residualised forward return columns to the label panel.

    For each horizon h and date T, fits:

        ret_fwd_{h}d ~ intercept + log_float_mcap + industry_dummies
                       [+ market_ret_{h}d if provided]

    and stores the OLS residual as ``residual_ret_{h}d``.

    Parameters
    ----------
    panel : pd.DataFrame
        Label panel containing:
          - symbol, date
          - ret_fwd_{h}d for each horizon
          - industry_code (from P4B features or label join)
          - float_mcap_yi OR cs_log_float_mcap (for size control)
    horizons : list[int] | None
        Label horizons.  Defaults to [1, 3, 5, 10, 20].
    industry_col : str
        Column containing the industry code.  Stocks with _UNKNOWN industry
        are included but their dummies contribute to the intercept.
    mcap_col : str | None
        Column for market cap.  If None, tries ``float_mcap_yi`` then
        ``cs_log_float_mcap`` (the latter is used directly without logging).
    market_ret_cols : dict[str, str] | None
        Mapping horizon_str → market return column name.
        e.g. {"5": "excess_vs_csi300_5d"}.  When provided, the market
        forward return is *added back* as a regressor (the excess-vs-CSI300
        label already subtracts the market return, so this should NOT be
        used on excess labels — use raw ret_fwd_{h}d as input here).
    min_stocks_per_date : int
        Minimum number of stocks required on a date to run the regression.
        Dates with fewer stocks get NaN residuals.

    Returns
    -------
    pd.DataFrame
        Panel with ``residual_ret_{h}d`` columns added.
    """
    from quant_platform.labels.builder import DEFAULT_HORIZONS
    horizons = horizons or DEFAULT_HORIZONS

    df = panel.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date

    # Determine size column
    _mcap_col = _resolve_mcap_col(df, mcap_col)

    for h in horizons:
        ret_col = f"ret_fwd_{h}d"
        res_col = f"residual_ret_{h}d"

        if ret_col not in df.columns:
            logger.warning("residualise_returns: %s not in panel — skipping", ret_col)
            df[res_col] = np.nan
            continue

        # Market return column for this horizon (optional)
        mkt_col = None
        if market_ret_cols:
            mkt_col = market_ret_cols.get(str(h))
            if mkt_col and mkt_col not in df.columns:
                logger.warning(
                    "residualise_returns: market_ret_col %s not in panel — ignoring", mkt_col
                )
                mkt_col = None

        residuals = _residualise_horizon(
            df, ret_col, industry_col, _mcap_col, mkt_col, min_stocks_per_date
        )
        df[res_col] = residuals

        n_valid = df[res_col].notna().sum()
        logger.info(
            "residual_ret_%dd: %d/%d valid residuals (%.1f%% coverage)",
            h, n_valid, len(df), 100 * n_valid / len(df),
        )

    return df


def _resolve_mcap_col(df: pd.DataFrame, mcap_col: str | None) -> str | None:
    """Find the most appropriate size column in the panel."""
    if mcap_col and mcap_col in df.columns:
        return mcap_col
    for candidate in ("float_mcap_yi", "total_mcap_yi", "cs_log_float_mcap", "cs_log_mcap_rank"):
        if candidate in df.columns:
            return candidate
    return None


def _residualise_horizon(
    df: pd.DataFrame,
    ret_col: str,
    industry_col: str,
    mcap_col: str | None,
    market_ret_col: str | None,
    min_stocks: int,
) -> pd.Series:
    """
    Run per-date OLS and return the full residual Series aligned to df.index.
    """
    from sklearn.linear_model import LinearRegression

    residuals = pd.Series(np.nan, index=df.index)

    for date, grp in df.groupby("date"):
        sub = grp[[ret_col]].copy()

        # Drop rows with NaN in the dependent variable
        if ret_col in grp.columns:
            sub = grp[grp[ret_col].notna()].copy()
        if len(sub) < min_stocks:
            continue

        y = sub[ret_col].values

        # Build regressors
        X_parts = [np.ones((len(sub), 1))]   # intercept

        # Market return regressor
        if market_ret_col and market_ret_col in sub.columns:
            mkt = sub[market_ret_col].fillna(0).values.reshape(-1, 1)
            X_parts.append(mkt)

        # Size: log(float_mcap_yi), or use the column directly if already logged
        if mcap_col and mcap_col in sub.columns:
            mcap_vals = sub[mcap_col].values
            if mcap_col in ("float_mcap_yi", "total_mcap_yi"):
                mcap_vals = np.log(np.clip(mcap_vals, 1e-6, None))
            # Standardise to avoid scale dominance
            mcap_vals = (mcap_vals - np.nanmean(mcap_vals)) / max(np.nanstd(mcap_vals, ddof=1), 1e-9)
            X_parts.append(mcap_vals.reshape(-1, 1))

        # Industry dummies (pandas get_dummies is clean and fast)
        if industry_col in sub.columns:
            ind_series = sub[industry_col].fillna("_UNKNOWN")
            ind_dummies = pd.get_dummies(ind_series, drop_first=True, dtype=float)
            if not ind_dummies.empty:
                X_parts.append(ind_dummies.values)

        X = np.hstack(X_parts)

        # Mask rows with any NaN in X
        valid_mask = ~np.any(np.isnan(X), axis=1)
        if valid_mask.sum() < min_stocks:
            continue

        X_fit = X[valid_mask]
        y_fit = y[valid_mask]

        try:
            reg = LinearRegression(fit_intercept=False)   # intercept already in X
            reg.fit(X_fit, y_fit)
            y_pred = reg.predict(X_fit)
            resid  = y_fit - y_pred
        except Exception as exc:
            logger.debug("residualiser: OLS failed on %s: %s", date, exc)
            continue

        # Write residuals back to the result Series
        valid_idx = sub.index[valid_mask]
        residuals.loc[valid_idx] = resid

    return residuals
