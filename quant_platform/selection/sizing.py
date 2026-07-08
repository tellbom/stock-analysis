"""
quant_platform.selection.sizing
================================
SR-05: holding horizon and suggested position weight.

Pure post-inference formatting layer -- adds ``holding_horizon_days`` and
``suggested_weight`` to an already-selected actionable frame. Does not
retrain models or touch feature/label pipelines.

Industry exposure caps reuse ``SelectionConfig.exposure_warning_threshold``
(the same constant ``ExposureMonitor`` flags against) rather than
introducing a new exposure knob.
"""

from __future__ import annotations

import pandas as pd

from quant_platform.selection.config import SelectionConfig

TILT_METHODS = ("equal", "confidence")
DEFAULT_HORIZON_DAYS = 3


def _apply_industry_cap(
    weights: pd.Series,
    industry: pd.Series,
    threshold: float,
    max_iter: int = 20,
) -> pd.Series:
    """
    Scale down any industry whose weight share exceeds *threshold*, and
    redistribute the excess proportionally across the remaining
    (non-capped) rows. Repeats until stable or *max_iter* is reached.
    """
    weights = weights.astype(float).copy()

    for _ in range(max_iter):
        industry_sum = weights.groupby(industry).transform("sum")
        over_mask = industry_sum > threshold + 1e-12
        if not over_mask.any():
            break

        scale = (threshold / industry_sum).where(over_mask, 1.0)
        capped = weights * scale
        excess = weights.sum() - capped.sum()
        weights = capped

        free_mask = ~over_mask
        free_total = weights[free_mask].sum()
        if excess > 1e-12 and free_total > 1e-12:
            weights.loc[free_mask] += excess * (weights[free_mask] / free_total)
        else:
            break

    return weights


def attach_holding_and_weight(
    actionable_df: pd.DataFrame,
    *,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    tilt: str = "equal",
    industry_col: str = "industry_code",
    confidence_col: str = "confidence",
    config: SelectionConfig | None = None,
) -> pd.DataFrame:
    """
    Attach ``holding_horizon_days`` (constant) and ``suggested_weight`` to
    every row of *actionable_df*.

    tilt="equal" : 1/n for all rows.
    tilt="confidence" : proportional to *confidence_col* (requires that
        column to be present and non-negative), then capped per industry
        against ``config.exposure_warning_threshold`` and renormalized.

    Weights always sum to 1 across the input frame.
    """
    if tilt not in TILT_METHODS:
        raise ValueError(f"unknown tilt {tilt!r}; expected one of {TILT_METHODS}")
    if actionable_df.empty:
        raise ValueError("attach_holding_and_weight requires a non-empty frame")

    df = actionable_df.copy()
    n = len(df)
    df["holding_horizon_days"] = horizon_days

    if tilt == "equal":
        raw_weight = pd.Series(1.0 / n, index=df.index)
    else:
        if confidence_col not in df.columns:
            raise ValueError(
                f"tilt='confidence' requires column '{confidence_col}'"
            )
        conf = df[confidence_col].astype(float).clip(lower=0.0)
        raw_weight = (
            conf / conf.sum() if conf.sum() > 1e-12 else pd.Series(1.0 / n, index=df.index)
        )

    if tilt == "confidence" and industry_col in df.columns:
        cfg = config or SelectionConfig()
        weight = _apply_industry_cap(raw_weight, df[industry_col], cfg.exposure_warning_threshold)
    else:
        weight = raw_weight

    df["suggested_weight"] = weight
    return df
