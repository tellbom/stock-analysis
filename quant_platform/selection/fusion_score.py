"""
quant_platform.selection.fusion_score
======================================
SR-01 score contract: hybrid = gate filter x smooth fusion.

GF-08 found gate_first does not beat smooth fusion, and the fixated
fixed_weight/gate_first arms are statistically tied.  Per GF-08 section 6.2
(the pre-registered fallback) the strategy layer ranks off a SMOOTH score
computed over (base_pct, recent_pct) / (base_rank, recent_rank), while the
gate's tiers are used only as a hard actionable/non-actionable filter.

This module is purely post-inference: it consumes the columns produced by
``quant_platform.selection.gate_fusion.gate_first_fusion`` (or an equivalent
base+recent scored frame) and never imports/edits gate_fusion internals
beyond its public ``ACTIONABLE_TIERS`` constant.
"""

from __future__ import annotations

import pandas as pd

from quant_platform.selection.gate_fusion import ACTIONABLE_TIERS

#: Smooth fusion methods.  Default is "rrf" (GF-08: best Rank IC, weight-free,
#: nothing to overfit).  "recent_only" and "fixed_weight" are the
#: pre-registered alternates decided at SR-07, not selected by default.
FUSION_METHODS = ("rrf", "recent_only", "fixed_weight")

DEFAULT_METHOD = "rrf"


def fusion_score(
    frame: pd.DataFrame,
    *,
    method: str = DEFAULT_METHOD,
    k: int = 60,
    base_weight: float = 0.7,
) -> pd.DataFrame:
    """
    Compute a smooth fusion score over base/recent model outputs.

    Required columns depend on ``method``:
      * "rrf"          -- base_rank, recent_rank
      * "recent_only"  -- recent_pct
      * "fixed_weight" -- base_pct, recent_pct

    Adds ``fusion_score_col`` and ``fusion_rank`` (1 = best).  Does not
    mutate gate columns; returns a copy sorted by fusion_rank.
    """
    if method not in FUSION_METHODS:
        raise ValueError(f"unknown fusion method {method!r}; expected one of {FUSION_METHODS}")

    df = frame.copy()

    if method == "rrf":
        required = {"base_rank", "recent_rank"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"fusion_score(method='rrf') missing columns: {sorted(missing)}")
        df["fusion_score_col"] = (1.0 / (k + df["base_rank"].astype(float))) + (
            1.0 / (k + df["recent_rank"].astype(float))
        )
    elif method == "recent_only":
        if "recent_pct" not in df.columns:
            raise ValueError("fusion_score(method='recent_only') missing column: recent_pct")
        df["fusion_score_col"] = df["recent_pct"].astype(float)
    else:  # fixed_weight
        required = {"base_pct", "recent_pct"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"fusion_score(method='fixed_weight') missing columns: {sorted(missing)}")
        if not 0.0 <= base_weight <= 1.0:
            raise ValueError(f"base_weight must be in [0, 1], got {base_weight}")
        df["fusion_score_col"] = (base_weight * df["base_pct"].astype(float)) + (
            (1.0 - base_weight) * df["recent_pct"].astype(float)
        )

    df["fusion_rank"] = df["fusion_score_col"].rank(ascending=False, method="first").astype(int)
    return df.sort_values("fusion_rank").reset_index(drop=True)


def actionable_pool(
    fused_gate_frame: pd.DataFrame,
    *,
    method: str = DEFAULT_METHOD,
    k: int = 60,
    base_weight: float = 0.7,
) -> pd.DataFrame:
    """
    Apply the SR-01 hybrid contract to a gate-tiered frame.

    ``fused_gate_frame`` must already carry ``gate_tier`` (i.e. have been
    through ``gate_first_fusion``).  RISK_VETO / E_REJECT / D_OBSERVE / etc
    are excluded -- only ``ACTIONABLE_TIERS`` (A_MAIN, B_SHORT_BOOST) remain
    -- and the survivors are re-ranked by the smooth fusion score, not by
    the gate's ``final_rank``.  This is the frame downstream
    ``IndustryNeutralRanker`` should consume, ranking on ``fusion_score_col``.
    """
    if "gate_tier" not in fused_gate_frame.columns:
        raise ValueError("actionable_pool requires a 'gate_tier' column (run gate_first_fusion first)")

    pool = fused_gate_frame[fused_gate_frame["gate_tier"].isin(ACTIONABLE_TIERS)].copy()
    return fusion_score(pool, method=method, k=k, base_weight=base_weight)
