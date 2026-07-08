"""
quant_platform.selection.reco_schema
=====================================
SR-06: recommendation output hygiene.

The user-facing recommendation CSV must not carry forward-return label
columns (``ret_fwd_*``, their ``_cs``/``_bin`` variants, or any other
``*_fwd_*`` future series).  Shipping those in a recommendation is a
presentation-layer leak and an audit red flag -- even though they are (almost
certainly) not model inputs.

This module defines an explicit allow-list schema for the recommendation
CSV and a writer that only ever emits allow-listed columns, regardless of
what extra columns the upstream scored/ranked frame happens to carry.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

#: Columns that are always safe to ship in a user-facing recommendation.
#: Only columns in this list (and present in the frame) are written --
#: silently ignoring anything else, so a stray upstream column (label,
#: raw feature, debug field) can never leak into the shipped file.
RECOMMENDATION_SCHEMA: tuple[str, ...] = (
    "symbol",
    "date",
    "industry_code",
    "industry_name",
    "model_score",
    "global_score",
    "fusion_score_col",
    "fusion_rank",
    "gate_tier",
    "gate_reason",
    "industry_rank",
    "industry_neutral_score",
    "industry_size",
    "selected",
    "selection_reason",
    "exposure_flag",
    "confidence",
    "confidence_label",
    "holding_horizon_days",
    "suggested_weight",
)

#: Column-name patterns that must never appear in a shipped recommendation,
#: even if someone extends RECOMMENDATION_SCHEMA by mistake.  Defense in
#: depth on top of the allow-list.
_FORBIDDEN_PATTERNS = (
    re.compile(r"^ret_fwd_"),
    re.compile(r".*_fwd_.*"),
    re.compile(r".*_bin$"),
    re.compile(r".*_cs$"),
)


def _is_forbidden(col: str) -> bool:
    return any(p.match(col) for p in _FORBIDDEN_PATTERNS)


def recommendation_columns(df: pd.DataFrame) -> list[str]:
    """Return the allow-listed columns of *df*, in schema order."""
    cols = [c for c in RECOMMENDATION_SCHEMA if c in df.columns]
    leaked = [c for c in cols if _is_forbidden(c)]
    if leaked:
        # Should be unreachable given the allow-list above, but fail loudly
        # rather than ship a leak if the schema is ever misedited.
        raise ValueError(f"RECOMMENDATION_SCHEMA contains forbidden column(s): {leaked}")
    return cols


def write_recommendation_csv(df: pd.DataFrame, path: Path | str) -> Path:
    """Write only the allow-listed recommendation columns to *path*."""
    cols = recommendation_columns(df)
    out = Path(path)
    df[cols].to_csv(out, index=False)
    return out
