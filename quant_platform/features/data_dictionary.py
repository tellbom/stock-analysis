"""
features.data_dictionary
========================
Feature and label data dictionary (T1.8).

Every column produced by the pipeline is documented here:
  - Formula / transform
  - Source columns (inputs)
  - ``known_at`` semantics (when is this value knowable at date T?)
  - Warm-up rows

Outputs a human-readable text file and a machine-readable Parquet.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from quant_platform.features.registry import TECHNICAL_SPECS, CROSS_SECTIONAL_SPECS
from quant_platform.core.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ColumnDoc:
    name:       str
    category:   str      # "technical" | "cross_sectional" | "fundamental" | "label"
    formula:    str      # human-readable computation description
    inputs:     str      # raw columns consumed
    known_at:   str      # timing semantics
    warmup:     int      # rows masked to NaN at series start
    notes:      str = "" # additional caveats


# ---------------------------------------------------------------------------
# Column documentation catalogue
# ---------------------------------------------------------------------------

FEATURE_DOCS: list[ColumnDoc] = [
    # --- from TECHNICAL_SPECS ---
    *[
        ColumnDoc(
            name=s.name,
            category="technical",
            formula=s.transform,
            inputs=", ".join(s.inputs),
            known_at=f"End of trading day T (uses only OHLCV[T] and earlier)",
            warmup=s.warmup,
        )
        for s in TECHNICAL_SPECS
    ],
    # --- from CROSS_SECTIONAL_SPECS ---
    *[
        ColumnDoc(
            name=s.name,
            category="cross_sectional",
            formula=s.transform,
            inputs=", ".join(s.inputs),
            known_at=(
                "End of trading day T (computed across universe members at T; "
                "no look-back across time)"
            ),
            warmup=0,
            notes=(
                "Cross-sectional rank/zscore is computed using the point-in-time "
                "universe membership.  Using today's universe for a past date "
                "introduces survivorship bias."
            ),
        )
        for s in CROSS_SECTIONAL_SPECS
    ],
    # --- Fundamental features (T1.4) ---
    ColumnDoc(
        name="fund_revenue",
        category="fundamental",
        formula="Most recently announced quarterly/annual revenue (CNY)",
        inputs="fundamentals.revenue (announce_date ≤ T)",
        known_at=(
            "The announce_date of the most recent available report where "
            "announce_date ≤ T.  JOIN KEY IS announce_date, NOT period_end."
        ),
        warmup=0,
        notes=(
            "Fundamental rows are accepted only from sources with exact "
            "announce_date semantics; estimated disclosure dates are not written."
        ),
    ),
    ColumnDoc(
        name="fund_net_profit",
        category="fundamental",
        formula="Most recently announced net profit attributable to shareholders (CNY)",
        inputs="fundamentals.net_profit (announce_date ≤ T)",
        known_at="Same as fund_revenue — as-of announce_date",
        warmup=0,
    ),
    ColumnDoc(
        name="fund_eps",
        category="fundamental",
        formula="Earnings per share (EPS) from most recent announcement",
        inputs="fundamentals.eps (announce_date ≤ T)",
        known_at="Same as fund_revenue",
        warmup=0,
    ),
    ColumnDoc(
        name="fund_roe",
        category="fundamental",
        formula="Return on equity (%) from most recent announcement",
        inputs="fundamentals.roe (announce_date ≤ T)",
        known_at="Same as fund_revenue",
        warmup=0,
    ),
    ColumnDoc(
        name="fund_lag_days",
        category="fundamental",
        formula="(feature_date T) − (announce_date of most recent announcement)",
        inputs="feature date T, fundamentals.announce_date",
        known_at="Derived at feature construction time",
        warmup=0,
        notes=(
            "High lag_days (> 90) indicates stale fundamentals.  "
            "Consider masking rows where fund_lag_days > 180."
        ),
    ),
]

LABEL_DOCS: list[ColumnDoc] = [
    ColumnDoc(
        name="ret_fwd_{h}d",
        category="label",
        formula="close(T+1+h) / close(T+1) - 1",
        inputs="close prices at T+1 and T+1+h",
        known_at=(
            "NOT known at T — this is a FUTURE value. "
            "T+1 execution assumption: earliest buy is at T+1 close."
        ),
        warmup=0,
        notes=(
            "Last h rows per symbol are NaN (no valid future window). "
            "These rows must be EXCLUDED from training — never imputed. "
            "Matches Qlib Alpha158 convention: Ref($close,-2)/Ref($close,-1)-1."
        ),
    ),
    ColumnDoc(
        name="ret_fwd_{h}d_cs",
        category="label",
        formula="Decile rank (0–9) of ret_fwd_{h}d within the cross-section on date T",
        inputs="ret_fwd_{h}d across universe at T",
        known_at="NOT known at T — future return used for ranking",
        warmup=0,
        notes=(
            "Cross-sectional label is safe to use because it ranks FUTURE returns "
            "against other stocks' FUTURE returns — no stock sees another's future."
        ),
    ),
    ColumnDoc(
        name="ret_fwd_{h}d_bin",
        category="label",
        formula="1 if ret_fwd_{h}d > cross-sectional median, else 0",
        inputs="ret_fwd_{h}d",
        known_at="NOT known at T — future",
        warmup=0,
    ),
    ColumnDoc(
        name="vol_fwd_{h}d",
        category="label",
        formula="std(daily returns from T+1 to T+1+h, ddof=1)",
        inputs="close prices T+1 … T+1+h",
        known_at="NOT known at T",
        warmup=0,
        notes="Risk label: forward realised volatility.",
    ),
    ColumnDoc(
        name="mdd_fwd_{h}d",
        category="label",
        formula="max drawdown of close prices from T+1 to T+1+h",
        inputs="close prices T+1 … T+1+h",
        known_at="NOT known at T",
        warmup=0,
        notes="Risk label: forward maximum drawdown (always ≤ 0).",
    ),
]


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

def build_data_dictionary(
    store_root: Path | str,
    horizons: list[int] | None = None,
) -> pd.DataFrame:
    """
    Build and write the data dictionary.

    Writes:
      <store_root>/data_dictionary.txt    (human-readable)
      <store_root>/data_dictionary.parquet (machine-readable)

    Returns the DataFrame for programmatic inspection.
    """
    store_root = Path(store_root)
    horizons   = horizons or [1, 5, 20]

    rows = []

    for doc in FEATURE_DOCS:
        rows.append({
            "name":     doc.name,
            "category": doc.category,
            "formula":  doc.formula,
            "inputs":   doc.inputs,
            "known_at": doc.known_at,
            "warmup":   doc.warmup,
            "notes":    doc.notes,
        })

    for doc in LABEL_DOCS:
        for h in horizons:
            rows.append({
                "name":     doc.name.replace("{h}", str(h)),
                "category": doc.category,
                "formula":  doc.formula.replace("{h}", str(h)),
                "inputs":   doc.inputs.replace("{h}", str(h)),
                "known_at": doc.known_at,
                "warmup":   doc.warmup,
                "notes":    doc.notes,
            })

    df = pd.DataFrame(rows)

    # Write Parquet
    parquet_path = store_root / "data_dictionary.parquet"
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path, index=False)

    # Write text
    txt_path = store_root / "data_dictionary.txt"
    lines = [
        "=" * 70,
        f"DATA DICTIONARY — generated {dt.datetime.now().isoformat(timespec='seconds')}",
        f"Total columns: {len(df)} ({df['category'].value_counts().to_dict()})",
        "=" * 70,
        "",
    ]
    for _, row in df.iterrows():
        lines += [
            f"[{row['category'].upper()}] {row['name']}",
            f"  Formula  : {row['formula']}",
            f"  Inputs   : {row['inputs']}",
            f"  Known at : {row['known_at']}",
            f"  Warmup   : {row['warmup']} rows",
        ]
        if row["notes"]:
            lines.append(f"  Notes    : {row['notes']}")
        lines.append("")

    txt_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(
        "Data dictionary written: %d columns → %s, %s",
        len(df), parquet_path, txt_path,
    )
    return df
