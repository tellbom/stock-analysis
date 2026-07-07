"""
features.event
==============
Event-driven feature builder (P4C-03).

Currently implements lockup expiry features from the silver lockup table.

PIT correctness (critical)
--------------------------
Upcoming lockup expiry dates are announced at IPO or at the time of the
private placement, so they are *public knowledge at all prior dates*.
Feature at date T:
  - ``days_to_next_unlock`` uses unlock_date > T only (strictly future)
  - ``unlock_size_ratio`` uses the NEXT unlock after T

Leakage trap to avoid: do NOT use unlock_date == T (the unlock itself).
The unlock event at date T is a contemporaneous event — we conservatively
treat it as unavailable until T+1.

Feature columns produced
------------------------
  days_to_next_unlock  int    Calendar days until the next lock-up expiry
                               after date T.  Default 999 when no unlock
                               event is scheduled within the next 180 days.
  unlock_size_ratio    float  shares_million(next_unlock) / float_mcap_yi
                               (supply pressure proxy).  0 when no unlock
                               within 30 days.  NaN when float_mcap unavailable.

Signal hypothesis
-----------------
Large imminent unlocks → supply pressure → negative short-horizon return.
Expected IC direction: negative vs ret_fwd_1d to ret_fwd_5d.

Spec list: LOCKUP_SPECS (for the feature registry)
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from quant_platform.features.registry import FeatureSpec
from quant_platform.core.logging import get_logger

logger = get_logger(__name__)

# Default horizon for unlock_size_ratio (only non-zero within this many days)
_RATIO_WINDOW_DAYS = 30

# Default value when no unlock is scheduled within MAX_FORWARD_DAYS
_MAX_FORWARD_DAYS = 180
_DEFAULT_DAYS     = 999


# ---------------------------------------------------------------------------
# FeatureSpec declarations
# ---------------------------------------------------------------------------

LOCKUP_SPECS: list[FeatureSpec] = [
    FeatureSpec(
        "days_to_next_unlock", "event",
        ("unlock_date",), 0,
        f"calendar_days_to_next_unlock (999 if none in {_MAX_FORWARD_DAYS}d)",
        warmup=0,
    ),
    FeatureSpec(
        "unlock_size_ratio", "event",
        ("unlock_date", "shares_million", "float_mcap_yi"), 0,
        f"shares_unlocking/float_mcap_yi (0 if no unlock in {_RATIO_WINDOW_DAYS}d)",
        warmup=0,
    ),
]


ANNOUNCEMENT_EVENT_SPECS: list[FeatureSpec] = [
    FeatureSpec("announcement_count_3d", "announcement_events", ("announce_date",), 3, "count announcements PIT 3d", warmup=0),
    FeatureSpec("announcement_count_5d", "announcement_events", ("announce_date",), 5, "count announcements PIT 5d", warmup=0),
    FeatureSpec("announcement_count_10d", "announcement_events", ("announce_date",), 10, "count announcements PIT 10d", warmup=0),
    FeatureSpec("has_announcement_3d", "announcement_events", ("announce_date",), 3, "any announcement PIT 3d", warmup=0),
    FeatureSpec("has_major_event_10d", "announcement_events", ("title", "category"), 10, "major event announcement PIT 10d", warmup=0),
    FeatureSpec("has_risk_announcement_5d", "announcement_events", ("title", "category"), 5, "risk announcement PIT 5d", warmup=0),
    FeatureSpec("has_financial_report_30d", "announcement_events", ("title", "category"), 30, "financial report announcement PIT 30d", warmup=0),
    FeatureSpec("has_reduction_notice_30d", "announcement_events", ("title", "category"), 30, "reduction notice PIT 30d", warmup=0),
]

DRAGON_TIGER_SPECS: list[FeatureSpec] = [
    FeatureSpec("has_dragon_tiger_5d", "dragon_tiger", ("available_date",), 5, "any dragon tiger event PIT 5d", warmup=0),
    FeatureSpec("dragon_tiger_count_10d", "dragon_tiger", ("available_date",), 10, "count dragon tiger events PIT 10d", warmup=0),
    FeatureSpec("dragon_tiger_net_buy_5d", "dragon_tiger", ("net_buy_amount",), 5, "sum net buy PIT 5d", warmup=0),
    FeatureSpec("dragon_tiger_net_buy_rank_5d", "dragon_tiger", ("net_buy_amount",), 5, "cross-sectional rank of 5d net buy", warmup=0),
    FeatureSpec("institution_net_buy_5d", "dragon_tiger", ("institution_buy_amount", "institution_sell_amount"), 5, "sum institution net buy PIT 5d", warmup=0),
    FeatureSpec("institution_net_buy_rank_5d", "dragon_tiger", ("institution_buy_amount", "institution_sell_amount"), 5, "cross-sectional rank of institution net buy", warmup=0),
]

BLOCK_TRADE_SPECS: list[FeatureSpec] = [
    FeatureSpec("block_trade_count_20d", "block_trade", ("available_date",), 20, "count block trades PIT 20d", warmup=0),
    FeatureSpec("block_trade_amount_20d", "block_trade", ("amount",), 20, "sum block trade amount PIT 20d", warmup=0),
    FeatureSpec("block_trade_amount_rank_20d", "block_trade", ("amount",), 20, "cross-sectional rank of block amount", warmup=0),
    FeatureSpec("block_trade_discount_mean_20d", "block_trade", ("discount_rate",), 20, "mean discount rate PIT 20d", warmup=0),
    FeatureSpec("has_large_discount_block_trade_20d", "block_trade", ("discount_rate",), 20, "discount <= -10% PIT 20d", warmup=0),
]

EVENT3_SPECS: list[FeatureSpec] = ANNOUNCEMENT_EVENT_SPECS + DRAGON_TIGER_SPECS + BLOCK_TRADE_SPECS


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_lockup_features(
    panel: pd.DataFrame,
    lockup_panel: pd.DataFrame,
    valuation_panel: pd.DataFrame | None = None,
    max_forward_days: int = _MAX_FORWARD_DAYS,
    ratio_window_days: int = _RATIO_WINDOW_DAYS,
) -> pd.DataFrame:
    """
    Add lock-up expiry features to the universe panel.

    Parameters
    ----------
    panel : pd.DataFrame
        Universe panel with [symbol, date, ...].
    lockup_panel : pd.DataFrame
        Concatenated silver lockup data for all symbols.
        Columns: symbol, unlock_date, lock_type, shares_million, ratio_pct.
    valuation_panel : pd.DataFrame | None
        For float_mcap_yi normalisation.  If None, unlock_size_ratio
        uses shares_million directly (unnormalised, less informative).
    max_forward_days : int
        ``days_to_next_unlock`` is capped at this value when no event is
        found within the window.
    ratio_window_days : int
        ``unlock_size_ratio`` is only non-zero when the next unlock is
        within this many days.

    Returns
    -------
    pd.DataFrame
        Panel with lockup feature columns added.
    """
    df = panel.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date

    if lockup_panel.empty:
        logger.info("build_lockup_features: empty lockup panel — features will default to 999/0")
        df["days_to_next_unlock"] = _DEFAULT_DAYS
        df["unlock_size_ratio"]   = 0.0
        return df

    lk = lockup_panel.copy()
    lk["unlock_date"] = pd.to_datetime(lk["unlock_date"]).dt.date

    # Merge valuation for float_mcap
    if valuation_panel is not None and not valuation_panel.empty:
        val = valuation_panel.copy()
        val["date"] = pd.to_datetime(val["date"]).dt.date
        mcap = val[["symbol", "date", "float_mcap_yi"]].copy()
        df = df.merge(mcap, on=["symbol", "date"], how="left")
    else:
        df["float_mcap_yi"] = np.nan

    # For each (symbol, date) row, find the next upcoming unlock after T
    # Vectorised approach: for each symbol, precompute sorted unlock dates
    # then join by nearest future date.

    days_vals  = np.full(len(df), _DEFAULT_DAYS, dtype=float)
    ratio_vals = np.zeros(len(df), dtype=float)

    # Group lockups by symbol for fast lookup
    lk_by_sym: dict[str, pd.DataFrame] = {
        sym: grp.sort_values("unlock_date")
        for sym, grp in lk.groupby("symbol")
    }

    df_np_date = df["date"].values
    df_np_sym  = df["symbol"].values
    df_np_mcap = df["float_mcap_yi"].values if "float_mcap_yi" in df.columns else np.full(len(df), np.nan)

    for idx in range(len(df)):
        sym  = df_np_sym[idx]
        date = df_np_date[idx]

        if sym not in lk_by_sym:
            continue

        sym_lk = lk_by_sym[sym]
        # Strictly future: unlock_date > date (not >= date)
        future = sym_lk[sym_lk["unlock_date"] > date]
        if future.empty:
            continue

        next_event = future.iloc[0]
        days = (next_event["unlock_date"] - date).days

        if days <= max_forward_days:
            days_vals[idx] = float(days)

        if days <= ratio_window_days:
            shares = float(next_event.get("shares_million", 0) or 0)
            mcap   = float(df_np_mcap[idx])
            if not np.isnan(mcap) and mcap > 0:
                # Convert shares_million to 亿股 for consistency: 1M shares = 0.01亿股
                # float_mcap_yi is in 亿元; assume price ~ 10-50 CNY/share
                # Better: use ratio_pct directly when available
                ratio_pct = float(next_event.get("ratio_pct", 0) or 0)
                ratio_vals[idx] = ratio_pct / 100.0   # as a fraction
            elif shares > 0:
                ratio_vals[idx] = shares   # fallback: absolute millions of shares

    df["days_to_next_unlock"] = days_vals
    df["unlock_size_ratio"]   = ratio_vals

    # Drop working column
    if "float_mcap_yi" in df.columns and "float_mcap_yi" not in panel.columns:
        df = df.drop(columns=["float_mcap_yi"])

    n_upcoming = (df["days_to_next_unlock"] < _DEFAULT_DAYS).sum()
    logger.info(
        "build_lockup_features: %d rows with upcoming unlock (within %dd)",
        n_upcoming, max_forward_days,
    )
    return df.sort_values(["date", "symbol"]).reset_index(drop=True)


def load_lockup_panel(
    store_root: Path | str,
    symbols: list[str],
) -> pd.DataFrame:
    """Load and concatenate silver lockup Parquets for all symbols."""
    from quant_platform.store.lake import lockup_path

    store_root = Path(store_root)
    frames = []
    for symbol in symbols:
        p = lockup_path(store_root, symbol)
        if p.exists():
            try:
                df = pd.read_parquet(p)
                df["unlock_date"] = pd.to_datetime(df["unlock_date"]).dt.date
                frames.append(df)
            except Exception as exc:
                logger.warning("Could not load lockup for %s: %s", symbol, exc)

    if not frames:
        return pd.DataFrame()
    return (
        pd.concat(frames, ignore_index=True)
          .sort_values(["symbol", "unlock_date"])
          .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Event3 builder
# ---------------------------------------------------------------------------

_MAJOR_TOKENS = ("重大", "重组", "收购", "并购", "控制权", "停牌", "复牌", "诉讼", "仲裁")
_RISK_TOKENS = ("风险", "退市", "st", "处罚", "问询", "监管", "立案", "诉讼", "冻结", "质押")
_FINANCIAL_TOKENS = ("年度报告", "半年度报告", "季度报告", "财务报告", "业绩快报", "业绩预告")
_REDUCTION_TOKENS = ("减持", "持股变动", "股份变动")


def _contains_any(text: object, tokens: tuple[str, ...]) -> bool:
    value = "" if pd.isna(text) else str(text).lower()
    return any(tok.lower() in value for tok in tokens)


def _window_mask(dates: pd.Series, as_of: dt.date, days: int) -> pd.Series:
    start = as_of - dt.timedelta(days=days - 1)
    return (dates >= start) & (dates <= as_of)


def _rank_nonzero_by_date(df: pd.DataFrame, col: str, out_col: str) -> None:
    def _rank(s: pd.Series) -> pd.Series:
        out = pd.Series(0.0, index=s.index)
        nz = s.fillna(0) != 0
        if nz.any():
            out.loc[nz] = s.loc[nz].rank(method="average", ascending=True, pct=True)
        return out

    df[out_col] = df.groupby("date")[col].transform(_rank)


def build_event3_features(
    panel: pd.DataFrame,
    announcement_panel: pd.DataFrame | None = None,
    dragon_tiger_panel: pd.DataFrame | None = None,
    block_trade_panel: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Add recent-only announcement, dragon tiger and block trade features."""
    df = panel.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    for spec in EVENT3_SPECS:
        df[spec.name] = 0.0

    ann = announcement_panel.copy() if announcement_panel is not None else pd.DataFrame()
    dtg = dragon_tiger_panel.copy() if dragon_tiger_panel is not None else pd.DataFrame()
    blk = block_trade_panel.copy() if block_trade_panel is not None else pd.DataFrame()

    if not ann.empty:
        ann["announce_date"] = pd.to_datetime(ann["announce_date"], errors="coerce").dt.date
        ann["_text"] = (
            ann.get("title", "").fillna("").astype(str)
            + " "
            + ann.get("category", "").fillna("").astype(str)
        )
        ann["_is_major"] = ann["_text"].map(lambda x: _contains_any(x, _MAJOR_TOKENS))
        ann["_is_risk"] = ann["_text"].map(lambda x: _contains_any(x, _RISK_TOKENS))
        ann["_is_financial"] = ann["_text"].map(lambda x: _contains_any(x, _FINANCIAL_TOKENS))
        ann["_is_reduction"] = ann["_text"].map(lambda x: _contains_any(x, _REDUCTION_TOKENS))

    if not dtg.empty:
        dtg["available_date"] = pd.to_datetime(dtg["available_date"], errors="coerce").dt.date
        for col in ("net_buy_amount", "institution_buy_amount", "institution_sell_amount"):
            if col not in dtg.columns:
                dtg[col] = 0.0
            dtg[col] = pd.to_numeric(dtg[col], errors="coerce").fillna(0.0)
        dtg["_institution_net"] = dtg["institution_buy_amount"] - dtg["institution_sell_amount"]

    if not blk.empty:
        blk["available_date"] = pd.to_datetime(blk["available_date"], errors="coerce").dt.date
        for col in ("amount", "discount_rate"):
            if col not in blk.columns:
                blk[col] = 0.0
            blk[col] = pd.to_numeric(blk[col], errors="coerce")

    ann_by_sym = {sym: grp.dropna(subset=["announce_date"]) for sym, grp in ann.groupby("symbol")} if not ann.empty else {}
    dtg_by_sym = {sym: grp.dropna(subset=["available_date"]) for sym, grp in dtg.groupby("symbol")} if not dtg.empty else {}
    blk_by_sym = {sym: grp.dropna(subset=["available_date"]) for sym, grp in blk.groupby("symbol")} if not blk.empty else {}

    for idx, row in df.iterrows():
        sym = str(row["symbol"]).zfill(6)
        as_of = row["date"]

        a = ann_by_sym.get(sym)
        if a is not None and not a.empty:
            for days, col in [(3, "announcement_count_3d"), (5, "announcement_count_5d"), (10, "announcement_count_10d")]:
                mask = _window_mask(a["announce_date"], as_of, days)
                df.at[idx, col] = float(mask.sum())
            df.at[idx, "has_announcement_3d"] = float(df.at[idx, "announcement_count_3d"] > 0)
            df.at[idx, "has_major_event_10d"] = float((_window_mask(a["announce_date"], as_of, 10) & a["_is_major"]).any())
            df.at[idx, "has_risk_announcement_5d"] = float((_window_mask(a["announce_date"], as_of, 5) & a["_is_risk"]).any())
            df.at[idx, "has_financial_report_30d"] = float((_window_mask(a["announce_date"], as_of, 30) & a["_is_financial"]).any())
            df.at[idx, "has_reduction_notice_30d"] = float((_window_mask(a["announce_date"], as_of, 30) & a["_is_reduction"]).any())

        d = dtg_by_sym.get(sym)
        if d is not None and not d.empty:
            mask5 = _window_mask(d["available_date"], as_of, 5)
            mask10 = _window_mask(d["available_date"], as_of, 10)
            df.at[idx, "has_dragon_tiger_5d"] = float(mask5.any())
            df.at[idx, "dragon_tiger_count_10d"] = float(mask10.sum())
            df.at[idx, "dragon_tiger_net_buy_5d"] = float(d.loc[mask5, "net_buy_amount"].sum())
            df.at[idx, "institution_net_buy_5d"] = float(d.loc[mask5, "_institution_net"].sum())

        b = blk_by_sym.get(sym)
        if b is not None and not b.empty:
            mask20 = _window_mask(b["available_date"], as_of, 20)
            recent = b.loc[mask20]
            df.at[idx, "block_trade_count_20d"] = float(len(recent))
            df.at[idx, "block_trade_amount_20d"] = float(recent["amount"].fillna(0).sum())
            discounts = recent["discount_rate"].dropna()
            df.at[idx, "block_trade_discount_mean_20d"] = float(discounts.mean()) if not discounts.empty else 0.0
            df.at[idx, "has_large_discount_block_trade_20d"] = float((discounts <= -0.10).any()) if not discounts.empty else 0.0

    _rank_nonzero_by_date(df, "dragon_tiger_net_buy_5d", "dragon_tiger_net_buy_rank_5d")
    _rank_nonzero_by_date(df, "institution_net_buy_5d", "institution_net_buy_rank_5d")
    _rank_nonzero_by_date(df, "block_trade_amount_20d", "block_trade_amount_rank_20d")
    return df.sort_values(["date", "symbol"]).reset_index(drop=True)


def _load_event_panel(store_root: Path | str, symbols: list[str], path_func, date_cols: tuple[str, ...]) -> pd.DataFrame:
    frames = []
    for symbol in symbols:
        path = path_func(store_root, symbol)
        if path.exists():
            try:
                df = pd.read_parquet(path)
                for col in date_cols:
                    if col in df.columns:
                        df[col] = pd.to_datetime(df[col], errors="coerce").dt.date
                frames.append(df)
            except Exception as exc:
                logger.warning("Could not load event3 data for %s from %s: %s", symbol, path, exc)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_announcement_events_panel(store_root: Path | str, symbols: list[str]) -> pd.DataFrame:
    from quant_platform.store.lake import announcement_events_path

    return _load_event_panel(store_root, symbols, announcement_events_path, ("announce_date", "event_date"))


def load_dragon_tiger_panel(store_root: Path | str, symbols: list[str]) -> pd.DataFrame:
    from quant_platform.store.lake import dragon_tiger_path

    return _load_event_panel(store_root, symbols, dragon_tiger_path, ("trade_date", "available_date"))


def load_block_trade_panel(store_root: Path | str, symbols: list[str]) -> pd.DataFrame:
    from quant_platform.store.lake import block_trade_path

    return _load_event_panel(store_root, symbols, block_trade_path, ("trade_date", "available_date"))
