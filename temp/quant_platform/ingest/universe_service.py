"""
ingest.universe_service
=======================
Universe Service: fetch, persist, and query index constituents.

T0.2 scope
----------
- Fetch current CSI 300 constituents via AKShare (fail loudly if unavailable).
- Persist to Parquet with an effective-date schema so survivorship-aware queries
  are possible once historical data is added.
- Expose ``get_symbols_as_of(date)`` — returns the correct member list for any
  past date.  In v1 (``has_effective_dates=False``) this always returns current
  members and tags the result with a survivorship-bias warning.
- All AKShare calls go through ``core.fetch.safe_call``; never fabricate data.

Effective-date schema
---------------------
Parquet file: ``<store_root>/universe/<universe_key>/membership.parquet``

Columns
  symbol      str       6-digit code, e.g. "600519"
  in_date     date      first day the symbol is a constituent  (inclusive)
  out_date    date|None last day the symbol is a constituent   (inclusive)
                        NULL means "still a member today"
  name        str       display name at time of entry

When ``has_effective_dates=False`` the service writes:
  in_date  = fetch_date (today)
  out_date = None        (open-ended, i.e. still in index)

This is structurally correct but historically incomplete — the quality report
(T0.8) must surface this gap.

Fetch strategy
--------------
AKShare endpoints tried in order:
  1. ``index_stock_cons_csindex(symbol=index_code)``   — csindex.com.cn XLS
  2. ``index_stock_cons(symbol=index_code)``           — Sina HTML scrape

Both may be blocked depending on the runtime environment (403 / network policy).
If ALL endpoints fail the service raises ``UniverseFetchError`` — no silent
fallback or fabricated data.
"""

from __future__ import annotations

import datetime as dt
import os
from pathlib import Path
from typing import Optional

import pandas as pd

from quant_platform.core.fetch import safe_call
from quant_platform.core.logging import get_logger
from quant_platform.core.universe import UniverseConfig, get_universe

logger = get_logger(__name__)


class UniverseFetchError(RuntimeError):
    """Raised when all AKShare endpoints fail for a universe fetch."""


# ---------------------------------------------------------------------------
# Internal: AKShare fetch attempts
# ---------------------------------------------------------------------------

def _fetch_csindex(index_code: str) -> pd.DataFrame | None:
    """Try csindex.com.cn XLS endpoint."""
    try:
        import akshare as ak
    except ImportError:
        logger.error("akshare is not installed — cannot fetch constituents")
        return None

    df = safe_call(
        ak.index_stock_cons_csindex,
        symbol=index_code,
        label=f"csindex constituents {index_code}",
        retries=2,
    )
    if df is None or df.empty:
        return None

    # Normalise columns: csindex returns 成分券代码 / 成分券名称
    code_col = next((c for c in df.columns if "成分券代码" in c or "code" in c.lower()), None)
    name_col = next((c for c in df.columns if "成分券名称" in c or "name" in c.lower()), None)
    if code_col is None:
        logger.warning("csindex response missing constituent code column; columns: %s", list(df.columns))
        return None

    out = pd.DataFrame({
        "symbol": df[code_col].astype(str).str.strip().str.zfill(6),
        "name":   df[name_col].astype(str).str.strip() if name_col else "",
    })
    return out[out["symbol"].str.len() == 6].reset_index(drop=True)


def _fetch_sina(index_code: str) -> pd.DataFrame | None:
    """Try Sina HTML scrape endpoint."""
    try:
        import akshare as ak
    except ImportError:
        return None

    df = safe_call(
        ak.index_stock_cons,
        symbol=index_code,
        label=f"sina constituents {index_code}",
        retries=2,
    )
    if df is None or df.empty:
        return None

    # Sina returns 品种代码 / 品种名称 (or similar)
    code_col = next(
        (c for c in df.columns if any(k in c for k in ("代码", "code", "Code"))),
        None,
    )
    name_col = next(
        (c for c in df.columns if any(k in c for k in ("名称", "name", "Name"))),
        None,
    )
    if code_col is None:
        logger.warning("Sina response missing code column; columns: %s", list(df.columns))
        return None

    out = pd.DataFrame({
        "symbol": df[code_col].astype(str).str.strip().str.zfill(6),
        "name":   df[name_col].astype(str).str.strip() if name_col else "",
    })
    return out[out["symbol"].str.len() == 6].reset_index(drop=True)


def _fetch_constituents(cfg: UniverseConfig) -> pd.DataFrame:
    """
    Fetch current constituents for *cfg*.

    Tries each AKShare endpoint in order; raises UniverseFetchError if all fail.
    Returns DataFrame with columns: symbol (str), name (str).
    """
    if cfg.index_code is None:
        raise UniverseFetchError(
            f"Universe '{cfg.key}' has no index_code; "
            "manual constituent population is not yet implemented."
        )

    attempts = [
        ("csindex", _fetch_csindex),
        ("sina",    _fetch_sina),
    ]

    for source_name, fn in attempts:
        logger.info("Trying %s for universe '%s' (index %s) …", source_name, cfg.key, cfg.index_code)
        try:
            result = fn(cfg.index_code)
        except Exception as exc:
            logger.warning("%s raised unexpectedly: %s", source_name, exc)
            result = None

        if result is not None and not result.empty:
            logger.info(
                "Fetched %d constituents for '%s' via %s",
                len(result), cfg.key, source_name,
            )
            return result

        logger.warning("%s returned no data for '%s'", source_name, cfg.key)

    raise UniverseFetchError(
        f"All AKShare endpoints failed for universe '{cfg.key}' "
        f"(index_code='{cfg.index_code}'). "
        "This is likely a network restriction (403) in the current environment. "
        "Fix the network or provide constituent data manually via "
        "UniverseService.load_from_csv()."
    )


# ---------------------------------------------------------------------------
# Membership Parquet helpers
# ---------------------------------------------------------------------------

def _membership_path(store_root: Path, universe_key: str) -> Path:
    return store_root / "universe" / universe_key / "membership.parquet"


def _read_membership(path: Path) -> pd.DataFrame:
    """Read existing membership Parquet, or return empty frame with correct schema."""
    schema_empty = pd.DataFrame({
        "symbol":   pd.Series(dtype="str"),
        "in_date":  pd.Series(dtype="object"),   # date objects
        "out_date": pd.Series(dtype="object"),
        "name":     pd.Series(dtype="str"),
        "source":   pd.Series(dtype="str"),
    })
    if not path.exists():
        return schema_empty
    try:
        df = pd.read_parquet(path)
        return df
    except Exception as exc:
        logger.error("Failed to read membership file %s: %s", path, exc)
        return schema_empty


def _write_membership(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info("Membership written → %s (%d rows)", path, len(df))


# ---------------------------------------------------------------------------
# Public: UniverseService
# ---------------------------------------------------------------------------

class UniverseService:
    """
    Manages constituent membership for one universe.

    Parameters
    ----------
    universe_key : str
        One of the keys in ``UNIVERSE_REGISTRY``, e.g. ``"csi300"``.
    store_root : Path | str
        Root of the Parquet data lake (same root used by other store modules).
    """

    def __init__(self, universe_key: str, store_root: Path | str) -> None:
        self.cfg = get_universe(universe_key)          # raises KeyError for unknown keys
        self.store_root = Path(store_root)
        self._membership_path = _membership_path(self.store_root, self.cfg.key)

    # ------------------------------------------------------------------
    # Fetch & persist
    # ------------------------------------------------------------------

    def fetch_and_save(self) -> pd.DataFrame:
        """
        Fetch current constituents from AKShare and append to membership Parquet.

        - Raises ``UniverseFetchError`` if all AKShare endpoints fail.
        - Existing open-ended rows (out_date IS NULL) for symbols that have LEFT
          the index are closed as of yesterday.
        - New symbols get in_date = today.
        - Symbols unchanged carry over with their original in_date.

        Returns the full membership DataFrame after the update.
        """
        fetch_date = dt.date.today()
        raw = _fetch_constituents(self.cfg)   # raises on failure — never fabricates
        new_symbols = set(raw["symbol"].tolist())

        existing = _read_membership(self._membership_path)

        if existing.empty:
            # First fetch — write current members as open-ended rows
            rows = raw.copy()
            rows["in_date"]  = fetch_date
            rows["out_date"] = None
            rows["source"]   = "current_only"
            membership = rows[["symbol", "in_date", "out_date", "name", "source"]]
            logger.info(
                "First fetch for '%s': %d symbols, in_date=%s, out_date=None (open)",
                self.cfg.key, len(membership), fetch_date,
            )
        else:
            # Close rows for symbols that have left the index
            still_open = existing["out_date"].isna()
            was_member = set(existing.loc[still_open, "symbol"].tolist())

            removed = was_member - new_symbols
            added   = new_symbols - was_member

            if removed:
                yesterday = fetch_date - dt.timedelta(days=1)
                mask = still_open & existing["symbol"].isin(removed)
                existing.loc[mask, "out_date"] = yesterday
                logger.info("Closed %d symbols removed from %s: %s", len(removed), self.cfg.key, sorted(removed)[:5])

            if added:
                name_map = raw.set_index("symbol")["name"].to_dict()
                new_rows = pd.DataFrame([
                    {"symbol": s, "in_date": fetch_date, "out_date": None,
                     "name": name_map.get(s, ""), "source": "current_only"}
                    for s in sorted(added)
                ])
                existing = pd.concat([existing, new_rows], ignore_index=True)
                logger.info("Added %d new symbols to %s: %s", len(added), self.cfg.key, sorted(added)[:5])

            membership = existing

        _write_membership(membership, self._membership_path)
        return membership

    def load_from_csv(self, csv_path: str | Path, *, has_effective_dates: bool = False) -> pd.DataFrame:
        """
        Load constituents from a user-supplied CSV instead of AKShare.

        Expected columns: ``symbol`` (required), ``name`` (optional),
        ``in_date`` (optional), ``out_date`` (optional).

        Use this when AKShare is unavailable (e.g. network restrictions).
        Set ``has_effective_dates=True`` only when the CSV contains real
        historical in/out dates — this controls the survivorship-bias flag
        in quality reports.
        """
        df = pd.read_csv(csv_path, dtype=str)
        if "symbol" not in df.columns:
            raise ValueError(f"CSV must have a 'symbol' column; found: {list(df.columns)}")

        df["symbol"] = df["symbol"].str.strip().str.zfill(6)

        today = dt.date.today()
        if "in_date" not in df.columns:
            df["in_date"] = today
        if "out_date" not in df.columns:
            df["out_date"] = None
        if "name" not in df.columns:
            df["name"] = ""

        df["source"] = "csv_import"
        membership = df[["symbol", "in_date", "out_date", "name", "source"]].copy()
        _write_membership(membership, self._membership_path)

        # Persist the has_effective_dates flag alongside the data
        flag_path = self._membership_path.parent / "meta.txt"
        flag_path.write_text(f"has_effective_dates={has_effective_dates}\n")

        logger.info(
            "Loaded %d symbols from CSV '%s' (has_effective_dates=%s)",
            len(membership), csv_path, has_effective_dates,
        )
        return membership

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_symbols_as_of(self, as_of: dt.date | None = None) -> list[str]:
        """
        Return the list of constituent symbols as of *as_of* (defaults to today).

        When ``has_effective_dates=False`` (v1), returns current members and
        logs a survivorship-bias warning — the caller receives real data, not
        a lie, but should understand the limitation.

        Raises FileNotFoundError if membership has never been fetched.
        """
        as_of = as_of or dt.date.today()

        if not self._membership_path.exists():
            raise FileNotFoundError(
                f"Membership file not found for universe '{self.cfg.key}'. "
                "Run fetch_and_save() first (or load_from_csv() if AKShare is unavailable)."
            )

        df = _read_membership(self._membership_path)

        # Check effective-date capability
        has_eff = self._has_effective_dates()
        if not has_eff:
            logger.warning(
                "Universe '%s' does not have historical effective dates. "
                "get_symbols_as_of(%s) returns CURRENT members — "
                "results may be subject to survivorship bias. "
                "%s",
                self.cfg.key, as_of, self.cfg.survivorship_note,
            )
            # Still apply date logic against what we have (in_date filter at minimum)
        
        # Members as of as_of:
        #   in_date <= as_of  AND  (out_date IS NULL OR out_date >= as_of)
        in_date_col  = pd.to_datetime(df["in_date"]).dt.date
        out_date_raw = df["out_date"]

        def _to_date(v):
            if v is None or (isinstance(v, float) and pd.isna(v)):
                return None
            try:
                return pd.to_datetime(v).date()
            except Exception:
                return None

        out_dates = out_date_raw.apply(_to_date)

        mask = (
            (in_date_col <= as_of)
            & (out_dates.isna() | (out_dates >= as_of))
        )
        symbols = sorted(df.loc[mask, "symbol"].unique().tolist())
        logger.info(
            "Universe '%s' as of %s: %d symbols", self.cfg.key, as_of, len(symbols)
        )
        return symbols

    def membership_df(self) -> pd.DataFrame:
        """Return the raw membership DataFrame (all rows, all history)."""
        return _read_membership(self._membership_path)

    def _has_effective_dates(self) -> bool:
        """Check the persisted has_effective_dates flag."""
        flag_path = self._membership_path.parent / "meta.txt"
        if not flag_path.exists():
            return False
        content = flag_path.read_text()
        return "has_effective_dates=True" in content

    def survivorship_status(self) -> dict:
        """
        Return a dict summarising survivorship-bias status for quality reports.
        Always call this from T0.8 quality report generation.
        """
        has_eff = self._has_effective_dates()
        return {
            "universe_key":        self.cfg.key,
            "has_effective_dates": has_eff,
            "survivorship_bias_risk": not has_eff,
            "note": (
                self.cfg.survivorship_note if not has_eff
                else "Historical membership tracked — survivorship bias mitigated."
            ),
        }
