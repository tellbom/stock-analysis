"""
ingest.industry_collector
=========================
A-share industry classification collector (P4B-03).

Stores industry membership as a slowly-changing-dimension (SCD) table so
that historical label construction can look up the correct industry for
any past date — critical for industry-neutral labels and sector-relative
features.

Data sources
------------
1. **Eastmoney push2 ``eastmoney_stock_info``** (primary): returns the
   ``industry`` field (东财行业名, e.g. "食品饮料") for each stock.
   Rate-limited with a 1-second minimum interval via ``_em_get()``.

2. **Eastmoney slist ``eastmoney_concept_blocks``** (supplemental):
   returns all board/concept/region tags for a stock (BK codes + names).
   Used to populate ``concept_tags``.

SCD design
----------
Each row represents an (symbol, effective_date) record.  When the
industry changes, a new row is inserted with the new ``effective_date``;
the previous row's ``out_date`` is set to that date.  A NULL ``out_date``
means the record is still current.

Silver table: ``silver/industry_map.parquet``
Schema:  symbol, industry_code, industry_name, concept_tags,
         effective_date, out_date

PIT query
---------
    def get_industry_as_of(df: pd.DataFrame, symbol: str, as_of: dt.date) -> str:
        rows = df[(df.symbol == symbol)
                  & (df.effective_date <= as_of)
                  & ((df.out_date.isna()) | (df.out_date > as_of))]
        return rows.iloc[-1]["industry_code"] if len(rows) else ""

Usage
-----
    from quant_platform.ingest.industry_collector import IndustryCollector
    ic = IndustryCollector(store_root=Path("/data/lake"))
    ic.run(symbols=csi300_symbols)
"""

from __future__ import annotations

import datetime as dt
import random
import time
from pathlib import Path

import pandas as pd
import requests

from quant_platform.core.logging import get_logger
from quant_platform.store.lake import industry_map_path, init_lake
from quant_platform.store.schemas import enforce_industry_map

logger = get_logger(__name__)

_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"

# Eastmoney throttle — max 1 req/sec, random jitter
_EM_MIN_INTERVAL = 1.0
_em_last_call: list[float] = [0.0]
_em_session = requests.Session()
_em_session.headers.update({"User-Agent": _UA})


def _em_get(url: str, params: dict | None = None, timeout: int = 15) -> requests.Response:
    """Throttled Eastmoney GET — enforces ≥1s between calls."""
    wait = _EM_MIN_INTERVAL - (time.time() - _em_last_call[0])
    if wait > 0:
        time.sleep(wait + random.uniform(0.1, 0.4))
    try:
        return _em_session.get(url, params=params, timeout=timeout)
    finally:
        _em_last_call[0] = time.time()


def _fetch_em_stock_info(code: str) -> dict:
    """
    Fetch basic info for one stock from Eastmoney push2 API.
    Returns {industry_name, industry_code} or empty dict on failure.
    """
    market = 1 if code.startswith("6") else 0
    url = "https://push2.eastmoney.com/api/qt/stock/get"
    params = {
        "fltt": "2", "invt": "2",
        "fields": "f57,f58,f127,f128",
        "secid": f"{market}.{code}",
    }
    try:
        r = _em_get(url, params=params)
        d = r.json().get("data") or {}
        return {
            "industry_name": str(d.get("f127", "") or ""),
            "industry_code": str(d.get("f128", "") or ""),
        }
    except Exception as exc:
        logger.debug("%s: eastmoney_stock_info failed: %s", code, exc)
        return {}


def _fetch_em_concept_blocks(code: str) -> list[str]:
    """
    Fetch all board/concept/region tags for one stock (Eastmoney slist).
    Returns a list of board names (e.g. ["食品饮料", "白酒Ⅲ", "贵州板块"]).
    """
    market = 1 if code.startswith("6") else 0
    url = "https://push2.eastmoney.com/api/qt/slist/get"
    params = {
        "fltt": "2", "invt": "2",
        "secid": f"{market}.{code}",
        "spt": "3", "pi": "0", "pz": "200", "po": "1",
        "fields": "f12,f14",
    }
    try:
        r = _em_get(url, params=params,
                    headers={"Referer": "https://quote.eastmoney.com/"})
        diff = (r.json().get("data") or {}).get("diff") or {}
        items = diff.values() if isinstance(diff, dict) else diff
        return [it.get("f14", "") for it in items if it.get("f14")]
    except Exception as exc:
        logger.debug("%s: eastmoney_concept_blocks failed: %s", code, exc)
        return []


class IndustryCollector:
    """
    Collect and maintain the industry classification SCD table.

    Parameters
    ----------
    store_root : Path | str
    fetch_concepts : bool
        Also fetch concept/board tags (doubles the Eastmoney calls).
        Default True — worth the extra time for downstream sector features.
    """

    def __init__(
        self,
        store_root: Path | str,
        fetch_concepts: bool = True,
    ) -> None:
        self.store_root    = Path(store_root)
        self.fetch_concepts = fetch_concepts
        init_lake(self.store_root)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        symbols: list[str],
        as_of: dt.date | None = None,
    ) -> pd.DataFrame:
        """
        Refresh the industry classification for all symbols.

        For each symbol, the current industry is fetched from Eastmoney.
        If it differs from the last stored record for that symbol, a new
        SCD row is inserted with today as ``effective_date`` and the
        previous row's ``out_date`` is set.

        Parameters
        ----------
        symbols : list[str]
            6-digit A-share codes.
        as_of : date | None
            Date to record as ``effective_date``.  Defaults to today.

        Returns
        -------
        pd.DataFrame
            The updated industry map table.
        """
        as_of = as_of or dt.date.today()
        logger.info("IndustryCollector.run: %d symbols, as_of=%s", len(symbols), as_of)

        existing = self._load()
        new_rows: list[dict] = []
        changes: int = 0

        for i, symbol in enumerate(symbols):
            info = _fetch_em_stock_info(symbol)
            if not info.get("industry_name"):
                logger.debug("%s: no industry returned", symbol)
                continue

            industry_name = info["industry_name"]
            industry_code = info.get("industry_code", "") or _name_to_code(industry_name)

            # Fetch concept tags (optional)
            concept_tags_str = ""
            if self.fetch_concepts:
                tags = _fetch_em_concept_blocks(symbol)
                concept_tags_str = "|".join(tags[:30])   # cap at 30 to limit size

            # Check against last known record
            sym_rows = existing[existing["symbol"] == symbol]
            current = sym_rows[sym_rows["out_date"].isna()]

            if not current.empty:
                last_industry = current.iloc[-1]["industry_name"]
                if last_industry == industry_name:
                    # No change; optionally refresh concept_tags in place
                    if self.fetch_concepts and concept_tags_str:
                        idx = current.index[-1]
                        existing.at[idx, "concept_tags"] = concept_tags_str
                    continue

                # Industry changed: close the old record
                idx_to_close = current.index[-1]
                existing.at[idx_to_close, "out_date"] = as_of
                changes += 1

            # Insert new row
            new_rows.append({
                "symbol":         symbol,
                "industry_code":  industry_code,
                "industry_name":  industry_name,
                "concept_tags":   concept_tags_str,
                "effective_date": as_of,
                "out_date":       None,
            })

            if (i + 1) % 50 == 0:
                logger.info("  ... %d/%d processed (%d changes so far)", i + 1, len(symbols), changes)

        if new_rows:
            new_df = pd.DataFrame(new_rows)
            combined = pd.concat([existing, new_df], ignore_index=True)
        else:
            combined = existing

        combined = enforce_industry_map(combined)
        self._save(combined)
        logger.info(
            "IndustryCollector done: %d new/changed records, total %d rows",
            len(new_rows), len(combined),
        )
        return combined

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load(self) -> pd.DataFrame:
        p = industry_map_path(self.store_root)
        if not p.exists():
            return pd.DataFrame(columns=[
                "symbol", "industry_code", "industry_name",
                "concept_tags", "effective_date", "out_date",
            ])
        df = pd.read_parquet(p)
        df["effective_date"] = pd.to_datetime(df["effective_date"]).dt.date
        if "out_date" in df.columns:
            df["out_date"] = pd.to_datetime(df["out_date"], errors="coerce").dt.date
        return df

    def _save(self, df: pd.DataFrame) -> None:
        p = industry_map_path(self.store_root)
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(p, index=False)
        logger.info("Industry map saved → %s (%d rows)", p, len(df))


def _name_to_code(name: str) -> str:
    """
    Deterministic 4-char hex code derived from the industry name.
    Used when Eastmoney does not return a numeric code (f128 is empty).
    Not a real exchange code — just a stable surrogate key.
    """
    import hashlib
    return hashlib.md5(name.encode()).hexdigest()[:4].upper()


def _date_series(series: pd.Series) -> pd.Series:
    """Return a Python-date object Series, preserving missing values as None."""
    return pd.to_datetime(series, errors="coerce").map(
        lambda x: x.date() if pd.notna(x) else None
    )


# ---------------------------------------------------------------------------
# PIT query helper
# ---------------------------------------------------------------------------

def get_industry_as_of(
    industry_map: pd.DataFrame,
    symbol: str,
    as_of: dt.date,
) -> dict:
    """
    Return the industry classification for ``symbol`` as-of ``as_of``.

    Returns a dict with keys: industry_code, industry_name, concept_tags.
    Returns empty strings if no record is found.
    """
    effective_date = _date_series(industry_map["effective_date"])
    out_date = (
        _date_series(industry_map["out_date"])
        if "out_date" in industry_map.columns
        else pd.Series([None] * len(industry_map), index=industry_map.index)
    )
    rows = industry_map[
        (industry_map["symbol"] == symbol)
        & (effective_date <= as_of)
        & (out_date.isna() | (out_date > as_of))
    ]
    if rows.empty:
        return {"industry_code": "", "industry_name": "", "concept_tags": ""}
    last = rows.iloc[-1]
    return {
        "industry_code": str(last.get("industry_code", "")),
        "industry_name": str(last.get("industry_name", "")),
        "concept_tags":  str(last.get("concept_tags", "")),
    }


def load_industry_map(store_root: Path | str) -> pd.DataFrame:
    """Load the industry SCD table; return empty DataFrame if not yet collected."""
    p = industry_map_path(Path(store_root))
    if not p.exists():
        return pd.DataFrame(columns=[
            "symbol", "industry_code", "industry_name",
            "concept_tags", "effective_date", "out_date",
        ])
    df = pd.read_parquet(p)
    df["effective_date"] = pd.to_datetime(df["effective_date"]).dt.date
    if "out_date" in df.columns:
        df["out_date"] = pd.to_datetime(df["out_date"], errors="coerce").dt.date
    return df
