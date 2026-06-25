"""
ingest.lockup_collector
=======================
Lockup expiry (限售解禁) data collector (P4C-03).

Fetches upcoming lock-up expiry events for A-share symbols from Eastmoney
datacenter (report: RPT_LIFT_STAGE).  Lockup expiry dates are *public
knowledge at all prior dates* — they are announced at IPO or at the time
of the private placement.  This makes the data legitimately PIT-safe: a
feature at date T that looks forward to the *next* unlock event after T
uses only information that was known to the market at T.

Silver schema
-------------
symbol, unlock_date, lock_type, shares_million, ratio_pct

  symbol         : 6-digit A-share code
  unlock_date    : date the lock-up expires (future or past)
    lock_type      : e.g. "首发机构配售股份", "定向增发"
    shares_million : number of shares unlocking (百万股)
  ratio_pct      : shares unlocking / float shares (%)

Collection strategy
-------------------
Collects forward-looking events (next 180 calendar days from today).
Events in the past are also retained so the silver table can be used to
build historical features by looking forward from any past date.

Rate limiting: Eastmoney datacenter is rate-limited.  All requests go
through the module-level ``_em_get()`` throttle (≥1s per call).

Usage
-----
    from quant_platform.ingest.lockup_collector import LockupCollector
    lc = LockupCollector(store_root=Path("/data/lake"))
    lc.run(symbols=csi300_symbols)
"""

from __future__ import annotations

import datetime as dt
import random
import time
from pathlib import Path

import pandas as pd
import requests

from quant_platform.core.logging import get_logger
from quant_platform.store.lake import lockup_dir, lockup_path, init_lake

logger = get_logger(__name__)

_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
_DATACENTER_URL = "https://datacenter-web.eastmoney.com/api/data/v1/get"
_FORWARD_DAYS = 180   # collect events up to this many days in the future

_EM_MIN_INTERVAL = 1.0
_em_last_call: list[float] = [0.0]
_em_session = requests.Session()
_em_session.headers.update({"User-Agent": _UA})


def _em_get(url: str, params: dict | None = None, timeout: int = 15) -> requests.Response:
    """Throttled Eastmoney datacenter GET."""
    wait = _EM_MIN_INTERVAL - (time.time() - _em_last_call[0])
    if wait > 0:
        time.sleep(wait + random.uniform(0.1, 0.4))
    try:
        return _em_session.get(url, params=params, timeout=timeout)
    finally:
        _em_last_call[0] = time.time()


def _fetch_lockup_events(
    code: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Fetch lock-up expiry events for one symbol from Eastmoney datacenter.

    Both historical records (already unlocked) and upcoming records are
    fetched using the RPT_LIFT_STAGE report with a date range filter.

    Returns a DataFrame with silver schema columns, or empty on failure.
    """
    params = {
        "reportName": "RPT_LIFT_STAGE",
        "columns":    "ALL",
        "filter":     (
            f'(SECURITY_CODE="{code}")'
            f'(FREE_DATE>=\'{start_date}\')(FREE_DATE<=\'{end_date}\')'
        ),
        "pageNumber": "1",
        "pageSize":   "50",
        "sortColumns": "FREE_DATE",
        "sortTypes":  "1",
        "source":     "WEB",
        "client":     "WEB",
    }
    try:
        r = _em_get(_DATACENTER_URL, params=params)
        data = r.json().get("result", {}) or {}
        rows = data.get("data") or []
    except Exception as exc:
        logger.debug("%s: lockup fetch failed: %s", code, exc)
        return pd.DataFrame()

    if not rows:
        return pd.DataFrame()

    records = []
    for row in rows:
        unlock_date_raw = str(row.get("FREE_DATE", "") or "")[:10]
        if not unlock_date_raw or unlock_date_raw == "None":
            continue
        try:
            shares = row.get("CURRENT_FREE_SHARES")
            if shares is None:
                shares = row.get("FREE_SHARES_NUM")
            if shares is None:
                shares = row.get("FREE_SHARES")
            lock_type = (
                row.get("FREE_SHARES_TYPE")
                or row.get("LIMITED_STOCK_TYPE")
                or ""
            )
            records.append({
                "symbol":          code,
                "unlock_date":     unlock_date_raw,
                "lock_type":       str(lock_type),
                "shares_million":  float(shares or 0) / 1e6,
                "ratio_pct":       float(row.get("FREE_RATIO", 0) or 0),
            })
        except (TypeError, ValueError):
            continue

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["unlock_date"] = pd.to_datetime(df["unlock_date"], errors="coerce").dt.date
    return df.dropna(subset=["unlock_date"]).sort_values("unlock_date").reset_index(drop=True)


class LockupCollector:
    """
    Collect lock-up expiry events for A-share symbols.

    Parameters
    ----------
    store_root : Path | str
    forward_days : int
        Collect events up to this many calendar days in the future.
    """

    def __init__(
        self,
        store_root: Path | str,
        forward_days: int = _FORWARD_DAYS,
    ) -> None:
        self.store_root   = Path(store_root)
        self.forward_days = forward_days
        init_lake(self.store_root)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        symbols: list[str],
        as_of: dt.date | None = None,
    ) -> dict[str, int]:
        """
        Collect lock-up expiry data for all symbols.

        For each symbol, fetches events within [today-365d, today+forward_days].
        New rows are appended to existing silver Parquet; duplicate
        (symbol, unlock_date) pairs are deduplicated.

        Parameters
        ----------
        symbols : list[str]
            6-digit A-share codes.
        as_of : date | None
            Reference date (default: today).

        Returns
        -------
        dict[str, int]
            symbol → number of new rows written (0 = no events or no change).
        """
        as_of = as_of or dt.date.today()
        start = (as_of - dt.timedelta(days=365)).isoformat()
        end   = (as_of + dt.timedelta(days=self.forward_days)).isoformat()

        logger.info(
            "LockupCollector.run: %d symbols, range %s → %s",
            len(symbols), start, end,
        )
        results: dict[str, int] = {}

        for i, symbol in enumerate(symbols):
            try:
                n = self._collect_one(symbol, start, end)
                results[symbol] = n
            except Exception as exc:
                logger.error("%s: lockup collection failed: %s", symbol, exc)
                results[symbol] = 0

            if (i + 1) % 50 == 0:
                logger.info("  ... %d/%d done", i + 1, len(symbols))

        n_ok = sum(1 for v in results.values() if v > 0)
        logger.info(
            "LockupCollector done: %d/%d symbols had lockup data", n_ok, len(symbols)
        )
        return results

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _collect_one(self, symbol: str, start: str, end: str) -> int:
        """Collect lockup events for one symbol; return number of rows written."""
        out_path = lockup_path(self.store_root, symbol)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        new_df = _fetch_lockup_events(symbol, start, end)
        if new_df.empty:
            return 0

        if out_path.exists():
            existing = pd.read_parquet(out_path)
            existing["unlock_date"] = pd.to_datetime(existing["unlock_date"]).dt.date
            combined = pd.concat([existing, new_df], ignore_index=True)
        else:
            combined = new_df

        combined = (
            combined.drop_duplicates(subset=["symbol", "unlock_date", "lock_type"])
                    .sort_values("unlock_date")
                    .reset_index(drop=True)
        )
        combined.to_parquet(out_path, index=False)
        logger.debug("%s: %d lockup rows → %s", symbol, len(combined), out_path)
        return len(new_df)


def load_lockup(store_root: Path | str, symbol: str) -> pd.DataFrame:
    """Load the lockup silver Parquet for one symbol."""
    p = lockup_path(Path(store_root), symbol)
    if not p.exists():
        return pd.DataFrame(columns=[
            "symbol", "unlock_date", "lock_type", "shares_million", "ratio_pct"
        ])
    df = pd.read_parquet(p)
    df["unlock_date"] = pd.to_datetime(df["unlock_date"]).dt.date
    return df.sort_values("unlock_date").reset_index(drop=True)
