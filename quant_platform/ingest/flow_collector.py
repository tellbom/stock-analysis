"""
ingest.flow_collector
=====================
Daily capital flow collector for the CSI 300 universe (P4B-06).

Fetches main/super-large/large/medium/small net inflow per stock from
Eastmoney push2his API (``stock_fund_flow_120d``).  This is the primary
short-horizon signal source: capital flow is orthogonal to price-derived
technicals and carries short-horizon predictive content in A-share markets.

Source: Eastmoney push2his.eastmoney.com
Rate limiting: enforced at ≥1 second per request via shared ``_em_get()``.
For 300 symbols this takes ~5 minutes — run as a nightly batch.

Silver schema
-------------
symbol, date, main_net, small_net, mid_net, large_net, super_net

All values in 元 (yuan).  NOT 万元.  Use / 1e4 or / 1e8 to convert.

Incremental design
------------------
push2his returns the most recent 120 calendar days for each symbol.
The collector tracks the last stored date in the silver Parquet.
- If last date < today − 60 days: full re-fetch (risk of gap)
- Otherwise: extend the tail only (append new rows)

Usage
-----
    from quant_platform.ingest.flow_collector import FundFlowCollector
    fc = FundFlowCollector(store_root=Path("/data/lake"))
    fc.run(symbols=csi300_symbols)
"""

from __future__ import annotations

import datetime as dt
import random
import time
from pathlib import Path

import pandas as pd
import requests

from quant_platform.core.logging import get_logger
from quant_platform.store.lake import fund_flow_path, fund_flow_dir, init_lake
from quant_platform.store.schemas import enforce_fund_flow

logger = get_logger(__name__)

_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
_PUSH2HIS_URL = "https://push2his.eastmoney.com/api/qt/stock/fflow/daykline/get"

# Throttle: ≥1s between Eastmoney push2his calls
_EM_MIN_INTERVAL = 1.0
_em_last_call: list[float] = [0.0]
_em_session = requests.Session()
_em_session.headers.update({
    "User-Agent": _UA,
    "Referer": "https://quote.eastmoney.com/",
    "Origin": "https://quote.eastmoney.com",
})

# Days of gap before we trigger a full re-fetch instead of incremental
_REFETCH_THRESHOLD_DAYS = 60


def _em_get(url: str, params: dict | None = None, timeout: int = 15) -> requests.Response:
    """Throttled Eastmoney GET."""
    wait = _EM_MIN_INTERVAL - (time.time() - _em_last_call[0])
    if wait > 0:
        time.sleep(wait + random.uniform(0.1, 0.4))
    try:
        return _em_session.get(url, params=params, timeout=timeout)
    finally:
        _em_last_call[0] = time.time()


def _fetch_push2his(code: str) -> pd.DataFrame:
    """
    Fetch up to 120 days of daily fund flow for one symbol.

    Returns a DataFrame with columns:
        date, main_net, small_net, mid_net, large_net, super_net
    or empty DataFrame on failure.

    Units: 元 (yuan) — as returned by the API.
    """
    market = 1 if code.startswith("6") else 0
    params = {
        "secid":   f"{market}.{code}",
        "fields1": "f1,f2,f3,f7",
        "fields2": "f51,f52,f53,f54,f55,f56",
        "lmt":     "120",
    }
    try:
        r = _em_get(_PUSH2HIS_URL, params=params)
        klines = r.json().get("data", {}).get("klines", []) or []
    except Exception as exc:
        logger.debug("%s: push2his fetch failed: %s", code, exc)
        return pd.DataFrame()

    if not klines:
        return pd.DataFrame()

    rows = []
    for line in klines:
        parts = line.split(",")
        if len(parts) < 6:
            continue
        try:
            rows.append({
                "date":      parts[0],           # YYYY-MM-DD
                "main_net":  float(parts[1]) if parts[1] not in ("-", "") else 0.0,
                "small_net": float(parts[2]) if parts[2] not in ("-", "") else 0.0,
                "mid_net":   float(parts[3]) if parts[3] not in ("-", "") else 0.0,
                "large_net": float(parts[4]) if parts[4] not in ("-", "") else 0.0,
                "super_net": float(parts[5]) if parts[5] not in ("-", "") else 0.0,
            })
        except (ValueError, IndexError):
            continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df.sort_values("date").reset_index(drop=True)


class FundFlowCollector:
    """
    Collect daily capital flow data for A-share symbols.

    Parameters
    ----------
    store_root : Path | str
    """

    def __init__(self, store_root: Path | str) -> None:
        self.store_root = Path(store_root)
        init_lake(self.store_root)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        symbols: list[str],
        overwrite: bool = False,
    ) -> dict[str, int]:
        """
        Fetch and store capital flow for all symbols.

        Parameters
        ----------
        symbols : list[str]
            6-digit A-share codes.
        overwrite : bool
            If True, ignore existing data and overwrite from scratch.

        Returns
        -------
        dict[str, int]
            symbol → number of new rows written.
        """
        logger.info("FundFlowCollector.run: %d symbols", len(symbols))
        results: dict[str, int] = {}

        for i, symbol in enumerate(symbols):
            try:
                n = self._collect_one(symbol, overwrite)
                results[symbol] = n
            except Exception as exc:
                logger.error("%s: fund flow collection failed: %s", symbol, exc)
                results[symbol] = 0

            if (i + 1) % 50 == 0:
                logger.info("  ... %d/%d done", i + 1, len(symbols))

        n_ok = sum(1 for v in results.values() if v > 0)
        logger.info(
            "FundFlowCollector done: %d/%d symbols had new data",
            n_ok, len(symbols),
        )
        return results

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _collect_one(self, symbol: str, overwrite: bool) -> int:
        """Collect one symbol's fund flow; return number of new rows written."""
        out_path = fund_flow_path(self.store_root, symbol)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        existing = pd.DataFrame()
        if out_path.exists() and not overwrite:
            try:
                existing = pd.read_parquet(out_path)
                existing["date"] = pd.to_datetime(existing["date"]).dt.date
            except Exception:
                existing = pd.DataFrame()

        # Decide whether to do a full re-fetch
        do_full = overwrite or existing.empty
        if not do_full and not existing.empty:
            last_date = existing["date"].max()
            gap = (dt.date.today() - last_date).days
            if gap > _REFETCH_THRESHOLD_DAYS:
                logger.info(
                    "%s: gap=%d days > threshold=%d, doing full re-fetch",
                    symbol, gap, _REFETCH_THRESHOLD_DAYS,
                )
                do_full = True

        # Fetch from Eastmoney
        new_df = _fetch_push2his(symbol)
        if new_df.empty:
            return 0

        # Enforce schema
        new_df["symbol"] = symbol
        new_df = enforce_fund_flow(new_df, symbol)

        if do_full or existing.empty:
            combined = new_df
        else:
            # Only keep genuinely new rows
            known_dates = set(existing["date"].values)
            actually_new = new_df[~new_df["date"].isin(known_dates)]
            if actually_new.empty:
                return 0
            combined = pd.concat([existing, actually_new], ignore_index=True)

        combined = (
            combined.sort_values("date")
                    .drop_duplicates(subset=["symbol", "date"], keep="last")
                    .reset_index(drop=True)
        )
        combined.to_parquet(out_path, index=False)
        n_new = len(new_df) if do_full else len(actually_new)
        logger.debug("%s: wrote %d new rows → %s", symbol, n_new, out_path)
        return n_new


def load_fund_flow(store_root: Path | str, symbol: str) -> pd.DataFrame:
    """Load the silver fund flow Parquet for one symbol."""
    p = fund_flow_path(Path(store_root), symbol)
    if not p.exists():
        return pd.DataFrame(columns=[
            "symbol", "date", "main_net", "small_net",
            "mid_net", "large_net", "super_net",
        ])
    df = pd.read_parquet(p)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df.sort_values("date").reset_index(drop=True)
