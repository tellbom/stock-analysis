"""
ingest.margin_collector
=======================
Daily margin trading (融资融券) collector for the CSI 300 universe (P4B-08).

Fetches daily 融资余额, 融资买入额, 融券余额 per stock from Eastmoney's
datacenter API (RPTA_WEB_RZRQ_GGMX).  This data is released with a
1-business-day delay — always apply a 1-day lag when joining to features.

Not all CSI 300 stocks are margin-eligible; non-eligible stocks are
silently omitted (empty Parquet).  The feature builder handles NaN.

Silver schema
-------------
symbol, date, rzye, rzmre, rzche, rqye, rqmcl, rqchl, rzrqye

All monetary values in 元 (yuan).

Usage
-----
    from quant_platform.ingest.margin_collector import MarginCollector
    mc = MarginCollector(store_root=Path("/data/lake"))
    mc.run(symbols=csi300_symbols)
"""

from __future__ import annotations

import datetime as dt
import random
import time
from pathlib import Path

import pandas as pd
import requests

from quant_platform.core.logging import get_logger
from quant_platform.store.lake import margin_path, margin_dir, init_lake
from quant_platform.store.schemas import enforce_margin

logger = get_logger(__name__)

_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
_DATACENTER_URL = "https://datacenter-web.eastmoney.com/api/data/v1/get"

_EM_MIN_INTERVAL = 1.0
_em_last_call: list[float] = [0.0]
_em_session = requests.Session()
_em_session.headers.update({"User-Agent": _UA})

# Days of lookback when doing a fresh fetch
_DEFAULT_LOOKBACK_DAYS = 90


def _em_get(url: str, params: dict | None = None, timeout: int = 15) -> requests.Response:
    """Throttled Eastmoney datacenter GET."""
    wait = _EM_MIN_INTERVAL - (time.time() - _em_last_call[0])
    if wait > 0:
        time.sleep(wait + random.uniform(0.1, 0.4))
    try:
        return _em_session.get(url, params=params, timeout=timeout)
    finally:
        _em_last_call[0] = time.time()


def _fetch_margin(
    code: str,
    start_date: str,
    end_date: str,
    page_size: int = 90,
) -> pd.DataFrame:
    """
    Fetch margin trading data for one symbol from Eastmoney datacenter.

    Returns a DataFrame with the silver schema columns, or empty if the
    symbol is not margin-eligible or the API returns no data.
    """
    params = {
        "reportName": "RPTA_WEB_RZRQ_GGMX",
        "columns":    "ALL",
        "filter":     f'(SCODE="{code}")'
                      f'(DATE>=\'{start_date}\')(DATE<=\'{end_date}\')',
        "pageNumber": "1",
        "pageSize":   str(page_size),
        "sortColumns": "DATE",
        "sortTypes":  "-1",
        "source":     "WEB",
        "client":     "WEB",
    }
    try:
        r = _em_get(_DATACENTER_URL, params=params)
        data = r.json().get("result", {}) or {}
        rows = data.get("data") or []
    except Exception as exc:
        logger.debug("%s: margin fetch failed: %s", code, exc)
        return pd.DataFrame()

    if not rows:
        return pd.DataFrame()

    records = []
    for row in rows:
        try:
            records.append({
                "date":    str(row.get("DATE", ""))[:10],
                "rzye":    float(row.get("RZYE") or 0),
                "rzmre":   float(row.get("RZMRE") or 0),
                "rzche":   float(row.get("RZCHE") or 0),
                "rqye":    float(row.get("RQYE") or 0),
                "rqmcl":   float(row.get("RQMCL") or 0),
                "rqchl":   float(row.get("RQCHL") or 0),
                "rzrqye":  float(row.get("RZRQYE") or 0),
            })
        except (TypeError, ValueError):
            continue

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df = df.dropna(subset=["date"])
    return df.sort_values("date").reset_index(drop=True)


class MarginCollector:
    """
    Collect daily margin trading data for A-share symbols.

    Parameters
    ----------
    store_root : Path | str
    lookback_days : int
        How many calendar days to fetch on a fresh (non-incremental) run.
    """

    def __init__(
        self,
        store_root: Path | str,
        lookback_days: int = _DEFAULT_LOOKBACK_DAYS,
    ) -> None:
        self.store_root    = Path(store_root)
        self.lookback_days = lookback_days
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
        Fetch and store margin data for all symbols.

        Returns
        -------
        dict[str, int]
            symbol → number of new rows written (0 if not margin-eligible
            or no new data).
        """
        logger.info("MarginCollector.run: %d symbols", len(symbols))
        results: dict[str, int] = {}

        for i, symbol in enumerate(symbols):
            try:
                n = self._collect_one(symbol, overwrite)
                results[symbol] = n
            except Exception as exc:
                logger.error("%s: margin collection failed: %s", symbol, exc)
                results[symbol] = 0

            if (i + 1) % 50 == 0:
                logger.info("  ... %d/%d done", i + 1, len(symbols))

        n_ok = sum(1 for v in results.values() if v > 0)
        logger.info(
            "MarginCollector done: %d/%d symbols had new data (rest not eligible or up-to-date)",
            n_ok, len(symbols),
        )
        return results

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _collect_one(self, symbol: str, overwrite: bool) -> int:
        """Collect one symbol's margin data; return number of new rows."""
        out_path = margin_path(self.store_root, symbol)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        existing = pd.DataFrame()
        fetch_start: str
        today = dt.date.today().isoformat()

        if out_path.exists() and not overwrite:
            try:
                existing = pd.read_parquet(out_path)
                existing["date"] = pd.to_datetime(existing["date"]).dt.date
                # Incremental: fetch from day after last known date
                last_date = existing["date"].max()
                fetch_start = (
                    pd.to_datetime(last_date) + pd.Timedelta(days=1)
                ).strftime("%Y-%m-%d")
                if fetch_start > today:
                    return 0   # already up to date
            except Exception:
                existing = pd.DataFrame()
                fetch_start = (
                    dt.date.today() - dt.timedelta(days=self.lookback_days)
                ).isoformat()
        else:
            fetch_start = (
                dt.date.today() - dt.timedelta(days=self.lookback_days)
            ).isoformat()

        new_df = _fetch_margin(symbol, fetch_start, today)
        if new_df.empty:
            return 0   # not margin-eligible or no data in range

        new_df["symbol"] = symbol
        new_df = enforce_margin(new_df, symbol)

        if existing.empty:
            combined = new_df
        else:
            combined = pd.concat([existing, new_df], ignore_index=True)

        combined = (
            combined.sort_values("date")
                    .drop_duplicates(subset=["symbol", "date"], keep="last")
                    .reset_index(drop=True)
        )
        combined.to_parquet(out_path, index=False)
        logger.debug("%s: wrote %d new margin rows → %s", symbol, len(new_df), out_path)
        return len(new_df)


def load_margin(store_root: Path | str, symbol: str) -> pd.DataFrame:
    """Load the silver margin Parquet for one symbol."""
    p = margin_path(Path(store_root), symbol)
    if not p.exists():
        return pd.DataFrame(columns=[
            "symbol", "date", "rzye", "rzmre", "rzche",
            "rqye", "rqmcl", "rqchl", "rzrqye",
        ])
    df = pd.read_parquet(p)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df.sort_values("date").reset_index(drop=True)
