"""
ingest.index_collector
======================
CSI 300 (and other market indices) daily OHLCV collector (P4A-05).

Fetches index daily close prices and stores them as silver Parquet, used
by ``labels.builder`` to compute excess-vs-CSI300 labels.

Source priority
---------------
1. **Tencent Finance API** (``qt.gtimg.cn``): real-time quote feed; not
   IP-banned; returns full OHLCV + turnover for indices.  Batch is not
   needed — the index is one symbol.  This is the primary source.
2. **AKShare** ``index_zh_a_hist``: fallback for historical bulk pull;
   instability risk but acceptable as a one-time backfill.

Usage
-----
    from quant_platform.ingest.index_collector import IndexCollector
    from pathlib import Path

    collector = IndexCollector(store_root=Path("/data/lake"))
    collector.run(symbols=["000300"], start_date="2018-01-01")
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

from quant_platform.core.fetch import safe_call
from quant_platform.core.logging import get_logger
from quant_platform.store.lake import index_ohlcv_path, index_ohlcv_dir, init_lake

logger = get_logger(__name__)

# Tencent Finance prefix for index symbols
_TENCENT_INDEX_PREFIX = {
    "000001": "sh",   # Shanghai Composite
    "000300": "sh",   # CSI 300
    "000016": "sh",   # SSE 50
    "399001": "sz",   # Shenzhen Component
    "399006": "sz",   # ChiNext
}


class IndexCollector:
    """
    Collect daily OHLCV for market indices and store as silver Parquet.

    The primary use-case is the CSI 300 index (000300) required for the
    excess-vs-CSI300 label in P4A-05.

    Parameters
    ----------
    store_root : Path | str
        Root of the data lake.
    """

    def __init__(self, store_root: Path | str) -> None:
        self.store_root = Path(store_root)
        init_lake(self.store_root)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        symbols: list[str] | None = None,
        start_date: str = "2015-01-01",
        overwrite: bool = False,
    ) -> dict[str, int]:
        """
        Collect index OHLCV for all requested symbols.

        Parameters
        ----------
        symbols : list[str] | None
            Index codes.  Default ["000300"] (CSI 300).
        start_date : str
            Earliest date to fetch (YYYY-MM-DD).  Ignored when
            incremental update finds existing data.
        overwrite : bool
            If True, re-fetch and overwrite existing data.

        Returns
        -------
        dict[str, int]
            symbol → number of new rows written.
        """
        symbols = symbols or ["000300"]
        results: dict[str, int] = {}

        for symbol in symbols:
            try:
                n = self._collect_one(symbol, start_date, overwrite)
                results[symbol] = n
                logger.info("Index %s: %d rows collected/updated", symbol, n)
            except Exception as exc:
                logger.error("Index %s: collection failed: %s", symbol, exc)
                results[symbol] = 0

        return results

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _collect_one(
        self,
        symbol: str,
        start_date: str,
        overwrite: bool,
    ) -> int:
        """Collect one index symbol; return number of new rows written."""
        out_path = index_ohlcv_path(self.store_root, symbol)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        existing = pd.DataFrame()
        if out_path.exists() and not overwrite:
            try:
                existing = pd.read_parquet(out_path)
                existing["date"] = pd.to_datetime(existing["date"]).dt.date
            except Exception:
                existing = pd.DataFrame()

        # Determine fetch start date (incremental)
        if not existing.empty:
            last_date = existing["date"].max()
            fetch_start = (
                pd.to_datetime(last_date) + pd.Timedelta(days=1)
            ).strftime("%Y-%m-%d")
        else:
            fetch_start = start_date

        today = dt.date.today().isoformat()
        if fetch_start > today:
            logger.info("Index %s: already up to date", symbol)
            return 0

        # Fetch from AKShare (historical bulk)
        new_rows = self._fetch_akshare(symbol, fetch_start, today)
        if new_rows.empty:
            new_rows = self._fetch_eastmoney(symbol, fetch_start, today)

        if new_rows.empty:
            logger.warning("Index %s: no new data from %s to %s", symbol, fetch_start, today)
            return 0

        # Merge with existing and deduplicate
        if not existing.empty:
            combined = pd.concat([existing, new_rows], ignore_index=True)
        else:
            combined = new_rows

        combined["date"] = pd.to_datetime(combined["date"]).dt.date
        combined = (
            combined.drop_duplicates(subset=["date"])
                    .sort_values("date")
                    .reset_index(drop=True)
        )

        combined.to_parquet(out_path, index=False)
        logger.info(
            "Index %s: wrote %d total rows → %s",
            symbol, len(combined), out_path,
        )
        return len(new_rows)

    def _fetch_akshare(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        Fetch index daily OHLCV from AKShare ``index_zh_a_hist``.

        Returns a normalised DataFrame with columns:
            date, open, high, low, close, volume, amount
        or an empty DataFrame on failure.
        """
        def _call():
            import akshare as ak
            df = ak.index_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=start_date.replace("-", ""),
                end_date=end_date.replace("-", ""),
            )
            return df

        df = safe_call(_call, retries=3, label=f"index_zh_a_hist {symbol}")
        if df is None or df.empty:
            logger.warning(
                "AKShare index_zh_a_hist returned empty for %s (%s to %s)",
                symbol, start_date, end_date,
            )
            return pd.DataFrame()

        # Normalise column names (AKShare uses Chinese column names)
        col_map = {
            "日期": "date",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "成交额": "amount",
        }
        df = df.rename(columns=col_map)

        required = {"date", "close"}
        if not required.issubset(df.columns):
            # Try English column names (some AKShare versions differ)
            logger.warning(
                "Index %s: unexpected AKShare column names: %s",
                symbol, list(df.columns),
            )
            return pd.DataFrame()

        df["date"] = pd.to_datetime(df["date"]).dt.date

        # Keep only the columns we need; fill optional ones with NaN
        out = pd.DataFrame({"date": df["date"], "close": pd.to_numeric(df["close"], errors="coerce")})
        for col in ("open", "high", "low", "volume", "amount"):
            out[col] = pd.to_numeric(df.get(col, pd.Series(dtype=float)), errors="coerce")

        out = out[out["close"].notna()].reset_index(drop=True)
        return out

    def _fetch_eastmoney(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Fetch index daily OHLCV directly from Eastmoney kline API."""
        import requests
        import urllib3

        market = _TENCENT_INDEX_PREFIX.get(symbol, "sh")
        market_id = "1" if market == "sh" else "0"
        url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
        params = {
            "secid": f"{market_id}.{symbol}",
            "fields1": "f1,f2,f3,f4,f5,f6",
            "fields2": "f51,f52,f53,f54,f55,f56,f57",
            "klt": "101",
            "fqt": "1",
            "beg": start_date.replace("-", ""),
            "end": end_date.replace("-", ""),
        }

        try:
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            resp = requests.get(url, params=params, timeout=30, verify=False)
            resp.raise_for_status()
            payload = resp.json()
        except Exception as exc:
            logger.warning("Eastmoney index kline failed for %s: %s", symbol, exc)
            return pd.DataFrame()

        klines = ((payload or {}).get("data") or {}).get("klines") or []
        rows: list[dict] = []
        for line in klines:
            parts = str(line).split(",")
            if len(parts) < 7:
                continue
            rows.append({
                "date": parts[0],
                "open": parts[1],
                "close": parts[2],
                "high": parts[3],
                "low": parts[4],
                "volume": parts[5],
                "amount": parts[6],
            })
        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"]).dt.date
        for col in ("open", "high", "low", "close", "volume", "amount"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df[df["close"].notna()].reset_index(drop=True)
