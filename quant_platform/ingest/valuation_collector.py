"""
ingest.valuation_collector
==========================
Daily valuation and size collector for the CSI 300 universe (P4B-01).

Fetches PE_TTM, PB, total market cap, float market cap, and turnover rate
from the Tencent Finance API (qt.gtimg.cn).  This is the primary source
because:
  - Not IP-banned: TCP connection to qt.gtimg.cn is unrestricted.
  - Batch-safe: all 300 symbols fit in a single HTTP request.
  - Same-day safe: values are derived from the closing price; no
    announcement-date lag needed.
  - No API key required.

Known API trap (实测校准 2026-05-03)
--------------------------------------
Tencent field index 43 = 振幅% (NOT PB).
PB is at field index 46.  Many online tutorials get this wrong.

Silver schema
-------------
symbol, date, pe_ttm, pb, total_mcap_yi, float_mcap_yi, turnover_pct

Usage
-----
    from quant_platform.ingest.valuation_collector import ValuationCollector
    from pathlib import Path

    vc = ValuationCollector(store_root=Path("/data/lake"))
    result = vc.run(symbols=csi300_symbols)
"""

from __future__ import annotations

import datetime as dt
import time
import urllib.request
from pathlib import Path

import pandas as pd

from quant_platform.core.logging import get_logger
from quant_platform.store.lake import valuation_path, valuation_dir, init_lake
from quant_platform.store.schemas import enforce_valuation

logger = get_logger(__name__)

# Tencent Finance real-time quote endpoint
_TENCENT_URL = "https://qt.gtimg.cn/q={codes}"
_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"

# Maximum symbols per request (Tencent can handle 300+ in one call)
_BATCH_SIZE = 300


def _market_prefix(code: str) -> str:
    """Map a 6-digit A-share code to its Tencent market prefix."""
    if code.startswith(("6", "9")):
        return f"sh{code}"
    elif code.startswith("8"):
        return f"bj{code}"
    return f"sz{code}"


def _fetch_tencent_batch(codes: list[str]) -> dict[str, dict]:
    """
    Batch-fetch real-time quotes for up to _BATCH_SIZE symbols.

    Returns {code: {pe_ttm, pb, total_mcap_yi, float_mcap_yi, turnover_pct}}
    for successfully parsed symbols.  Unparseable symbols are omitted.

    Field index reference (实测 2026-05):
      vals[1]  = name
      vals[3]  = price
      vals[38] = turnover_pct %
      vals[39] = PE_TTM
      vals[43] = 振幅% (NOT PB — common mistake)
      vals[44] = total_mcap (亿元)
      vals[45] = float_mcap (亿元)
      vals[46] = PB
    """
    prefixed = ",".join(_market_prefix(c) for c in codes)
    url = _TENCENT_URL.format(codes=prefixed)
    req = urllib.request.Request(url, headers={"User-Agent": _UA})

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw = resp.read().decode("gbk", errors="replace")
    except Exception as exc:
        logger.warning("Tencent batch fetch failed: %s", exc)
        return {}

    result: dict[str, dict] = {}
    for line in raw.strip().split(";"):
        line = line.strip()
        if not line or "=" not in line or '"' not in line:
            continue
        try:
            key_part = line.split("=")[0]
            code = key_part.split("_")[-1][2:]   # strip sh/sz/bj prefix
            vals = line.split('"')[1].split("~")
            if len(vals) < 47:
                continue
            result[code] = {
                "pe_ttm":        float(vals[39]) if vals[39] else 0.0,
                "pb":            float(vals[46]) if vals[46] else 0.0,
                "total_mcap_yi": float(vals[44]) if vals[44] else 0.0,
                "float_mcap_yi": float(vals[45]) if vals[45] else 0.0,
                "turnover_pct":  float(vals[38]) if vals[38] else 0.0,
            }
        except (IndexError, ValueError):
            continue

    return result


class ValuationCollector:
    """
    Collect daily valuation data for a universe of A-share symbols.

    Parameters
    ----------
    store_root : Path | str
        Root of the data lake.
    """

    def __init__(self, store_root: Path | str) -> None:
        self.store_root = Path(store_root)
        init_lake(self.store_root)

    def run(
        self,
        symbols: list[str],
        date: dt.date | str | None = None,
        overwrite: bool = False,
    ) -> dict[str, bool]:
        """
        Fetch today's (or a specified date's) valuation for all symbols and
        append to per-symbol silver Parquet files.

        Parameters
        ----------
        symbols : list[str]
            6-digit A-share codes.
        date : date | str | None
            The trading date to record.  Defaults to today.
        overwrite : bool
            If True, overwrite existing rows for this date.

        Returns
        -------
        dict[str, bool]
            symbol → True if successfully written.
        """
        if date is None:
            record_date = dt.date.today()
        else:
            record_date = pd.to_datetime(date).date() if isinstance(date, str) else date

        logger.info(
            "ValuationCollector.run: %d symbols, date=%s", len(symbols), record_date
        )

        # Fetch in batches
        quotes: dict[str, dict] = {}
        for i in range(0, len(symbols), _BATCH_SIZE):
            batch = symbols[i : i + _BATCH_SIZE]
            batch_quotes = _fetch_tencent_batch(batch)
            quotes.update(batch_quotes)
            if i + _BATCH_SIZE < len(symbols):
                time.sleep(0.3)   # gentle pacing for large universes

        logger.info("Tencent batch: %d/%d symbols returned data", len(quotes), len(symbols))

        results: dict[str, bool] = {}
        for symbol in symbols:
            q = quotes.get(symbol)
            if q is None:
                logger.debug("%s: no quote returned from Tencent", symbol)
                results[symbol] = False
                continue
            try:
                success = self._write_one(symbol, record_date, q, overwrite)
                results[symbol] = success
            except Exception as exc:
                logger.error("%s: write failed: %s", symbol, exc)
                results[symbol] = False

        n_ok = sum(1 for v in results.values() if v)
        logger.info("ValuationCollector done: %d/%d written", n_ok, len(symbols))
        return results

    def _write_one(
        self,
        symbol: str,
        record_date: dt.date,
        quote: dict,
        overwrite: bool,
    ) -> bool:
        """Append one day's valuation data to the symbol's silver Parquet."""
        out_path = valuation_path(self.store_root, symbol)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        new_row = pd.DataFrame([{
            "symbol":         symbol,
            "date":           record_date,
            **quote,
        }])
        new_row = enforce_valuation(new_row, symbol)

        if out_path.exists():
            existing = pd.read_parquet(out_path)
            existing["date"] = pd.to_datetime(existing["date"]).dt.date
            if not overwrite:
                # Skip if this date is already present
                if record_date in existing["date"].values:
                    return True   # already up to date
            else:
                existing = existing[existing["date"] != record_date]
            combined = pd.concat([existing, new_row], ignore_index=True)
        else:
            combined = new_row

        combined = (
            combined.sort_values("date")
                    .drop_duplicates(subset=["symbol", "date"], keep="last")
                    .reset_index(drop=True)
        )
        combined.to_parquet(out_path, index=False)
        return True


def load_valuation(
    store_root: Path | str,
    symbol: str,
) -> pd.DataFrame:
    """
    Load the silver valuation Parquet for one symbol.
    Returns an empty DataFrame if not yet collected.
    """
    p = valuation_path(Path(store_root), symbol)
    if not p.exists():
        return pd.DataFrame(columns=["symbol", "date", "pe_ttm", "pb",
                                     "total_mcap_yi", "float_mcap_yi", "turnover_pct"])
    df = pd.read_parquet(p)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df.sort_values("date").reset_index(drop=True)
