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

import numpy as np
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
_VALUATION_COLUMNS = [
    "symbol", "date", "pe_ttm", "pb", "total_mcap_yi", "float_mcap_yi", "turnover_pct",
]


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


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str:
    """Find a source column by exact or substring match; raises if missing."""
    columns = [str(c).strip() for c in df.columns]
    lowered = {c.lower(): c for c in columns}
    for name in candidates:
        key = name.lower()
        if key in lowered:
            return lowered[key]
    for name in candidates:
        key = name.lower()
        for col in columns:
            if key in col.lower():
                return col
    raise ValueError(f"AKShare stock_value_em schema missing one of {candidates}; columns={columns}")


def normalise_stock_value_em(
    raw: pd.DataFrame,
    symbol: str,
    start_date: dt.date | str,
    end_date: dt.date | str,
    turnover: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Convert AKShare ``stock_value_em`` history to the valuation silver schema.

    AKShare market-cap fields are yuan; silver stores ``*_mcap_yi`` in
    hundred-million yuan.  ``turnover_pct`` is not present in stock_value_em
    and must be joined from OHLCV when doing historical backfill.
    """
    if raw is None or raw.empty:
        raise ValueError(f"{symbol}: empty AKShare stock_value_em response")

    date_col = _find_column(raw, ["date", "数据日期", "日期"])
    pe_col = _find_column(raw, ["pe_ttm", "PE_TTM", "PE(TTM)", "市盈率(TTM)", "滚动市盈率"])
    pb_col = _find_column(raw, ["pb", "PB", "市净率"])
    total_col = _find_column(raw, ["total_mcap", "总市值"])
    float_col = _find_column(raw, ["float_mcap", "流通市值"])

    df = pd.DataFrame({
        "symbol": symbol,
        "date": pd.to_datetime(raw[date_col], errors="coerce"),
        "pe_ttm": pd.to_numeric(raw[pe_col], errors="coerce"),
        "pb": pd.to_numeric(raw[pb_col], errors="coerce"),
        "total_mcap_yi": pd.to_numeric(raw[total_col], errors="coerce") / 1e8,
        "float_mcap_yi": pd.to_numeric(raw[float_col], errors="coerce") / 1e8,
    })

    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    df = df[(df["date"] >= start) & (df["date"] <= end)].copy()
    if df.empty:
        raise ValueError(f"{symbol}: no valuation rows in {start.date()} -> {end.date()}")

    df["date"] = df["date"].dt.date
    if turnover is not None and not turnover.empty:
        t = turnover.copy()
        t["date"] = pd.to_datetime(t["date"]).dt.date
        df = df.merge(t[["date", "turnover_pct"]], on="date", how="left")
    else:
        df["turnover_pct"] = np.nan

    return enforce_valuation(df[_VALUATION_COLUMNS], symbol)


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

    def backfill_history(
        self,
        symbols: list[str],
        start_date: dt.date | str,
        end_date: dt.date | str,
        dry_run: bool = True,
        min_success_rate: float = 0.95,
    ) -> dict:
        """
        Backfill historical valuation from AKShare ``stock_value_em``.

        The method validates schema, units, per-symbol row counts, and core
        field non-null rates before writing.  In ``dry_run`` mode it performs
        all fetch/validation work without touching silver files.
        """
        try:
            import akshare as ak
        except ImportError as exc:
            raise RuntimeError("akshare is required for historical valuation backfill") from exc

        results: dict[str, dict] = {}
        frames: dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            try:
                raw = ak.stock_value_em(symbol=symbol)
                turnover = self._load_ohlcv_turnover(symbol)
                df = normalise_stock_value_em(raw, symbol, start_date, end_date, turnover)
                stats = self._validate_history(symbol, df)
                results[symbol] = {"status": "ok", **stats}
                frames[symbol] = df
            except Exception as exc:
                logger.error("%s: valuation history validation failed: %s", symbol, exc)
                results[symbol] = {"status": "failed", "error": str(exc)}

        ok_symbols = [s for s, r in results.items() if r["status"] == "ok"]
        success_rate = len(ok_symbols) / len(symbols) if symbols else 1.0
        report = {
            "source": "akshare.stock_value_em",
            "dry_run": dry_run,
            "date_range": f"{start_date} -> {end_date}",
            "total_symbols": len(symbols),
            "succeeded": len(ok_symbols),
            "failed": len(symbols) - len(ok_symbols),
            "success_rate": success_rate,
            "results": results,
        }
        if success_rate < min_success_rate:
            raise RuntimeError(
                f"Valuation backfill success rate {success_rate:.1%} below "
                f"threshold {min_success_rate:.1%}"
            )
        if not dry_run:
            for symbol, df in frames.items():
                out_path = valuation_path(self.store_root, symbol)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                df.to_parquet(out_path, index=False)
        return report

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

    def _load_ohlcv_turnover(self, symbol: str) -> pd.DataFrame:
        from quant_platform.store.lake import ohlcv_path

        p = ohlcv_path(self.store_root, symbol)
        if not p.exists():
            return pd.DataFrame(columns=["date", "turnover_pct"])
        df = pd.read_parquet(p)
        if "turnover" not in df.columns:
            return pd.DataFrame(columns=["date", "turnover_pct"])
        df["date"] = pd.to_datetime(df["date"]).dt.date
        return df[["date", "turnover"]].rename(columns={"turnover": "turnover_pct"})

    @staticmethod
    def _validate_history(symbol: str, df: pd.DataFrame) -> dict:
        core = ["pe_ttm", "pb", "total_mcap_yi", "float_mcap_yi", "turnover_pct"]
        if df.empty:
            raise ValueError("empty normalised valuation frame")
        non_null = df[core].notna().mean().to_dict()
        bad = {k: v for k, v in non_null.items() if v < 0.95}
        if bad:
            raise ValueError(f"core field coverage below 95%: {bad}")
        if (df[["total_mcap_yi", "float_mcap_yi"]] <= 0).any().any():
            raise ValueError("market cap fields must be positive yi-yuan values")
        if df["date"].duplicated().any():
            raise ValueError("duplicate valuation dates")
        return {
            "rows": int(len(df)),
            "date_min": str(df["date"].min()),
            "date_max": str(df["date"].max()),
            "non_null": {k: round(float(v), 4) for k, v in non_null.items()},
            "units": {
                "total_mcap_yi": "100 million CNY",
                "float_mcap_yi": "100 million CNY",
            },
        }


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
