"""Block trade collector backed by Eastmoney datacenter via AKShare."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import requests

from quant_platform.core.logging import get_logger
from quant_platform.store.lake import block_trade_path, init_lake

logger = get_logger(__name__)
_DATACENTER_URL = "https://datacenter-web.eastmoney.com/api/data/v1/get"
_SESSION = requests.Session()
_SESSION.trust_env = False
_SESSION.headers.update({"User-Agent": "Mozilla/5.0", "Referer": "https://data.eastmoney.com/"})

_COLUMNS = [
    "symbol",
    "trade_date",
    "available_date",
    "price",
    "volume",
    "amount",
    "discount_rate",
    "buyer",
    "seller",
    "source",
    "raw_update_time",
    "fetched_at",
]


def _normalise_symbol(symbol: str) -> str:
    s = str(symbol).strip().upper()
    if "." in s:
        s = s.split(".", 1)[0]
    if s.startswith(("SH", "SZ", "BJ")):
        s = s[2:]
    return s.zfill(6)


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=_COLUMNS)


def _next_trading_day_map(calendar_dates: list[dt.date]) -> dict[dt.date, dt.date]:
    dates = sorted(pd.to_datetime(calendar_dates).date)
    return {dates[i]: dates[i + 1] for i in range(len(dates) - 1)}


def _compute_available_date(trade_dates: pd.Series, next_map: dict) -> pd.Series:
    """
    T+1 availability. FIXED (review finding #9): previously fell back to
    SAME-DAY availability (`.fillna(trade_date)`) for any trade_date at or
    beyond the end of the supplied trading_dates calendar -- a PIT-safety
    violation at the calendar boundary. Falls back to trade_date + 1
    CALENDAR day instead (never same-day), and logs a warning so callers
    know to extend their trading-day calendar.
    """
    available = trade_dates.map(next_map)
    missing = available.isna() & trade_dates.notna()
    if missing.any():
        logger.warning(
            "_compute_available_date: %d trade_date(s) beyond the supplied "
            "trading_dates calendar; falling back to trade_date + 1 calendar "
            "day (extend the calendar passed to run() to avoid this)",
            int(missing.sum()),
        )
    fallback = trade_dates.map(lambda d: d + dt.timedelta(days=1) if pd.notna(d) else d)
    return available.where(~missing, fallback)


def _datacenter_get(params: dict, retries: int = 3) -> dict:
    import time

    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            resp = _SESSION.get(_DATACENTER_URL, params=params, timeout=25)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            last_exc = exc
            time.sleep(1.0 + attempt)
    raise RuntimeError(f"datacenter-web request failed: {last_exc}")


def _fetch_datacenter_pages(params: dict) -> pd.DataFrame:
    first = _datacenter_get({**params, "pageNumber": "1"})
    result = first.get("result") or {}
    pages = int(result.get("pages") or 1)
    rows = list(result.get("data") or [])
    for page in range(2, pages + 1):
        payload = _datacenter_get({**params, "pageNumber": str(page)})
        rows.extend((payload.get("result") or {}).get("data") or [])
    return pd.DataFrame(rows)


def _fetch_block_trades(start_date: str, end_date: str, trading_dates: list[dt.date]) -> pd.DataFrame:
    start_raw = start_date.replace("-", "")
    end_raw = end_date.replace("-", "")
    start = "-".join([start_raw[:4], start_raw[4:6], start_raw[6:]])
    end = "-".join([end_raw[:4], end_raw[4:6], end_raw[6:]])
    raw = _fetch_datacenter_pages({
        "sortColumns": "SECURITY_CODE",
        "sortTypes": "1",
        "pageSize": "5000",
        "reportName": "RPT_DATA_BLOCKTRADE",
        "columns": (
            "TRADE_DATE,SECURITY_CODE,SECUCODE,SECURITY_NAME_ABBR,CHANGE_RATE,CLOSE_PRICE,"
            "DEAL_PRICE,PREMIUM_RATIO,DEAL_VOLUME,DEAL_AMT,TURNOVER_RATE,BUYER_NAME,SELLER_NAME,"
            "BUYER_CODE,SELLER_CODE"
        ),
        "source": "WEB",
        "client": "WEB",
        "filter": f"(SECURITY_TYPE_WEB=1)(TRADE_DATE>='{start}')(TRADE_DATE<='{end}')",
    })
    if raw is None or raw.empty:
        return _empty_frame()

    df = raw.rename(
        columns={
            "SECURITY_CODE": "symbol",
            "TRADE_DATE": "trade_date",
            "DEAL_PRICE": "price",
            "DEAL_VOLUME": "volume",
            "DEAL_AMT": "amount",
            "PREMIUM_RATIO": "discount_rate",
            "BUYER_NAME": "buyer",
            "SELLER_NAME": "seller",
        }
    ).copy()
    out = pd.DataFrame()
    out["symbol"] = df["symbol"].astype(str).map(_normalise_symbol)
    out["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce").dt.date
    out["price"] = pd.to_numeric(df.get("price"), errors="coerce")
    out["volume"] = pd.to_numeric(df.get("volume"), errors="coerce")
    out["amount"] = pd.to_numeric(df.get("amount"), errors="coerce")
    out["discount_rate"] = pd.to_numeric(df.get("discount_rate"), errors="coerce")
    out["buyer"] = df.get("buyer", pd.Series("", index=df.index)).fillna("").astype(str)
    out["seller"] = df.get("seller", pd.Series("", index=df.index)).fillna("").astype(str)
    out = out.dropna(subset=["trade_date"]).copy()
    next_map = _next_trading_day_map(trading_dates)
    out["available_date"] = _compute_available_date(out["trade_date"], next_map)
    out["source"] = "datacenter-web"
    out["raw_update_time"] = pd.NaT
    out["fetched_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    return out[_COLUMNS].sort_values(["symbol", "trade_date"]).reset_index(drop=True)


class BlockTradeCollector:
    def __init__(self, store_root: Path | str) -> None:
        self.store_root = Path(store_root)
        init_lake(self.store_root)

    def run(
        self,
        symbols: list[str],
        *,
        start_date: str,
        end_date: str,
        trading_dates: list[dt.date],
        overwrite: bool = False,
    ) -> dict[str, int]:
        wanted = {_normalise_symbol(s) for s in symbols}
        fetched = _fetch_block_trades(start_date, end_date, trading_dates)
        if not fetched.empty:
            fetched = fetched[fetched["symbol"].isin(wanted)].copy()

        results: dict[str, int] = {}
        for symbol in sorted(wanted):
            sym_df = fetched[fetched["symbol"] == symbol].copy() if not fetched.empty else _empty_frame()
            results[symbol] = self._write_one(symbol, sym_df, overwrite)
        return results

    def _write_one(self, symbol: str, fetched: pd.DataFrame, overwrite: bool) -> int:
        out_path = block_trade_path(self.store_root, symbol)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        existing = pd.DataFrame(columns=_COLUMNS)
        if out_path.exists() and not overwrite:
            existing = pd.read_parquet(out_path)
            for col in ("trade_date", "available_date"):
                if col in existing.columns:
                    existing[col] = pd.to_datetime(existing[col], errors="coerce").dt.date

        combined = fetched if overwrite or existing.empty else pd.concat([existing, fetched], ignore_index=True)
        if not combined.empty:
            combined = (
                combined[_COLUMNS]
                .drop_duplicates(subset=["symbol", "trade_date", "price", "volume", "buyer", "seller"], keep="last")
                .sort_values(["trade_date", "amount"])
                .reset_index(drop=True)
            )
        else:
            combined = _empty_frame()
        combined.to_parquet(out_path, index=False)
        return int(len(fetched))


def load_block_trade(store_root: Path | str, symbol: str) -> pd.DataFrame:
    path = block_trade_path(store_root, _normalise_symbol(symbol))
    if not path.exists():
        return _empty_frame()
    df = pd.read_parquet(path)
    for col in ("trade_date", "available_date"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.date
    return df
