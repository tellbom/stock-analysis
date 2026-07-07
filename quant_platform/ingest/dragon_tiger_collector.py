"""Dragon Tiger List collector backed by Eastmoney datacenter via AKShare."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import requests

from quant_platform.core.logging import get_logger
from quant_platform.store.lake import dragon_tiger_path, init_lake

logger = get_logger(__name__)
_DATACENTER_URL = "https://datacenter-web.eastmoney.com/api/data/v1/get"
_SESSION = requests.Session()
_SESSION.trust_env = False
_SESSION.headers.update({"User-Agent": "Mozilla/5.0", "Referer": "https://data.eastmoney.com/"})

_COLUMNS = [
    "symbol",
    "trade_date",
    "available_date",
    "reason",
    "buy_amount",
    "sell_amount",
    "net_buy_amount",
    "institution_buy_amount",
    "institution_sell_amount",
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
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            resp = _SESSION.get(_DATACENTER_URL, params=params, timeout=25)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            last_exc = exc
            time_sleep = 1.0 + attempt
            import time

            time.sleep(time_sleep)
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


def _fetch_dragon_tiger(start_date: str, end_date: str, trading_dates: list[dt.date]) -> pd.DataFrame:
    start = "-".join([start_date.replace("-", "")[:4], start_date.replace("-", "")[4:6], start_date.replace("-", "")[6:]])
    end = "-".join([end_date.replace("-", "")[:4], end_date.replace("-", "")[4:6], end_date.replace("-", "")[6:]])
    detail = _fetch_datacenter_pages({
        "sortColumns": "SECURITY_CODE,TRADE_DATE",
        "sortTypes": "1,-1",
        "pageSize": "500",
        "reportName": "RPT_DAILYBILLBOARD_DETAILSNEW",
        "columns": (
            "SECURITY_CODE,SECUCODE,SECURITY_NAME_ABBR,TRADE_DATE,EXPLAIN,CLOSE_PRICE,CHANGE_RATE,"
            "BILLBOARD_NET_AMT,BILLBOARD_BUY_AMT,BILLBOARD_SELL_AMT,BILLBOARD_DEAL_AMT,ACCUM_AMOUNT,"
            "DEAL_NET_RATIO,DEAL_AMOUNT_RATIO,TURNOVERRATE,FREE_MARKET_CAP,EXPLANATION"
        ),
        "source": "WEB",
        "client": "WEB",
        "filter": f"(TRADE_DATE<='{end}')(TRADE_DATE>='{start}')",
    })
    inst = _fetch_datacenter_pages({
        "sortColumns": "NET_BUY_AMT",
        "sortTypes": "-1",
        "pageSize": "500",
        "reportName": "RPT_ORGANIZATION_TRADE_DETAILS",
        "columns": "SECURITY_CODE,SECURITY_NAME_ABBR,CLOSE_PRICE,CHANGE_RATE,BUY_TIMES,SELL_TIMES,BUY_AMT,SELL_AMT,NET_BUY_AMT,ACCUM_AMOUNT,DEAL_NET_RATIO,TURNOVERRATE,FREE_MARKET_CAP,EXPLANATION,TRADE_DATE",
        "source": "WEB",
        "client": "WEB",
        "filter": f"(TRADE_DATE<='{end}')(TRADE_DATE>='{start}')",
    })
    if (detail is None or detail.empty) and (inst is None or inst.empty):
        return _empty_frame()

    frames = []
    if detail is not None and not detail.empty:
        df = detail.rename(
            columns={
                "SECURITY_CODE": "symbol",
                "TRADE_DATE": "trade_date",
                "EXPLANATION": "reason",
                "BILLBOARD_BUY_AMT": "buy_amount",
                "BILLBOARD_SELL_AMT": "sell_amount",
                "BILLBOARD_NET_AMT": "net_buy_amount",
            }
        ).copy()
        out = pd.DataFrame()
        out["symbol"] = df["symbol"].astype(str).map(_normalise_symbol)
        out["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce").dt.date
        out["reason"] = df.get("reason", pd.Series("", index=df.index)).fillna("").astype(str)
        for col in ("buy_amount", "sell_amount", "net_buy_amount"):
            out[col] = pd.to_numeric(df.get(col), errors="coerce")
        out["institution_buy_amount"] = pd.NA
        out["institution_sell_amount"] = pd.NA
        frames.append(out)

    combined = pd.concat(frames, ignore_index=True) if frames else _empty_frame()

    if inst is not None and not inst.empty:
        idf = inst.rename(
            columns={
                "SECURITY_CODE": "symbol",
                "TRADE_DATE": "trade_date",
                "BUY_AMT": "institution_buy_amount",
                "SELL_AMT": "institution_sell_amount",
            }
        ).copy()
        inst_norm = pd.DataFrame()
        inst_norm["symbol"] = idf["symbol"].astype(str).map(_normalise_symbol)
        inst_norm["trade_date"] = pd.to_datetime(idf["trade_date"], errors="coerce").dt.date
        inst_norm["institution_buy_amount"] = pd.to_numeric(idf.get("institution_buy_amount"), errors="coerce")
        inst_norm["institution_sell_amount"] = pd.to_numeric(idf.get("institution_sell_amount"), errors="coerce")
        inst_norm = (
            inst_norm.dropna(subset=["trade_date"])
            .groupby(["symbol", "trade_date"], as_index=False)[["institution_buy_amount", "institution_sell_amount"]]
            .sum(min_count=1)
        )
        if combined.empty:
            combined = inst_norm.copy()
            combined["reason"] = ""
            combined["buy_amount"] = pd.NA
            combined["sell_amount"] = pd.NA
            combined["net_buy_amount"] = pd.NA
        else:
            combined = combined.merge(inst_norm, on=["symbol", "trade_date"], how="left", suffixes=("", "_inst"))
            for col in ("institution_buy_amount", "institution_sell_amount"):
                alt = f"{col}_inst"
                if alt in combined.columns:
                    combined[col] = combined[alt].combine_first(combined[col])
                    combined = combined.drop(columns=[alt])

    combined = combined.dropna(subset=["trade_date"]).copy()
    next_map = _next_trading_day_map(trading_dates)
    combined["available_date"] = _compute_available_date(combined["trade_date"], next_map)
    combined["source"] = "datacenter-web"
    combined["raw_update_time"] = pd.NaT
    combined["fetched_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    return (
        combined[_COLUMNS]
        .drop_duplicates(subset=["symbol", "trade_date", "reason"], keep="last")
        .sort_values(["symbol", "trade_date"])
        .reset_index(drop=True)
    )


class DragonTigerCollector:
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
        fetched = _fetch_dragon_tiger(start_date, end_date, trading_dates)
        if not fetched.empty:
            fetched = fetched[fetched["symbol"].isin(wanted)].copy()

        results: dict[str, int] = {}
        for symbol in sorted(wanted):
            sym_df = fetched[fetched["symbol"] == symbol].copy() if not fetched.empty else _empty_frame()
            results[symbol] = self._write_one(symbol, sym_df, overwrite)
        return results

    def _write_one(self, symbol: str, fetched: pd.DataFrame, overwrite: bool) -> int:
        out_path = dragon_tiger_path(self.store_root, symbol)
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
                .drop_duplicates(subset=["symbol", "trade_date", "reason"], keep="last")
                .sort_values(["trade_date", "reason"])
                .reset_index(drop=True)
            )
        else:
            combined = _empty_frame()
        combined.to_parquet(out_path, index=False)
        return int(len(fetched))


def load_dragon_tiger(store_root: Path | str, symbol: str) -> pd.DataFrame:
    path = dragon_tiger_path(store_root, _normalise_symbol(symbol))
    if not path.exists():
        return _empty_frame()
    df = pd.read_parquet(path)
    for col in ("trade_date", "available_date"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.date
    return df
