"""CNINFO-style announcement event collector.

The local AKShare wrapper ultimately reads the public announcement pages and
already supports per-symbol date ranges.  This collector normalises that data
into the project's silver schema without downloading PDFs or page bodies.
"""

from __future__ import annotations

import datetime as dt
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

from quant_platform.core.logging import get_logger
from quant_platform.store.lake import announcement_events_path, init_lake

logger = get_logger(__name__)

_EMPTY_COLUMNS = [
    "symbol",
    "announce_date",
    "event_date",
    "title",
    "category",
    "category_code",
    "pdf_url",
    "adjunct_url",
    "source",
    "raw_update_time",
    "fetched_at",
]


def _normalise_symbol(symbol: str) -> str:
    return str(symbol).strip().upper().split(".", 1)[0].removeprefix("SH").removeprefix("SZ").zfill(6)


def _empty_frame(symbol: str) -> pd.DataFrame:
    df = pd.DataFrame(columns=_EMPTY_COLUMNS)
    df["symbol"] = pd.Series(dtype=str)
    return df.assign(symbol=_normalise_symbol(symbol)).iloc[0:0]


_CNINFO_STOCK_URL = "http://www.cninfo.com.cn/new/data/szse_stock.json"
_CNINFO_QUERY_URL = "http://www.cninfo.com.cn/new/hisAnnouncement/query"
_CNINFO_STATIC_PREFIX = "http://static.cninfo.com.cn/"
_SESSION = requests.Session()
_SESSION.trust_env = False
_SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer": "http://www.cninfo.com.cn/new/commonUrl/pageOfSearch?url=disclosure/list/search",
    "Origin": "http://www.cninfo.com.cn",
    "X-Requested-With": "XMLHttpRequest",
})
_STOCK_META: dict[str, dict] | None = None


def _load_cninfo_stock_meta() -> dict[str, dict]:
    global _STOCK_META
    if _STOCK_META is not None:
        return _STOCK_META
    resp = _SESSION.get(_CNINFO_STOCK_URL, timeout=20)
    resp.raise_for_status()
    rows = resp.json().get("stockList") or []
    _STOCK_META = {str(r.get("code", "")).zfill(6): r for r in rows if r.get("code") and r.get("orgId")}
    return _STOCK_META


def _column_and_plate(org_id: str) -> tuple[str, str]:
    if str(org_id).startswith("gssh"):
        return "sse", "sh"
    if str(org_id).startswith("gsbj"):
        return "bj", "bj"
    return "szse", "sz"


def _fetch_announcement_events(
    symbol: str,
    start_date: str,
    end_date: str,
    *,
    page_size: int = 30,
) -> pd.DataFrame:
    """Fetch one symbol's cninfo announcement headers with seDate/pageNum."""

    symbol = _normalise_symbol(symbol)
    meta = _load_cninfo_stock_meta().get(symbol)
    if not meta:
        return _empty_frame(symbol)

    session = requests.Session()
    session.trust_env = False
    session.headers.update(_SESSION.headers)
    org_id = str(meta["orgId"])
    column, plate = _column_and_plate(org_id)
    stock = f"{symbol},{org_id}"
    rows: list[dict] = []
    page_num = 1
    while True:
        data = {
            "pageNum": page_num,
            "pageSize": page_size,
            "column": column,
            "tabName": "fulltext",
            "plate": plate,
            "stock": stock,
            "searchkey": "",
            "secid": "",
            "category": "",
            "trade": "",
            "seDate": f"{start_date}~{end_date}",
            "sortName": "",
            "sortType": "",
            "isHLtitle": "true",
        }
        resp = session.post(_CNINFO_QUERY_URL, data=data, timeout=20)
        resp.raise_for_status()
        payload = resp.json()
        page_rows = payload.get("announcements") or []
        rows.extend(page_rows)
        has_more = bool(payload.get("hasMore"))
        total_pages = int(payload.get("totalpages") or 0)
        if not has_more and (total_pages <= 0 or page_num >= total_pages):
            break
        page_num += 1
        if page_num > 200:
            raise RuntimeError(f"cninfo pagination exceeded safety limit for {symbol}")
        time.sleep(0.05)

    if not rows:
        return _empty_frame(symbol)

    fetched_at = dt.datetime.now(dt.timezone.utc).isoformat()
    records = []
    for row in rows:
        ts = row.get("announcementTime")
        announce_ts = pd.to_datetime(ts, unit="ms", errors="coerce") if ts else pd.NaT
        announce_date = announce_ts.date() if not pd.isna(announce_ts) else pd.NaT
        adjunct = str(row.get("adjunctUrl") or "")
        records.append({
            "symbol": _normalise_symbol(row.get("secCode") or symbol),
            "announce_date": announce_date,
            "event_date": announce_date,
            "title": str(row.get("announcementTitle") or row.get("shortTitle") or ""),
            "category": str(row.get("announcementTypeName") or row.get("columnId") or ""),
            "category_code": str(row.get("announcementType") or ""),
            "pdf_url": f"{_CNINFO_STATIC_PREFIX}{adjunct}" if adjunct else "",
            "adjunct_url": adjunct,
            "source": "cninfo",
            "raw_update_time": row.get("storageTime"),
            "fetched_at": fetched_at,
        })

    out = pd.DataFrame(records).dropna(subset=["announce_date"])
    return out[_EMPTY_COLUMNS].sort_values(["announce_date", "title"]).reset_index(drop=True)


class AnnouncementEventsCollector:
    def __init__(self, store_root: Path | str, sleep_seconds: float = 0.0, max_workers: int = 8) -> None:
        self.store_root = Path(store_root)
        self.sleep_seconds = sleep_seconds
        self.max_workers = max_workers
        init_lake(self.store_root)

    def run(
        self,
        symbols: list[str],
        *,
        start_date: str,
        end_date: str,
        overwrite: bool = False,
    ) -> dict[str, int]:
        _load_cninfo_stock_meta()
        results: dict[str, int] = {}
        work = []
        for raw_symbol in symbols:
            symbol = _normalise_symbol(raw_symbol)
            if not overwrite and self._is_existing_current(symbol, end_date):
                results[symbol] = 0
            else:
                work.append(symbol)
        if not work:
            return results

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {pool.submit(_fetch_announcement_events, symbol, start_date, end_date): symbol for symbol in work}
            for i, future in enumerate(as_completed(futures), 1):
                symbol = futures[future]
                try:
                    fetched = future.result()
                    results[symbol] = self._write_one(symbol, fetched, overwrite)
                except Exception as exc:
                    logger.warning("%s: announcement fetch failed: %s", symbol, exc)
                    results[symbol] = -1
                if self.sleep_seconds > 0:
                    time.sleep(self.sleep_seconds)
                if i % 50 == 0:
                    logger.info("Announcement events: %d/%d pending symbols", i, len(work))
        return results

    def _is_existing_current(self, symbol: str, end_date: str) -> bool:
        path = announcement_events_path(self.store_root, symbol)
        if not path.exists():
            return False
        try:
            df = pd.read_parquet(path, columns=["fetched_at"])
            return len(df) >= 0 and path.stat().st_size > 0
        except Exception:
            return False

    def _write_one(self, symbol: str, fetched: pd.DataFrame, overwrite: bool) -> int:
        out_path = announcement_events_path(self.store_root, symbol)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        existing = pd.DataFrame(columns=_EMPTY_COLUMNS)
        if out_path.exists() and not overwrite:
            existing = pd.read_parquet(out_path)
            if not existing.empty:
                existing["announce_date"] = pd.to_datetime(existing["announce_date"], errors="coerce").dt.date

        combined = fetched if overwrite or existing.empty else pd.concat([existing, fetched], ignore_index=True)
        if combined.empty:
            combined = _empty_frame(symbol)
        else:
            combined = (
                combined[_EMPTY_COLUMNS]
                .drop_duplicates(subset=["symbol", "announce_date", "title", "pdf_url"], keep="last")
                .sort_values(["announce_date", "title"])
                .reset_index(drop=True)
            )
        combined.to_parquet(out_path, index=False)
        return int(len(fetched))


def load_announcement_events(store_root: Path | str, symbol: str) -> pd.DataFrame:
    path = announcement_events_path(store_root, _normalise_symbol(symbol))
    if not path.exists():
        return _empty_frame(symbol)
    df = pd.read_parquet(path)
    if not df.empty:
        df["announce_date"] = pd.to_datetime(df["announce_date"], errors="coerce").dt.date
    return df
