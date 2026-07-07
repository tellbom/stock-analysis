"""
Fund-flow provider abstraction and optional multi-source adapters.

Providers return a canonical frame with ``trade_date``.  The collector keeps a
``date`` alias when writing silver so existing feature builders continue to
work while reports can audit the new schema.
"""

from __future__ import annotations

import datetime as dt
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import pandas as pd

import json
import time

import urllib3

from quant_platform.ingest.flow_collector import (
    _FETCH_BACKOFF_SECONDS,
    _FETCH_RETRIES,
    _fetch_push2his_result,
    _market_code,
    _normalise_symbol,
)


CANONICAL_FUND_FLOW_COLUMNS = [
    "symbol",
    "trade_date",
    "main_net",
    "main_net_rate",
    "super_net",
    "super_net_rate",
    "large_net",
    "large_net_rate",
    "medium_net",
    "mid_net",
    "medium_net_rate",
    "small_net",
    "small_net_rate",
    "close",
    "pct_change",
    "source",
    "raw_update_time",
    "fetched_at",
]

EMDATAH5_SOURCE = "eastmoney_emdatah5_zjlx"
_EMDATAH5_URL = "https://emdatah5.eastmoney.com/dc/ZJLX/getDBHistoryData"
_EMDATAH5_UT = "b2884a393a59ad64002292a3e90d46a5"
_EMDATAH5_HTTP = urllib3.PoolManager(num_pools=1, maxsize=1, retries=0)


@dataclass(frozen=True)
class ProviderResult:
    frame: pd.DataFrame
    missing_fields: list[str]
    retry_count: int = 0
    same_source_risk: str = ""

    @property
    def ok(self) -> bool:
        return not self.frame.empty


class FundFlowProvider(ABC):
    name: str = "base"
    same_source_risk: str = ""

    @abstractmethod
    def fetch_symbol(self, symbol: str, days: int = 120) -> ProviderResult:
        """Fetch one symbol and return canonical fund-flow columns."""

    def _canonicalise(
        self,
        df: pd.DataFrame,
        symbol: str,
        *,
        date_col: str,
        mapping: dict[str, str],
        missing_ok: bool = True,
    ) -> ProviderResult:
        symbol = _normalise_symbol(symbol)
        out = pd.DataFrame()
        out["symbol"] = [symbol] * len(df)
        out["trade_date"] = pd.to_datetime(df[date_col], errors="coerce").dt.date
        for canonical in CANONICAL_FUND_FLOW_COLUMNS:
            if canonical in ("symbol", "trade_date", "source", "raw_update_time", "fetched_at"):
                continue
            src = mapping.get(canonical)
            if src and src in df.columns:
                out[canonical] = pd.to_numeric(df[src], errors="coerce")
            else:
                out[canonical] = pd.NA
        out["source"] = df["source"].astype(str) if "source" in df.columns else self.name
        out["raw_update_time"] = df["raw_update_time"] if "raw_update_time" in df.columns else None
        out["fetched_at"] = (
            df["fetched_at"]
            if "fetched_at" in df.columns
            else dt.datetime.now(dt.timezone.utc).isoformat()
        )
        out = out.dropna(subset=["trade_date"]).sort_values("trade_date").reset_index(drop=True)
        missing = [
            c for c in CANONICAL_FUND_FLOW_COLUMNS
            if c not in out.columns or out[c].isna().all()
        ]
        if not missing_ok and missing:
            raise ValueError(f"{self.name}: missing canonical fields {missing}")
        return ProviderResult(out[CANONICAL_FUND_FLOW_COLUMNS], missing, same_source_risk=self.same_source_risk)


class NativeEastmoneyFundFlowProvider(FundFlowProvider):
    name = "native_eastmoney"
    same_source_risk = "Eastmoney push2his host"

    def fetch_symbol(self, symbol: str, days: int = 120) -> ProviderResult:
        df, error, retries = _fetch_push2his_result(symbol)
        if df.empty:
            raise RuntimeError(error or "empty native_eastmoney response")
        if days and len(df) > days:
            df = df.tail(days)
        return self._canonicalise(
            df,
            symbol,
            date_col="date",
            mapping={
                "main_net": "main_net",
                "super_net": "super_net",
                "large_net": "large_net",
                "medium_net": "mid_net",
                "mid_net": "mid_net",
                "small_net": "small_net",
                "main_net_rate": "main_net_rate",
                "small_net_rate": "small_net_rate",
                "medium_net_rate": "medium_net_rate",
                "large_net_rate": "large_net_rate",
                "super_net_rate": "super_net_rate",
                "close": "close",
                "pct_change": "pct_change",
            },
        )


class EastmoneyH5FundFlowProvider(FundFlowProvider):
    name = "eastmoney_emdatah5"
    same_source_risk = "Eastmoney H5 emdatah5 ZJLX host"

    def fetch_symbol(self, symbol: str, days: int = 120) -> ProviderResult:
        code = _normalise_symbol(symbol)
        market = _market_code(code)
        params = {
            "secid": f"{market}.{code}",
            "fields1": "f1,f2,f3,f7",
            "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63",
            "ut": _EMDATAH5_UT,
        }
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Referer": f"https://emdatah5.eastmoney.com/dc/zjlx/stock?fc={market}.{code}",
            "Accept": "application/json,text/plain,*/*",
        }

        last_error = ""
        retry_count = 0
        data: dict[str, Any] = {}
        klines: list[str] = []
        for attempt in range(_FETCH_RETRIES):
            retry_count = attempt + 1
            try:
                response = _EMDATAH5_HTTP.request(
                    "GET",
                    _EMDATAH5_URL,
                    fields=params,
                    headers=headers,
                    timeout=urllib3.Timeout(connect=5.0, read=15.0),
                    preload_content=True,
                )
                status_code = getattr(response, "status", 200)
                body = getattr(response, "data", b"")
                text = body.decode("utf-8", errors="replace") if isinstance(body, bytes) else str(body)
                if isinstance(status_code, int) and status_code != 200:
                    last_error = f"HTTP {status_code}: {text[:200]}"
                    _EMDATAH5_HTTP.clear()
                    time.sleep(_FETCH_BACKOFF_SECONDS[min(attempt, len(_FETCH_BACKOFF_SECONDS) - 1)])
                    continue
                payload = json.loads(text)
                data = payload.get("data") or {}
                klines = data.get("klines") or []
                if not klines:
                    last_error = f"empty klines for secid={market}.{code}; payload_head={str(payload)[:300]}"
                    _EMDATAH5_HTTP.clear()
                    time.sleep(_FETCH_BACKOFF_SECONDS[min(attempt, len(_FETCH_BACKOFF_SECONDS) - 1)])
                    continue
                break
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                _EMDATAH5_HTTP.clear()
                time.sleep(_FETCH_BACKOFF_SECONDS[min(attempt, len(_FETCH_BACKOFF_SECONDS) - 1)])
        else:
            raise RuntimeError(last_error or "unknown emdatah5 fetch failure")

        fetched_at = dt.datetime.now(dt.timezone.utc).isoformat()
        raw_update_time = (
            data.get("updateTime")
            or data.get("update_time")
            or data.get("lastUpdateTime")
            or data.get("last_update_time")
        )
        rows = []
        for line in klines:
            parts = str(line).split(",")
            if len(parts) < 13:
                continue
            rows.append({
                "symbol": code,
                "trade_date": parts[0],
                "date": parts[0],
                "main_net": _parse_h5_number(parts[1]),
                "small_net": _parse_h5_number(parts[2]),
                "medium_net": _parse_h5_number(parts[3]),
                "mid_net": _parse_h5_number(parts[3]),
                "large_net": _parse_h5_number(parts[4]),
                "super_net": _parse_h5_number(parts[5]),
                "main_net_rate": _parse_h5_number(parts[6]),
                "small_net_rate": _parse_h5_number(parts[7]),
                "medium_net_rate": _parse_h5_number(parts[8]),
                "large_net_rate": _parse_h5_number(parts[9]),
                "super_net_rate": _parse_h5_number(parts[10]),
                "close": _parse_h5_number(parts[11]),
                "pct_change": _parse_h5_number(parts[12]),
                "source": EMDATAH5_SOURCE,
                "raw_update_time": raw_update_time,
                "fetched_at": fetched_at,
            })

        if not rows:
            raise RuntimeError("all emdatah5 klines failed to parse")
        df = pd.DataFrame(rows)
        df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce").dt.date
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
        df = df.dropna(subset=["trade_date"]).sort_values("trade_date").reset_index(drop=True)
        if days and len(df) > days:
            df = df.tail(days).reset_index(drop=True)

        result = self._canonicalise(
            df,
            code,
            date_col="trade_date",
            mapping={
                "main_net": "main_net",
                "small_net": "small_net",
                "medium_net": "medium_net",
                "mid_net": "mid_net",
                "large_net": "large_net",
                "super_net": "super_net",
                "main_net_rate": "main_net_rate",
                "small_net_rate": "small_net_rate",
                "medium_net_rate": "medium_net_rate",
                "large_net_rate": "large_net_rate",
                "super_net_rate": "super_net_rate",
                "close": "close",
                "pct_change": "pct_change",
            },
        )
        required = [
            "main_net",
            "small_net",
            "medium_net",
            "large_net",
            "super_net",
            "main_net_rate",
            "small_net_rate",
            "medium_net_rate",
            "large_net_rate",
            "super_net_rate",
            "close",
            "pct_change",
        ]
        missing_required = [c for c in required if c in result.missing_fields]
        if missing_required:
            raise RuntimeError(f"eastmoney_emdatah5 missing required fields {missing_required}")
        return ProviderResult(
            result.frame,
            result.missing_fields,
            retry_count=retry_count,
            same_source_risk=self.same_source_risk,
        )


def _parse_h5_number(value: str | None) -> float | None:
    if value in ("", "-", None):
        return None
    return float(value)


class AKShareFundFlowProvider(FundFlowProvider):
    name = "akshare"
    same_source_risk = "AKShare stock_individual_fund_flow uses Eastmoney push2his"

    def fetch_symbol(self, symbol: str, days: int = 120) -> ProviderResult:
        try:
            import akshare as ak
        except ImportError as exc:
            raise ImportError("akshare is not installed") from exc

        code = _normalise_symbol(symbol)
        market = "sh" if _market_code(code) == 1 else "bj" if _market_code(code) == 2 else "sz"
        df = ak.stock_individual_fund_flow(stock=code, market=market)
        if df is None or df.empty:
            raise RuntimeError("empty akshare stock_individual_fund_flow response")
        if days and len(df) > days:
            df = df.tail(days)

        cols = {str(c): c for c in df.columns}
        mapping = {
            "main_net": cols.get("主力净流入-净额"),
            "main_net_rate": cols.get("主力净流入-净占比"),
            "super_net": cols.get("超大单净流入-净额"),
            "super_net_rate": cols.get("超大单净流入-净占比"),
            "large_net": cols.get("大单净流入-净额"),
            "large_net_rate": cols.get("大单净流入-净占比"),
            "medium_net": cols.get("中单净流入-净额"),
            "medium_net_rate": cols.get("中单净流入-净占比"),
            "small_net": cols.get("小单净流入-净额"),
            "small_net_rate": cols.get("小单净流入-净占比"),
        }
        date_col = cols.get("日期") or cols.get("date") or cols.get("trade_date")
        if not date_col:
            raise RuntimeError(f"akshare response has no date column: {list(df.columns)}")
        return self._canonicalise(df, code, date_col=date_col, mapping=mapping)


class ADataFundFlowProvider(FundFlowProvider):
    name = "adata"
    same_source_risk = "adata documented stock.market.get_capital_flow data source may be Eastmoney"

    def fetch_symbol(self, symbol: str, days: int = 120) -> ProviderResult:
        try:
            import adata
        except ImportError as exc:
            raise ImportError("adata is not installed; optional dependency: pip install adata") from exc

        code = _normalise_symbol(symbol)
        fn = getattr(getattr(adata, "stock").market, "get_capital_flow")
        try:
            df = fn(stock_code=code)
        except TypeError:
            df = fn(code)
        if df is None or df.empty:
            raise RuntimeError("empty adata get_capital_flow response")
        if days and len(df) > days:
            df = df.tail(days)
        return self._canonicalise_guess(df, code)

    def _canonicalise_guess(self, df: pd.DataFrame, symbol: str) -> ProviderResult:
        return _canonicalise_by_guess(self, df, symbol)


class QStockFundFlowProvider(FundFlowProvider):
    name = "qstock"
    same_source_risk = "qstock data module aggregates public web sources including Eastmoney/THS/Sina"

    def fetch_symbol(self, symbol: str, days: int = 120) -> ProviderResult:
        try:
            import qstock as qs
        except ImportError as exc:
            raise ImportError("qstock is not installed; optional dependency: pip install qstock") from exc

        code = _normalise_symbol(symbol)
        candidates = [
            "stock_money",
            "stock_fund_flow",
            "money_flow",
            "fund_flow",
            "ths_money",
        ]
        last_error: Exception | None = None
        for name in candidates:
            fn = getattr(qs, name, None)
            if not callable(fn):
                continue
            try:
                df = fn(code)
                if df is not None and not df.empty:
                    if days and len(df) > days:
                        df = df.tail(days)
                    return _canonicalise_by_guess(self, df, code)
            except Exception as exc:
                last_error = exc
        if last_error:
            raise RuntimeError(f"qstock provider failed: {last_error}") from last_error
        raise RuntimeError("qstock provider has no recognised fund-flow function")


def _canonicalise_by_guess(provider: FundFlowProvider, df: pd.DataFrame, symbol: str) -> ProviderResult:
    cols = {str(c).lower(): c for c in df.columns}
    raw_cols = {str(c): c for c in df.columns}

    def pick(*names: str) -> Any:
        for name in names:
            if name in raw_cols:
                return raw_cols[name]
            if name.lower() in cols:
                return cols[name.lower()]
        return None

    date_col = pick("trade_date", "date", "日期", "净流入日期")
    if not date_col:
        raise RuntimeError(f"{provider.name}: no recognisable date column: {list(df.columns)}")
    mapping = {
        "main_net": pick("main_net", "主力净流入", "主力净流入-净额", "主力净额"),
        "main_net_rate": pick("main_net_rate", "主力净流入占比", "主力净流入-净占比"),
        "super_net": pick("super_net", "超大单净流入", "超大单净流入-净额"),
        "super_net_rate": pick("super_net_rate", "超大单净流入占比", "超大单净流入-净占比"),
        "large_net": pick("large_net", "大单净流入", "大单净流入-净额"),
        "large_net_rate": pick("large_net_rate", "大单净流入占比", "大单净流入-净占比"),
        "medium_net": pick("medium_net", "mid_net", "中单净流入", "中单净流入-净额"),
        "medium_net_rate": pick("medium_net_rate", "中单净流入占比", "中单净流入-净占比"),
        "small_net": pick("small_net", "小单净流入", "小单净流入-净额"),
        "small_net_rate": pick("small_net_rate", "小单净流入占比", "小单净流入-净占比"),
    }
    return provider._canonicalise(df, symbol, date_col=date_col, mapping=mapping)


def default_fund_flow_providers() -> list[FundFlowProvider]:
    return [
        EastmoneyH5FundFlowProvider(),
    ]
