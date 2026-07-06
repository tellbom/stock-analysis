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

from quant_platform.ingest.flow_collector import (
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
    "medium_net_rate",
    "small_net",
    "small_net_rate",
    "source",
    "raw_update_time",
    "fetched_at",
]


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
        out["source"] = self.name
        out["raw_update_time"] = None
        out["fetched_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
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
                "small_net": "small_net",
            },
        )


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
        ADataFundFlowProvider(),
        QStockFundFlowProvider(),
        NativeEastmoneyFundFlowProvider(),
        AKShareFundFlowProvider(),
    ]
