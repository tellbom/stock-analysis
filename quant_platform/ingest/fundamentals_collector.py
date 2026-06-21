"""
ingest.fundamentals_collector
==============================
PIT (Point-in-Time) fundamentals collector.

T0.7 scope
----------
Collect financial summary data per symbol, store with both ``announce_date``
and ``period_end``, and expose an ``as_of`` query so features can join
fundamentals correctly without lookahead bias.

PIT design (the most important correctness decision in the project)
-------------------------------------------------------------------
Every fundamental figure is known to the market only when **announced**,
which is typically 1–4 weeks after the period ends.  Joining on ``period_end``
(the reporting quarter) instead of ``announce_date`` leaks future information
into training features — this is the single most common source of silent
lookahead bias in financial ML.

Schema (silver/fundamentals/<symbol>.parquet):
  symbol          str    6-digit code
  announce_date   date   date the report was disclosed to the market  ← JOIN KEY
  period_end      date   last day of the reporting period (e.g. 2023-09-30)
  period_type     str    "Q1" | "H1" | "Q3" | "annual"
  source          str    AKShare endpoint name used
  <metric cols>   float  financial metrics (revenue, net_profit, roe, eps, …)

Fetch strategy
--------------
``stock_yjyg_em`` (业绩预告) and ``stock_yjkb_em`` (业绩快报) are collected
by explicit report-period request date (e.g. ``date="20251231"``) and exact
``公告日期`` from the endpoint response.  Endpoints without exact
``announce_date`` are intentionally excluded; no estimated disclosure dates
are written.

All endpoints return 403 in the sandbox.  The collector fails loudly with
``FundamentalsFetchError``; tests use monkeypatching.

Network failure handling
------------------------
- Fail loudly: raises ``FundamentalsFetchError`` when all endpoints fail.
- No silent fallback or estimated data is written to the Parquet.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

from quant_platform.core.fetch import safe_call
from quant_platform.core.logging import get_logger
from quant_platform.store.lake import fundamentals_path, fundamentals_dir, init_lake
from quant_platform.store.schemas import enforce_fundamentals

logger = get_logger(__name__)

# Reporting period end dates for standard quarters
_PERIOD_ENDS = {
    "Q1":     "-03-31",
    "H1":     "-06-30",
    "Q3":     "-09-30",
    "annual": "-12-31",
}


class FundamentalsFetchError(RuntimeError):
    """Raised when all AKShare endpoints fail for a fundamentals fetch."""


# ---------------------------------------------------------------------------
# Column normalisation helpers
# ---------------------------------------------------------------------------

def _normalise_yjyg(
    df: pd.DataFrame,
    symbol: str,
    period_end: str | dt.date | None = None,
) -> pd.DataFrame:
    """
    Normalise stock_yjyg_em (业绩预告) output.
    Returns columns: symbol, announce_date, period_end, period_type, source,
                     forecast_metric, forecast_value, forecast_change_pct.
    """
    df = df.copy()
    rename = {
        "公告日期": "announce_date",
        "报告日期": "period_end",
        "预测指标": "forecast_metric",
        "业绩变动": "forecast_text",
        "预测数值": "forecast_value",
        "业绩变动幅度": "forecast_change_pct",
        "业绩变动原因": "forecast_reason",
        "净利润下限":  "net_profit_low",
        "净利润上限":  "net_profit_high",
        "同比增长下限": "yoy_low",
        "同比增长上限": "yoy_high",
        "预告类型":   "forecast_type",
        "上年同期值": "prior_value",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    df["symbol"] = symbol
    df["source"] = "yjyg_em"
    if period_end is not None:
        df["period_end"] = _parse_period_end(period_end)

    for col in ("announce_date", "period_end"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.date

    if "period_end" in df.columns:
        df["period_type"] = df["period_end"].apply(_infer_period_type)

    return df


def _normalise_yjkb(
    df: pd.DataFrame,
    symbol: str,
    period_end: str | dt.date | None = None,
) -> pd.DataFrame:
    """
    Normalise stock_yjkb_em (业绩快报) output.
    Returns columns: symbol, announce_date, period_end, period_type, source,
                     revenue, net_profit, eps, roe.
    """
    df = df.copy()
    rename = {
        "公告日期": "announce_date",
        "报告日期": "period_end",
        "营业收入": "revenue",
        "营业收入-营业收入": "revenue",
        "归母净利润": "net_profit",
        "净利润-净利润": "net_profit",
        "每股收益": "eps",
        "净资产收益率": "roe",
        "每股净资产": "nav_per_share",
        "所处行业": "industry",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    df["symbol"] = symbol
    df["source"] = "yjkb_em"
    if period_end is not None:
        df["period_end"] = _parse_period_end(period_end)

    for col in ("announce_date", "period_end"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.date

    if "period_end" in df.columns:
        df["period_type"] = df["period_end"].apply(_infer_period_type)

    return df


def _infer_period_type(period_end: dt.date | None) -> str:
    """Infer Q1/H1/Q3/annual from the month of period_end."""
    if period_end is None:
        return "unknown"
    m = period_end.month
    return {3: "Q1", 6: "H1", 9: "Q3", 12: "annual"}.get(m, "unknown")


def _parse_period_end(value: str | dt.date) -> dt.date:
    """Parse the report-period date passed to AKShare, e.g. 20251231."""
    parsed = pd.to_datetime(value, format="%Y%m%d", errors="coerce")
    if pd.isna(parsed):
        parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"Invalid report period date: {value!r}")
    return parsed.date()


# ---------------------------------------------------------------------------
# AKShare fetch attempts
# ---------------------------------------------------------------------------

def _fetch_yjyg(symbol: str, years: int) -> pd.DataFrame | None:
    """Fetch 业绩预告 for recent periods (has announce_date + period_end)."""
    try:
        import akshare as ak
    except ImportError:
        return None

    today = dt.date.today()
    dfs = []
    for yr in range(today.year, today.year - years - 1, -1):
        for q_end in ("1231", "0930", "0630", "0331"):
            date_str = f"{yr}{q_end}"
            df = safe_call(
                ak.stock_yjyg_em,
                date=date_str,
                label=f"yjyg_em {symbol} {date_str}",
                retries=2,
            )
            if df is not None and not df.empty:
                # Filter to this symbol
                code_col = next(
                    (c for c in df.columns if any(k in c for k in ("代码", "code", "Code"))),
                    None,
                )
                if code_col:
                    df = df[df[code_col].astype(str).str.strip().str.zfill(6) == symbol]
                if not df.empty:
                    dfs.append(_normalise_yjyg(df, symbol, period_end=date_str))
    return pd.concat(dfs, ignore_index=True) if dfs else None


def _fetch_yjkb(symbol: str, years: int) -> pd.DataFrame | None:
    """Fetch 业绩快报 for recent periods (has announce_date + period_end)."""
    try:
        import akshare as ak
    except ImportError:
        return None

    today = dt.date.today()
    dfs = []
    for yr in range(today.year, today.year - years - 1, -1):
        for q_end in ("1231", "0930", "0630", "0331"):
            date_str = f"{yr}{q_end}"
            df = safe_call(
                ak.stock_yjkb_em,
                date=date_str,
                label=f"yjkb_em {symbol} {date_str}",
                retries=2,
            )
            if df is not None and not df.empty:
                code_col = next(
                    (c for c in df.columns if any(k in c for k in ("代码", "code", "Code"))),
                    None,
                )
                if code_col:
                    df = df[df[code_col].astype(str).str.strip().str.zfill(6) == symbol]
                if not df.empty:
                    dfs.append(_normalise_yjkb(df, symbol, period_end=date_str))
    return pd.concat(dfs, ignore_index=True) if dfs else None


# ---------------------------------------------------------------------------
# Public: FundamentalsCollector
# ---------------------------------------------------------------------------

class FundamentalsCollector:
    """
    Collect PIT fundamentals for one symbol and write to the silver lake.

    Parameters
    ----------
    store_root : Path | str
        Root of the Parquet data lake.
    years : int
        How many years of history to collect.  Default 3.
    """

    def __init__(self, store_root: Path | str, years: int = 3) -> None:
        self.store_root = Path(store_root)
        self.years = years
        init_lake(self.store_root)

    def collect(self, symbol: str) -> pd.DataFrame:
        """
        Fetch fundamentals for *symbol* and write to Parquet.

        Merges only sources with exact announce_date and report-period
        semantics.  Raises ``FundamentalsFetchError`` if all exact sources
        fail — never writes fabricated or estimated data.

        Returns the stored DataFrame.
        """
        frames = []

        # --- Priority 1: yjyg (has exact announce_date) ---
        df_yg = _fetch_yjyg(symbol, self.years)
        if df_yg is not None and not df_yg.empty:
            frames.append(df_yg)
            logger.info("%s: yjyg_em → %d rows", symbol, len(df_yg))

        # --- Priority 2: yjkb (has exact announce_date) ---
        df_kb = _fetch_yjkb(symbol, self.years)
        if df_kb is not None and not df_kb.empty:
            frames.append(df_kb)
            logger.info("%s: yjkb_em → %d rows", symbol, len(df_kb))

        if not frames:
            raise FundamentalsFetchError(
                f"All fundamentals endpoints failed for '{symbol}'. "
                "Network may be restricted (403). No data written."
            )

        combined = pd.concat(frames, ignore_index=True)
        combined = enforce_fundamentals(combined, symbol)

        # Drop duplicate (symbol, announce_date, period_end) keeping last source
        combined = combined.drop_duplicates(
            subset=["symbol", "announce_date", "period_end"], keep="last"
        ).sort_values("announce_date").reset_index(drop=True)

        path = fundamentals_path(self.store_root, symbol)
        path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(path, index=False)
        logger.info("Fundamentals written → %s (%d rows)", path, len(combined))
        return combined

    def collect_universe(self, symbols: list[str]) -> dict[str, bool]:
        """
        Collect fundamentals for all *symbols*.
        Returns dict mapping symbol → success (True) | failure (False).
        Failures are logged but do not stop the run.
        """
        results: dict[str, bool] = {}
        for symbol in symbols:
            try:
                self.collect(symbol)
                results[symbol] = True
            except FundamentalsFetchError as exc:
                logger.error("FundamentalsFetchError for %s: %s", symbol, exc)
                results[symbol] = False
            except Exception as exc:
                logger.error("Unexpected error for %s: %s", symbol, exc)
                results[symbol] = False
        n_ok  = sum(results.values())
        n_bad = len(results) - n_ok
        logger.info("Fundamentals collection done: %d OK, %d failed", n_ok, n_bad)
        return results


# ---------------------------------------------------------------------------
# Public: PIT as-of query
# ---------------------------------------------------------------------------

def query_fundamentals_as_of(
    store_root: Path | str,
    symbol: str,
    as_of: dt.date,
) -> pd.DataFrame:
    """
    Return fundamental rows for *symbol* that were **known** as of *as_of*.

    Filters to rows where ``announce_date <= as_of``, i.e. only data the
    market actually had access to on that date.  This is the correct join
    for feature construction (T1.4) — joining on ``period_end`` would leak
    future information.

    Returns empty DataFrame if no Parquet exists or no rows satisfy the filter.
    Raises ValueError if the Parquet exists but is unreadable.
    """
    path = fundamentals_path(Path(store_root), symbol)
    if not path.exists():
        logger.debug("No fundamentals Parquet for %s (run collect first)", symbol)
        return pd.DataFrame()

    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        raise ValueError(f"Cannot read fundamentals for {symbol}: {exc}") from exc

    df["announce_date"] = pd.to_datetime(df["announce_date"]).dt.date
    mask = df["announce_date"] <= as_of
    result = df[mask].sort_values("announce_date").reset_index(drop=True)
    logger.debug(
        "%s as_of %s: %d/%d rows satisfy announce_date <= as_of",
        symbol, as_of, len(result), len(df),
    )
    return result


def get_latest_fundamentals_as_of(
    store_root: Path | str,
    symbol: str,
    as_of: dt.date,
) -> pd.Series | None:
    """
    Return the single most-recent fundamental row known as of *as_of*.
    Returns None if no data is available.  Used by T1.4 feature builder.
    """
    df = query_fundamentals_as_of(store_root, symbol, as_of)
    if df.empty:
        return None
    return df.sort_values("announce_date").iloc[-1]
