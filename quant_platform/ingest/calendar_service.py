"""
ingest.calendar_service
=======================
Trading Calendar Service for A-shares (Shanghai / Shenzhen exchanges).

T0.3 scope
----------
- Build and persist the master A-share trading calendar to Parquet.
- Expose ``is_trading_day(date)`` and ``trading_days_in_range(start, end)``.
- Used by the OHLCV collector (T0.5) for gap detection and by the label
  builder (T1.6) for forward-return shifting.

Data source strategy
--------------------
Primary:  ``exchange_calendars`` (PyPI, installable offline) — XSHG calendar.
          Covers 2006-06-21 → current year + 1.
          Accuracy: ~99%+; occasional Spring Festival make-up day (补班) may be
          off by ±1 day. This is noted in the quality report.

Fallback: ``akshare.tool_trade_date_hist_sina`` — exact SSE official dates,
          but requires live access to finance.sina.com.cn (403 in sandboxed
          environments). Attempted first; silently skipped if unavailable.

NOTE: The fallback order is reversed from typical expectation — AKShare is
*more accurate* but *less reliable*; exchange_calendars is always available.
When AKShare succeeds, its data wins and overwrites the Parquet cache.

Parquet schema
--------------
File:    ``<store_root>/calendar/trading_calendar.parquet``
Columns:
  date          date      the calendar date
  is_trading    bool      True if the exchange is open
  source        str       "exchange_calendars_XSHG" | "akshare_sina"

The full date range is stored (not just trading days) so gap checks can
identify missing dates vs. genuine non-trading days.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

from quant_platform.core.logging import get_logger

logger = get_logger(__name__)

# Parquet location relative to store root
_CALENDAR_SUBPATH = "calendar/trading_calendar.parquet"

# Accuracy note written into the quality report
CALENDAR_ACCURACY_NOTE = (
    "Trading calendar source: exchange_calendars (XSHG). "
    "Accuracy is ~99%+. Known limitation: occasional Spring Festival make-up "
    "trading days (补班) may be misclassified by ±1 day. "
    "For highest accuracy, run in an environment where AKShare / Sina Finance "
    "is reachable so the official SSE calendar can be used instead."
)


# ---------------------------------------------------------------------------
# Internal: build from exchange_calendars (always available)
# ---------------------------------------------------------------------------

def _build_from_exchange_calendars(
    start: dt.date,
    end: dt.date,
) -> pd.DataFrame:
    """
    Build a full date-range calendar DataFrame using exchange_calendars XSHG.
    Returns a frame with columns: date (date), is_trading (bool), source (str).
    """
    try:
        import exchange_calendars as ec
    except ImportError:
        raise ImportError(
            "exchange_calendars is not installed. "
            "Run: pip install exchange_calendars"
        )

    cal = ec.get_calendar("XSHG")
    cal_start = cal.sessions[0].date()
    cal_end   = cal.sessions[-1].date()

    # Clamp requested range to what the calendar actually covers
    eff_start = max(start, cal_start)
    eff_end   = min(end,   cal_end)

    if eff_start > eff_end:
        raise ValueError(
            f"Requested range {start}→{end} falls entirely outside "
            f"XSHG calendar coverage {cal_start}→{cal_end}."
        )

    if eff_start != start or eff_end != end:
        logger.warning(
            "Requested calendar range %s→%s clamped to XSHG coverage %s→%s",
            start, end, eff_start, eff_end,
        )

    trading_set = {s.date() for s in cal.sessions}

    all_dates = pd.date_range(eff_start, eff_end, freq="D")
    df = pd.DataFrame({
        "date":       [d.date() for d in all_dates],
        "is_trading": [d.date() in trading_set for d in all_dates],
        "source":     "exchange_calendars_XSHG",
    })
    logger.info(
        "Built calendar from exchange_calendars XSHG: %d days (%d trading) %s→%s",
        len(df), df["is_trading"].sum(), eff_start, eff_end,
    )
    return df


# ---------------------------------------------------------------------------
# Internal: build from AKShare (requires live Sina access)
# ---------------------------------------------------------------------------

def _build_from_akshare(start: dt.date, end: dt.date) -> pd.DataFrame | None:
    """
    Attempt to fetch the official SSE trading calendar from AKShare/Sina.
    Returns None (with a log warning) if AKShare is unavailable — never raises.
    """
    try:
        import akshare as ak
    except ImportError:
        logger.warning("akshare not installed; skipping AKShare calendar fetch")
        return None

    from quant_platform.core.fetch import safe_call

    raw = safe_call(
        ak.tool_trade_date_hist_sina,
        label="akshare sina trading calendar",
        retries=2,
    )
    if raw is None or raw.empty:
        logger.warning("AKShare calendar fetch returned no data (network likely blocked)")
        return None

    # Normalise
    raw = raw.copy()
    date_col = raw.columns[0]          # always "trade_date" but be defensive
    raw["date"] = pd.to_datetime(raw[date_col]).dt.date

    trading_set = set(raw["date"].tolist())
    all_dates = pd.date_range(start, end, freq="D")

    df = pd.DataFrame({
        "date":       [d.date() for d in all_dates],
        "is_trading": [d.date() in trading_set for d in all_dates],
        "source":     "akshare_sina",
    })
    logger.info(
        "Built calendar from AKShare/Sina: %d days (%d trading) %s→%s",
        len(df), df["is_trading"].sum(), start, end,
    )
    return df


# ---------------------------------------------------------------------------
# Public: CalendarService
# ---------------------------------------------------------------------------

class CalendarService:
    """
    Manages the master A-share trading calendar.

    Parameters
    ----------
    store_root : Path | str
        Root of the Parquet data lake.
    """

    def __init__(self, store_root: Path | str) -> None:
        self.store_root  = Path(store_root)
        self._parquet_path = self.store_root / _CALENDAR_SUBPATH
        self._cache: pd.DataFrame | None = None   # in-memory cache after first load

    # ------------------------------------------------------------------
    # Build & persist
    # ------------------------------------------------------------------

    def build_and_save(
        self,
        start: dt.date | str | None = None,
        end:   dt.date | str | None = None,
    ) -> pd.DataFrame:
        """
        Build the trading calendar and write it to Parquet.

        Tries AKShare first (most accurate); falls back to exchange_calendars
        if AKShare is unavailable.  Fails loudly only if *both* sources fail.

        Parameters
        ----------
        start : date | str, optional
            First date to include.  Defaults to 2010-01-01.
        end : date | str, optional
            Last date to include.  Defaults to today + 365 days (covers near future).
        """
        start = _as_date(start) if start else dt.date(2010, 1, 1)
        end   = _as_date(end)   if end   else dt.date.today() + dt.timedelta(days=365)

        logger.info("Building trading calendar %s → %s", start, end)

        # 1. Try AKShare (authoritative)
        df = _build_from_akshare(start, end)

        # 2. Fall back to exchange_calendars
        if df is None:
            logger.info("Using exchange_calendars as calendar source")
            df = _build_from_exchange_calendars(start, end)
            # _build_from_exchange_calendars raises if it also fails, which
            # propagates to the caller — that's the "fail loudly" contract.

        self._write(df)
        self._cache = df
        return df

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def is_trading_day(self, date: dt.date | str) -> bool:
        """Return True if *date* is an A-share trading day."""
        date = _as_date(date)
        df = self._load()
        row = df[df["date"] == date]
        if row.empty:
            raise KeyError(
                f"{date} is outside the calendar range. "
                "Run build_and_save() with a wider date range."
            )
        return bool(row.iloc[0]["is_trading"])

    def trading_days_in_range(
        self,
        start: dt.date | str,
        end:   dt.date | str,
    ) -> list[dt.date]:
        """
        Return sorted list of trading days in [start, end] inclusive.

        Raises KeyError if the range extends beyond the stored calendar.
        """
        start = _as_date(start)
        end   = _as_date(end)
        df = self._load()

        in_range = df[(df["date"] >= start) & (df["date"] <= end)]
        if in_range.empty:
            raise KeyError(
                f"No calendar data for {start}→{end}. "
                "Run build_and_save() first."
            )
        missing_boundary = (
            in_range["date"].min() > start or in_range["date"].max() < end
        )
        if missing_boundary:
            raise KeyError(
                f"Calendar only covers {in_range['date'].min()}→{in_range['date'].max()}, "
                f"but {start}→{end} was requested. Extend with build_and_save()."
            )
        return sorted(in_range[in_range["is_trading"]]["date"].tolist())

    def next_trading_day(self, date: dt.date | str) -> dt.date:
        """Return the first trading day strictly after *date*."""
        date = _as_date(date)
        df = self._load()
        after = df[(df["date"] > date) & (df["is_trading"])]
        if after.empty:
            raise KeyError(
                f"No trading day found after {date} in the stored calendar. "
                "Extend with build_and_save()."
            )
        return after.iloc[0]["date"]

    def prev_trading_day(self, date: dt.date | str) -> dt.date:
        """Return the most recent trading day strictly before *date*."""
        date = _as_date(date)
        df = self._load()
        before = df[(df["date"] < date) & (df["is_trading"])]
        if before.empty:
            raise KeyError(
                f"No trading day found before {date} in the stored calendar."
            )
        return before.iloc[-1]["date"]

    def calendar_df(self) -> pd.DataFrame:
        """Return the full calendar DataFrame (all dates, not just trading days)."""
        return self._load().copy()

    def source(self) -> str:
        """Return the data source string recorded in the calendar Parquet."""
        df = self._load()
        return str(df["source"].iloc[0]) if not df.empty else "unknown"

    def coverage(self) -> dict:
        """Summary dict for quality reports."""
        df = self._load()
        trading = df[df["is_trading"]]
        return {
            "source":               self.source(),
            "first_date":           str(df["date"].min()),
            "last_date":            str(df["date"].max()),
            "total_calendar_days":  len(df),
            "total_trading_days":   int(df["is_trading"].sum()),
            "accuracy_note":        CALENDAR_ACCURACY_NOTE,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load(self) -> pd.DataFrame:
        if self._cache is not None:
            return self._cache
        if not self._parquet_path.exists():
            raise FileNotFoundError(
                f"Calendar Parquet not found at {self._parquet_path}. "
                "Run CalendarService.build_and_save() first."
            )
        df = pd.read_parquet(self._parquet_path)
        df["date"] = pd.to_datetime(df["date"]).dt.date
        self._cache = df
        return df

    def _write(self, df: pd.DataFrame) -> None:
        self._parquet_path.parent.mkdir(parents=True, exist_ok=True)
        # Convert date objects to strings for clean Parquet serialisation
        out = df.copy()
        out["date"] = out["date"].astype(str)
        out.to_parquet(self._parquet_path, index=False)
        logger.info(
            "Calendar written → %s (%d rows, %d trading days)",
            self._parquet_path, len(out), int(df["is_trading"].sum()),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _as_date(v: dt.date | str) -> dt.date:
    if isinstance(v, dt.date) and not isinstance(v, dt.datetime):
        return v
    return pd.to_datetime(v).date()
