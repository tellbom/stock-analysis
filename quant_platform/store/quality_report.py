"""
store.quality_report
====================
Data quality validation for the Parquet lake.

T0.8 scope
----------
Run after each ingestion pass to surface data problems before any feature or
model code runs.  Checks are grouped by layer:

  1. Universe     — member count, survivorship-bias flag
  2. Calendar     — source, date range, weekend invariant
  3. OHLCV        — calendar gaps, duplicates, NaN audit, price sanity
  4. Catalog      — success/failed/pending counts, failed symbol list
  5. Fundamentals — PIT integrity, heuristic-date row count

Report output
-------------
- Returns a ``QualityReport`` dataclass (structured, machine-readable).
- Writes a human-readable text summary to ``<store_root>/quality_report.txt``
  (overwritten on every run).
- ``QualityReport.has_errors`` is True if any ERROR-level finding exists.
- Findings have severity: ``INFO`` | ``WARN`` | ``ERROR``.

Design constraints (CLAUDE.md)
-------------------------------
- No new dependencies beyond what is already in the lake stack.
- Each check is a small, independently callable function.
- The aggregator ``run_quality_report()`` calls them all; individual checks
  can also be called in isolation for targeted debugging.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import duckdb
import pandas as pd

from quant_platform.core.logging import get_logger
from quant_platform.store.lake import (
    ohlcv_dir, fundamentals_dir, calendar_path,
    catalog_path, universe_root,
)

logger = get_logger(__name__)

Severity = Literal["INFO", "WARN", "ERROR"]


# ---------------------------------------------------------------------------
# Finding
# ---------------------------------------------------------------------------

@dataclass
class Finding:
    severity: Severity
    check:    str       # which check produced this
    message:  str

    def __str__(self) -> str:
        return f"[{self.severity:5s}] {self.check}: {self.message}"


# ---------------------------------------------------------------------------
# QualityReport
# ---------------------------------------------------------------------------

@dataclass
class QualityReport:
    generated_at:  str = ""
    store_root:    str = ""
    universe_key:  str = ""
    findings:      list[Finding] = field(default_factory=list)
    stats:         dict          = field(default_factory=dict)

    # ---- convenience accessors ----

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "ERROR"]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "WARN"]

    @property
    def has_errors(self) -> bool:
        return bool(self.errors)

    def add(self, severity: Severity, check: str, message: str) -> None:
        self.findings.append(Finding(severity, check, message))

    def summary_lines(self) -> list[str]:
        lines = [
            "=" * 70,
            f"QUALITY REPORT — {self.generated_at}",
            f"store_root   : {self.store_root}",
            f"universe_key : {self.universe_key}",
            "=" * 70,
            "",
        ]
        for f in self.findings:
            lines.append(str(f))
        lines += [
            "",
            "-" * 70,
            f"Total: {len(self.findings)} findings  |  "
            f"{len(self.errors)} errors  |  {len(self.warnings)} warnings",
            "PASS" if not self.has_errors else "FAIL — fix errors before training",
            "=" * 70,
        ]
        return lines

    def write_text(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self.summary_lines()), encoding="utf-8")

    def print_summary(self) -> None:
        for line in self.summary_lines():
            print(line)


# ---------------------------------------------------------------------------
# Check 1 — Universe
# ---------------------------------------------------------------------------

def check_universe(
    report: QualityReport,
    store_root: Path,
    universe_key: str,
) -> None:
    """Check universe membership: count, date range, survivorship flag."""
    check = "universe"
    membership_path = universe_root(store_root) / universe_key / "membership.parquet"

    if not membership_path.exists():
        report.add("ERROR", check,
                   f"Membership Parquet not found for '{universe_key}'. "
                   "Run UniverseService.fetch_and_save() or load_from_csv() first.")
        return

    df = pd.read_parquet(membership_path)
    total = len(df)
    open_members = df["out_date"].isna().sum()

    report.stats["universe_total_rows"]  = int(total)
    report.stats["universe_open_members"] = int(open_members)

    report.add("INFO", check,
               f"Membership rows: {total}, open (out_date=NULL): {open_members}")

    # Survivorship bias flag
    meta_path = membership_path.parent / "meta.txt"
    has_eff = False
    if meta_path.exists():
        has_eff = "has_effective_dates=True" in meta_path.read_text()

    if not has_eff:
        from quant_platform.core.universe import get_universe
        try:
            cfg = get_universe(universe_key)
            note = cfg.survivorship_note
        except KeyError:
            note = "Unknown universe — cannot retrieve survivorship note."
        report.add("WARN", check,
                   "SURVIVORSHIP BIAS: universe uses current constituents only. "
                   f"{note}")
        report.stats["survivorship_bias"] = True
    else:
        report.add("INFO", check,
                   "Historical effective dates present — survivorship bias mitigated.")
        report.stats["survivorship_bias"] = False


# ---------------------------------------------------------------------------
# Check 2 — Calendar
# ---------------------------------------------------------------------------

def check_calendar(report: QualityReport, store_root: Path) -> None:
    """Check trading calendar: source, range, weekend invariant."""
    check = "calendar"
    cal_path = calendar_path(store_root)

    if not cal_path.exists():
        report.add("ERROR", check,
                   "Calendar Parquet not found. "
                   "Run CalendarService.build_and_save() first.")
        return

    from quant_platform.ingest.calendar_service import CalendarService, CALENDAR_ACCURACY_NOTE
    svc = CalendarService(store_root)
    cov = svc.coverage()

    report.stats["calendar"] = cov
    report.add("INFO", check,
               f"Source: {cov['source']} | "
               f"Range: {cov['first_date']} → {cov['last_date']} | "
               f"Trading days: {cov['total_trading_days']} / {cov['total_calendar_days']}")

    # Accuracy note (always shown — lets the user know limitations)
    report.add("INFO", check, f"Accuracy note: {cov['accuracy_note']}")

    # Weekend invariant
    df = svc.calendar_df()
    weekend_trading = df[
        pd.to_datetime(df["date"]).dt.dayofweek >= 5
    ]["is_trading"].sum()
    if weekend_trading > 0:
        report.add("ERROR", check,
                   f"{weekend_trading} weekend days are marked as trading days — "
                   "calendar data is corrupt.")
    else:
        report.add("INFO", check, "Weekend invariant: no weekend is a trading day. ✓")

    # Warn if calendar doesn't reach today
    try:
        last = dt.date.fromisoformat(cov["last_date"])
        if last < dt.date.today():
            report.add("WARN", check,
                       f"Calendar ends on {last}, which is before today "
                       f"({dt.date.today()}). Re-run build_and_save() to extend.")
    except (ValueError, KeyError):
        pass


# ---------------------------------------------------------------------------
# Check 3 — OHLCV
# ---------------------------------------------------------------------------

def check_ohlcv(
    report: QualityReport,
    store_root: Path,
    universe_key: str,
    sample_symbols: list[str] | None = None,
) -> None:
    """
    Check OHLCV silver layer: coverage, gaps, duplicates, NaN, price sanity.

    Parameters
    ----------
    sample_symbols : list[str] | None
        If given, per-symbol gap and sanity checks run only on these symbols.
        If None, uses all Parquet files found in the OHLCV directory.
    """
    check = "ohlcv"
    ohlcv_d = ohlcv_dir(store_root)

    if not ohlcv_d.exists() or not list(ohlcv_d.glob("*.parquet")):
        report.add("WARN", check,
                   "No OHLCV Parquet files found. "
                   "Run OHLCVCollector before quality checks.")
        return

    parquet_files = list(ohlcv_d.glob("*.parquet"))
    symbols_on_disk = sorted(p.stem for p in parquet_files)
    report.stats["ohlcv_symbol_count"] = len(symbols_on_disk)
    report.add("INFO", check,
               f"OHLCV files found: {len(parquet_files)} symbols")

    # Universe coverage gap
    membership_path = universe_root(store_root) / universe_key / "membership.parquet"
    if membership_path.exists():
        uni_df = pd.read_parquet(membership_path)
        universe_symbols = set(uni_df[uni_df["out_date"].isna()]["symbol"].tolist())
        missing_ohlcv = universe_symbols - set(symbols_on_disk)
        if missing_ohlcv:
            report.add("WARN", check,
                       f"{len(missing_ohlcv)} universe symbols have no OHLCV data: "
                       f"{sorted(missing_ohlcv)[:10]}"
                       f"{'…' if len(missing_ohlcv) > 10 else ''}")
        else:
            report.add("INFO", check,
                       "All universe symbols have OHLCV data. ✓")

    # Load all OHLCV via DuckDB for aggregate checks
    glob = str(ohlcv_d / "*.parquet")
    con = duckdb.connect()
    try:
        _check_ohlcv_aggregate(report, con, glob, check)
        _check_ohlcv_per_symbol(
            report, con, glob, store_root,
            sample_symbols or symbols_on_disk[:20],  # cap at 20 for speed
            check,
        )
    finally:
        con.close()


def _check_ohlcv_aggregate(
    report: QualityReport,
    con: duckdb.DuckDBPyConnection,
    glob: str,
    check: str,
) -> None:
    """Aggregate checks: row count, date range, duplicates, NaN, price sanity."""

    # Row count and date range
    agg = con.execute(f"""
        SELECT
            count(*)                       AS total_rows,
            count(DISTINCT symbol)         AS symbols,
            min(date::DATE)                AS min_date,
            max(date::DATE)                AS max_date,
            sum(CASE WHEN close IS NULL OR close <= 0 THEN 1 ELSE 0 END) AS bad_close,
            sum(CASE WHEN high < low THEN 1 ELSE 0 END)                  AS high_lt_low,
            sum(CASE WHEN volume < 0 THEN 1 ELSE 0 END)                  AS neg_volume,
            sum(CASE WHEN open  IS NULL THEN 1 ELSE 0 END)               AS null_open,
            sum(CASE WHEN high  IS NULL THEN 1 ELSE 0 END)               AS null_high,
            sum(CASE WHEN low   IS NULL THEN 1 ELSE 0 END)               AS null_low,
            sum(CASE WHEN close IS NULL THEN 1 ELSE 0 END)               AS null_close,
            sum(CASE WHEN volume IS NULL THEN 1 ELSE 0 END)              AS null_volume
        FROM read_parquet('{glob}')
    """).fetchdf().iloc[0]

    total = int(agg["total_rows"])
    report.stats["ohlcv_total_rows"]  = total
    report.stats["ohlcv_date_range"]  = f"{agg['min_date']} → {agg['max_date']}"
    report.add("INFO", check,
               f"Total rows: {total:,} | Date range: {agg['min_date']} → {agg['max_date']}")

    # Duplicate (symbol, date)
    dup = con.execute(f"""
        SELECT count(*) AS dup_count
        FROM (
            SELECT symbol, date, count(*) AS cnt
            FROM read_parquet('{glob}')
            GROUP BY symbol, date
            HAVING cnt > 1
        )
    """).fetchone()[0]
    if dup > 0:
        report.add("ERROR", check,
                   f"{dup} duplicate (symbol, date) pairs found — "
                   "run enforce_ohlcv() to deduplicate.")
    else:
        report.add("INFO", check, "No duplicate (symbol, date) pairs. ✓")

    # NaN audit
    null_cols = {
        "open": int(agg["null_open"]), "high": int(agg["null_high"]),
        "low":  int(agg["null_low"]),  "close": int(agg["null_close"]),
        "volume": int(agg["null_volume"]),
    }
    report.stats["ohlcv_null_counts"] = null_cols
    any_nulls = {k: v for k, v in null_cols.items() if v > 0}
    if any_nulls:
        null_pct = {k: f"{v/total*100:.1f}%" for k, v in any_nulls.items()}
        report.add("WARN", check,
                   f"NULL values in OHLCV: {null_pct}. "
                   "Consider forward-filling or dropping before feature construction.")
    else:
        report.add("INFO", check, "No NULL values in OHLCV price/volume columns. ✓")

    # Price sanity
    bad_close  = int(agg["bad_close"])
    high_lt_low = int(agg["high_lt_low"])
    neg_vol    = int(agg["neg_volume"])

    if bad_close > 0:
        report.add("ERROR", check,
                   f"{bad_close} rows have close <= 0 or NULL close — data corrupt.")
    if high_lt_low > 0:
        report.add("ERROR", check,
                   f"{high_lt_low} rows have high < low — impossible OHLCV.")
    if neg_vol > 0:
        report.add("ERROR", check,
                   f"{neg_vol} rows have negative volume.")
    if bad_close == 0 and high_lt_low == 0 and neg_vol == 0:
        report.add("INFO", check, "Price/volume sanity checks passed. ✓")


def _check_ohlcv_per_symbol(
    report: QualityReport,
    con: duckdb.DuckDBPyConnection,
    glob: str,
    store_root: Path,
    symbols: list[str],
    check: str,
) -> None:
    """Per-symbol calendar gap check (sampled to keep runtime bounded)."""
    cal_path = calendar_path(store_root)
    if not cal_path.exists():
        report.add("WARN", check,
                   "Calendar not found — skipping per-symbol gap check. "
                   "Run CalendarService.build_and_save() first.")
        return

    cal_parq = str(cal_path)
    total_gaps = 0
    gap_details: list[str] = []

    for symbol in symbols:
        gaps = con.execute(f"""
            SELECT count(*) AS gap_count
            FROM read_parquet('{cal_parq}') c
            LEFT JOIN read_parquet('{glob}') o
                ON c.date::DATE = o.date::DATE AND o.symbol = '{symbol}'
            WHERE c.is_trading = true
              AND c.date::DATE BETWEEN (
                    SELECT min(date)::DATE FROM read_parquet('{glob}')
                    WHERE symbol = '{symbol}'
                  )
                  AND (
                    SELECT max(date)::DATE FROM read_parquet('{glob}')
                    WHERE symbol = '{symbol}'
                  )
              AND o.date IS NULL
        """).fetchone()[0]

        if gaps > 0:
            total_gaps += gaps
            gap_details.append(f"{symbol}:{gaps}")

    if gap_details:
        report.add("WARN", check,
                   f"Calendar gaps found in {len(gap_details)}/{len(symbols)} sampled symbols "
                   f"(total missing trading days: {total_gaps}). "
                   f"Symbols: {gap_details[:10]}"
                   f"{'…' if len(gap_details) > 10 else ''}")
        report.stats["ohlcv_gap_symbols_sampled"] = gap_details
    else:
        report.add("INFO", check,
                   f"No calendar gaps in {len(symbols)} sampled symbols. ✓")


# ---------------------------------------------------------------------------
# Check 4 — Catalog
# ---------------------------------------------------------------------------

def check_catalog(report: QualityReport, store_root: Path) -> None:
    """Check collector catalog: status counts, failed symbols."""
    check = "catalog"
    cat_path = catalog_path(store_root)

    if not cat_path.exists():
        report.add("WARN", check,
                   "Catalog Parquet not found — "
                   "no collection run has completed yet.")
        return

    from quant_platform.ingest.catalog import CollectorCatalog
    cat = CollectorCatalog(store_root)
    s = cat.summary()
    report.stats["catalog"] = s
    by_status = s.get("by_status", {})

    report.add("INFO", check,
               f"Catalog entries: {s.get('total', 0)} | "
               f"by status: {by_status}")

    n_failed = by_status.get("failed", 0)
    if n_failed > 0:
        failed_syms = [
            row["symbol"]
            for _, row in cat.load().iterrows()
            if row.get("status") == "failed"
        ]
        report.add("WARN", check,
                   f"{n_failed} symbols in FAILED state: "
                   f"{sorted(failed_syms)[:10]}"
                   f"{'…' if len(failed_syms) > 10 else ''}")

    n_pending = by_status.get("pending", 0) + by_status.get("in_progress", 0)
    if n_pending > 0:
        report.add("WARN", check,
                   f"{n_pending} symbols still pending/in_progress — "
                   "collection may be incomplete or a crash occurred. "
                   "Run CatalogDrivenCollector to resume.")

    if n_failed == 0 and n_pending == 0:
        report.add("INFO", check, "All catalog entries are in SUCCESS state. ✓")


# ---------------------------------------------------------------------------
# Check 5 — Fundamentals
# ---------------------------------------------------------------------------

def check_fundamentals(
    report: QualityReport,
    store_root: Path,
    sample_symbols: list[str] | None = None,
) -> None:
    """Check PIT fundamentals: coverage, heuristic rows, date ordering."""
    check = "fundamentals"
    fund_d = fundamentals_dir(store_root)

    if not fund_d.exists() or not list(fund_d.glob("*.parquet")):
        report.add("INFO", check,
                   "No fundamentals Parquet files found — "
                   "T0.7 collection not yet run (optional for T1.4).")
        return

    parquet_files = list(fund_d.glob("*.parquet"))
    report.stats["fundamentals_symbol_count"] = len(parquet_files)
    report.add("INFO", check, f"Fundamentals files: {len(parquet_files)} symbols")

    symbols = sample_symbols or [p.stem for p in parquet_files[:10]]
    heuristic_count = 0
    pit_violations  = 0
    total_rows      = 0

    for symbol in symbols:
        path = fund_d / f"{symbol}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        total_rows += len(df)

        # Count heuristic announce_date rows
        if "source" in df.columns:
            heuristic_count += int(
                df["source"].astype(str).str.contains("heuristic").sum()
            )

        # PIT invariant: announce_date must not precede period_end
        # (a company can't announce before the period closes)
        if "announce_date" in df.columns and "period_end" in df.columns:
            ad = pd.to_datetime(df["announce_date"]).dt.date
            pe = pd.to_datetime(df["period_end"]).dt.date
            mask_valid = ad.notna() & pe.notna()
            violations = int((ad[mask_valid] < pe[mask_valid]).sum())
            pit_violations += violations

    report.stats["fundamentals_total_rows_sampled"] = total_rows
    report.stats["fundamentals_heuristic_rows"] = heuristic_count

    if heuristic_count > 0:
        report.add("WARN", check,
                   f"{heuristic_count} rows use heuristic announce_date "
                   f"(period_end + 45 days) from stock_financial_abstract. "
                   "These rows may introduce up to 45 days of lookahead bias. "
                   "Use yjyg_em / yjkb_em endpoints for exact announce_dates.")

    if pit_violations > 0:
        report.add("ERROR", check,
                   f"{pit_violations} rows have announce_date < period_end — "
                   "impossible: a company cannot announce before the period ends. "
                   "Data is corrupt; check normalisation logic.")
    elif total_rows > 0:
        report.add("INFO", check,
                   f"PIT invariant (announce_date >= period_end) holds for all "
                   f"{total_rows} sampled rows. ✓")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_quality_report(
    store_root: Path | str,
    universe_key: str = "csi300",
    sample_symbols: list[str] | None = None,
    write_file: bool = True,
) -> QualityReport:
    """
    Run all quality checks and return a ``QualityReport``.

    Parameters
    ----------
    store_root : Path | str
        Root of the Parquet data lake.
    universe_key : str
        Which universe to check membership against.  Default ``"csi300"``.
    sample_symbols : list[str] | None
        If given, per-symbol checks (OHLCV gaps, fundamentals PIT) run only
        on these symbols.  If None, checks up to 20 symbols for OHLCV gaps
        and up to 10 for fundamentals.
    write_file : bool
        If True, write the report to ``<store_root>/quality_report.txt``.

    Returns
    -------
    QualityReport
        ``report.has_errors`` is True if any ERROR-level finding exists.
    """
    root = Path(store_root)
    report = QualityReport(
        generated_at=dt.datetime.now().isoformat(timespec="seconds"),
        store_root=str(root),
        universe_key=universe_key,
    )

    logger.info("Running quality report for %s (universe=%s)", root, universe_key)

    check_universe(report, root, universe_key)
    check_calendar(report, root)
    check_ohlcv(report, root, universe_key, sample_symbols)
    check_catalog(report, root)
    check_fundamentals(report, root, sample_symbols)

    if write_file:
        out_path = root / "quality_report.txt"
        report.write_text(out_path)
        logger.info("Quality report written → %s", out_path)

    return report
