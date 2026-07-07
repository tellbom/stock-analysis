from __future__ import annotations

import datetime as dt
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quant_platform.evaluation.coverage_gate import CoverageGateConfig
from quant_platform.ingest.flow_collector import (
    FundFlowCollector,
    FundFlowFailure,
    FundFlowRouteError,
    _normalise_symbol,
)
from quant_platform.ingest.fund_flow_providers import EMDATAH5_SOURCE, EastmoneyH5FundFlowProvider
from quant_platform.store.lake import fund_flow_path


STORE_ROOT = Path("models/data")
REPORT_DIR = STORE_ROOT / "reports"
CSI300_MEMBERSHIP = STORE_ROOT / "universe/csi300/membership.parquet"
SMOKE_SYMBOLS = ["600000", "000001", "300750", "688981"]
FIELD_COLUMNS = [
    "symbol",
    "trade_date",
    "main_net",
    "small_net",
    "medium_net",
    "mid_net",
    "large_net",
    "super_net",
    "main_net_rate",
    "small_net_rate",
    "medium_net_rate",
    "large_net_rate",
    "super_net_rate",
    "close",
    "pct_change",
    "source",
    "raw_update_time",
    "fetched_at",
]
REPORT_PREFIX = "fund_flow_emdatah5"


def _load_csi300_symbols() -> list[str]:
    df = pd.read_parquet(CSI300_MEMBERSHIP)
    if "out_date" in df.columns:
        df = df[df["out_date"].isna()]
    return sorted(df["symbol"].astype(str).str.zfill(6).unique().tolist())


def _empty_failed_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "symbol",
            "provider",
            "error_type",
            "error_message",
            "retry_count",
            "latest_success_provider",
            "reported_at",
        ]
    )


def _write_failures(failures: list[FundFlowFailure]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / f"{REPORT_PREFIX}_failed_symbols.csv"
    if failures:
        df = pd.DataFrame([f.__dict__ for f in failures])
        df["reported_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    else:
        df = _empty_failed_frame()
    df.to_csv(path, index=False)


def _summarise_symbol(symbol: str, success: bool, error: str = "") -> dict:
    path = fund_flow_path(STORE_ROOT, symbol)
    if success and path.exists():
        df = pd.read_parquet(path)
        date_col = "trade_date" if "trade_date" in df.columns else "date"
        dates = pd.to_datetime(df[date_col], errors="coerce").dropna().dt.date
        fields = ",".join([c for c in FIELD_COLUMNS if c in df.columns])
        return {
            "symbol": symbol,
            "rows": int(len(df)),
            "min_date": min(dates).isoformat() if not dates.empty else "",
            "max_date": max(dates).isoformat() if not dates.empty else "",
            "fields": fields,
            "success": True,
            "error": "",
        }
    return {
        "symbol": symbol,
        "rows": 0,
        "min_date": "",
        "max_date": "",
        "fields": "",
        "success": False,
        "error": error,
    }


def collect_symbols(
    symbols: list[str],
    *,
    collector: FundFlowCollector,
    failures: list[FundFlowFailure],
    label: str,
    per_symbol_sleep: float = 2.0,
) -> pd.DataFrame:
    rows: list[dict] = []
    total = len(symbols)
    for idx, raw_symbol in enumerate(symbols, start=1):
        symbol = _normalise_symbol(raw_symbol)
        try:
            frame, _provider, _missing, provider_failures = collector._fetch_with_provider_route(symbol)
            collector._write_one(symbol, frame, overwrite=False)
            failures.extend(provider_failures)
            rows.append(_summarise_symbol(symbol, True))
        except Exception as exc:
            if isinstance(exc, FundFlowRouteError):
                failures.extend(exc.failures)
            else:
                failures.append(
                    FundFlowFailure(
                        symbol=symbol,
                        provider="eastmoney_emdatah5",
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                        retry_count=3,
                    )
                )
            rows.append(_summarise_symbol(symbol, False, f"{type(exc).__name__}: {exc}"))

        if idx % 10 == 0 or idx == total:
            print(f"{label}: {idx}/{total} done")
        time.sleep(per_symbol_sleep)
    return pd.DataFrame(rows)


def _load_fund_flow_panel(symbols: list[str]) -> pd.DataFrame:
    frames = []
    for symbol in symbols:
        path = fund_flow_path(STORE_ROOT, symbol)
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        if "date" not in df.columns and "trade_date" in df.columns:
            df["date"] = df["trade_date"]
        if "source" in df.columns:
            df = df[df["source"].astype(str) == EMDATAH5_SOURCE].copy()
        if df.empty:
            continue
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_coverage_gate(symbols: list[str]) -> tuple[pd.DataFrame, dict]:
    cfg = CoverageGateConfig()
    panel = _load_fund_flow_panel(symbols)
    if panel.empty:
        gate = pd.DataFrame(
            [
                {
                    "field": "fund_flow",
                    "covered_symbols": 0,
                    "latest_available_date": "",
                    "missing_rate": 1.0,
                    "recent_symbol_coverage": 0,
                    "recent_20d_avg_symbol_coverage": 0.0,
                    "available_trading_days": 0,
                    "is_allowed_for_recent_model": False,
                    "rejection_reason": "no-data",
                }
            ]
        )
        return gate, {
            "covered_symbols": 0,
            "latest_available_date": "",
            "recent_symbol_coverage": 0,
            "recent_20d_avg_symbol_coverage": 0.0,
            "available_trading_days": 0,
            "field_missing_rate": 1.0,
            "allowed_recent": False,
            "rejection_reason": "no-data",
        }

    latest = max(panel["date"])
    dates = sorted(panel["date"].dropna().unique())
    last_20 = dates[-20:]
    latest_slice = panel[panel["date"] == latest]
    recent_symbol_coverage = int(latest_slice["symbol"].nunique())
    daily_coverage = panel[panel["date"].isin(last_20)].groupby("date")["symbol"].nunique()
    recent_20d_avg = float(daily_coverage.mean()) if not daily_coverage.empty else 0.0
    available_days = int(len(dates))
    as_of = dt.date.today()
    latest_lag_days = (as_of - latest).days

    reasons: list[str] = []
    if recent_symbol_coverage < cfg.recent_symbol_threshold:
        reasons.append("recent-symbol-coverage-low")
    if recent_20d_avg < cfg.recent_20d_symbol_threshold:
        reasons.append("recent-20d-coverage-low")
    if available_days < cfg.min_recent_trading_days:
        reasons.append("recent-trading-days-low")
    if latest_lag_days > cfg.latest_lag_days:
        reasons.append("latest-date-stale")

    allowed_recent = not reasons
    rejection_reason = "; ".join(reasons) if reasons else "allowed"

    rows = []
    for field in FIELD_COLUMNS:
        missing_rate = float(panel[field].isna().mean()) if field in panel.columns else 1.0
        rows.append(
            {
                "field": field,
                "covered_symbols": int(panel["symbol"].nunique()),
                "latest_available_date": latest.isoformat(),
                "missing_rate": missing_rate,
                "recent_symbol_coverage": recent_symbol_coverage,
                "recent_20d_avg_symbol_coverage": recent_20d_avg,
                "available_trading_days": available_days,
                "is_allowed_for_recent_model": allowed_recent,
                "rejection_reason": rejection_reason,
            }
        )
    gate = pd.DataFrame(rows)
    key_fields = [c for c in FIELD_COLUMNS if c in panel.columns and c != "raw_update_time"]
    field_missing_rate = float(panel[key_fields].isna().mean().mean()) if key_fields else 1.0
    return gate, {
        "covered_symbols": int(panel["symbol"].nunique()),
        "latest_available_date": latest.isoformat(),
        "recent_symbol_coverage": recent_symbol_coverage,
        "recent_20d_avg_symbol_coverage": recent_20d_avg,
        "available_trading_days": available_days,
        "field_missing_rate": field_missing_rate,
        "allowed_recent": allowed_recent,
        "rejection_reason": rejection_reason,
    }


def _required_fields_present(fields: str) -> bool:
    required = {
        "symbol",
        "trade_date",
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
        "source",
        "raw_update_time",
        "fetched_at",
    }
    return required.issubset(set(str(fields).split(",")))


def _smoke_row_passed(row: pd.Series) -> bool:
    if not bool(row.get("success")):
        return False
    if int(row.get("rows") or 0) < 100:
        return False
    if not _required_fields_present(str(row.get("fields") or "")):
        return False
    return bool(str(row.get("max_date") or ""))


def write_reports(
    *,
    full_results: pd.DataFrame,
    failures: list[FundFlowFailure],
    symbols: list[str],
    stopped_reason: str = "",
) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    success_df = full_results[full_results["success"]].copy()
    success_df.to_csv(REPORT_DIR / f"{REPORT_PREFIX}_success_symbols.csv", index=False)
    _write_failures(failures)

    gate, summary = build_coverage_gate(symbols)
    gate.to_csv(REPORT_DIR / f"{REPORT_PREFIX}_coverage_gate.csv", index=False)

    success_count = int(success_df["symbol"].nunique()) if not success_df.empty else 0
    failed_count = int(len(set(symbols) - set(success_df["symbol"].astype(str))))
    allowed = bool(summary["allowed_recent"])
    rejection = "" if allowed else str(summary["rejection_reason"])
    if stopped_reason and not rejection:
        rejection = stopped_reason

    lines = [
        "# Eastmoney H5 Fund Flow Backfill Report",
        "",
        f"Generated: {dt.datetime.now().isoformat(timespec='seconds')}",
        "",
        "- Provider: EastmoneyH5FundFlowProvider",
        "- Source: eastmoney_emdatah5_zjlx",
        f"- Success symbols: {success_count}",
        f"- Failed symbols: {failed_count}",
        f"- covered_symbols: {summary['covered_symbols']}",
        f"- latest_available_date: {summary['latest_available_date']}",
        f"- Recent valid-day coverage symbols: {summary['recent_symbol_coverage']}",
        f"- Recent 20 trading-day average coverage symbols: {summary['recent_20d_avg_symbol_coverage']:.2f}",
        f"- available_trading_days: {summary.get('available_trading_days', 0)}",
        f"- Field missing rate: {summary['field_missing_rate']:.6f}",
        f"- is_allowed_for_recent_model: {'yes' if allowed else 'no'}",
    ]
    if not allowed:
        lines.append(f"- rejection_reason: {rejection or 'coverage gate rejected fund_flow'}")
    if stopped_reason:
        lines.append(f"- stopped_reason: {stopped_reason}")
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- models/data/reports/{REPORT_PREFIX}_success_symbols.csv",
            f"- models/data/reports/{REPORT_PREFIX}_failed_symbols.csv",
            f"- models/data/reports/{REPORT_PREFIX}_coverage_gate.csv",
        ]
    )
    (REPORT_DIR / f"{REPORT_PREFIX}_backfill_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    failures: list[FundFlowFailure] = []
    collector = FundFlowCollector(STORE_ROOT, providers=[EastmoneyH5FundFlowProvider()])

    smoke = collect_symbols(SMOKE_SYMBOLS, collector=collector, failures=failures, label="smoke")
    smoke = smoke[["symbol", "rows", "min_date", "max_date", "success", "error", "fields"]]
    print(smoke.to_string(index=False))
    smoke_success_rate = float(smoke.apply(_smoke_row_passed, axis=1).mean()) if not smoke.empty else 0.0

    csi300 = _load_csi300_symbols()
    if smoke_success_rate < 0.75:
        write_reports(
            full_results=smoke,
            failures=failures,
            symbols=SMOKE_SYMBOLS,
            stopped_reason=f"smoke success rate {smoke_success_rate:.0%} < 75%",
        )
        return

    full = collect_symbols(csi300, collector=collector, failures=failures, label="csi300_full")
    write_reports(full_results=full, failures=failures, symbols=csi300)


if __name__ == "__main__":
    main()
