"""
Feature coverage gate for base/recent model separation.

The gate is deliberately data-shape based: it does not decide whether a
feature has alpha.  It decides whether a feature is eligible for a long-history
base model, a short-window recent model, or prediction-only monitoring.

Source purity (review finding #3)
----------------------------------
This module intentionally stays family/data-shape based and does not
inspect a `source` column itself -- by the time a feature column reaches
here, per-row provenance has usually already been aggregated away (e.g.
cs_main_flow_rank_1d is a cross-sectional rank, not a single source
value). Source-purity filtering (e.g. excluding stale/wrong-provider
fund_flow rows) is enforced UPSTREAM instead, in the per-family panel
loader -- see quant_platform.features.flow.load_flow_panel's
`allowed_sources` parameter, which defaults to only the currently-trusted
EMDATAH5 provider. Callers building a panel for this gate should load
flow/event data through those family-specific loaders (which filter by
source) rather than reading silver parquet directly.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


STABLE_FAMILIES = {
    "technical",
    "cross_sectional",
    "valuation",
    "industry",
    "margin",
    "raw_aux",
    "fundamental",
}

SHORT_HISTORY_FAMILIES = {
    "flow",
    "sector_flow",
    "concept_flow",
    "proxy_flow",
    "event",
    "announcement",
    "announcement_events",
    "dragon_tiger",
    "block_trade",
}

FUTURE_FIELD_TOKENS = (
    "future",
    "fwd",
    "forward_return",
    "post_return",
    "after_return",
    "event_return",
)

PIT_RISK_TOKENS = (
    "pit_risk",
    "no_announce",
    "heuristic_announce",
)


@dataclass(frozen=True)
class CoverageGateConfig:
    recent_window_days: int = 120
    recent_symbol_threshold: int = 250
    recent_20d_symbol_threshold: int = 250
    base_missing_threshold: float = 0.30
    latest_lag_days: int = 2
    min_recent_trading_days: int = 80


@dataclass(frozen=True)
class FeatureCoverageRow:
    feature_name: str
    feature_family: str
    source: str
    latest_available_date: str | None
    overall_missing_rate: float
    recent_symbol_coverage: int
    recent_20d_avg_symbol_coverage: float
    available_trading_days: int
    date_coverage_rate: float
    is_allowed_for_base_model: bool
    is_allowed_for_recent_model: bool
    is_allowed_for_prediction_only: bool
    rejection_reason: str


def _normalise_dates(panel: pd.DataFrame) -> pd.Series:
    if "date" in panel.columns:
        return pd.to_datetime(panel["date"], errors="coerce").dt.date
    if "trade_date" in panel.columns:
        return pd.to_datetime(panel["trade_date"], errors="coerce").dt.date
    raise ValueError("coverage gate requires a 'date' or 'trade_date' column")


def _has_token(name: str, tokens: Iterable[str]) -> bool:
    lowered = name.lower()
    return any(tok in lowered for tok in tokens)


def _source_for_family(family: str) -> str:
    return {
        "flow": "silver/fund_flow",
        "sector_flow": "silver/sector_fund_flow",
        "concept_flow": "silver/concept_fund_flow",
        "proxy_flow": "silver/sector_fund_flow",
        "event": "silver/lockup",
        "announcement": "silver/announcement_events",
        "announcement_events": "cninfo",
        "dragon_tiger": "datacenter-web",
        "block_trade": "datacenter-web",
        "fundamental": "silver/fundamentals",
        "valuation": "silver/valuation",
        "industry": "silver/industry_map",
        "margin": "silver/margin",
    }.get(family, "feature_panel")


def compute_feature_coverage_report(
    panel: pd.DataFrame,
    feature_cols: list[str],
    *,
    family_by_col: dict[str, str] | None = None,
    as_of_date: dt.date | str | None = None,
    config: CoverageGateConfig | None = None,
) -> pd.DataFrame:
    """Return a gate report with one row per candidate feature."""
    if panel.empty:
        return pd.DataFrame(columns=[f.name for f in FeatureCoverageRow.__dataclass_fields__.values()])

    cfg = config or CoverageGateConfig()
    family_by_col = family_by_col or {}

    df = panel.copy()
    df["_gate_date"] = _normalise_dates(df)
    df = df.dropna(subset=["_gate_date"])
    if df.empty:
        return pd.DataFrame()

    end_date = (
        pd.to_datetime(as_of_date).date()
        if as_of_date is not None
        else max(df["_gate_date"])
    )
    recent_start = end_date - dt.timedelta(days=cfg.recent_window_days)
    recent = df[(df["_gate_date"] >= recent_start) & (df["_gate_date"] <= end_date)]
    recent_dates = sorted(recent["_gate_date"].dropna().unique())
    last_20_dates = recent_dates[-20:]

    rows: list[FeatureCoverageRow] = []
    for col in feature_cols:
        if col not in df.columns:
            continue

        family = family_by_col.get(col, "raw_aux")
        source = _source_for_family(family)
        non_null = df[df[col].notna()]
        latest = max(non_null["_gate_date"]) if not non_null.empty else None
        overall_missing = float(df[col].isna().mean())
        available_days = int(non_null["_gate_date"].nunique())
        total_days = max(int(df["_gate_date"].nunique()), 1)
        date_coverage_rate = available_days / total_days

        if not recent.empty:
            latest_slice = recent[recent["_gate_date"] == end_date]
            recent_symbol_coverage = int(latest_slice[col].notna().sum())
        else:
            recent_symbol_coverage = 0

        if last_20_dates:
            daily = (
                recent[recent["_gate_date"].isin(last_20_dates)]
                .groupby("_gate_date")[col]
                .apply(lambda s: int(s.notna().sum()))
            )
            recent_20d_avg = float(daily.mean()) if not daily.empty else 0.0
        else:
            recent_20d_avg = 0.0

        reasons: list[str] = []
        has_future_field = _has_token(col, FUTURE_FIELD_TOKENS)
        has_pit_risk = _has_token(col, PIT_RISK_TOKENS)
        latest_ok = bool(latest and (end_date - latest).days <= cfg.latest_lag_days)

        if has_future_field:
            reasons.append("future-field")
        if has_pit_risk:
            reasons.append("pit-risk")
        if latest is None:
            reasons.append("no-data")

        stable_family = family in STABLE_FAMILIES
        short_family = family in SHORT_HISTORY_FAMILIES

        base_allowed = (
            stable_family
            and overall_missing <= cfg.base_missing_threshold
            and recent_symbol_coverage >= cfg.recent_symbol_threshold
            and not has_future_field
            and not has_pit_risk
        )
        if not base_allowed:
            if short_family:
                reasons.append("short-history-family")
            elif overall_missing > cfg.base_missing_threshold:
                reasons.append(f"base-missing>{cfg.base_missing_threshold:.0%}")
            if recent_symbol_coverage < cfg.recent_symbol_threshold:
                reasons.append("base-recent-symbol-coverage-low")

        recent_allowed = (
            (stable_family or short_family)
            and recent_symbol_coverage >= cfg.recent_symbol_threshold
            and recent_20d_avg >= cfg.recent_20d_symbol_threshold
            and available_days >= cfg.min_recent_trading_days
            and latest_ok
            and not has_future_field
            and not has_pit_risk
        )
        if not recent_allowed:
            if recent_symbol_coverage < cfg.recent_symbol_threshold:
                reasons.append("recent-symbol-coverage-low")
            if recent_20d_avg < cfg.recent_20d_symbol_threshold:
                reasons.append("recent-20d-coverage-low")
            if available_days < cfg.min_recent_trading_days:
                reasons.append("recent-trading-days-low")
            if not latest_ok and latest is not None:
                reasons.append("latest-date-stale")

        prediction_only = bool(latest is not None and not base_allowed and not recent_allowed)
        reason = "; ".join(dict.fromkeys(reasons)) if reasons else "allowed"

        rows.append(
            FeatureCoverageRow(
                feature_name=col,
                feature_family=family,
                source=source,
                latest_available_date=latest.isoformat() if latest else None,
                overall_missing_rate=overall_missing,
                recent_symbol_coverage=recent_symbol_coverage,
                recent_20d_avg_symbol_coverage=recent_20d_avg,
                available_trading_days=available_days,
                date_coverage_rate=date_coverage_rate,
                is_allowed_for_base_model=base_allowed,
                is_allowed_for_recent_model=recent_allowed,
                is_allowed_for_prediction_only=prediction_only,
                rejection_reason=reason,
            )
        )

    return pd.DataFrame([asdict(r) for r in rows])


def select_features_by_gate(
    gate_report: pd.DataFrame,
    *,
    model_path: str = "base",
) -> list[str]:
    """Return feature names allowed for ``base`` or ``recent`` model path."""
    if gate_report.empty:
        return []
    if model_path == "base":
        mask = gate_report["is_allowed_for_base_model"]
    elif model_path == "recent":
        mask = gate_report["is_allowed_for_recent_model"]
    else:
        raise ValueError("model_path must be 'base' or 'recent'")
    return gate_report.loc[mask, "feature_name"].astype(str).tolist()


def write_coverage_gate_report(
    gate_report: pd.DataFrame,
    out_dir: Path | str,
    *,
    prefix: str = "coverage_gate",
) -> tuple[Path, Path]:
    """Write CSV and compact markdown reports."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / f"{prefix}.csv"
    md_path = out / f"{prefix}.md"
    gate_report.to_csv(csv_path, index=False)

    lines = [
        "# Feature Coverage Gate",
        "",
        f"Generated: {dt.datetime.now().isoformat(timespec='seconds')}",
        "",
    ]
    if gate_report.empty:
        lines.append("No feature candidates were evaluated.")
    else:
        lines.extend([
            f"- Features checked: {len(gate_report)}",
            f"- Base model allowed: {int(gate_report['is_allowed_for_base_model'].sum())}",
            f"- Recent model allowed: {int(gate_report['is_allowed_for_recent_model'].sum())}",
            f"- Prediction-only: {int(gate_report['is_allowed_for_prediction_only'].sum())}",
            "",
            "## Rejections",
            "",
            "| Feature | Family | Reason |",
            "|---|---|---|",
        ])
        rejected = gate_report[
            ~gate_report["is_allowed_for_base_model"]
            | ~gate_report["is_allowed_for_recent_model"]
        ]
        for _, row in rejected.head(80).iterrows():
            lines.append(
                f"| {row['feature_name']} | {row['feature_family']} | "
                f"{row['rejection_reason']} |"
            )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return csv_path, md_path
