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
from typing import Iterable, TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from quant_platform.features.registry import FeatureMetadata


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

#: Family assigned to columns absent from ``family_by_col``.  It is in neither
#: STABLE_FAMILIES nor SHORT_HISTORY_FAMILIES, so an untagged feature is
#: fail-closed: locked out of the base model and the recent model, left as
#: prediction-only.  Entering base requires an explicit, trusted family tag.
#: (GF-04)
UNTAGGED_FAMILY = "unknown"

#: Short-history families whose signal is routed to the risk/veto/downgrade
#: channel as structured event flags -- they must NOT drive the recent alpha
#: model (recent_pct).  A subset of SHORT_HISTORY_FAMILIES.
EVENT_FAMILIES = {
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
    "label",
    "target",
)

PIT_RISK_TOKENS = (
    "pit_risk",
    "no_announce",
    "heuristic_announce",
)


@dataclass(frozen=True)
class CoverageGateConfig:
    recent_window_days: int = 120
    # GF-06: coverage thresholds are expressed as a fraction of the universe
    # size (ratio takes priority); the absolute *_threshold fields are retained
    # as a fallback used only when the corresponding ratio is None or the
    # universe size is unknown.
    recent_symbol_ratio: float | None = 0.83
    recent_20d_symbol_ratio: float | None = 0.83
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
    is_pit_safe: bool
    known_at: str
    history_start: str | None
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


def _effective_threshold(ratio: float | None, absolute: int, universe_size: int) -> float:
    """Resolve a coverage threshold (GF-06).

    ``ratio`` (fraction of the universe) takes priority; the absolute count is
    the fallback used only when no ratio is set or the universe size is unknown.
    """
    if ratio is not None and universe_size > 0:
        return ratio * universe_size
    return float(absolute)


def compute_feature_coverage_report(
    panel: pd.DataFrame,
    feature_cols: list[str],
    *,
    family_by_col: dict[str, str] | None = None,
    feature_metadata: dict[str, "FeatureMetadata"] | None = None,
    as_of_date: dt.date | str | None = None,
    config: CoverageGateConfig | None = None,
) -> pd.DataFrame:
    """Return a gate report with one row per candidate feature.

    ``feature_metadata`` (GF-04): when provided, base-model eligibility is gated
    on typed provenance -- a column that is unregistered (absent from the dict),
    not ``pit_safe``, or missing a ``known_at`` rule cannot enter the base
    model.  Name-token detection remains a secondary guard.  When omitted, the
    legacy ``family_by_col`` path is used (name tokens only).
    """
    if panel.empty:
        return pd.DataFrame(columns=[f.name for f in FeatureCoverageRow.__dataclass_fields__.values()])

    cfg = config or CoverageGateConfig()
    family_by_col = family_by_col or {}

    df = panel.copy()
    df["_gate_date"] = _normalise_dates(df)
    df = df.dropna(subset=["_gate_date"])
    if df.empty:
        return pd.DataFrame()

    # GF-06: coverage thresholds scale with the universe size.
    universe_size = int(df["symbol"].nunique()) if "symbol" in df.columns else 0
    base_cov_threshold = _effective_threshold(
        cfg.recent_symbol_ratio, cfg.recent_symbol_threshold, universe_size
    )
    recent_cov_threshold = _effective_threshold(
        cfg.recent_20d_symbol_ratio, cfg.recent_20d_symbol_threshold, universe_size
    )

    end_date = (
        pd.to_datetime(as_of_date).date()
        if as_of_date is not None
        else max(df["_gate_date"])
    )
    # GF-06: the recent window is the trailing N *trading* days actually present
    # in the panel (up to end_date), not a natural-day calendar cutoff -- this
    # keeps it consistent with min_recent_trading_days and holiday-stable.
    panel_dates = sorted(d for d in df["_gate_date"].dropna().unique() if d <= end_date)
    recent_dates = panel_dates[-cfg.recent_window_days:]
    recent = df[df["_gate_date"].isin(recent_dates)]
    last_20_dates = recent_dates[-20:]

    rows: list[FeatureCoverageRow] = []
    for col in feature_cols:
        if col not in df.columns:
            continue

        # GF-04: resolve typed provenance.  With metadata, an unregistered
        # column is fail-closed (pit_safe False, known_at ""); without it, fall
        # back to the legacy family_by_col path (name-token PIT guard only).
        if feature_metadata is not None:
            meta = feature_metadata.get(col)
            unregistered = meta is None
            has_meta = meta is not None
            family = meta.family if has_meta else UNTAGGED_FAMILY
            source = meta.source if has_meta else _source_for_family(family)
            pit_safe = bool(meta.pit_safe) if has_meta else False
            known_at = (meta.known_at or "") if has_meta else ""
            history_start = meta.history_start if has_meta else None
        else:
            unregistered = False
            has_meta = False
            family = family_by_col.get(col, UNTAGGED_FAMILY)
            source = _source_for_family(family)
            pit_safe = True            # legacy: no typed PIT info to gate on
            known_at = "legacy"
            history_start = None

        non_null = df[df[col].notna()]
        latest = max(non_null["_gate_date"]) if not non_null.empty else None
        overall_missing = float(df[col].isna().mean())
        available_days = int(non_null["_gate_date"].nunique())
        total_days = max(int(df["_gate_date"].nunique()), 1)
        date_coverage_rate = available_days / total_days

        if not recent.empty:
            latest_slice = recent[recent["_gate_date"] == end_date]
            recent_symbol_coverage = int(latest_slice[col].notna().sum())
            # GF-06: distinct trading days *within the recent window* on which
            # this feature is populated -- the metric the recent gate needs,
            # rather than the lifetime `available_days` count.
            recent_trading_days = int(recent.loc[recent[col].notna(), "_gate_date"].nunique())
        else:
            recent_symbol_coverage = 0
            recent_trading_days = 0

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
        known_at_present = bool(known_at.strip())

        if has_future_field:
            reasons.append("future-field")
        if has_pit_risk:
            reasons.append("pit-risk")
        if latest is None:
            reasons.append("no-data")

        stable_family = family in STABLE_FAMILIES
        short_family = family in SHORT_HISTORY_FAMILIES
        event_family = family in EVENT_FAMILIES
        untagged = family == UNTAGGED_FAMILY

        # GF-04: base admission additionally requires typed pit_safe + a declared
        # known_at rule.  GF-06: gate coverage on the 20-day average vs the
        # universe-scaled threshold, not the single as-of-day snapshot.
        base_allowed = (
            stable_family
            and overall_missing <= cfg.base_missing_threshold
            and recent_20d_avg >= base_cov_threshold
            and pit_safe
            and known_at_present
            and not has_future_field
            and not has_pit_risk
        )
        if not base_allowed:
            if unregistered:
                reasons.append("unregistered")
            elif untagged:
                reasons.append("untagged-fail-closed")
            elif short_family:
                reasons.append("short-history-family")
            elif overall_missing > cfg.base_missing_threshold:
                reasons.append(f"base-missing>{cfg.base_missing_threshold:.0%}")
            if has_meta and not pit_safe:
                reasons.append("not-pit-safe")
            if has_meta and not known_at_present:
                reasons.append("known-at-missing")
            if stable_family and recent_20d_avg < base_cov_threshold:
                reasons.append("base-recent-coverage-low")

        # GF-04b: the recent *alpha* model has the same typed-PIT bar as base
        # (registered + pit_safe + known_at + no leak tokens).  Event families
        # are excluded entirely -- their signal belongs in the risk/veto channel,
        # never in recent_pct.
        recent_alpha_eligible = stable_family or (short_family and not event_family)
        recent_allowed = (
            recent_alpha_eligible
            and recent_20d_avg >= recent_cov_threshold
            and recent_trading_days >= cfg.min_recent_trading_days
            and latest_ok
            and pit_safe
            and known_at_present
            and not has_future_field
            and not has_pit_risk
        )
        if not recent_allowed:
            if unregistered:
                reasons.append("unregistered")
            elif untagged:
                reasons.append("untagged-fail-closed")
            elif event_family:
                reasons.append("event-family-risk-channel-only")
            if has_meta and not pit_safe:
                reasons.append("not-pit-safe")
            if has_meta and not known_at_present:
                reasons.append("known-at-missing")
            if recent_alpha_eligible and recent_20d_avg < recent_cov_threshold:
                reasons.append("recent-20d-coverage-low")
            if recent_alpha_eligible and recent_trading_days < cfg.min_recent_trading_days:
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
                is_pit_safe=pit_safe,
                known_at=known_at,
                history_start=history_start,
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
