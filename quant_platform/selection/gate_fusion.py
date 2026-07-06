"""
Gate-first fusion for D+3 base/recent model outputs.

The fusion is intentionally not a weighted ensemble.  The base model sets the
direction; the recent model confirms, boosts, or downgrades; risk flags can
veto or push a name into observation.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class GateFusionConfig:
    tier_a_base_pct: float = 0.80
    tier_a_recent_pct: float = 0.60
    tier_b_base_pct: float = 0.60
    tier_b_recent_pct: float = 0.85
    tier_c_base_pct: float = 0.80
    tier_c_recent_pct: float = 0.35
    tier_d_recent_pct: float = 0.90
    tier_d_base_pct: float = 0.50
    tier_e_base_pct: float = 0.60
    tier_e_recent_pct: float = 0.60
    risk_veto_tokens: tuple[str, ...] = (
        "high_unlock",
        "earnings_down",
        "pit_risk",
        "extreme_outflow",
        "risk_warning",
    )
    risk_downgrade_tokens: tuple[str, ...] = (
        "unlock",
        "stale",
        "coverage",
    )


_TIER_PRIORITY = {
    "A_MAIN": 0,
    "B_SHORT_BOOST": 1,
    "C_DOWNGRADE_OBSERVE": 2,
    "D_OBSERVE": 3,
    "RISK_DOWNGRADE": 4,
    "E_REJECT": 5,
    "RISK_VETO": 6,
    "UNCLASSIFIED": 7,
}


def _text_has_any(value: object, tokens: tuple[str, ...]) -> bool:
    text = "" if pd.isna(value) else str(value).lower()
    return any(tok.lower() in text for tok in tokens)


def gate_first_fusion(
    scored: pd.DataFrame,
    *,
    config: GateFusionConfig | None = None,
) -> pd.DataFrame:
    """
    Apply gate-first tiers to a scored DataFrame.

    Required columns:
      symbol, trade_date/date, base_score, base_pct, recent_score, recent_pct
    Optional columns:
      base_rank, recent_rank, risk_flags, event_flags
    """
    cfg = config or GateFusionConfig()
    required = {"symbol", "base_score", "base_pct", "recent_score", "recent_pct"}
    missing = required - set(scored.columns)
    if missing:
        raise ValueError(f"gate fusion missing required columns: {sorted(missing)}")

    df = scored.copy()
    if "trade_date" not in df.columns and "date" in df.columns:
        df["trade_date"] = df["date"]
    if "trade_date" not in df.columns:
        df["trade_date"] = None

    for col in ("risk_flags", "event_flags"):
        if col not in df.columns:
            df[col] = ""

    if "base_rank" not in df.columns:
        df["base_rank"] = df["base_score"].rank(ascending=False, method="first").astype(int)
    if "recent_rank" not in df.columns:
        df["recent_rank"] = df["recent_score"].rank(ascending=False, method="first").astype(int)

    df["short_boost"] = df["recent_pct"] - df["base_pct"]
    tiers: list[str] = []
    reasons: list[str] = []

    for _, row in df.iterrows():
        combined_risk = f"{row.get('risk_flags', '')};{row.get('event_flags', '')}"
        veto = _text_has_any(combined_risk, cfg.risk_veto_tokens)
        downgrade = _text_has_any(combined_risk, cfg.risk_downgrade_tokens)
        base_pct = float(row["base_pct"])
        recent_pct = float(row["recent_pct"])

        if veto:
            tiers.append("RISK_VETO")
            reasons.append(f"risk veto: {combined_risk}".strip(";"))
        elif downgrade:
            tiers.append("RISK_DOWNGRADE")
            reasons.append(f"risk downgrade: {combined_risk}".strip(";"))
        elif base_pct >= cfg.tier_a_base_pct and recent_pct >= cfg.tier_a_recent_pct:
            tiers.append("A_MAIN")
            reasons.append("base strong, recent confirms")
        elif base_pct >= cfg.tier_b_base_pct and recent_pct >= cfg.tier_b_recent_pct:
            tiers.append("B_SHORT_BOOST")
            reasons.append("recent model strongly boosts an acceptable base name")
        elif base_pct >= cfg.tier_c_base_pct and recent_pct < cfg.tier_c_recent_pct:
            tiers.append("C_DOWNGRADE_OBSERVE")
            reasons.append("base strong but recent model does not confirm")
        elif recent_pct >= cfg.tier_d_recent_pct and base_pct < cfg.tier_d_base_pct:
            tiers.append("D_OBSERVE")
            reasons.append("recent strong but base model is weak")
        elif base_pct < cfg.tier_e_base_pct and recent_pct < cfg.tier_e_recent_pct:
            tiers.append("E_REJECT")
            reasons.append("base and recent are both weak")
        else:
            tiers.append("UNCLASSIFIED")
            reasons.append("falls between configured gate tiers")

    df["gate_tier"] = tiers
    df["gate_reason"] = reasons
    df["_tier_priority"] = df["gate_tier"].map(_TIER_PRIORITY).fillna(99)

    df = df.sort_values(
        ["_tier_priority", "base_pct", "recent_pct", "short_boost"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)
    df["final_rank"] = range(1, len(df) + 1)
    return df.drop(columns=["_tier_priority"])


def write_gate_fusion_outputs(
    fused: pd.DataFrame,
    out_dir: Path | str,
    *,
    prefix: str = "D3_fused",
    top_n: int = 50,
) -> tuple[Path, Path]:
    """Write ranked CSV and a markdown fusion report."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / f"{prefix}_ranked.csv"
    md_path = out / f"{prefix}_fusion_report.md"
    fused.to_csv(csv_path, index=False)

    counts = fused["gate_tier"].value_counts().to_dict() if "gate_tier" in fused.columns else {}
    lines = [
        "# D3 Gate-First Fusion Report",
        "",
        f"Generated: {dt.datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Pool Counts",
        "",
        f"- Tier A main recommendations: {counts.get('A_MAIN', 0)}",
        f"- Tier B short-boost candidates: {counts.get('B_SHORT_BOOST', 0)}",
        f"- Tier C downgraded observations: {counts.get('C_DOWNGRADE_OBSERVE', 0)}",
        f"- Tier D recent-only observations: {counts.get('D_OBSERVE', 0)}",
        f"- Risk vetoes: {counts.get('RISK_VETO', 0)}",
        f"- Risk downgrades: {counts.get('RISK_DOWNGRADE', 0)}",
        f"- Rejected double-weak: {counts.get('E_REJECT', 0)}",
        "",
        f"## Top {top_n}",
        "",
        "| Rank | Symbol | Base Pct | Recent Pct | Boost | Tier | Reason |",
        "|---:|---|---:|---:|---:|---|---|",
    ]
    for _, row in fused.head(top_n).iterrows():
        lines.append(
            f"| {int(row['final_rank'])} | {row['symbol']} | "
            f"{float(row['base_pct']):.3f} | {float(row['recent_pct']):.3f} | "
            f"{float(row['short_boost']):+.3f} | {row['gate_tier']} | "
            f"{row['gate_reason']} |"
        )

    md_path.write_text("\n".join(lines), encoding="utf-8")
    return csv_path, md_path
