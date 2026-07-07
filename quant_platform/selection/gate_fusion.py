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


# Ordering rationale (GF-01/GF-02):
#   UNCLASSIFIED holds decent-base / neutral-recent names that fall between the
#   configured primary tiers.  It sits in the MIDDLE (below the strong-base
#   observe tier, above weak-base D_OBSERVE and the reject/veto tiers) and is
#   ranked by base_pct within the tier -- never dumped last.  RISK_VETO is
#   strictly last: a vetoed name must rank below everything actionable.
_TIER_PRIORITY = {
    "A_MAIN": 0,
    "B_SHORT_BOOST": 1,
    "C_DOWNGRADE_OBSERVE": 2,
    "UNCLASSIFIED": 3,
    "D_OBSERVE": 4,
    "RISK_DOWNGRADE": 5,
    "E_REJECT": 6,
    "RISK_VETO": 7,
}


def _parse_flag_codes(*values: object) -> list[tuple[str, bool]]:
    """Parse ``;``-separated structured flag codes into ``(name, has_time)``.

    Format: ``name[:detail][@known_at]``.  Matching uses ``name`` exactly, so a
    descriptive flag such as ``coverage_ok`` never trips the ``coverage`` token
    (GF-05).  ``has_time`` is True only when a non-empty ``@`` timing component
    (known_at / event_time / announce_time) is present; the risk/veto/downgrade
    channel admits a flag only when it is a registered code *and* carries an
    explicit time -- an untimed or unregistered event cannot drive a veto or
    downgrade, and events never touch ``recent_pct`` (GF-04b req #3).
    """
    out: list[tuple[str, bool]] = []
    for value in values:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            continue
        for part in str(value).split(";"):
            part = part.strip()
            if not part:
                continue
            code, _, time_part = part.partition("@")
            name = code.split(":", 1)[0].strip().lower()
            if name:
                out.append((name, bool(time_part.strip())))
    return out


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
    veto_set = {t.lower() for t in cfg.risk_veto_tokens}
    downgrade_set = {t.lower() for t in cfg.risk_downgrade_tokens}
    tiers: list[str] = []
    reasons: list[str] = []

    for _, row in df.iterrows():
        flag_pairs = _parse_flag_codes(row.get("risk_flags", ""), row.get("event_flags", ""))
        # only registered codes that carry an explicit time may drive the channel
        veto = any(name in veto_set and timed for name, timed in flag_pairs)
        downgrade = any(name in downgrade_set and timed for name, timed in flag_pairs)
        combined_risk = f"{row.get('risk_flags', '')};{row.get('event_flags', '')}"
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

    # GF-07: within-tier ordering is tier-dependent.  For recent-led tiers
    # (B_SHORT_BOOST, D_OBSERVE) base_pct is weak/uninformative, so lead with
    # recent_pct; all other tiers lead with base_pct ("base sets direction").
    _RECENT_LED_TIERS = {"B_SHORT_BOOST", "D_OBSERVE"}
    recent_led = df["gate_tier"].isin(_RECENT_LED_TIERS)
    df["_sort_primary"] = df["base_pct"].where(~recent_led, df["recent_pct"])
    df["_sort_secondary"] = df["recent_pct"].where(~recent_led, df["base_pct"])

    df = df.sort_values(
        ["_tier_priority", "_sort_primary", "_sort_secondary", "short_boost"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)
    df["final_rank"] = range(1, len(df) + 1)
    return df.drop(columns=["_tier_priority", "_sort_primary", "_sort_secondary"])


#: Tiers safe to treat as buyable.  Everything else (observe/downgrade/reject/
#: veto) is routed to a separate file so a "top-K by rank" consumer can never
#: ingest a non-actionable name.  (GF-03)
ACTIONABLE_TIERS = ("A_MAIN", "B_SHORT_BOOST")


def write_gate_fusion_outputs(
    fused: pd.DataFrame,
    out_dir: Path | str,
    *,
    prefix: str = "D3_fused",
    top_n: int = 50,
) -> tuple[Path, Path]:
    """Write the actionable ranked CSV and a markdown fusion report.

    Three files are written:
      * ``{prefix}_ranked.csv``  -- actionable A/B tiers only (the buyable pool)
      * ``{prefix}_observe.csv`` -- observe/downgrade/reject/veto tiers
      * ``{prefix}_fusion_report.md`` -- human-readable summary

    Only the first and the report path are returned (backward-compatible tuple).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / f"{prefix}_ranked.csv"
    observe_path = out / f"{prefix}_observe.csv"
    md_path = out / f"{prefix}_fusion_report.md"

    if "gate_tier" in fused.columns:
        is_actionable = fused["gate_tier"].isin(ACTIONABLE_TIERS)
        actionable = fused[is_actionable]
        observe = fused[~is_actionable]
    else:  # no tiers computed -- fall back to writing the whole frame as-is
        actionable = fused
        observe = fused.iloc[0:0]

    actionable.to_csv(csv_path, index=False)
    observe.to_csv(observe_path, index=False)

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
        f"- Unclassified (mid base / neutral recent): {counts.get('UNCLASSIFIED', 0)}",
        f"- Risk vetoes: {counts.get('RISK_VETO', 0)}",
        f"- Risk downgrades: {counts.get('RISK_DOWNGRADE', 0)}",
        f"- Rejected double-weak: {counts.get('E_REJECT', 0)}",
        "",
        f"Actionable pool: {len(actionable)}  |  Observe/veto pool: {len(observe)}",
        "",
        f"## Top {top_n} (actionable only)",
        "",
        "| Rank | Symbol | Base Pct | Recent Pct | Boost | Tier | Reason |",
        "|---:|---|---:|---:|---:|---|---|",
    ]
    for _, row in actionable.head(top_n).iterrows():
        lines.append(
            f"| {int(row['final_rank'])} | {row['symbol']} | "
            f"{float(row['base_pct']):.3f} | {float(row['recent_pct']):.3f} | "
            f"{float(row['short_boost']):+.3f} | {row['gate_tier']} | "
            f"{row['gate_reason']} |"
        )

    md_path.write_text("\n".join(lines), encoding="utf-8")
    return csv_path, md_path
