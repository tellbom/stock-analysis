"""
SR-07 strategy bake-off harness.

Extends `scripts/gf08_bakeoff.py` (imported, not reimplemented) into a
strategy-layer bake-off over the SAME pre-registered 20 dates: does the
SR-01 hybrid score + SR-02 turnover-aware hysteresis reduce turnover
without giving up net return, vs a hybrid-score-only baseline?

Reuses gf08_bakeoff's date selection, panel building, and metric fns
verbatim -- per the SR task doc's guardrails, SR-07 must not reselect
dates, reimplement metrics, or touch frozen gate logic.

The "+confidence" arm from the SR task doc is deliberately not run as a
separate arm: SR-03's confidence signal is banner-only (it never changes
the selected set), so a "+confidence" arm would be selection-identical to
`turnover_aware` and produce duplicate rows. It is noted as a scoping
decision in the results report instead.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant_platform.selection.config import SelectionConfig, StrategyType  # noqa: E402
from quant_platform.selection.confidence import _dispersion_signal  # noqa: E402
from quant_platform.selection.fusion_score import actionable_pool  # noqa: E402
from quant_platform.selection.gate_fusion import gate_first_fusion  # noqa: E402
from quant_platform.selection.strategies import TurnoverAwareStrategy  # noqa: E402
from scripts import gf08_bakeoff as gf08  # noqa: E402
from scripts import run_d3_gate_once as d3  # noqa: E402

STORE_ROOT = gf08.STORE_ROOT
REPORT_DIR = STORE_ROOT / "reports/sr"
LABEL_COL = gf08.LABEL_COL
SECONDARY_LABEL_COL = gf08.SECONDARY_LABEL_COL
CS_LABEL_COL = gf08.CS_LABEL_COL
CS_SECONDARY_LABEL_COL = gf08.CS_SECONDARY_LABEL_COL

ARMS = ("baseline", "turnover_aware")

# Target size matches K=20, the K the promotion rule evaluates turnover
# and net return against. Sizing this to 50 (the other reported K) would
# let every true top-20 name slip into the selected set anyway (well
# inside enter_rank), collapsing the arm's top-20 slice back to the plain
# fusion_rank top-20 and making hysteresis invisible at the boundary that
# actually matters.
TARGET_SIZE = 20
ENTER_RANK = 15
KEEP_RANK = 30
MAX_TURNOVER = 0.5
PROMOTION_TURNOVER_CEILING = 0.50
PROMOTION_NET_TOLERANCE = 0.002


def build_hybrid_pool(fused_input: pd.DataFrame) -> pd.DataFrame:
    """SR-01 hybrid pool for one date: gate tiers filter, smooth score ranks."""
    gate = gate_first_fusion(fused_input)
    return actionable_pool(gate)


def _rank_with_selection_priority(pool: pd.DataFrame, selected: set) -> pd.DataFrame:
    """
    Build the turnover_aware arm's per-date frame.

    Uses the same universe and the same underlying score (fusion_score_col)
    as baseline -- so Rank IC is directly comparable -- but arm_rank pushes
    the currently-selected (post-hysteresis) set to the front, so top-K
    metrics reflect the actual held portfolio rather than a fresh re-rank.
    """
    df = pool.copy()
    df["_is_selected"] = df["symbol"].isin(selected)
    df = df.sort_values(["_is_selected", "fusion_rank"], ascending=[False, True])
    df["arm_rank"] = range(1, len(df) + 1)
    df["arm_score"] = df["fusion_score_col"].astype(float)
    return df.drop(columns=["_is_selected"]).reset_index(drop=True)


def apply_sr_arms(pool: pd.DataFrame, *, prior_selected: set) -> tuple[dict, set]:
    """Build both arms' per-date frames for one date. Returns (arms, new_selected)."""
    baseline = pool.copy()
    baseline["arm_rank"] = baseline["fusion_rank"]
    baseline["arm_score"] = baseline["fusion_score_col"].astype(float)

    config = SelectionConfig(
        strategy=StrategyType.TURNOVER_AWARE,
        max_total=TARGET_SIZE,
        enter_rank=ENTER_RANK,
        keep_rank=KEEP_RANK,
        max_turnover=MAX_TURNOVER,
    )
    strategy = TurnoverAwareStrategy(prior_selected=prior_selected)
    selected, _reasons = strategy.select(
        pool, config,
        industry_col="industry_code", score_col="fusion_score_col", symbol_col="symbol",
    )
    turnover_aware = _rank_with_selection_priority(pool, selected)

    return {"baseline": baseline, "turnover_aware": turnover_aware}, selected


def decide_sr_promotion(aggregate: pd.DataFrame, weak_aggregate: pd.DataFrame) -> dict:
    """
    Apply the SR-07 promotion rule (frozen in docs/strategy_layer_sr_task.md
    section "SR-07"): adopt turnover_aware iff
      1. one-way top-20 turnover <= 0.50, AND
      2. net Top-20 >= baseline - 0.002, AND
      3. no worse than baseline in the weak-regime split on net Top-20.
    If (1) holds but (2) fails, defer to reviewers instead of silently
    picking (per the doc's explicit fallback instruction).
    """
    net_metric = "top20_ret_fwd_3d_net"
    turnover_metric = "turnover_top20"

    def _mean(df: pd.DataFrame, arm: str, metric: str) -> float:
        row = df[(df["arm"] == arm) & (df["metric"] == metric)]
        return float(row["mean"].iloc[0]) if not row.empty else float("nan")

    baseline_net = _mean(aggregate, "baseline", net_metric)
    ta_net = _mean(aggregate, "turnover_aware", net_metric)
    ta_turnover = _mean(aggregate, "turnover_aware", turnover_metric)
    weak_baseline_net = _mean(weak_aggregate, "baseline", net_metric)
    weak_ta_net = _mean(weak_aggregate, "turnover_aware", net_metric)

    turnover_pass = bool(not np.isnan(ta_turnover) and ta_turnover <= PROMOTION_TURNOVER_CEILING)
    net_pass = bool(
        not np.isnan(baseline_net) and not np.isnan(ta_net)
        and ta_net >= baseline_net - PROMOTION_NET_TOLERANCE
    )
    weak_pass = bool(
        not np.isnan(weak_baseline_net) and not np.isnan(weak_ta_net)
        and weak_ta_net >= weak_baseline_net
    )

    if turnover_pass and net_pass and weak_pass:
        recommendation = "promote"
    elif turnover_pass and not net_pass:
        recommendation = "defer_to_reviewers"
    else:
        recommendation = "reject"

    return {
        "recommendation": recommendation,
        "baseline_top20_net": baseline_net,
        "turnover_aware_top20_net": ta_net,
        "turnover_aware_turnover_top20": ta_turnover,
        "weak_regime_baseline_top20_net": weak_baseline_net,
        "weak_regime_turnover_aware_top20_net": weak_ta_net,
        "turnover_pass": turnover_pass,
        "net_pass": net_pass,
        "weak_regime_pass": weak_pass,
    }


def write_sr_results_markdown(path: Path, aggregate: pd.DataFrame, weak_aggregate: pd.DataFrame, decision: dict, metas: list) -> None:
    metrics = [
        "rank_ic_ret_fwd_3d",
        "top20_ret_fwd_3d_net",
        "top50_ret_fwd_3d_net",
        "precision_at_20_excess",
        "turnover_top20",
    ]
    lines = [
        "# SR-07 Strategy Bake-off Results",
        "",
        f"- decision: **{decision['recommendation']}**",
        f"- baseline_top20_net: {decision['baseline_top20_net']:.6f}",
        f"- turnover_aware_top20_net: {decision['turnover_aware_top20_net']:.6f}",
        f"- turnover_aware_turnover_top20: {decision['turnover_aware_turnover_top20']:.6f}",
        f"- weak_regime_baseline_top20_net: {decision['weak_regime_baseline_top20_net']:.6f}",
        f"- weak_regime_turnover_aware_top20_net: {decision['weak_regime_turnover_aware_top20_net']:.6f}",
        f"- turnover_pass (<= {PROMOTION_TURNOVER_CEILING}): {decision['turnover_pass']}",
        f"- net_pass (>= baseline - {PROMOTION_NET_TOLERANCE}): {decision['net_pass']}",
        f"- weak_regime_pass: {decision['weak_regime_pass']}",
        "",
        "## Scoping note",
        "",
        "The optional `+confidence` arm from the task doc is not run "
        "separately: SR-03 confidence is banner-only metadata and never "
        "changes the selected set, so it would be selection-identical to "
        "`turnover_aware`.",
        "",
        "## Summary Metrics (all dates)",
        "",
        "| Metric | Arm | Mean | Std | N | ICIR |",
        "|---|---|---:|---:|---:|---:|",
    ]
    subset = aggregate[aggregate["metric"].isin(metrics)]
    for _, row in subset.iterrows():
        lines.append(
            f"| {row['metric']} | {row['arm']} | {row['mean']:.6f} | "
            f"{row['std']:.6f} | {int(row['n'])} | {row['icir']:.6f} |"
        )
    lines += [
        "",
        "## Weak-regime Summary Metrics",
        "",
        "| Metric | Arm | Mean | Std | N | ICIR |",
        "|---|---|---:|---:|---:|---:|",
    ]
    weak_subset = weak_aggregate[weak_aggregate["metric"].isin(metrics)]
    for _, row in weak_subset.iterrows():
        lines.append(
            f"| {row['metric']} | {row['arm']} | {row['mean']:.6f} | "
            f"{row['std']:.6f} | {int(row['n'])} | {row['icir']:.6f} |"
        )
    lines += [
        "",
        "## Date Runs",
        "",
        "| Date | Rows | Label Rows | Base Features | Recent Features |",
        "|---|---:|---:|---:|---:|",
    ]
    for meta in metas:
        lines.append(
            f"| {meta['date']} | {meta['universe_rows']} | {meta['label_rows']} | "
            f"{meta['base_feature_count']} | {meta['recent_feature_count']} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_sr_bakeoff(args: argparse.Namespace) -> dict:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    details_dir = out_dir / "details"
    details_dir.mkdir(parents=True, exist_ok=True)

    symbols = d3._load_symbols()
    labels = gf08._load_label_panel(symbols)
    dates = (
        [pd.to_datetime(d).date() for d in args.dates.split(",")]
        if args.dates
        else gf08.choose_preregistered_dates(
            labels,
            universe_n=len(symbols),
            count=args.date_count,
            min_gap_trading_days=args.min_gap_trading_days,
            min_coverage_ratio=args.min_coverage_ratio,
        )
    )

    max_date = max(dates)
    base_panel, base_feature_set_id = d3._build_panel(symbols, max_date, include_flow=False, include_lockup=False)
    recent_panel, recent_feature_set_id = d3._build_panel(symbols, max_date, include_flow=True, include_lockup=True)

    per_date_rows: list = []
    metas: list = []
    skipped: list = []
    dispersion_by_date: dict = {}
    top_by_arm: dict = {arm: {20: {}, 50: {}} for arm in ARMS}
    prior_selected: set = set()

    for as_of in dates:
        try:
            fused_input, _detail, meta = gf08.build_date_inputs(
                symbols=symbols,
                base_panel=base_panel,
                recent_panel=recent_panel,
                labels=labels,
                as_of=as_of,
                out_dir=out_dir,
            )
        except Exception as exc:
            skipped.append({"date": as_of.isoformat(), "reason": str(exc)})
            continue

        pool = build_hybrid_pool(fused_input)
        dispersion_by_date[as_of.isoformat()] = _dispersion_signal(pool["fusion_score_col"])

        arms, prior_selected = apply_sr_arms(pool, prior_selected=prior_selected)
        for arm, arm_frame in arms.items():
            labelled = gf08._merge_arm_labels(arm_frame, labels, as_of)
            labelled.insert(0, "arm", arm)
            labelled.to_csv(details_dir / f"{as_of.isoformat()}_{arm}.csv", index=False)
            per_date_rows.append(gf08.compute_arm_date_metrics(labelled, arm))
            for k in (20, 50):
                top_by_arm[arm][k][as_of.isoformat()] = (
                    labelled.sort_values("arm_rank")["symbol"].head(k).astype(str).tolist()
                )

        meta["base_feature_set_id"] = base_feature_set_id
        meta["recent_feature_set_id"] = recent_feature_set_id
        metas.append(meta)

    gf08._add_turnover_metrics(per_date_rows, top_by_arm)
    per_date = pd.concat(per_date_rows, ignore_index=True) if per_date_rows else pd.DataFrame()
    aggregate = gf08.aggregate_metrics(per_date)

    # Median split on per-date pool dispersion (same signal SR-03 uses) --
    # this is a PIT-safe, best-effort regime proxy; it never calls the
    # forward-label-dependent RegimeAnalyser live.
    median_dispersion = float(np.median(list(dispersion_by_date.values()))) if dispersion_by_date else float("nan")
    weak_dates = {d for d, disp in dispersion_by_date.items() if disp < median_dispersion}
    weak_per_date = per_date[per_date["date"].isin(weak_dates)]
    weak_aggregate = gf08.aggregate_metrics(weak_per_date)

    decision = decide_sr_promotion(aggregate, weak_aggregate)

    per_date.to_csv(out_dir / "sr07_per_date_metrics.csv", index=False)
    aggregate.to_csv(out_dir / "sr07_aggregate_metrics.csv", index=False)
    weak_aggregate.to_csv(out_dir / "sr07_weak_regime_aggregate_metrics.csv", index=False)
    (out_dir / "sr07_run_meta.json").write_text(
        json.dumps(
            {
                "dates": [d.isoformat() for d in dates],
                "completed_dates": [m["date"] for m in metas],
                "skipped": skipped,
                "median_dispersion": median_dispersion,
                "weak_regime_dates": sorted(weak_dates),
                "decision": decision,
                "base_feature_set_id": base_feature_set_id,
                "recent_feature_set_id": recent_feature_set_id,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    write_sr_results_markdown(out_dir / "sr07_results.md", aggregate, weak_aggregate, decision, metas)
    return decision


def parse_args(argv: list | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SR-07 strategy bake-off")
    parser.add_argument("--out-dir", default=str(REPORT_DIR))
    parser.add_argument("--date-count", type=int, default=20)
    parser.add_argument("--min-gap-trading-days", type=int, default=3)
    parser.add_argument("--min-coverage-ratio", type=float, default=0.83)
    parser.add_argument("--dates", default="", help="Comma-separated YYYY-MM-DD dates; overrides auto selection")
    return parser.parse_args(argv)


def main(argv: list | None = None) -> int:
    decision = run_sr_bakeoff(parse_args(argv))
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
