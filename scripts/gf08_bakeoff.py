"""
GF-08 multi-date gate-fusion bake-off harness.

This module is intentionally additive: it imports the D3 one-shot helpers and
the frozen gate-first implementation, then compares alternative ranking arms
over a pre-registered set of out-of-sample dates.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant_platform.cli import (  # noqa: E402
    _build_feature_cols,
    _coverage_gate_config_for_universe,
    _feature_family_lookup,
)
from quant_platform.evaluation.coverage_gate import (  # noqa: E402
    compute_feature_coverage_report,
    select_features_by_gate,
    write_coverage_gate_report,
)
from quant_platform.features.registry import feature_metadata_lookup  # noqa: E402
from quant_platform.selection.gate_fusion import gate_first_fusion  # noqa: E402
from quant_platform.store.lake import label_path  # noqa: E402
from scripts import run_d3_gate_once as d3  # noqa: E402


STORE_ROOT = ROOT / "models/data"
REPORT_DIR = STORE_ROOT / "reports/gf08"
LABEL_COL = "ret_fwd_3d"
SECONDARY_LABEL_COL = "ret_fwd_5d"
CS_LABEL_COL = "ret_fwd_3d_cs"
CS_SECONDARY_LABEL_COL = "ret_fwd_5d_cs"
ARMS = (
    "base_only",
    "recent_only",
    "gate_first",
    "rrf",
    "fixed_weight_0.3",
    "fixed_weight_0.5",
    "fixed_weight_0.7",
    "recent_heavy",
    "gate_first_noveto",
)


def add_rrf_rank(frame: pd.DataFrame, *, k: int = 60) -> pd.DataFrame:
    """Rank by reciprocal-rank fusion over base and recent ranks."""
    df = frame.copy()
    df["arm_score"] = (1.0 / (k + df["base_rank"].astype(float))) + (
        1.0 / (k + df["recent_rank"].astype(float))
    )
    df["arm_rank"] = df["arm_score"].rank(ascending=False, method="first").astype(int)
    return df.sort_values("arm_rank").reset_index(drop=True)


def add_fixed_weight_rank(frame: pd.DataFrame, *, base_weight: float) -> pd.DataFrame:
    """Rank by a fixed base/recent percentile blend."""
    df = frame.copy()
    recent_weight = 1.0 - base_weight
    df["arm_score"] = (base_weight * df["base_pct"].astype(float)) + (
        recent_weight * df["recent_pct"].astype(float)
    )
    df["arm_rank"] = df["arm_score"].rank(ascending=False, method="first").astype(int)
    return df.sort_values("arm_rank").reset_index(drop=True)


def _add_score_rank(frame: pd.DataFrame, score_col: str) -> pd.DataFrame:
    df = frame.copy()
    df["arm_score"] = df[score_col].astype(float)
    df["arm_rank"] = df["arm_score"].rank(ascending=False, method="first").astype(int)
    return df.sort_values("arm_rank").reset_index(drop=True)


def _add_gate_rank(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    df["arm_rank"] = df["final_rank"].astype(int)
    df["arm_score"] = -df["arm_rank"].astype(float)
    return df.sort_values("arm_rank").reset_index(drop=True)


def compute_arm_date_metrics(
    frame: pd.DataFrame,
    arm: str,
    *,
    ks: tuple[int, ...] = (20, 50),
    cost_bps: float = 10,
) -> pd.DataFrame:
    """Compute one date's long-format metric rows for one arm."""
    df = frame.dropna(subset=["arm_score", LABEL_COL]).copy()
    if df.empty:
        return pd.DataFrame(columns=["date", "arm", "metric", "value"])
    if "arm_rank" not in df.columns:
        df["arm_rank"] = df["arm_score"].rank(ascending=False, method="first").astype(int)

    date = str(pd.to_datetime(df["trade_date"].iloc[0]).date())
    rows: list[dict] = []

    if df["arm_score"].nunique(dropna=True) >= 2 and df[LABEL_COL].nunique(dropna=True) >= 2:
        ric, _ = spearmanr(df["arm_score"], df[LABEL_COL])
        rows.append({"date": date, "arm": arm, "metric": "rank_ic_ret_fwd_3d", "value": float(ric)})

    for label in (LABEL_COL, CS_LABEL_COL, SECONDARY_LABEL_COL, CS_SECONDARY_LABEL_COL):
        if label not in df.columns:
            continue
        valid = df.dropna(subset=[label]).sort_values("arm_rank")
        if valid.empty:
            continue
        suffix = label
        for k in ks:
            if len(valid) < k:
                continue
            top = valid.head(k)
            gross = float(top[label].mean())
            rows.append({"date": date, "arm": arm, "metric": f"top{k}_{suffix}_gross", "value": gross})
            rows.append({
                "date": date,
                "arm": arm,
                "metric": f"top{k}_{suffix}_net",
                "value": gross - (cost_bps / 10000.0),
            })

    universe_mean = float(df[LABEL_COL].mean())
    ranked = df.sort_values("arm_rank")
    for k in ks:
        if len(ranked) < k:
            continue
        top = ranked.head(k)
        precision = float((top[LABEL_COL] > universe_mean).mean())
        rows.append({"date": date, "arm": arm, "metric": f"precision_at_{k}_excess", "value": precision})

    return pd.DataFrame(rows)


def compute_turnover(top_symbols_by_date: dict[str, list[str]], *, k: int) -> float:
    """Average one-way top-K turnover across consecutive dates."""
    dates = sorted(top_symbols_by_date)
    if len(dates) < 2:
        return float("nan")
    values = []
    for prev, cur in zip(dates, dates[1:]):
        prev_set = set(top_symbols_by_date[prev][:k])
        cur_set = set(top_symbols_by_date[cur][:k])
        values.append(1.0 - (len(prev_set & cur_set) / float(k)))
    return float(np.mean(values)) if values else float("nan")


def _load_label_panel(symbols: list[str]) -> pd.DataFrame:
    frames = []
    cols = ["symbol", "date", LABEL_COL, SECONDARY_LABEL_COL]
    for symbol in symbols:
        path = label_path(STORE_ROOT, "forward_returns", symbol)
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        keep = [c for c in cols if c in frame.columns]
        if keep:
            frames.append(frame[keep])
    if not frames:
        return pd.DataFrame(columns=cols + [CS_LABEL_COL, CS_SECONDARY_LABEL_COL])

    labels = pd.concat(frames, ignore_index=True)
    labels["symbol"] = labels["symbol"].astype(str).str.zfill(6)
    labels["date"] = pd.to_datetime(labels["date"]).dt.date
    for raw, cs in ((LABEL_COL, CS_LABEL_COL), (SECONDARY_LABEL_COL, CS_SECONDARY_LABEL_COL)):
        if raw in labels.columns:
            labels[cs] = labels[raw] - labels.groupby("date")[raw].transform("mean")
    return labels.sort_values(["date", "symbol"]).reset_index(drop=True)


def choose_preregistered_dates(
    labels: pd.DataFrame,
    *,
    universe_n: int,
    count: int = 20,
    min_gap_trading_days: int = 3,
    min_coverage_ratio: float = 0.83,
) -> list[dt.date]:
    threshold = math.ceil(universe_n * min_coverage_ratio)
    counts = labels.dropna(subset=[LABEL_COL]).groupby("date")["symbol"].nunique()
    eligible = sorted(counts[counts >= threshold].index)
    chosen: list[dt.date] = []
    last_idx: int | None = None
    for idx in range(len(eligible) - 1, -1, -1):
        if last_idx is None or (last_idx - idx) >= min_gap_trading_days:
            chosen.append(eligible[idx])
            last_idx = idx
        if len(chosen) == count:
            break
    if len(chosen) < count:
        raise RuntimeError(
            f"Only found {len(chosen)} eligible dates; need {count}. "
            f"threshold={threshold}, min_gap={min_gap_trading_days}"
        )
    return list(reversed(chosen))


def _feature_selection(panel: pd.DataFrame, symbols: list[str], as_of: dt.date, *, model_path: str) -> tuple[list[str], pd.DataFrame]:
    panel_to_date = panel[panel["date"] <= as_of].copy()
    candidates = _build_feature_cols(panel_to_date)
    family_by_col = _feature_family_lookup()
    gate = compute_feature_coverage_report(
        panel_to_date,
        candidates,
        family_by_col=family_by_col,
        feature_metadata=feature_metadata_lookup(),
        as_of_date=as_of,
        config=_coverage_gate_config_for_universe(len(symbols)),
    )
    features = select_features_by_gate(gate, model_path=model_path)
    if model_path == "base":
        features = [c for c in features if family_by_col.get(c) != "flow"]
    return features, gate


def _recent_train_start(panel: pd.DataFrame, as_of: dt.date) -> dt.date:
    dates = sorted(
        d for d in panel.loc[(panel["date"] < as_of) & panel[LABEL_COL].notna(), "date"].unique()
    )
    if not dates:
        raise RuntimeError("no recent training dates with observable ret_fwd_3d")
    return dates[max(0, len(dates) - 120)]


def build_date_inputs(
    *,
    symbols: list[str],
    base_panel: pd.DataFrame,
    recent_panel: pd.DataFrame,
    labels: pd.DataFrame,
    as_of: dt.date,
    out_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    base_features, base_gate = _feature_selection(base_panel, symbols, as_of, model_path="base")
    recent_features, recent_gate = _feature_selection(recent_panel, symbols, as_of, model_path="recent")
    coverage_dir = out_dir / "coverage"
    write_coverage_gate_report(base_gate, coverage_dir, prefix=f"GF08_base_coverage_gate_{as_of.isoformat()}")
    write_coverage_gate_report(recent_gate, coverage_dir, prefix=f"GF08_recent_coverage_gate_{as_of.isoformat()}")

    base_scored, base_meta, _, _, _ = d3._score_model(
        base_panel[base_panel["date"] <= as_of].copy(),
        as_of,
        base_features,
        train_start=None,
        params=d3.BASE_PARAMS,
        score_prefix="base",
    )
    if len(recent_features) == 0:
        raise RuntimeError("coverage gate rejected all recent features")
    recent_scored, recent_meta, _, _, _ = d3._score_model(
        recent_panel[recent_panel["date"] <= as_of].copy(),
        as_of,
        recent_features,
        train_start=_recent_train_start(recent_panel, as_of),
        params=d3.RECENT_PARAMS,
        score_prefix="recent",
    )
    fused_input = base_scored.merge(
        recent_scored[["symbol", "trade_date", "recent_score", "recent_rank", "recent_pct"]],
        on=["symbol", "trade_date"],
        how="inner",
    )
    risk = d3._risk_flags(recent_panel[recent_panel["date"] <= as_of].copy(), as_of)
    fused_input = fused_input.merge(risk, on=["symbol", "trade_date"], how="left")
    fused_input[["risk_flags", "event_flags"]] = fused_input[["risk_flags", "event_flags"]].fillna("")

    date_labels = labels[labels["date"] == as_of].copy().rename(columns={"date": "trade_date"})
    detail = fused_input.merge(date_labels, on=["symbol", "trade_date"], how="left")
    meta = {
        "date": as_of.isoformat(),
        "base_meta": base_meta,
        "recent_meta": recent_meta,
        "base_feature_count": len(base_features),
        "recent_feature_count": len(recent_features),
        "universe_rows": len(detail),
        "label_rows": int(detail[LABEL_COL].notna().sum()),
    }
    return fused_input, detail, meta


def apply_arms(fused_input: pd.DataFrame) -> dict[str, pd.DataFrame]:
    gate = gate_first_fusion(fused_input)
    noveto_input = fused_input.copy()
    noveto_input["risk_flags"] = ""
    noveto_input["event_flags"] = ""
    gate_noveto = gate_first_fusion(noveto_input)

    return {
        "base_only": _add_score_rank(fused_input, "base_pct"),
        "recent_only": _add_score_rank(fused_input, "recent_pct"),
        "gate_first": _add_gate_rank(gate),
        "rrf": add_rrf_rank(fused_input),
        "fixed_weight_0.3": add_fixed_weight_rank(fused_input, base_weight=0.3),
        "fixed_weight_0.5": add_fixed_weight_rank(fused_input, base_weight=0.5),
        "fixed_weight_0.7": add_fixed_weight_rank(fused_input, base_weight=0.7),
        "recent_heavy": add_fixed_weight_rank(fused_input, base_weight=0.3),
        "gate_first_noveto": _add_gate_rank(gate_noveto),
    }


def _merge_arm_labels(arm_frame: pd.DataFrame, labels: pd.DataFrame, as_of: dt.date) -> pd.DataFrame:
    date_labels = labels[labels["date"] == as_of].copy().rename(columns={"date": "trade_date"})
    return arm_frame.merge(
        date_labels[["symbol", "trade_date", LABEL_COL, CS_LABEL_COL, SECONDARY_LABEL_COL, CS_SECONDARY_LABEL_COL]],
        on=["symbol", "trade_date"],
        how="left",
    )


def veto_metrics(gate_frame: pd.DataFrame, labels: pd.DataFrame, as_of: dt.date) -> pd.DataFrame:
    labelled = _merge_arm_labels(gate_frame, labels, as_of)
    universe_mean = float(labelled[LABEL_COL].mean())
    risk = labelled[labelled["gate_tier"].isin(["RISK_VETO", "RISK_DOWNGRADE"])].copy()
    excess = float(risk[LABEL_COL].mean() - universe_mean) if not risk.empty else float("nan")
    date = as_of.isoformat()
    return pd.DataFrame(
        [
            {"date": date, "arm": "gate_first", "metric": "veto_count", "value": float(len(risk))},
            {"date": date, "arm": "gate_first", "metric": "veto_realized_excess", "value": excess},
        ]
    )


def aggregate_metrics(per_date: pd.DataFrame) -> pd.DataFrame:
    cols = ["arm", "metric", "mean", "std", "n", "icir"]
    if "value" not in per_date.columns:
        return pd.DataFrame(columns=cols)
    rows = []
    for (arm, metric), grp in per_date.dropna(subset=["value"]).groupby(["arm", "metric"]):
        vals = grp["value"].astype(float)
        std = float(vals.std(ddof=1)) if len(vals) > 1 else float("nan")
        mean = float(vals.mean())
        row = {"arm": arm, "metric": metric, "mean": mean, "std": std, "n": int(len(vals))}
        if metric == "rank_ic_ret_fwd_3d" and std and not np.isnan(std):
            row["icir"] = mean / std * math.sqrt(len(vals))
        else:
            row["icir"] = float("nan")
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows).sort_values(["metric", "arm"]).reset_index(drop=True)


def _add_turnover_metrics(per_date: list[pd.DataFrame], top_by_arm: dict[str, dict[int, dict[str, list[str]]]]) -> None:
    for arm, by_k in top_by_arm.items():
        for k, date_map in by_k.items():
            per_date.append(
                pd.DataFrame([{
                    "date": "ALL",
                    "arm": arm,
                    "metric": f"turnover_top{k}",
                    "value": compute_turnover(date_map, k=k),
                }])
            )


def decide_promotion(per_date: pd.DataFrame, aggregate: pd.DataFrame) -> dict:
    ic = aggregate[aggregate["metric"] == "rank_ic_ret_fwd_3d"].set_index("arm")
    gate_icir = float(ic.loc["gate_first", "icir"]) if "gate_first" in ic.index else float("nan")
    alt_icir = ic.drop(index=["gate_first"], errors="ignore")["icir"].max()

    veto = aggregate[(aggregate["arm"] == "gate_first") & (aggregate["metric"] == "veto_realized_excess")]
    veto_excess = float(veto["mean"].iloc[0]) if not veto.empty else float("nan")

    net_metric = "top20_ret_fwd_3d_net"
    net = aggregate[aggregate["metric"] == net_metric].set_index("arm")
    gate_net = float(net.loc["gate_first", "mean"]) if "gate_first" in net.index else float("nan")
    smooth_arms = ["rrf", "fixed_weight_0.3", "fixed_weight_0.5", "fixed_weight_0.7", "recent_heavy"]
    smooth_net = net.reindex(smooth_arms).dropna(subset=["mean"])
    best_smooth = str(smooth_net["mean"].idxmax()) if not smooth_net.empty else ""
    best_smooth_net = float(smooth_net.loc[best_smooth, "mean"]) if best_smooth else float("nan")

    p_value = float("nan")
    if best_smooth:
        pivot = per_date[per_date["metric"] == net_metric].pivot(index="date", columns="arm", values="value")
        if {"gate_first", best_smooth}.issubset(pivot.columns):
            diff = (pivot["gate_first"] - pivot[best_smooth]).dropna()
            if len(diff) > 1:
                p_value = 1.0 if np.allclose(diff, 0.0) else float(wilcoxon(diff).pvalue)

    ic_pass = bool(not np.isnan(gate_icir) and not np.isnan(alt_icir) and gate_icir >= alt_icir - 0.02)
    veto_pass = bool(not np.isnan(veto_excess) and veto_excess <= 0)
    net_beats = bool(not np.isnan(gate_net) and not np.isnan(best_smooth_net) and gate_net >= best_smooth_net)
    net_tie = bool(not np.isnan(p_value) and p_value > 0.05)

    if ic_pass and veto_pass and net_beats:
        recommendation = "promote"
    elif ic_pass and veto_pass and net_tie:
        recommendation = "hybrid"
    else:
        recommendation = "reject"

    return {
        "recommendation": recommendation,
        "gate_icir": gate_icir,
        "max_alternative_icir": float(alt_icir),
        "veto_realized_excess": veto_excess,
        "gate_top20_net": gate_net,
        "best_smooth_arm": best_smooth,
        "best_smooth_top20_net": best_smooth_net,
        "wilcoxon_p_gate_vs_best_smooth": p_value,
        "ic_pass": ic_pass,
        "veto_pass": veto_pass,
        "net_beats": net_beats,
        "net_tie": net_tie,
    }


def write_preregistration(path: Path, *, dates: list[dt.date], universe_n: int) -> None:
    data = {
        "date_list": [d.isoformat() for d in dates],
        "horizon": LABEL_COL,
        "secondary_horizon": SECONDARY_LABEL_COL,
        "universe": "CSI300",
        "universe_n": universe_n,
        "arms": list(ARMS),
        "k": [20, 50],
        "cost_bps": 10,
        "coverage_config": "CoverageGateConfig() defaults via _coverage_gate_config_for_universe",
        "promotion_rule": "GF-08 plan section 6",
        "result_location": str(REPORT_DIR),
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_summary_markdown(path: Path, aggregate: pd.DataFrame, decision: dict, metas: list[dict]) -> None:
    metrics = [
        "rank_ic_ret_fwd_3d",
        "top20_ret_fwd_3d_net",
        "top50_ret_fwd_3d_net",
        "precision_at_20_excess",
        "turnover_top20",
        "veto_realized_excess",
    ]
    lines = [
        "# GF-08 Bake-off Results",
        "",
        f"- decision: **{decision['recommendation']}**",
        f"- gate_icir: {decision['gate_icir']:.6f}",
        f"- max_alternative_icir: {decision['max_alternative_icir']:.6f}",
        f"- veto_realized_excess: {decision['veto_realized_excess']:.6f}",
        f"- best_smooth_arm: {decision['best_smooth_arm']}",
        f"- gate_top20_net: {decision['gate_top20_net']:.6f}",
        f"- best_smooth_top20_net: {decision['best_smooth_top20_net']:.6f}",
        f"- wilcoxon_p_gate_vs_best_smooth: {decision['wilcoxon_p_gate_vs_best_smooth']:.6f}",
        "",
        "## Summary Metrics",
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


def run_bakeoff(args: argparse.Namespace) -> dict:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    details_dir = out_dir / "details"
    details_dir.mkdir(parents=True, exist_ok=True)

    symbols = d3._load_symbols()
    labels = _load_label_panel(symbols)
    dates = (
        [pd.to_datetime(d).date() for d in args.dates.split(",")]
        if args.dates
        else choose_preregistered_dates(
            labels,
            universe_n=len(symbols),
            count=args.date_count,
            min_gap_trading_days=args.min_gap_trading_days,
            min_coverage_ratio=args.min_coverage_ratio,
        )
    )
    write_preregistration(out_dir / "preregistration.json", dates=dates, universe_n=len(symbols))

    max_date = max(dates)
    base_panel, base_feature_set_id = d3._build_panel(symbols, max_date, include_flow=False, include_lockup=False)
    recent_panel, recent_feature_set_id = d3._build_panel(symbols, max_date, include_flow=True, include_lockup=True)

    per_date_rows: list[pd.DataFrame] = []
    metas: list[dict] = []
    skipped: list[dict] = []
    top_by_arm: dict[str, dict[int, dict[str, list[str]]]] = {
        arm: {20: {}, 50: {}} for arm in ARMS
    }

    for as_of in dates:
        try:
            fused_input, detail, meta = build_date_inputs(
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

        arms = apply_arms(fused_input)
        for arm, arm_frame in arms.items():
            labelled = _merge_arm_labels(arm_frame, labels, as_of)
            labelled.insert(0, "arm", arm)
            labelled.to_csv(details_dir / f"{as_of.isoformat()}_{arm}.csv", index=False)
            per_date_rows.append(compute_arm_date_metrics(labelled, arm))
            for k in (20, 50):
                top_by_arm[arm][k][as_of.isoformat()] = (
                    labelled.sort_values("arm_rank")["symbol"].head(k).astype(str).tolist()
                )

        gate_detail = arms["gate_first"].merge(
            detail[["symbol", "trade_date", LABEL_COL, CS_LABEL_COL, SECONDARY_LABEL_COL, CS_SECONDARY_LABEL_COL]],
            on=["symbol", "trade_date"],
            how="left",
        )
        gate_detail.to_csv(details_dir / f"{as_of.isoformat()}_gate_first_fused.csv", index=False)
        per_date_rows.append(veto_metrics(arms["gate_first"], labels, as_of))
        meta["base_feature_set_id"] = base_feature_set_id
        meta["recent_feature_set_id"] = recent_feature_set_id
        metas.append(meta)

    _add_turnover_metrics(per_date_rows, top_by_arm)
    per_date = pd.concat(per_date_rows, ignore_index=True) if per_date_rows else pd.DataFrame()
    aggregate = aggregate_metrics(per_date)
    decision = decide_promotion(per_date, aggregate)

    per_date.to_csv(out_dir / "gf08_per_date_metrics.csv", index=False)
    aggregate.to_csv(out_dir / "gf08_aggregate_metrics.csv", index=False)
    (out_dir / "gf08_run_meta.json").write_text(
        json.dumps(
            {
                "dates": [d.isoformat() for d in dates],
                "completed_dates": [m["date"] for m in metas],
                "skipped": skipped,
                "decision": decision,
                "base_feature_set_id": base_feature_set_id,
                "recent_feature_set_id": recent_feature_set_id,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    write_summary_markdown(out_dir / "gf08_results.md", aggregate, decision, metas)
    return decision


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GF-08 gate-fusion bake-off")
    parser.add_argument("--out-dir", default=str(REPORT_DIR))
    parser.add_argument("--date-count", type=int, default=20)
    parser.add_argument("--min-gap-trading-days", type=int, default=3)
    parser.add_argument("--min-coverage-ratio", type=float, default=0.83)
    parser.add_argument("--dates", default="", help="Comma-separated YYYY-MM-DD dates; overrides auto selection")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    decision = run_bakeoff(parse_args(argv))
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
