"""Read-only-lake, exact-label D3/D5 paired research comparison.

Each cohort is fitted at its signal close, selects before reading outcomes,
and has independent capital. Outputs are retrospective research, never NAV
or authorization for a frozen paper-trading plan.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quant_platform.features import industry as industry_module
from quant_platform.features.flow import build_flow_features, load_flow_panel
from quant_platform.features.registry import DEFAULT_SPECS, compute_feature_set_id
from quant_platform.features.valuation import load_valuation_panel
from quant_platform.selection.fusion_score import fusion_score
from quant_platform.selection.ridge_baseline import ORIGINAL_FEATURES, fit_predict_ridge32, score_frame
from quant_platform.selection.robust_ridge import fit_predict_robust_ridge
from quant_platform.training.lgbm_model import fit_final_model
from scripts import next_week_d3_prediction as weekly
from scripts.compare_original_hybrid import FLOW_FEATURES, combine, fingerprint, select
from scripts.run_d3_gate_once import RECENT_PARAMS
from scripts.run_dual_horizon_rrf import observable_target
from temp.latest_dual_horizon import _fast_join_industry

ARMS = ["ordinary", "hybrid_no_flow", "hybrid_flow", "robust_ridge_v1", "reversal_3d_rule"]
CORE = ["ordinary", "robust_ridge_v1", "reversal_3d_rule"]


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=str), encoding="utf-8")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def control_gate_snapshot(path: Path | None) -> str | None:
    """Hash an explicitly supplied gate; keep machine-local locations external."""
    return sha(path) if path is not None else None


def mature_panel(panel: pd.DataFrame, signal: dt.date, horizon: int) -> pd.DataFrame:
    """Censor stored/recomputed labels by their explicit exchange-calendar end."""
    label, end = f"ret_fwd_{horizon}d", f"label_end_{horizon}d"
    out = panel.loc[panel.date <= signal].copy()
    out.loc[out[end].isna() | (out[end] > signal), label] = np.nan
    return out


def cohort_net(gross: float, one_way_bps: float) -> float:
    # Same additive cost convention as the existing fixed-cohort comparison.
    return float(gross - 2 * one_way_bps / 10000)


def paired_interval(values: np.ndarray) -> tuple[float | None, float | None]:
    """Two-cohort moving-block bootstrap; descriptive, not a promotion test."""
    values = np.asarray(values, dtype=float)
    if len(values) < 4:
        return None, None
    rng = np.random.default_rng(20261025)
    starts = rng.integers(0, len(values), size=(5000, (len(values) + 1) // 2))
    idx = np.stack([starts, (starts + 1) % len(values)], axis=-1).reshape(5000, -1)
    means = values[idx[:, :len(values)]].mean(axis=1)
    lo, hi = np.quantile(means, [.025, .975])
    return float(lo), float(hi)


def load_snapshot(store: Path, as_of: dt.date):
    membership = pd.read_parquet(store / "universe/csi300/membership.parquet")
    symbols = sorted(membership.symbol.astype(str).str.zfill(6).unique())
    coverage, price_frames = [], []
    for symbol in symbols:
        p = store / f"silver/ohlcv/{symbol}.parquet"
        f = store / f"silver/fund_flow/{symbol}.parquet"
        prices = pd.read_parquet(p, columns=["symbol", "date", "close"])
        prices.date = pd.to_datetime(prices.date).dt.date
        flow = pd.read_parquet(f, columns=["date"]) if f.exists() else pd.DataFrame(columns=["date"])
        last_flow = pd.to_datetime(flow.date).max().date() if len(flow) else None
        usable = as_of in set(prices.date) and last_flow is not None and last_flow >= as_of
        coverage.append({"symbol": symbol, "price_last": prices.date.max(), "flow_last": last_flow,
                         "included": usable, "reason": "" if usable else "no complete snapshot at cutoff"})
        if usable:
            price_frames.append(prices)
    active = [r["symbol"] for r in coverage if r["included"]]
    if len(active) < 250:
        raise RuntimeError("Insufficient snapshot coverage")
    old_store, old_join = weekly.STORE_ROOT, industry_module._join_industry
    weekly.STORE_ROOT, industry_module._join_industry = store, _fast_join_industry
    try:
        panel = weekly.build_panel(active, compute_feature_set_id(DEFAULT_SPECS), max_date=as_of)
    finally:
        weekly.STORE_ROOT, industry_module._join_industry = old_store, old_join
    panel.symbol = panel.symbol.astype(str).str.zfill(6)
    panel.date = pd.to_datetime(panel.date).dt.date
    calendar = pd.read_parquet(store / "calendar/trading_calendar.parquet")
    calendar.date = pd.to_datetime(calendar.date).dt.date
    prices = pd.concat(price_frames, ignore_index=True)
    prices.symbol = prices.symbol.astype(str).str.zfill(6)
    for h in (3, 5):
        panel = panel.drop(columns=[f"ret_fwd_{h}d", f"label_end_{h}d"], errors="ignore").merge(
            observable_target(prices, as_of, h, calendar), on=["symbol", "date"], validate="one_to_one")
    flow = load_flow_panel(store, active)
    valuation = load_valuation_panel(store, active)
    flow.date = pd.to_datetime(flow.date).dt.date
    valuation.date = pd.to_datetime(valuation.date).dt.date
    enriched = build_flow_features(panel[["symbol", "date"]], flow[flow.date <= as_of], valuation[valuation.date <= as_of])
    panel = panel.drop(columns=FLOW_FEATURES, errors="ignore").merge(enriched, on=["symbol", "date"], validate="one_to_one")
    panel = panel.replace([np.inf, -np.inf], np.nan).sort_values(["symbol", "date"]).reset_index(drop=True)
    if panel.duplicated(["symbol", "date"]).any():
        raise ValueError("Duplicate panel keys")
    return panel, calendar, pd.DataFrame(coverage), membership


def score_signal(panel, calendar, signal, h, n_jobs):
    label = f"ret_fwd_{h}d"
    censored = mature_panel(panel, signal, h)
    base, base_meta, base_model = fit_predict_ridge32(censored, signal, label)
    robust, robust_meta, robust_model = fit_predict_robust_ridge(
        censored, signal, label, trading_days=calendar.loc[calendar.is_trading, "date"].tolist())
    if set(base.symbol) != set(robust.symbol):
        raise RuntimeError("New baseline and ordinary scoring universes differ")
    pred = censored[(censored.date == signal) & censored.symbol.isin(base.symbol)].copy()
    if len(pred) < 250:
        raise RuntimeError(f"{signal}: only {len(pred)} scoreable symbols")
    frames = {"ordinary": (base, "base_score"), "robust_ridge_v1": (robust, "base_score"),
              "reversal_3d_rule": (score_frame(pred, pred.reversal_3d.to_numpy(), "base"), "base_score")}
    train = censored[censored.date < signal].dropna(subset=ORIGINAL_FEATURES + [label])
    recent_dates = sorted(train.date.unique())[-120:]
    recent = train[train.date.isin(recent_dates)].dropna(subset=FLOW_FEATURES)
    metadata = {"signal": signal, "horizon": h, "ordinary": base_meta, "robust": robust_meta,
                "recent_usable_dates": recent.date.nunique(), "recent_rows": len(recent),
                "train_row_label_hash": fingerprint(train, ["symbol", "date", label]),
                "recent_row_label_hash": fingerprint(recent, ["symbol", "date", label]),
                "score_universe_hash": fingerprint(pred, ["symbol", "date"]), "flow_skip": None}
    if recent.date.nunique() >= 80 and pred[FLOW_FEATURES].notna().all().all():
        for name, features in (("hybrid_no_flow", ORIGINAL_FEATURES), ("hybrid_flow", ORIGINAL_FEATURES + FLOW_FEATURES)):
            model = fit_final_model(recent, features, label, {**RECENT_PARAMS, "n_jobs": n_jobs})
            scores = score_frame(pred, model.predict(pred[features]), "recent")
            frames[name] = (fusion_score(combine(base, scores)), "fusion_score_col")
    else:
        metadata["flow_skip"] = "fewer than 80 usable recent dates or incomplete prediction flow"
    attributes = pred[["symbol", "industry_code", "industry_name", "close"]]
    selections = {}
    for arm, (frame, score_col) in frames.items():
        picks, _ = select(frame, attributes, score_col, 10)
        selections[arm] = picks
    metadata["unknown_industry_count"] = int(pred.industry_code.eq("_UNKNOWN").sum())
    return frames, selections, metadata, {"ordinary": base_model, "robust_ridge_v1": robust_model}


def summarise(periods: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (h, group), sample in periods.groupby(["horizon", "comparison"]):
        ordinary = sample[sample.arm == "ordinary"].set_index("signal")
        for arm, p in sample.groupby("arm"):
            p = p.sort_values("signal").set_index("signal")
            delta = p.net_return - ordinary.net_return
            lo, hi = paired_interval(delta.to_numpy())
            blocks = [x for x in np.array_split(delta.to_numpy(), 5) if len(x)]
            rows.append({"horizon": h, "comparison": group, "arm": arm,
                         "cohorts": len(p), "first_signal": p.index.min(), "last_signal": p.index.max(),
                         "mean_gross": p.gross_return.mean(), "mean_net": p.net_return.mean(),
                         "mean_rank_ic": p.rank_ic.mean(), "delta_vs_ordinary": delta.mean(),
                         "paired_ci_low": lo, "paired_ci_high": hi,
                         "positive_delta_blocks": sum(float(x.mean()) > 0 for x in blocks),
                         "worst_cohort": p.net_return.min(), "winning_cohort_fraction": (p.net_return > 0).mean(),
                         "verdict": "RESEARCH_ONLY_NO_PROMOTION"})
    return pd.DataFrame(rows)


def run(args):
    out = args.out_dir
    if out.exists() and any(out.iterdir()):
        raise RuntimeError("Use a new output directory; existing research is immutable")
    gate = args.control_gate
    gate_hash = control_gate_snapshot(gate)
    out.mkdir(parents=True, exist_ok=True)
    print("Loading and checking snapshot provenance", flush=True)
    panel, calendar, coverage, membership = load_snapshot(args.store, args.as_of)
    coverage.to_csv(out / "data_coverage.csv", index=False, encoding="utf-8-sig")
    days = sorted(calendar.loc[calendar.is_trading & (calendar.date <= args.as_of), "date"].unique())
    # Six exchange sessions between signals: D5 holding intervals cannot overlap.
    last = len(days) - 7
    signals = [days[i] for i in range(max(0, last - 6 * (args.cohorts - 1)), last + 1, 6)]
    plan = {"version": "paired-d3-d5-v1", "created_on": "2026-10-25", "as_of": args.as_of,
            "arms": ARMS, "new_baseline": "robust_ridge_v1", "labels": ["ret_fwd_3d", "ret_fwd_5d"],
            "signal_dates": signals, "signal_step_sessions": 6, "one_way_cost_bps": args.cost_bps,
            "cost_convention": "gross return minus fee+slippage at both entry and exit; independent cohort capital",
            "selection": {"top_k_per_industry": 3, "max_total": 50, "sector_cap_tech": 10},
            "new_baseline_contract": {"features": ORIGINAL_FEATURES, "window": 252, "alpha": 10, "clip_quantiles": [.01, .99], "tuned_on_this_run": False},
            "comparison_sets": {"core": CORE, "all_arms": ARMS},
            "control_gate_sha256_before": gate_hash, "mutates_old_paper_plan": False,
            "registration": "active in this independent research comparison; not promoted or backfilled into old cohorts",
            "prospective_entry": None, "latest_scoring_is_retrospective": True,
            "limits": ["current membership, not historical constituent membership", "revised QFQ snapshot, not archived point-in-time vintage",
                       "daily close fill assumption, no suspension/limit-up execution proof", "short flow history", "not a clean untouched holdout",
                       "bootstrap interval descriptive; no multiple-trial adjustment", "no stop8 valuation"]}
    write_json(out / "comparison_plan.json", plan)
    periods, pick_records, audits, omitted = [], [], [], []
    for h in (3, 5):
        label = f"ret_fwd_{h}d"
        for signal in signals:
            print(f"D{h} signal {signal}", flush=True)
            frames, selections, metadata, _ = score_signal(panel, calendar, signal, h, args.n_jobs)
            audits.append(metadata)
            outcome = panel[panel.date == signal][["symbol", label]].set_index("symbol")[label]
            for group, required in (("core", CORE), ("all_arms", ARMS)):
                if not set(required).issubset(selections):
                    omitted.append({"horizon": h, "signal": signal, "comparison": group, "reason": metadata["flow_skip"]})
                    continue
                equal_n = min(len(selections[a]) for a in required)
                picks_by_arm = {a: selections[a].head(equal_n).copy() for a in required}
                missing = sorted({s for p in picks_by_arm.values() for s in p.symbol if pd.isna(outcome.get(s, np.nan))})
                if equal_n == 0 or missing:
                    omitted.append({"horizon": h, "signal": signal, "comparison": group,
                                    "reason": "missing selected-stock outcome or empty selection", "symbols": missing})
                    continue
                pos = days.index(signal)
                for arm, picks in picks_by_arm.items():
                    realized = picks.symbol.map(outcome)
                    frame, score_col = frames[arm]
                    pairs = frame[["symbol", score_col]].assign(realized=frame.symbol.map(outcome)).dropna()
                    ic = float(spearmanr(pairs[score_col], pairs.realized).statistic)
                    periods.append({"horizon": h, "comparison": group, "signal": signal, "entry": days[pos + 1],
                                    "exit": days[pos + h + 1], "arm": arm, "pick_count": equal_n,
                                    "gross_return": realized.mean(), "net_return": cohort_net(realized.mean(), args.cost_bps),
                                    "rank_ic": ic, "score_universe": len(frame)})
                    for symbol, gross in zip(picks.symbol, realized):
                        pick_records.append({"horizon": h, "comparison": group, "signal": signal, "arm": arm,
                                             "symbol": symbol, "gross_return": gross, "weight": 1 / equal_n})
            pd.DataFrame(periods).to_csv(out / "cohort_returns.csv", index=False, encoding="utf-8-sig")
    periods = pd.DataFrame(periods)
    summary = summarise(periods)
    summary.to_csv(out / "summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(pick_records).to_csv(out / "historical_picks.csv", index=False, encoding="utf-8-sig")
    write_json(out / "training_audit.json", audits)
    write_json(out / "omitted_cohorts.json", omitted)
    latest_audits = []
    for h in (3, 5):
        frames, picks, meta, models = score_signal(panel, calendar, args.as_of, h, args.n_jobs)
        latest_audits.append(meta)
        for arm, p in picks.items():
            p.to_csv(out / f"D{h}_{arm}_snapshot_picks.csv", index=False, encoding="utf-8-sig")
        for arm, model in models.items():
            joblib.dump(model, out / f"D{h}_{arm}_snapshot.joblib")
    write_json(out / "snapshot_training_audit.json", latest_audits)
    gate_after = control_gate_snapshot(gate)
    if gate_after != gate_hash:
        raise RuntimeError("Old paper control gate changed during run; inspect concurrent update")
    write_json(out / "verification.json", {"gate_unchanged": True if gate_hash is not None else None,
               "control_gate_check": "verified" if gate_hash is not None else "not_requested",
               "control_gate_sha256": gate_hash,
               "labels": ["ret_fwd_3d", "ret_fwd_5d"], "included_symbols": int(coverage.included.sum()),
               "excluded_symbols": coverage.loc[~coverage.included, "symbol"].tolist(),
               "no_production_promotion": True, "no_entry_or_nav_written": True,
               "source_type": sorted(membership.source.dropna().unique()),
               "snapshot_date": args.as_of, "report_date": "2026-10-25"})
    write_json(out / "sha256.json", {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name != "sha256.json"})
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--as-of", type=dt.date.fromisoformat, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--control-gate", type=Path,
                        help="Optional existing paper-plan control gate; read and hash only, never modified")
    parser.add_argument("--cohorts", type=int, default=24)
    parser.add_argument("--cost-bps", type=float, default=17.5)
    parser.add_argument("--n-jobs", type=int, default=4)
    args = parser.parse_args()
    if args.cohorts < 5 or args.cost_bps < 0:
        parser.error("cohorts must be >=5 and costs nonnegative")
    run(args)
