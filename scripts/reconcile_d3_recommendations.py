"""Reconcile frozen D+3 recommendations over at most the latest calendar week.

No training, label-store update, or replacement of the delivered stock lists.
Prices are a separate CSV snapshot: symbol,date,close,source (000300 = index).
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quant_platform.features.industry import _join_industry
from quant_platform.selection.config import SelectionConfig, StrategyType
from quant_platform.selection.fusion_score import fusion_score
from quant_platform.selection.ranker import IndustryNeutralRanker


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def controlled_selection(frame: pd.DataFrame, score: str, count: int) -> pd.DataFrame:
    ranked = IndustryNeutralRanker(
        SelectionConfig(strategy=StrategyType.EQUAL_TOP_K, top_k=3, max_total=count),
        score_col=score,
    ).run(frame)
    selected = ranked[ranked.selected].sort_values([score, "symbol"], ascending=[False, True]).copy()
    selected["rank"] = np.arange(1, len(selected) + 1)
    return selected


def build_strategies(ordinary_path: Path, fusion_dir: Path, store: Path) -> tuple[dict, dict]:
    ordinary = json.loads(ordinary_path.read_text(encoding="utf-8"))
    mixed = json.loads((fusion_dir / "run_summary.json").read_text(encoding="utf-8"))
    if ordinary["as_of"] != mixed["as_of"]:
        raise ValueError("The two delivered recommendations have different signal dates")
    for data in (ordinary, mixed):
        if data["horizons"]["3"]["label_col"] != "ret_fwd_3d":
            raise ValueError("Only the exact ret_fwd_3d target is accepted")
    plain = pd.DataFrame(ordinary["horizons"]["3"]["selected"])
    fused = pd.read_csv(fusion_dir / "D3_picks.csv", dtype={"symbol": str})
    if fused.symbol.tolist() != [r["symbol"] for r in mixed["horizons"]["3"]["selected"]]:
        raise ValueError("Fusion CSV differs from its frozen run summary")
    fused["rank"] = fused.recommendation_rank
    count = len(fused)
    strategies = {"ordinary_delivered": plain, "mixed_delivered": fused,
                  "ordinary_equal_count": plain.head(count).copy()}
    scored = pd.read_csv(fusion_dir / "D3_gate.csv", dtype={"symbol": str})
    signal = dt.date.fromisoformat(mixed["as_of"])
    attributes = _join_industry(
        pd.DataFrame({"symbol": scored.symbol, "date": signal}),
        pd.read_parquet(store / "silver/industry_map.parquet"),
    )
    names = pd.read_parquet(store / "universe/csi300/membership.parquet")
    name_map = dict(zip(names.symbol.astype(str).str.zfill(6), names.name))
    scored = scored.merge(attributes, on="symbol", validate="one_to_one")
    scored["name"] = scored.symbol.map(name_map)
    # Keep the same existing risk exclusion, but remove the score-based Gate.
    scored = scored[~scored.gate_tier.isin(["RISK_VETO", "RISK_DOWNGRADE"])].copy()
    strategies["long_lgbm_equal_count"] = controlled_selection(scored, "base_score", count)
    strategies["recent_flow_equal_count"] = controlled_selection(scored, "recent_score", count)
    strategies["rrf_without_score_gate"] = controlled_selection(fusion_score(scored), "fusion_score_col", count)
    for key, rows in strategies.items():
        if rows.symbol.duplicated().any():
            raise ValueError(f"{key}: duplicate symbols")
    meta = {
        "signal_date": mixed["as_of"], "entry_date": mixed["entry_date"],
        "target_date": mixed["horizons"]["3"]["target_date"],
        "target": "ret_fwd_3d", "label_definition": "close(T+1+3)/close(T+1)-1",
        "ordinary_source": str(ordinary_path), "ordinary_sha256": sha256(ordinary_path),
        "fusion_source": str(fusion_dir / "run_summary.json"),
        "fusion_sha256": sha256(fusion_dir / "run_summary.json"),
        "controls_status": "Retrospective diagnostics selected from frozen signal-day scores; not delivered portfolios",
        "fusion_run": mixed,
    }
    return strategies, meta


def reconcile_basket(rows: pd.DataFrame, prices: pd.DataFrame, entry: dt.date,
                     observed: dt.date, target: dt.date, cost_per_side: float) -> tuple[dict, list[dict]]:
    if observed < entry:
        raise ValueError("Observation must be on or after the entry date")
    if not 0 <= cost_per_side < 0.1:
        raise ValueError("Invalid cost per side")
    price = prices.copy()
    price["date"] = pd.to_datetime(price.date).dt.date
    price = price[price.date <= observed]
    if price.duplicated(["symbol", "date"]).any():
        raise ValueError("Duplicate price rows")
    lookup = price.set_index(["symbol", "date"])["close"].to_dict()
    def valid(value):
        return value is not None and np.isfinite(value) and value > 0
    ia, ib = lookup.get(("000300", entry)), lookup.get(("000300", observed))
    benchmark = float(ib / ia - 1) if valid(ia) and valid(ib) else None
    detail = []
    for row in rows.to_dict("records"):
        symbol = row["symbol"]
        a, b = lookup.get((symbol, entry)), lookup.get((symbol, observed))
        gross = float(b / a - 1) if valid(a) and valid(b) else None
        detail.append({
            "rank": int(row["rank"]), "symbol": symbol, "name": row.get("name", ""),
            "industry_name": row.get("industry_name", ""),
            "entry_close": float(a) if valid(a) else None,
            "observed_close": float(b) if valid(b) else None,
            "gross_return": gross,
            "net_if_exit": gross - 2 * cost_per_side if gross is not None else None,
            "excess_if_exit": gross - 2 * cost_per_side - benchmark if gross is not None and benchmark is not None else None,
            "price_status": "complete" if gross is not None else "missing_entry_or_observation_price",
        })
    usable = [r for r in detail if r["gross_return"] is not None]
    complete = len(usable) == len(rows) and len(rows) > 0
    gross = float(np.mean([r["gross_return"] for r in usable])) if complete else None
    net = gross - 2 * cost_per_side if gross is not None else None
    return {
        "selected_count": len(rows), "priced_count": len(usable), "price_complete": complete,
        "gross_return": gross, "net_if_exit": net, "benchmark_return": benchmark,
        "excess_if_exit": net - benchmark if net is not None and benchmark is not None else None,
        "stock_win_rate": float(np.mean([r["gross_return"] > 0 for r in usable])) if complete else None,
        "stock_beat_index_rate": float(np.mean([r["gross_return"] > benchmark for r in usable])) if complete and benchmark is not None else None,
        "mature": observed >= target, "status": "full_D3" if observed >= target else "interim_D3",
        "missing_symbols": [r["symbol"] for r in detail if r["gross_return"] is None],
    }, detail


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ordinary-json", type=Path, required=True)
    parser.add_argument("--fusion-dir", type=Path, required=True)
    parser.add_argument("--store", type=Path, default=ROOT / "models/data")
    parser.add_argument("--prices-csv", type=Path)
    parser.add_argument("--as-of", type=dt.date.fromisoformat, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--cost-per-side", type=float, default=0.001)
    args = parser.parse_args()
    portfolios, meta = build_strategies(args.ordinary_json, args.fusion_dir, args.store)
    signal = dt.date.fromisoformat(meta["signal_date"])
    if not args.as_of - dt.timedelta(days=6) <= signal <= args.as_of:
        raise ValueError("Signal is outside the latest seven calendar days; refusing a wider backtest")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for key, rows in portfolios.items():
        rows[["rank", "symbol", "name", "industry_name"]].to_csv(args.out_dir / f"{key}_portfolio.csv", index=False, encoding="utf-8-sig")
    requested = sorted(set().union(*(set(f.symbol) for f in portfolios.values())))
    (args.out_dir / "requested_symbols.json").write_text(json.dumps(requested), encoding="utf-8")
    if args.prices_csv is None:
        print(f"Prepared frozen portfolios: {len(requested)} stocks", flush=True)
        return
    prices = pd.read_csv(args.prices_csv, dtype={"symbol": str})
    price_dates = pd.to_datetime(prices.date).dt.date
    prices = prices.loc[price_dates <= args.as_of].copy()
    entry, target = map(dt.date.fromisoformat, (meta["entry_date"], meta["target_date"]))
    calendar = pd.read_parquet(args.store / "calendar/trading_calendar.parquet")
    sessions = set(pd.to_datetime(calendar.loc[calendar.is_trading, "date"]).dt.date)
    if args.as_of not in sessions or entry not in sessions:
        raise ValueError("Entry and observation must be trading dates")
    summary = {k: v for k, v in meta.items() if k != "fusion_run"}
    summary.update({"as_of": args.as_of.isoformat(), "window_start": (args.as_of - dt.timedelta(days=6)).isoformat(),
                    "cost_per_side": args.cost_per_side, "cost_meaning": "Hypothetical full exit at observation close; 10bp each side by default",
                    "prices_sha256": sha256(args.prices_csv), "strategies": {}, "daily": []})
    summary["latest_available_price_date"] = max(pd.to_datetime(prices.date).dt.date).isoformat() if len(prices) else None
    summary["performance_status"] = "pending_price_reconciliation"
    all_detail = []
    days = sorted(d for d in sessions if entry <= d <= min(args.as_of, target))
    for key, rows in portfolios.items():
        last, detail = reconcile_basket(rows, prices, entry, min(args.as_of, target), target, args.cost_per_side)
        summary["strategies"][key] = last
        all_detail.extend({"strategy": key, **r} for r in detail)
        for day in days:
            metric, _ = reconcile_basket(rows, prices, entry, day, target, args.cost_per_side)
            summary["daily"].append({"strategy": key, "date": day.isoformat(), **metric})
    detail_frame = pd.DataFrame(all_detail)
    detail_frame.to_csv(args.out_dir / "stock_returns.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summary["daily"]).to_csv(args.out_dir / "daily_returns.csv", index=False, encoding="utf-8-sig")
    summary["ordinary_mixed_overlap"] = len(set(portfolios["ordinary_delivered"].symbol) & set(portfolios["mixed_delivered"].symbol))
    required = [summary["strategies"][k] for k in ("ordinary_delivered", "mixed_delivered")]
    if all(x["price_complete"] and x["benchmark_return"] is not None for x in required):
        summary["performance_status"] = "complete_D3" if all(x["mature"] for x in required) else "interim_prices_complete"
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"as_of": summary["as_of"], "strategies": summary["strategies"]}, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
