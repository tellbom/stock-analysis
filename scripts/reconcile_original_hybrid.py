"""Evaluate the controlled hybrid experiment over at most one calendar week.

Read frozen experiment picks and staged daily prices. Never train models or
change a recommendation. D3/D5 endpoints are capped at their own target dates.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd


def evaluate(out: Path, prices_path: Path, observed_through: dt.date, one_way_cost_bps: float) -> dict:
    train = json.loads((out / "training_summary.json").read_text(encoding="utf-8"))
    entry = dt.date.fromisoformat(train["entry_date"])
    if observed_through < entry or (observed_through - entry).days > 7:
        raise ValueError("Evaluation must cover at most one calendar week from entry")
    if not np.isfinite(one_way_cost_bps) or one_way_cost_bps < 0:
        raise ValueError("Invalid cost")
    prices = pd.read_csv(prices_path, dtype={"symbol": str})
    prices["symbol"] = prices.symbol.str.zfill(6)
    prices["date"] = pd.to_datetime(prices.date).dt.date
    prices = prices[prices.date.between(entry, observed_through)].copy()
    if prices.duplicated(["symbol", "date"]).any() or not np.isfinite(prices.close).all() or (prices.close <= 0).any():
        raise ValueError("Invalid or duplicate prices")
    picks = pd.read_csv(out / "all_picks.csv", dtype={"symbol": str})
    picks["symbol"] = picks.symbol.str.zfill(6)
    lookup = prices.set_index(["symbol", "date"]).close
    price_days = sorted(prices.loc[prices.symbol == "000300", "date"].unique())
    if not price_days or price_days[0] != entry or price_days[-1] != observed_through:
        raise ValueError("Missing index endpoint")
    cost = one_way_cost_bps / 10000.0
    stats, details, daily, overlaps = [], [], [], []
    for h in (3, 5):
        target = dt.date.fromisoformat(train["horizons"][str(h)]["target_date"])
        endpoint = min(observed_through, target)
        days = [d for d in price_days if d <= endpoint]
        if days[-1] != endpoint:
            raise ValueError("Missing target/observation day")
        selected = picks[picks.horizon == h].copy()
        need = {(s, d) for s in set(selected.symbol) | {"000300"} for d in days}
        missing = sorted(need - set(lookup.index))
        if missing:
            raise ValueError(f"Missing holding-week prices: {missing[:8]}")
        index_return = float(lookup["000300", endpoint] / lookup["000300", entry] - 1)
        baseline_set = set(selected[(selected.strategy == "ordinary") & selected.in_equal_count_comparison].symbol)
        for strategy, rows in selected.groupby("strategy", sort=False):
            for scope in ("equal_count", "full_selection"):
                subset = rows[rows.in_equal_count_comparison] if scope == "equal_count" else rows
                n = len(subset)
                if n == 0:
                    raise ValueError("Empty comparison")
                values = []
                for row in subset.itertuples():
                    start_price, end_price = float(lookup[row.symbol, entry]), float(lookup[row.symbol, endpoint])
                    value = end_price / start_price - 1
                    details.append({"horizon": h, "strategy": strategy, "strategy_name": row.strategy_name,
                                    "scope": scope, "rank": row.recommendation_rank, "symbol": row.symbol,
                                    "name": row.name, "industry_name": row.industry_name,
                                    "entry_close": start_price, "observed_close": end_price, "gross_return": value,
                                    "net_if_exit": value - 2 * cost, "excess_if_exit": value - 2 * cost - index_return,
                                    "entry_date": entry.isoformat(), "end_date": endpoint.isoformat()})
                    values.append(value)
                curve = []
                for date in days:
                    gross = float(np.mean([lookup[s, date] / lookup[s, entry] - 1 for s in subset.symbol]))
                    benchmark = float(lookup["000300", date] / lookup["000300", entry] - 1)
                    curve.append(1 + gross)
                    daily.append({"horizon": h, "strategy": strategy, "scope": scope, "date": date.isoformat(),
                                  "gross_return": gross, "net_if_exit": gross - 2 * cost, "benchmark_return": benchmark})
                gross = float(np.mean(values))
                stats.append({"horizon": h, "strategy": strategy, "strategy_name": rows.strategy_name.iloc[0],
                              "scope": scope, "count": n, "gross_return": gross, "net_if_exit": gross - 2 * cost,
                              "benchmark_return": index_return, "excess_if_exit": gross - 2 * cost - index_return,
                              "stock_win_rate": float(np.mean(np.array(values) > 0)),
                              "stock_beat_index_rate": float(np.mean(np.array(values) > index_return)),
                              "holding_week_max_drawdown": float(np.min(np.array(curve) / np.maximum.accumulate(curve) - 1)),
                              "entry_date": entry.isoformat(), "end_date": endpoint.isoformat(),
                              "target_date": target.isoformat(), "complete": endpoint == target,
                              "status": "完整持有期" if endpoint == target else "阶段收益，尚未到期"})
            equal_set = set(rows[rows.in_equal_count_comparison].symbol)
            overlaps.append({"horizon": h, "strategy": strategy, "count": len(equal_set),
                             "overlap_with_ordinary": len(equal_set & baseline_set),
                             "replacements_vs_ordinary": len(equal_set - baseline_set)})
    for h in (3, 5):
        for scope in ("equal_count", "full_selection"):
            rows = [s for s in stats if s["horizon"] == h and s["scope"] == scope]
            by_key = {s["strategy"]: s for s in rows}
            for s in rows:
                s["return_delta_vs_ordinary"] = s["net_if_exit"] - by_key["ordinary"]["net_if_exit"]
                s["flow_increment_vs_no_flow"] = by_key["hybrid_flow"]["net_if_exit"] - by_key["hybrid_no_flow"]["net_if_exit"]
                s["score_gate_increment"] = by_key["hybrid_flow_score_gate"]["net_if_exit"] - by_key["hybrid_flow"]["net_if_exit"]
    result = {"signal_date": train["as_of"], "entry_date": entry.isoformat(),
              "observed_through": observed_through.isoformat(), "one_way_cost_bps": one_way_cost_bps,
              "research_only": True, "retrospective_reconstruction": True,
              "cost_convention": "gross minus two one-way costs; early endpoints assume exit",
              "comparison_n_rule": "minimum selected count across four arms, fixed before reading returns",
              "strategies": stats, "overlaps": overlaps}
    for name, data in (("stock_returns", details), ("daily_returns", daily), ("selection_overlap", overlaps)):
        pd.DataFrame(data).to_csv(out / f"{name}.csv", index=False, encoding="utf-8-sig")
    prices.to_csv(out / "prices.csv", index=False, encoding="utf-8-sig")
    (out / "performance_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(pd.DataFrame(stats).loc[lambda f: f.scope == "equal_count", ["horizon", "strategy", "count", "net_if_exit", "excess_if_exit", "return_delta_vs_ordinary", "status"]].to_string(index=False))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--prices", type=Path, required=True)
    parser.add_argument("--observed-through", type=dt.date.fromisoformat, required=True)
    parser.add_argument("--one-way-cost-bps", type=float, default=10)
    args = parser.parse_args()
    evaluate(args.experiment_dir, args.prices, args.observed_through, args.one_way_cost_bps)


if __name__ == "__main__":
    main()
