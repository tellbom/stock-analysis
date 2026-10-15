"""Read-only, standard-library replay of the one-week stop-loss reference.

Uses committed frozen picks and four daily OHLC bars. No network, parquet,
training, or writes. It deliberately does not use the reviewed stop-loss code.
"""
from pathlib import Path
import csv
import hashlib
import json
import math
import statistics


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DATES = ("2026-10-08", "2026-10-09", "2026-10-12", "2026-10-13")


def read(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    manifest = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        if hashlib.sha256((HERE / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError("Evidence hash mismatch: " + name)
    picks_path = ROOT / "docs/reviews/2026-09-30-controlled/all_picks.csv"
    if hashlib.sha256(picks_path.read_bytes().replace(b"\r\n", b"\n")).hexdigest() != manifest["frozen_picks_sha256"]:
        raise RuntimeError("Frozen recommendation source changed")
    picks = [p for p in read(picks_path) if int(p["horizon"]) == 3]
    observed_sessions = tuple(p["date"] for p in read(HERE / "calendar_week.csv") if p["is_trading"] == "True")
    if observed_sessions != DATES:
        raise RuntimeError("Frozen exchange-calendar sessions differ from the holding window")
    prices = {(p["symbol"], p["date"]): p for p in read(HERE / "D3_prices.csv")}
    golden = read(HERE / "D3_reference_summary.csv")
    output = []
    for arm in dict.fromkeys(p["strategy"] for p in picks):
        for scope in ("equal_count", "full_selection"):
            basket = [p for p in picks if p["strategy"] == arm and (
                scope == "full_selection" or p["in_equal_count_comparison"] == "True")]
            for stop in (0.08, 0.05):
                for mode in ("ideal_threshold", "gap_aware_open"):
                    fixed, stopped_returns, trigger_count = [], [], 0
                    for pick in basket:
                        bars = [prices[pick["symbol"], day] for day in DATES]
                        entry = float(bars[0]["close"])
                        end = float(bars[-1]["close"])
                        exit_price = end
                        for bar in bars[1:]:
                            threshold = entry * (1 - stop)
                            if float(bar["low"]) <= threshold:
                                trigger_count += 1
                                exit_price = min(threshold, float(bar["open"])) if mode == "gap_aware_open" else threshold
                                break
                        fixed.append(end / entry - 1 - 0.0035)
                        stopped_returns.append(exit_price / entry - 1 - 0.0035)
                    result = {
                        "strategy": arm, "scope": scope, "stop_pct": stop,
                        "fill_mode": mode, "count": len(basket),
                        "n_stopped": trigger_count,
                        "baseline_net": statistics.mean(fixed),
                        "stop_net": statistics.mean(stopped_returns),
                        "delta_stop_minus_hold": statistics.mean(b - a for a, b in zip(fixed, stopped_returns)),
                    }
                    ref = next(r for r in golden if r["strategy"] == arm and r["scope"] == scope and float(r["stop_pct"]) == stop and r["fill_mode"] == mode)
                    for key in ("count", "n_stopped", "baseline_net", "stop_net", "delta_stop_minus_hold"):
                        if not math.isclose(float(result[key]), float(ref[key]), abs_tol=1e-12):
                            raise RuntimeError(f"Reference mismatch: {arm} {scope} {stop} {mode} {key}")
                    output.append(result)
    for r in output:
        if r["scope"] == "equal_count" and r["fill_mode"] == "gap_aware_open":
            print(f"{r['strategy']:26} stop={r['stop_pct']:.0%} N={r['count']} hold={r['baseline_net']:.4%} stop={r['stop_net']:.4%} delta={100*r['delta_stop_minus_hold']:.4f}pp")
    print("PASS: all 32 summary rows match; only the frozen D3 holding week was replayed")
    print("This is a retrospective reference, not acceptance of the reviewed production evaluator")


if __name__ == "__main__":
    main()
