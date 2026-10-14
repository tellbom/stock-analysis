"""Stage at most one calendar week of Tencent daily prices for an experiment."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime as dt
import json
from pathlib import Path
import time

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]


def fetch_week(symbols: list[str], start: dt.date, end: dt.date, output: Path, cache: Path) -> None:
    if end < start or (end - start).days > 7:
        raise ValueError("Only a holding-week price query is permitted")
    cache.mkdir(parents=True, exist_ok=True)
    calendar = pd.read_parquet(ROOT / "models/data/calendar/trading_calendar.parquet")
    calendar["date"] = pd.to_datetime(calendar.date).dt.date
    required = set(calendar.loc[calendar.is_trading & calendar.date.between(start, end), "date"])

    def fetch(symbol: str) -> tuple[list[dict], dict | None]:
        market = "sh" if symbol.startswith(("6", "9")) or symbol == "000300" else "sz"
        code = market + symbol
        session = requests.Session()
        session.trust_env = False
        url = f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={code},day,{start},{end},20,qfq"
        for attempt in range(3):
            try:
                path = cache / f"{code}.json"
                if attempt == 0 and path.exists():
                    payload = json.loads(path.read_text(encoding="utf-8"))
                else:
                    response = session.get(url, timeout=(8, 12))
                    response.raise_for_status()
                    payload = response.json()
                    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
                if payload.get("code") != 0:
                    raise ValueError("Nonzero provider status")
                data = payload.get("data", {}).get(code, {})
                field = "qfqday" if "qfqday" in data else "day"
                bars = data.get(field, [])
                rows = [{"symbol": symbol, "date": bar[0], "close": float(bar[2]),
                         "source": "tencent_daily_index" if symbol == "000300" else
                         ("tencent_daily_qfq" if field == "qfqday" else "tencent_qfq_request_day_field")}
                        for bar in bars if start <= dt.date.fromisoformat(bar[0]) <= end]
                dates = [dt.date.fromisoformat(row["date"]) for row in rows]
                if not required.issubset(dates) or len(set(dates)) != len(dates) or any(row["close"] <= 0 for row in rows):
                    raise ValueError("Missing/invalid daily prices")
                if symbol != "000300" and field == "day":
                    local = pd.read_parquet(ROOT / f"models/data/silver/ohlcv/{symbol}.parquet", columns=["date", "close"])
                    local["date"] = pd.to_datetime(local.date).dt.date
                    overlap = local[local.date.isin(dates)].sort_values("date")
                    if overlap.empty:
                        raise ValueError("Cannot validate alternate daily-bar shape")
                    last = overlap.iloc[-1]
                    provider_close = next(r["close"] for r in rows if r["date"] == last.date.isoformat())
                    if abs(provider_close - float(last.close)) > 0.010000001:
                        raise ValueError("Alternate daily-bar shape differs from local adjustment scale")
                return rows, None
            except Exception as error:
                if attempt == 2:
                    return [], {"symbol": symbol, "error": str(error)[:160]}
                time.sleep(0.4)
        raise AssertionError("unreachable")

    rows, failures = [], []
    with ThreadPoolExecutor(max_workers=3) as pool:
        tasks = {pool.submit(fetch, s): s for s in sorted(set(symbols) | {"000300"})}
        for i, future in enumerate(as_completed(tasks), 1):
            values, failure = future.result()
            rows.extend(values)
            if failure:
                failures.append(failure)
            if i % 20 == 0 or i == len(tasks):
                print(f"{i}/{len(tasks)} fetched; failures={len(failures)}", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values(["symbol", "date"]).to_csv(output, index=False, encoding="utf-8-sig")
    output.with_name("price_fetch_audit.json").write_text(json.dumps({
        "start": start.isoformat(), "end": end.isoformat(), "requested_symbols": len(set(symbols)),
        "required_dates": sorted(d.isoformat() for d in required), "rows": len(rows), "failures": failures,
        "provider": "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    if failures:
        raise RuntimeError(f"Incomplete holding-week price coverage: {len(failures)} symbols")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols-json", type=Path, required=True)
    parser.add_argument("--start", type=dt.date.fromisoformat, required=True)
    parser.add_argument("--end", type=dt.date.fromisoformat, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "temp/comparison_week_raw")
    args = parser.parse_args()
    fetch_week(json.loads(args.symbols_json.read_text(encoding="utf-8")), args.start, args.end, args.output, args.cache_dir)


if __name__ == "__main__":
    main()
