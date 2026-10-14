"""Refresh the current market snapshot with per-symbol backups and audits."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime as dt
import json
from pathlib import Path
import shutil
import sys
import threading

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quant_platform.ingest.ohlcv_collector import _fetch_symbol_raw
from quant_platform.ingest.flow_collector import FundFlowCollector
from quant_platform.ingest.fund_flow_providers import EastmoneyH5FundFlowProvider
from quant_platform.store.parquet_store import read_ohlcv, write_ohlcv

TARGET = dt.date(2026, 10, 14)
STORE = ROOT / "models/data"
OUT = ROOT / "temp/snapshot_20261014"
OUT.mkdir(parents=True, exist_ok=True)
symbols = pd.read_parquet(STORE / "universe/csi300/membership.parquet").symbol.astype(str).str.zfill(6).tolist()
local = threading.local()


def request(*args, **kwargs):
    if not hasattr(local, "session"):
        local.session = requests.Session()
        local.session.trust_env = False
    kwargs["timeout"] = (6, 12)
    return local.session.get(*args, **kwargs)


requests.get = request


def backup(path: Path, category: str):
    dest = OUT / "before" / category / path.name
    if path.exists() and not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)


def ohlcv(symbol):
    # P0(2026-10-14 审计)：此前手工重叠判断只比较近期 N 天，除权日在窗口外时
    # 漏判（如 300803）。现委托统一采集实现做全历史复权尺度检查，
    # 不在此处重复不充分的判断。
    from quant_platform.ingest.ohlcv_collector import collect_symbol
    try:
        path = STORE / "silver/ohlcv" / f"{symbol}.parquet"
        old = read_ohlcv(path)
        backup(path, "ohlcv")
        result = collect_symbol(
            symbol, STORE,
            start_date=old["date"].min() if not old.empty else TARGET - dt.timedelta(days=365 * 3),
            end_date=TARGET,
            deep_verify=True,  # 强制全历史 QFQ 尺度检查
        )
        if not result.success:
            return {"symbol": symbol, "status": "failed", "error": result.error[:180]}
        return {"symbol": symbol, "status": "updated", "latest": str(result.last_date),
                "adjustment_history_refreshed": result.error.startswith("scale_refreshed"),
                "rows": result.rows_total,
                "source": "ingest.ohlcv_collector.collect_symbol(deep_verify=True)"}
    except Exception as exc:
        return {"symbol": symbol, "status": "failed", "error": str(exc)[:180]}


collector = FundFlowCollector(STORE, providers=[EastmoneyH5FundFlowProvider()])


def flow(symbol):
    try:
        result = collector.providers[0].fetch_symbol(symbol, days=120)
        fresh = result.frame[result.frame.trade_date <= TARGET].copy()
        if fresh.empty or fresh.trade_date.max() != TARGET:
            return {"symbol": symbol, "status": "stale", "latest": str(fresh.trade_date.max())}
        if fresh[fresh.trade_date == TARGET][["main_net", "small_net", "super_net"]].isna().any().any():
            raise RuntimeError("Incomplete requested flow fields")
        # Writes are serialized by the orchestrating thread.
        return {"symbol": symbol, "status": "fetched", "latest": str(TARGET), "frame": fresh}
    except Exception as exc:
        return {"symbol": symbol, "status": "failed", "error": str(exc)[:180]}


mode = sys.argv[1]
records = []
with ThreadPoolExecutor(max_workers=4) as pool:
    jobs = [pool.submit(ohlcv if mode == "ohlcv" else flow, s) for s in symbols]
    for i, job in enumerate(as_completed(jobs), 1):
        record = job.result()
        if record["status"] == "fetched":
            frame = record.pop("frame")
            try:
                backup(STORE / "silver/fund_flow" / f"{record['symbol']}.parquet", "fund_flow")
                record["new_rows"] = collector._write_one(record["symbol"], frame, overwrite=False)
                record["status"] = "updated"
            except Exception as exc:
                record.update(status="failed", error=str(exc)[:180])
        records.append(record)
        if i % 20 == 0 or i == len(jobs):
            print(f"{mode}: {i}/{len(jobs)} processed; updated={sum(r['status']=='updated' for r in records)}", flush=True)
            (OUT / f"{mode}_update_audit.json").write_text(json.dumps({"target": str(TARGET), "records": records}, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"{mode} snapshot done", flush=True)
