"""Update the flow denominator and cached technical inputs for the same snapshot."""
import datetime as dt
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quant_platform.features.pipeline import FeaturePipeline
from quant_platform.features.registry import DEFAULT_SPECS, compute_feature_set_id
from quant_platform.store.parquet_store import read_ohlcv, write_parquet

TARGET = dt.date(2026, 10, 14)
STORE = ROOT / "models/data"
BACKUP = ROOT / "temp/snapshot_20261014/before"
syms = pd.read_parquet(STORE / "universe/csi300/membership.parquet").symbol.astype(str).str.zfill(6).tolist()
updates = []
feature_id = compute_feature_set_id(DEFAULT_SPECS)
for symbol in syms:
    prices = read_ohlcv(STORE / f"silver/ohlcv/{symbol}.parquet")
    path = STORE / f"silver/valuation/{symbol}.parquet"
    valuation = pd.read_parquet(path)
    valuation["date"] = pd.to_datetime(valuation.date).dt.date
    candidates = prices[(prices.date > valuation.date.max()) & (prices.date <= TARGET)].copy()
    if "outstanding_share" in candidates:
        candidates = candidates[candidates.outstanding_share.notna() & (candidates.outstanding_share > 0)]
        # Date-specific observed float shares times close; no future share
        # snapshot is backfilled into historical dates, and PE/PB is not filled.
        if not candidates.empty:
            fresh = pd.DataFrame({"symbol": symbol, "date": candidates.date,
                                  "float_mcap_yi": candidates.close * candidates.outstanding_share / 1e8,
                                  "mcap_source": "observed_ohlcv_close_times_float_shares"})
            old_backup = BACKUP / "valuation" / path.name
            if not old_backup.exists():
                old_backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, old_backup)
            write_parquet(pd.concat([valuation, fresh], ignore_index=True).sort_values("date"), path)
            updates.append({"symbol": symbol, "added_rows": len(fresh), "latest": str(fresh.date.max())})
    feature_path = STORE / f"features/{feature_id}/{symbol}.parquet"
    destination = BACKUP / "features" / feature_path.name
    if feature_path.exists() and not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(feature_path, destination)

(ROOT / "temp/snapshot_20261014/marketcap_update_audit.json").write_text(json.dumps(updates, indent=2), encoding="utf-8")
print(f"Current float market cap updated for {len(updates)} stocks", flush=True)
pipeline = FeaturePipeline(STORE, project_root=ROOT)
pipeline.run(syms, specs=DEFAULT_SPECS, end_date=TARGET, strict=True)
print("Technical feature snapshot complete", flush=True)
