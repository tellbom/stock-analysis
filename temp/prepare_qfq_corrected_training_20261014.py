"""Rebuild the corrected October 14 feature/label snapshot with backups."""
from pathlib import Path
import datetime as dt
import json
import shutil
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quant_platform.features.pipeline import FeaturePipeline, verify_feature_freshness
from quant_platform.features.registry import DEFAULT_SPECS, compute_feature_set_id
from quant_platform.labels.builder import build_labels, DEFAULT_HORIZONS

STORE = ROOT / 'models/data'
SNAPSHOT = ROOT / 'temp/snapshot_20261014_qfq_corrected'
ASOF = dt.date(2026, 10, 14)
symbols = pd.read_parquet(STORE / 'universe/csi300/membership.parquet').symbol.astype(str).str.zfill(6).tolist()
feature_id = compute_feature_set_id(DEFAULT_SPECS)
for symbol in symbols:
    for category, folder in (('features', STORE / f'features/{feature_id}'), ('labels', STORE / 'labels/forward_returns')):
        for suffix in ('.parquet', '.provenance.json'):
            source = folder / f'{symbol}{suffix}'
            dest = SNAPSHOT / f'before/{category}/{source.name}'
            if source.exists() and not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, dest)
print('Feature and label backups complete', flush=True)
pipeline = FeaturePipeline(STORE, project_root=ROOT)
pipeline.run(symbols, specs=DEFAULT_SPECS, end_date=ASOF, strict=True)
stale = verify_feature_freshness(STORE, feature_id, symbols)
if stale:
    raise RuntimeError(f'Stale rebuilt feature snapshots: {stale}')
print('All 300 feature fingerprints verified', flush=True)
results = build_labels(STORE, symbols, horizons=DEFAULT_HORIZONS, overwrite=True)
failed = [s for s, n in results.items() if n == 0]
if failed:
    raise RuntimeError(f'Label rebuild failed: {failed}')
for symbol in symbols:
    cols = pd.read_parquet(STORE / f'labels/forward_returns/{symbol}.parquet').columns
    if any(f'ret_fwd_{h}d' not in cols for h in DEFAULT_HORIZONS):
        raise RuntimeError(f'Existing label horizons lost: {symbol}')
(SNAPSHOT / 'feature_label_rebuild.json').write_text(json.dumps({
    'as_of': str(ASOF), 'feature_set_id': feature_id, 'symbols': len(symbols),
    'stale_features': stale, 'rebuilt_horizons': DEFAULT_HORIZONS,
    'label_rows': results, 'label_date_basis': 'exchange_calendar_complete_forward_window'
}, indent=2), encoding='utf-8')
print('All original label horizons rebuilt; ready to train', flush=True)
