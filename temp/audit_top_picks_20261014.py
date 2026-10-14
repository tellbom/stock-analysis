"""Read-only audit of the frozen October 14 scores and price decline."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quant_platform.features.cross_sectional import build_cross_sectional_features
from quant_platform.features.flow import build_flow_features, load_flow_panel
from quant_platform.features.registry import DEFAULT_SPECS, compute_feature_set_id
from quant_platform.features.valuation import load_valuation_panel
from quant_platform.selection.ridge_baseline import ORIGINAL_FEATURES
from quant_platform.selection.fusion_score import fusion_score
from scripts.compare_original_hybrid import FLOW_FEATURES, fingerprint, select

ASOF = dt.date(2026, 10, 14)
STORE = ROOT / 'models/data'
OUT = Path(os.environ.get('STOCK_WEEKLY_OUTPUT_DIR', ROOT / 'outputs/01a116ce-1f0c-7d41-8967-adcca880c1b8/2026-10-14_统一基线_D3_D5'))
BATCH = OUT / 'comparison'
summary = json.loads((BATCH / 'training_summary.json').read_text(encoding='utf-8'))
manifest = json.loads((OUT / 'model_manifest.json').read_text(encoding='utf-8'))
picks = pd.read_csv(BATCH / 'all_picks.csv', dtype={'symbol': str})
symbols = pd.read_parquet(STORE / 'universe/csi300/membership.parquet').symbol.astype(str).str.zfill(6).sort_values().tolist()
frames = []
for symbol in symbols:
    path = STORE / f'features/{compute_feature_set_id(DEFAULT_SPECS)}/{symbol}.parquet'
    f = pd.read_parquet(path)
    f['date'] = pd.to_datetime(f.date).dt.date
    frames.append(f[f.date == ASOF].copy())
pred = build_cross_sectional_features(pd.concat(frames, ignore_index=True))
pred['symbol'] = pred.symbol.astype(str).str.zfill(6)
pred = pred.sort_values(['symbol', 'date']).dropna(subset=ORIGINAL_FEATURES).reset_index(drop=True)
assert fingerprint(pred, ['symbol', 'date'] + ORIGINAL_FEATURES) == summary['prediction_feature_hash']
flow = load_flow_panel(STORE, symbols)
flow = flow[flow.date <= ASOF].copy()
valuation = load_valuation_panel(STORE, symbols)
valuation['date'] = pd.to_datetime(valuation.date).dt.date
valuation = valuation[valuation.date <= ASOF].copy()
flow_features = build_flow_features(pred[['symbol', 'date']], flow, valuation)
pred = pred.drop(columns=FLOW_FEATURES, errors='ignore').merge(
    flow_features[['symbol', 'date'] + FLOW_FEATURES], on=['symbol', 'date'], validate='one_to_one')
attributes = pd.read_csv(BATCH / 'prediction_attributes.csv', dtype={'symbol': str})
top_symbols = sorted(set(picks[(picks.recommendation_rank <= 5) & picks.strategy.isin(['ordinary', 'hybrid_flow'])].symbol))
report = {'signal_date': str(ASOF), 'prediction_rows': len(pred), 'feature_hash_matches': True,
          'source_hashes_match': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == v for p, v in manifest['runtime_source_hashes'].items()},
          'model_hashes_match': {p: hashlib.sha256((BATCH / p).read_bytes()).hexdigest() == v for p, v in manifest['model_hashes'].items()},
          'score_checks': [], 'selection_checks': [], 'ridge_contributions': [], 'flow_model_contributions': [], 'prices': []}
contribution_rows = []
for h in (3, 5):
    scored = {}
    for arm, stem, features, prefix in (
        ('ordinary', 'original_ridge', ORIGINAL_FEATURES, 'base'),
        ('hybrid_no_flow', 'recent_no_flow', ORIGINAL_FEATURES, 'recent'),
        ('hybrid_flow', 'recent_flow', ORIGINAL_FEATURES + FLOW_FEATURES, 'recent')):
        model = joblib.load(BATCH / f'D{h}_{stem}.joblib')
        values = model.predict(pred[features].to_numpy())
        score_stem = 'original' if arm == 'ordinary' else stem
        frozen = pd.read_csv(BATCH / f'D{h}_{score_stem}_scores.csv', dtype={'symbol': str}).set_index('symbol')
        error = float(np.max(np.abs(values - frozen.loc[pred.symbol, f'{prefix}_score'].to_numpy())))
        assert error < 1e-12, (h, arm, error)
        scored[arm] = pd.DataFrame({'symbol': pred.symbol, 'date': pred.date, 'trade_date': pred.date, f'{prefix}_score': values})
        scored[arm][f'{prefix}_rank'] = scored[arm][f'{prefix}_score'].rank(ascending=False, method='first').astype(int)
        scored[arm][f'{prefix}_pct'] = scored[arm][f'{prefix}_score'].rank(ascending=True, pct=True)
        report['score_checks'].append({'horizon': h, 'arm': arm, 'rows': len(values), 'max_abs_error': error})
        print('SCORE_CHECK', report['score_checks'][-1], flush=True)
        if arm == 'ordinary':
            transformed = model[:-1].transform(pred[features].to_numpy())
            estimator = model.named_steps['model']
            for symbol in top_symbols:
                i = int(np.flatnonzero(pred.symbol.to_numpy() == symbol)[0])
                terms = transformed[i] * estimator.coef_
                assert abs(float(terms.sum() + estimator.intercept_) - values[i]) < 1e-12
                positive = sorted(zip(features, terms), key=lambda item: item[1], reverse=True)[:6]
                negative = sorted(zip(features, terms), key=lambda item: item[1])[:4]
                row = {'horizon': h, 'symbol': symbol, 'score_pct': float(values[i] * 100),
                       'intercept_pp': float(estimator.intercept_ * 100),
                       'top_positive_contributions_pp': {k: float(v * 100) for k, v in positive},
                       'top_negative_contributions_pp': {k: float(v * 100) for k, v in negative},
                       'reversal_3d_contribution_pp': float(terms[features.index('reversal_3d')] * 100)}
                report['ridge_contributions'].append(row)
                for k, value, raw, standardized in zip(features, terms, pred.iloc[i][features], transformed[i]):
                    contribution_rows.append({'horizon': h, 'symbol': symbol, 'feature': k, 'raw_feature': float(raw), 'standardized_feature': float(standardized), 'contribution_pp': float(value * 100)})
                print('RIDGE_ATTRIBUTION', row, flush=True)
        elif arm == 'hybrid_flow':
            transformed = model[:-1].transform(pred[features].to_numpy())
            contributions = model.named_steps['lgbm'].booster_.predict(transformed, pred_contrib=True)
            assert np.max(np.abs(contributions.sum(axis=1) - values)) < 1e-12
            for symbol in top_symbols:
                i = int(np.flatnonzero(pred.symbol.to_numpy() == symbol)[0])
                row = {'horizon': h, 'symbol': symbol, 'score_pct': float(values[i] * 100),
                       'flow_contribution_sum_pp': float(contributions[i, 32:37].sum() * 100),
                       'top_positive_contributions_pp': {k: float(v * 100) for k, v in sorted(zip(features, contributions[i, :-1]), key=lambda item: item[1], reverse=True)[:5]}}
                report['flow_model_contributions'].append(row)
                print('FLOW_ATTRIBUTION', row, flush=True)
    for arm in ('ordinary', 'hybrid_no_flow', 'hybrid_flow'):
        if arm == 'ordinary':
            frame, score_col = scored[arm], 'base_score'
        else:
            combined = scored['ordinary'].merge(scored[arm].drop(columns='date'), on=['symbol', 'trade_date'], validate='one_to_one')
            combined['risk_flags'] = ''
            combined['event_flags'] = ''
            frame, score_col = fusion_score(combined), 'fusion_score_col'
        selected, _ = select(frame, attributes, score_col, sector_cap_tech=10)
        frozen = picks[(picks.horizon == h) & (picks.strategy == arm)].sort_values('recommendation_rank')
        assert selected.symbol.tolist() == frozen.symbol.tolist(), (h, arm)
        report['selection_checks'].append({'horizon': h, 'arm': arm, 'count': len(selected), 'ordered_picks_match': True})
for symbol in top_symbols:
    price = pd.read_parquet(STORE / f'silver/ohlcv/{symbol}.parquet').sort_values('date')
    price['date'] = pd.to_datetime(price.date).dt.date
    price = price[price.date <= ASOF].copy()
    c = price.close
    f = pred[pred.symbol == symbol].iloc[0]
    flow_tail = flow[flow.symbol == symbol].sort_values('date').tail(5)
    row = {'symbol': symbol, 'name': attributes.set_index('symbol').loc[symbol, 'name'], 'close': float(c.iloc[-1]),
           'return_5d_pct': float((c.iloc[-1]/c.iloc[-6]-1)*100), 'return_10d_pct': float((c.iloc[-1]/c.iloc[-11]-1)*100),
           'return_20d_pct': float((c.iloc[-1]/c.iloc[-21]-1)*100), 'last_20_peak_drawdown_pct': float((c.iloc[-1]/c.tail(20).max()-1)*100),
           'rsi_6': float(f.rsi_6), 'ma_20_distance_pct': float(f.ma_20 * 100),
           'flow_main_net_latest_yuan': float(flow_tail.main_net.iloc[-1]), 'flow_main_net_5d_yuan': float(flow_tail.main_net.sum()),
           'flow_features': {k: float(f[k]) for k in FLOW_FEATURES},
           'last_10_prices': [{'date': str(r.date), 'close': float(r.close)} for r in price.tail(10).itertuples()]}
    report['prices'].append(row)
audit = OUT / 'top_picks_audit'
audit.mkdir(exist_ok=True)
(audit / 'audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
pd.DataFrame(contribution_rows).to_csv(audit / 'ridge_contributions.csv', index=False, encoding='utf-8-sig')
pd.DataFrame([{k:v for k,v in r.items() if not isinstance(v,(dict,list))} for r in report['prices']]).to_csv(audit / 'price_declines.csv', index=False, encoding='utf-8-sig')
print('AUDIT_COMPLETE', str(audit), flush=True)
