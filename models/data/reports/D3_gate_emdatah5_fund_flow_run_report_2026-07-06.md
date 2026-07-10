# D3 Gate-First Run Report 2026-07-06

- requested_as_of_date: 2026-07-06
- actual_as_of_date: 2026-07-06
- actual date reason: requested date has sufficient OHLCV coverage
- label: ret_fwd_3d

## Data Coverage Summary

```json
{
  "ohlcv": {
    "requested_coverage": "299/300",
    "actual_coverage": "299/300"
  },
  "features": {
    "1474f2c4": {
      "path": "features/1474f2c4",
      "files": 300,
      "symbol_file_coverage": "300/300",
      "as_of_files": 0,
      "latest_min": "2026-06-18",
      "latest_max": "2026-06-18"
    },
    "d02a4ebf": {
      "path": "features/d02a4ebf",
      "files": 300,
      "symbol_file_coverage": "300/300",
      "as_of_files": 0,
      "latest_min": "2026-06-18",
      "latest_max": "2026-06-29"
    }
  },
  "labels": {
    "path": "labels/forward_returns",
    "files": 300,
    "symbol_file_coverage": "300/300",
    "as_of_files": 0,
    "latest_min": "2026-06-18",
    "latest_max": "2026-07-03"
  },
  "fund_flow": {
    "path": "silver/fund_flow",
    "files": 300,
    "symbol_file_coverage": "300/300",
    "as_of_files": 299,
    "latest_min": "2026-06-26",
    "latest_max": "2026-07-06"
  },
  "fundamentals": {
    "path": "silver/fundamentals",
    "files": 79,
    "symbol_file_coverage": "79/300",
    "as_of_files": 0,
    "latest_min": "2023-07-08",
    "latest_max": "2026-06-15"
  },
  "lockup": {
    "path": "silver/lockup",
    "files": 0,
    "symbol_file_coverage": "0/300",
    "as_of_files": 0,
    "latest_min": null,
    "latest_max": null
  },
  "announcement_events": {
    "path": "silver/announcement_events",
    "files": 300,
    "symbol_file_coverage": "300/300",
    "as_of_files": 0,
    "latest_min": "2026-04-28",
    "latest_max": "2026-07-03"
  },
  "valuation": {
    "path": "silver/valuation",
    "files": 300,
    "symbol_file_coverage": "300/300",
    "as_of_files": 0,
    "latest_min": "2026-06-29",
    "latest_max": "2026-06-29"
  },
  "margin": {
    "path": "silver/margin",
    "files": 295,
    "symbol_file_coverage": "295/300",
    "as_of_files": 0,
    "latest_min": "2026-06-18",
    "latest_max": "2026-06-24"
  }
}
```

## Feature Gate Summary

- base_feature_set_id: `80fd2338`
- recent_enhanced_feature_set_id: `80fd2338`
- base candidates: 52; base admitted: 35
- recent candidates: 57; recent admitted: 40
- gate_mode: fund_flow_enhanced_gate
- fund flow entered recent model: yes (cs_main_flow_rank_1d, cs_main_flow_rank_5d, cs_small_flow_rank_1d, cs_super_flow_rank_1d, cs_flow_reversal_5d)
- fund_flow coverage gate result: {'status': 'computed_this_run', 'recent_symbol_coverage': 299, 'latest_available_date': '2026-07-06', 'recent_20d_avg_symbol_coverage': 288.65, 'available_trading_days': 130, 'field_missing_rate': 0.8603629197277506, 'is_allowed_for_recent_model': True, 'rejection_reason': 'short-history-family'}
- base final feature list: `E:\stock-analysis\models\data\reports\D3_base_X_train_columns_2026-07-06.csv`
- recent final feature list: `E:\stock-analysis\models\data\reports\D3_recent_emdatah5_X_train_columns_2026-07-06.csv`
- fund flow forbidden from base model: yes
- base coverage report: `E:\stock-analysis\models\data\reports\D3_base_coverage_gate_2026-07-06.csv`
- recent coverage report: `E:\stock-analysis\models\data\reports\D3_recent_coverage_gate_2026-07-06.csv`

## Model Summary

- base_model_d3: {'train_start': '2023-01-03', 'train_end': '2026-06-29', 'prediction_date': '2026-07-06', 'x_train_rows': 250172, 'prediction_rows': 299, 'feature_count': 35}
- recent_enhanced_model_d3 status: trained
- recent_enhanced_model_d3: {'train_start': '2025-12-25', 'train_end': '2026-06-29', 'prediction_date': '2026-07-06', 'x_train_rows': 35874, 'prediction_rows': 299, 'feature_count': 40}
- base feature importance Top20: `E:\stock-analysis\models\data\reports\D3_base_feature_importance_top20_2026-07-06.csv`
- fund_flow feature importance: `E:\stock-analysis\models\data\reports\D3_recent_fund_flow_feature_importance_2026-07-06.csv`
- recent feature importance Top20: `E:\stock-analysis\models\data\reports\D3_recent_feature_importance_top20_2026-07-06.csv`

## Gate Summary

- tier counts: {'E_REJECT': 127, 'UNCLASSIFIED': 105, 'A_MAIN': 43, 'B_SHORT_BOOST': 12, 'C_DOWNGRADE_OBSERVE': 8, 'D_OBSERVE': 4}
- risk veto count: 0
- top20 intersections: {'base_recent_top20': 2, 'base_gate_top20': 12, 'recent_gate_top20': 3, 'all_three_top20': 2}
- recent boosted symbols: ['603259', '000725', '600276', '002311', '300274', '603296', '688981', '600999', '688223', '000338', '688472', '600026']
- recent downgraded symbols: ['301165', '688521', '600547', '600426', '300502', '000657', '601899', '000999']
- event risk vetoed symbols: []

## Base Top20

| Rank | Symbol | Score |
|---:|---|---:|
| 1 | 301165 | +0.046577 |
| 2 | 688082 | +0.039370 |
| 3 | 000975 | +0.032019 |
| 4 | 002371 | +0.024759 |
| 5 | 600176 | +0.024475 |
| 6 | 002460 | +0.021224 |
| 7 | 688012 | +0.020285 |
| 8 | 688506 | +0.019372 |
| 9 | 300308 | +0.016971 |
| 10 | 688521 | +0.015808 |
| 11 | 301308 | +0.015745 |
| 12 | 300450 | +0.015294 |
| 13 | 603986 | +0.015207 |
| 14 | 300498 | +0.014664 |
| 15 | 600547 | +0.012994 |
| 16 | 601066 | +0.012924 |
| 17 | 002625 | +0.012157 |
| 18 | 002532 | +0.011801 |
| 19 | 600660 | +0.011765 |
| 20 | 002028 | +0.011404 |

## Recent Top20

| Rank | Symbol | Score |
|---:|---|---:|
| 1 | 300661 | +0.032983 |
| 2 | 002028 | +0.029263 |
| 3 | 688506 | +0.027433 |
| 4 | 603259 | +0.025027 |
| 5 | 002422 | +0.024083 |
| 6 | 000938 | +0.023588 |
| 7 | 000725 | +0.022270 |
| 8 | 002916 | +0.021340 |
| 9 | 600276 | +0.019851 |
| 10 | 002311 | +0.017642 |
| 11 | 300274 | +0.016435 |
| 12 | 688183 | +0.015250 |
| 13 | 603296 | +0.014945 |
| 14 | 000100 | +0.014139 |
| 15 | 688256 | +0.013250 |
| 16 | 300866 | +0.013116 |
| 17 | 600584 | +0.011294 |
| 18 | 600118 | +0.010310 |
| 19 | 688981 | +0.009777 |
| 20 | 002353 | +0.009436 |

## Gate Fused Top20

| Rank | Symbol | Tier | Reason |
|---:|---|---|---|
| 1 | 688082 | A_MAIN | base strong, recent confirms |
| 2 | 000975 | A_MAIN | base strong, recent confirms |
| 3 | 002371 | A_MAIN | base strong, recent confirms |
| 4 | 002460 | A_MAIN | base strong, recent confirms |
| 5 | 688506 | A_MAIN | base strong, recent confirms |
| 6 | 300450 | A_MAIN | base strong, recent confirms |
| 7 | 603986 | A_MAIN | base strong, recent confirms |
| 8 | 300498 | A_MAIN | base strong, recent confirms |
| 9 | 601066 | A_MAIN | base strong, recent confirms |
| 10 | 002532 | A_MAIN | base strong, recent confirms |
| 11 | 600660 | A_MAIN | base strong, recent confirms |
| 12 | 002028 | A_MAIN | base strong, recent confirms |
| 13 | 002938 | A_MAIN | base strong, recent confirms |
| 14 | 300394 | A_MAIN | base strong, recent confirms |
| 15 | 601225 | A_MAIN | base strong, recent confirms |
| 16 | 688303 | A_MAIN | base strong, recent confirms |
| 17 | 300418 | A_MAIN | base strong, recent confirms |
| 18 | 605499 | A_MAIN | base strong, recent confirms |
| 19 | 603799 | A_MAIN | base strong, recent confirms |
| 20 | 000100 | A_MAIN | base strong, recent confirms |

## Verification Artifacts

- `E:\stock-analysis\models\data\reports\D3_base_ranked_2026-07-06.csv`
- `E:\stock-analysis\models\data\reports\D3_recent_ranked_2026-07-06.csv`
- `E:\stock-analysis\models\data\reports\D3_gate_fused_ranked_2026-07-06.csv`
- `E:\stock-analysis\models\data\reports\D3_hybrid_ranked_2026-07-06.csv`
- `E:\stock-analysis\models\data\reports\D3_hybrid_reco_cards_2026-07-06.json`
- `E:\stock-analysis\models\data\reports\D3_gate_emdatah5_fund_flow_run_report_2026-07-06.md`