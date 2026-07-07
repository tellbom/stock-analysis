# D3 Gate-First Run Report 2026-07-03

- requested_as_of_date: 2026-07-06
- actual_as_of_date: 2026-07-03
- actual date reason: requested_as_of_date 2026-07-06 has OHLCV coverage 0/300; using latest complete date 2026-07-03 with coverage 298/300
- label: ret_fwd_3d

## Data Coverage Summary

```json
{
  "ohlcv": {
    "requested_coverage": "0/300",
    "actual_coverage": "298/300"
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
    "as_of_files": 298,
    "latest_min": "2026-06-18",
    "latest_max": "2026-07-03"
  },
  "fund_flow": {
    "path": "silver/fund_flow",
    "files": 300,
    "symbol_file_coverage": "300/300",
    "as_of_files": 299,
    "latest_min": "2026-06-26",
    "latest_max": "2026-07-03"
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
    "as_of_files": 6,
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
- fund_flow coverage gate result: {'path': 'E:\\stock-analysis\\models\\data\\reports\\fund_flow_emdatah5_coverage_gate.csv', 'covered_symbols': 300, 'latest_available_date': '2026-07-06', 'recent_symbol_coverage': 299, 'recent_20d_avg_symbol_coverage': 292.1, 'available_trading_days': 130, 'field_missing_rate': 0.05555555555555555, 'is_allowed_for_recent_model': True, 'rejection_reason': 'allowed'}
- base final feature list: `E:\stock-analysis\models\data\reports\D3_base_X_train_columns_2026-07-03.csv`
- recent final feature list: `E:\stock-analysis\models\data\reports\D3_recent_emdatah5_X_train_columns_2026-07-03.csv`
- fund flow forbidden from base model: yes
- base coverage report: `E:\stock-analysis\models\data\reports\D3_base_coverage_gate_2026-07-03.csv`
- recent coverage report: `E:\stock-analysis\models\data\reports\D3_recent_coverage_gate_2026-07-03.csv`

## Model Summary

- base_model_d3: {'train_start': '2023-01-03', 'train_end': '2026-06-29', 'prediction_date': '2026-07-03', 'x_train_rows': 250172, 'prediction_rows': 298, 'feature_count': 35}
- recent_enhanced_model_d3 status: trained
- recent_enhanced_model_d3: {'train_start': '2025-12-25', 'train_end': '2026-06-29', 'prediction_date': '2026-07-03', 'x_train_rows': 35874, 'prediction_rows': 298, 'feature_count': 40}
- base feature importance Top20: `E:\stock-analysis\models\data\reports\D3_base_feature_importance_top20_2026-07-03.csv`
- fund_flow feature importance: `E:\stock-analysis\models\data\reports\D3_recent_fund_flow_feature_importance_2026-07-03.csv`
- recent feature importance Top20: `E:\stock-analysis\models\data\reports\D3_recent_feature_importance_top20_2026-07-03.csv`

## Gate Summary

- tier counts: {'E_REJECT': 140, 'UNCLASSIFIED': 85, 'A_MAIN': 49, 'C_DOWNGRADE_OBSERVE': 9, 'B_SHORT_BOOST': 8, 'D_OBSERVE': 7}
- risk veto count: 0
- top20 intersections: {'base_recent_top20': 3, 'base_gate_top20': 16, 'recent_gate_top20': 5, 'all_three_top20': 3}
- recent boosted symbols: ['002463', '300033', '002384', '000100', '300803', '601211', '300866', '688183']
- recent downgraded symbols: ['301165', '002625', '002594', '600547', '300316', '688396', '301269', '002179', '601899']
- event risk vetoed symbols: []

## Base Top20

| Rank | Symbol | Score |
|---:|---|---:|
| 1 | 301165 | +0.034196 |
| 2 | 002422 | +0.020770 |
| 3 | 000975 | +0.018621 |
| 4 | 688012 | +0.016622 |
| 5 | 300476 | +0.016615 |
| 6 | 000807 | +0.015003 |
| 7 | 002532 | +0.014943 |
| 8 | 601600 | +0.014728 |
| 9 | 002371 | +0.014355 |
| 10 | 002625 | +0.013740 |
| 11 | 688082 | +0.013658 |
| 12 | 688256 | +0.013361 |
| 13 | 600188 | +0.013231 |
| 14 | 002460 | +0.012510 |
| 15 | 688521 | +0.011964 |
| 16 | 002594 | +0.011414 |
| 17 | 600489 | +0.011374 |
| 18 | 600547 | +0.011237 |
| 19 | 688506 | +0.011220 |
| 20 | 300450 | +0.010691 |

## Recent Top20

| Rank | Symbol | Score |
|---:|---|---:|
| 1 | 000100 | +0.030757 |
| 2 | 002028 | +0.030487 |
| 3 | 300661 | +0.023973 |
| 4 | 603259 | +0.023725 |
| 5 | 688506 | +0.023624 |
| 6 | 002916 | +0.022616 |
| 7 | 600118 | +0.018191 |
| 8 | 300033 | +0.016034 |
| 9 | 600176 | +0.014604 |
| 10 | 300274 | +0.014434 |
| 11 | 002384 | +0.013359 |
| 12 | 002353 | +0.013234 |
| 13 | 600999 | +0.012277 |
| 14 | 002920 | +0.012174 |
| 15 | 600276 | +0.011972 |
| 16 | 300433 | +0.011101 |
| 17 | 002532 | +0.009947 |
| 18 | 300450 | +0.009371 |
| 19 | 300866 | +0.009309 |
| 20 | 300408 | +0.009146 |

## Gate Fused Top20

| Rank | Symbol | Tier | Reason |
|---:|---|---|---|
| 1 | 002422 | A_MAIN | base strong, recent confirms |
| 2 | 000975 | A_MAIN | base strong, recent confirms |
| 3 | 688012 | A_MAIN | base strong, recent confirms |
| 4 | 300476 | A_MAIN | base strong, recent confirms |
| 5 | 000807 | A_MAIN | base strong, recent confirms |
| 6 | 002532 | A_MAIN | base strong, recent confirms |
| 7 | 601600 | A_MAIN | base strong, recent confirms |
| 8 | 002371 | A_MAIN | base strong, recent confirms |
| 9 | 688082 | A_MAIN | base strong, recent confirms |
| 10 | 688256 | A_MAIN | base strong, recent confirms |
| 11 | 600188 | A_MAIN | base strong, recent confirms |
| 12 | 002460 | A_MAIN | base strong, recent confirms |
| 13 | 688521 | A_MAIN | base strong, recent confirms |
| 14 | 600489 | A_MAIN | base strong, recent confirms |
| 15 | 688506 | A_MAIN | base strong, recent confirms |
| 16 | 300450 | A_MAIN | base strong, recent confirms |
| 17 | 605499 | A_MAIN | base strong, recent confirms |
| 18 | 300661 | A_MAIN | base strong, recent confirms |
| 19 | 002353 | A_MAIN | base strong, recent confirms |
| 20 | 300502 | A_MAIN | base strong, recent confirms |

## Verification Artifacts

- `E:\stock-analysis\models\data\reports\D3_base_ranked_2026-07-03.csv`
- `E:\stock-analysis\models\data\reports\D3_recent_ranked_2026-07-03.csv`
- `E:\stock-analysis\models\data\reports\D3_gate_fused_ranked_2026-07-03.csv`
- `E:\stock-analysis\models\data\reports\D3_gate_emdatah5_fund_flow_run_report_2026-07-03.md`