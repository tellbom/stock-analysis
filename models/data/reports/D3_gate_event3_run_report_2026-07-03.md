# D3 Gate Event3 Run Report 2026-07-03

- gate_mode: `event3_enhanced_gate`
- requested_as_of_date: 2026-07-06
- actual_as_of_date: 2026-07-03
- actual date reason: requested_as_of_date 2026-07-06 has OHLCV coverage 0/300; using latest complete date 2026-07-03 with coverage 298/300
- label: ret_fwd_3d
- collector_summary: {'announcement_events': {'ok_files_or_empty': 300, 'failed': 0, 'rows_fetched': 58}, 'dragon_tiger': {'ok_files_or_empty': 300, 'failed': 0, 'rows_fetched': 222}, 'block_trade': {'ok_files_or_empty': 300, 'failed': 0, 'rows_fetched': 2718}}

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
    "files": 12,
    "symbol_file_coverage": "12/300",
    "as_of_files": 7,
    "latest_min": "2026-06-25",
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
  "announcement_events": {
    "path": "silver/announcement_events",
    "files": 300,
    "symbol_file_coverage": "300/300",
    "as_of_files": 6,
    "latest_min": "2026-04-28",
    "latest_max": "2026-07-03"
  },
  "dragon_tiger": {
    "path": "silver/dragon_tiger",
    "files": 300,
    "symbol_file_coverage": "300/300",
    "as_of_files": 8,
    "latest_min": "2026-01-06",
    "latest_max": "2026-07-03"
  },
  "block_trade": {
    "path": "silver/block_trade",
    "files": 300,
    "symbol_file_coverage": "300/300",
    "as_of_files": 15,
    "latest_min": "2026-01-08",
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
- recent_event_feature_set_id: `1da89e19`
- base candidates: 50; base admitted: 35
- recent candidates: 69; recent admitted: 54
- announcement_events entered recent model: yes
- dragon_tiger entered recent model: yes
- block_trade entered recent model: yes
- event3 features entered recent model: ['announcement_count_3d', 'announcement_count_5d', 'announcement_count_10d', 'has_announcement_3d', 'has_major_event_10d', 'has_risk_announcement_5d', 'has_financial_report_30d', 'has_reduction_notice_30d', 'has_dragon_tiger_5d', 'dragon_tiger_count_10d', 'dragon_tiger_net_buy_5d', 'dragon_tiger_net_buy_rank_5d', 'institution_net_buy_5d', 'institution_net_buy_rank_5d', 'block_trade_count_20d', 'block_trade_amount_20d', 'block_trade_amount_rank_20d', 'block_trade_discount_mean_20d', 'has_large_discount_block_trade_20d']
- event3 features entered base model: none
- margin supplement cross-validation only: yes; margin features in recent=none, base=none
- base X_train.columns: `E:\stock-analysis\models\data\reports\D3_base_X_train_columns_2026-07-03.csv`
- recent X_train.columns: `E:\stock-analysis\models\data\reports\D3_recent_event3_X_train_columns_2026-07-03.csv`
- base coverage report: `E:\stock-analysis\models\data\reports\D3_base_coverage_gate_2026-07-03.csv`
- recent coverage report: `E:\stock-analysis\models\data\reports\D3_recent_coverage_gate_2026-07-03.csv`
- event3 coverage gate: `E:\stock-analysis\models\data\reports\D3_event3_coverage_gate_2026-07-03.csv`
- event3 missing/non-zero rates: `E:\stock-analysis\models\data\reports\D3_event3_feature_missing_nonzero_2026-07-03.csv`

## Model Summary

- base_model_d3: {'train_start': '2023-01-03', 'train_end': '2026-06-29', 'prediction_date': '2026-07-03', 'x_train_rows': 250172, 'prediction_rows': 298, 'feature_count': 35}
- recent_enhanced_model_d3 status: trained
- recent_enhanced_model_d3: {'train_start': '2025-12-25', 'train_end': '2026-06-29', 'prediction_date': '2026-07-03', 'x_train_rows': 35874, 'prediction_rows': 298, 'feature_count': 54}
- base feature importance Top20: `E:\stock-analysis\models\data\reports\D3_base_feature_importance_top20_2026-07-03.csv`
- recent feature importance Top20: `E:\stock-analysis\models\data\reports\D3_recent_feature_importance_top20_2026-07-03.csv`
- event3 feature importance: `E:\stock-analysis\models\data\reports\D3_event3_feature_importance_2026-07-03.csv`

## Gate Summary

- tier counts: {'E_REJECT': 127, 'UNCLASSIFIED': 91, 'A_MAIN': 46, 'RISK_VETO': 10, 'C_DOWNGRADE_OBSERVE': 9, 'B_SHORT_BOOST': 8, 'D_OBSERVE': 7}
- risk veto count: 10
- top20 intersections: {'base_recent_top20': 1, 'base_gate_top20': 13, 'recent_gate_top20': 1, 'all_three_top20': 0}
- recent boosted symbols: ['002463', '300033', '002384', '300498', '688036', '601211', '300866', '688183']
- recent downgraded symbols: ['301165', '002371', '002625', '600489', '600547', '300316', '688396', '301269', '603986']
- event risk vetoed symbols: ['002460', '600549', '600588', '000100', '000963', '002352', '002602', '002049', '300760', '600221']

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
| 1 | 002916 | +0.030233 |
| 2 | 002028 | +0.027880 |
| 3 | 000100 | +0.026856 |
| 4 | 300502 | +0.025272 |
| 5 | 002384 | +0.023434 |
| 6 | 300274 | +0.018964 |
| 7 | 600176 | +0.018158 |
| 8 | 000776 | +0.017016 |
| 9 | 600999 | +0.016546 |
| 10 | 300033 | +0.016401 |
| 11 | 603259 | +0.016372 |
| 12 | 600026 | +0.016004 |
| 13 | 601211 | +0.015766 |
| 14 | 300442 | +0.015072 |
| 15 | 300433 | +0.015028 |
| 16 | 688183 | +0.011907 |
| 17 | 300866 | +0.011866 |
| 18 | 000338 | +0.011603 |
| 19 | 600276 | +0.011299 |
| 20 | 002460 | +0.010885 |

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
| 8 | 688082 | A_MAIN | base strong, recent confirms |
| 9 | 688256 | A_MAIN | base strong, recent confirms |
| 10 | 600188 | A_MAIN | base strong, recent confirms |
| 11 | 688521 | A_MAIN | base strong, recent confirms |
| 12 | 688506 | A_MAIN | base strong, recent confirms |
| 13 | 300450 | A_MAIN | base strong, recent confirms |
| 14 | 605499 | A_MAIN | base strong, recent confirms |
| 15 | 300661 | A_MAIN | base strong, recent confirms |
| 16 | 002353 | A_MAIN | base strong, recent confirms |
| 17 | 300502 | A_MAIN | base strong, recent confirms |
| 18 | 601066 | A_MAIN | base strong, recent confirms |
| 19 | 688303 | A_MAIN | base strong, recent confirms |
| 20 | 688223 | A_MAIN | base strong, recent confirms |

## Verification Artifacts

- `E:\stock-analysis\models\data\reports\D3_base_ranked_2026-07-03.csv`
- `E:\stock-analysis\models\data\reports\D3_recent_ranked_2026-07-03.csv`
- `E:\stock-analysis\models\data\reports\D3_gate_fused_ranked_2026-07-03.csv`
- `E:\stock-analysis\models\data\reports\D3_gate_event3_run_report_2026-07-03.md`