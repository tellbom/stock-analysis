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
    "files": 6,
    "symbol_file_coverage": "6/300",
    "as_of_files": 0,
    "latest_min": "2026-06-25",
    "latest_max": "2026-06-25"
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
    "files": 0,
    "symbol_file_coverage": "0/300",
    "as_of_files": 0,
    "latest_min": null,
    "latest_max": null
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
- recent candidates: 57; recent admitted: 35
- fund flow entered recent model: no (none)
- fund flow forbidden from base model: yes
- base coverage report: `/Users/fuziqiang/Desktop/stock-analysis/models/data/reports/D3_base_coverage_gate_2026-07-03.csv`
- recent coverage report: `/Users/fuziqiang/Desktop/stock-analysis/models/data/reports/D3_recent_coverage_gate_2026-07-03.csv`

## Model Summary

- base_model_d3: {'train_start': '2023-01-03', 'train_end': '2026-06-29', 'prediction_date': '2026-07-03', 'x_train_rows': 250172, 'prediction_rows': 298, 'feature_count': 35}
- recent_enhanced_model_d3 status: trained
- recent_enhanced_model_d3: {'train_start': '2025-12-25', 'train_end': '2026-06-29', 'prediction_date': '2026-07-03', 'x_train_rows': 35874, 'prediction_rows': 298, 'feature_count': 35}
- base feature importance Top20: `/Users/fuziqiang/Desktop/stock-analysis/models/data/reports/D3_base_feature_importance_top20_2026-07-03.csv`
- recent feature importance Top20: `/Users/fuziqiang/Desktop/stock-analysis/models/data/reports/D3_recent_feature_importance_top20_2026-07-03.csv`

## Gate Summary

- tier counts: {'E_REJECT': 129, 'UNCLASSIFIED': 96, 'A_MAIN': 43, 'C_DOWNGRADE_OBSERVE': 13, 'B_SHORT_BOOST': 10, 'D_OBSERVE': 7}
- risk veto count: 0
- top20 intersections: {'base_recent_top20': 1, 'base_gate_top20': 12, 'recent_gate_top20': 4, 'all_three_top20': 1}
- recent boosted symbols: ['002463', '300033', '002384', '600522', '002558', '000100', '002493', '601211', '300866', '688183']
- recent downgraded symbols: ['301165', '000975', '002625', '002594', '600489', '600547', '688506', '300316', '688396', '301269', '603799', '603986', '601899']
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
| 1 | 002028 | +0.030218 |
| 2 | 000100 | +0.025402 |
| 3 | 002384 | +0.019827 |
| 4 | 300433 | +0.018578 |
| 5 | 002353 | +0.015774 |
| 6 | 603259 | +0.015635 |
| 7 | 600118 | +0.015451 |
| 8 | 600176 | +0.015265 |
| 9 | 600999 | +0.014662 |
| 10 | 688183 | +0.013045 |
| 11 | 002916 | +0.012958 |
| 12 | 300274 | +0.012940 |
| 13 | 601211 | +0.012450 |
| 14 | 688303 | +0.011787 |
| 15 | 002463 | +0.011721 |
| 16 | 300866 | +0.011337 |
| 17 | 000776 | +0.011120 |
| 18 | 600276 | +0.010830 |
| 19 | 002532 | +0.010085 |
| 20 | 603501 | +0.009409 |

## Gate Fused Top20

| Rank | Symbol | Tier | Reason |
|---:|---|---|---|
| 1 | 002422 | A_MAIN | base strong, recent confirms |
| 2 | 688012 | A_MAIN | base strong, recent confirms |
| 3 | 300476 | A_MAIN | base strong, recent confirms |
| 4 | 000807 | A_MAIN | base strong, recent confirms |
| 5 | 002532 | A_MAIN | base strong, recent confirms |
| 6 | 601600 | A_MAIN | base strong, recent confirms |
| 7 | 002371 | A_MAIN | base strong, recent confirms |
| 8 | 688082 | A_MAIN | base strong, recent confirms |
| 9 | 600188 | A_MAIN | base strong, recent confirms |
| 10 | 002460 | A_MAIN | base strong, recent confirms |
| 11 | 688521 | A_MAIN | base strong, recent confirms |
| 12 | 300450 | A_MAIN | base strong, recent confirms |
| 13 | 605499 | A_MAIN | base strong, recent confirms |
| 14 | 300661 | A_MAIN | base strong, recent confirms |
| 15 | 002353 | A_MAIN | base strong, recent confirms |
| 16 | 300502 | A_MAIN | base strong, recent confirms |
| 17 | 601066 | A_MAIN | base strong, recent confirms |
| 18 | 688303 | A_MAIN | base strong, recent confirms |
| 19 | 688223 | A_MAIN | base strong, recent confirms |
| 20 | 600999 | A_MAIN | base strong, recent confirms |

## Verification Artifacts

- `/Users/fuziqiang/Desktop/stock-analysis/models/data/reports/D3_base_ranked_2026-07-03.csv`
- `/Users/fuziqiang/Desktop/stock-analysis/models/data/reports/D3_recent_ranked_2026-07-03.csv`
- `/Users/fuziqiang/Desktop/stock-analysis/models/data/reports/D3_gate_fused_ranked_2026-07-03.csv`
- `/Users/fuziqiang/Desktop/stock-analysis/models/data/reports/D3_gate_run_report_2026-07-03.md`