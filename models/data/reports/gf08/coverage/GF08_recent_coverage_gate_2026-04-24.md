# Feature Coverage Gate

Generated: 2026-07-08T22:41:41

- Features checked: 57
- Base model allowed: 47
- Recent model allowed: 52
- Prediction-only: 5

## Rejections

| Feature | Family | Reason |
|---|---|---|
| fund_revenue | fundamental | base-missing>30%; base-recent-coverage-low; recent-20d-coverage-low |
| fund_net_profit | fundamental | base-missing>30%; base-recent-coverage-low; recent-20d-coverage-low |
| fund_eps | fundamental | base-missing>30%; base-recent-coverage-low; recent-20d-coverage-low |
| fund_roe | fundamental | base-missing>30%; base-recent-coverage-low; recent-20d-coverage-low |
| fund_lag_days | fundamental | base-missing>30%; base-recent-coverage-low; recent-20d-coverage-low |
| cs_main_flow_rank_1d | flow | short-history-family |
| cs_main_flow_rank_5d | flow | short-history-family |
| cs_small_flow_rank_1d | flow | short-history-family |
| cs_super_flow_rank_1d | flow | short-history-family |
| cs_flow_reversal_5d | flow | short-history-family |