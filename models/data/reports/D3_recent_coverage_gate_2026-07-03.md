# Feature Coverage Gate

Generated: 2026-07-06T22:32:36

- Features checked: 57
- Base model allowed: 35
- Recent model allowed: 35
- Prediction-only: 22

## Rejections

| Feature | Family | Reason |
|---|---|---|
| fund_revenue | fundamental | base-missing>30%; base-recent-symbol-coverage-low; recent-symbol-coverage-low; recent-20d-coverage-low |
| fund_net_profit | fundamental | base-missing>30%; base-recent-symbol-coverage-low; recent-symbol-coverage-low; recent-20d-coverage-low |
| fund_eps | fundamental | base-missing>30%; base-recent-symbol-coverage-low; recent-symbol-coverage-low; recent-20d-coverage-low |
| fund_roe | fundamental | base-missing>30%; base-recent-symbol-coverage-low; recent-symbol-coverage-low; recent-20d-coverage-low |
| fund_lag_days | fundamental | base-missing>30%; base-recent-symbol-coverage-low; recent-symbol-coverage-low; recent-20d-coverage-low |
| pe_ttm | raw_aux | base-recent-symbol-coverage-low; recent-symbol-coverage-low; recent-20d-coverage-low; latest-date-stale |
| pb | raw_aux | base-recent-symbol-coverage-low; recent-symbol-coverage-low; recent-20d-coverage-low; latest-date-stale |
| turnover_pct | raw_aux | base-recent-symbol-coverage-low; recent-symbol-coverage-low; recent-20d-coverage-low; latest-date-stale |
| cs_log_float_mcap | valuation | base-recent-symbol-coverage-low; recent-symbol-coverage-low; recent-20d-coverage-low; latest-date-stale |
| cs_pe_ttm_rank | valuation | base-recent-symbol-coverage-low; recent-symbol-coverage-low; recent-20d-coverage-low; latest-date-stale |
| cs_pb_rank | valuation | base-recent-symbol-coverage-low; recent-symbol-coverage-low; recent-20d-coverage-low; latest-date-stale |
| cs_turnover_rank | valuation | base-recent-symbol-coverage-low; recent-symbol-coverage-low; recent-20d-coverage-low; latest-date-stale |
| cs_log_mcap_rank | valuation | base-recent-symbol-coverage-low; recent-symbol-coverage-low; recent-20d-coverage-low; latest-date-stale |
| pe_momentum_5d | valuation | base-recent-symbol-coverage-low; recent-symbol-coverage-low; recent-20d-coverage-low; latest-date-stale |
| ind_rank_turnover | industry | base-recent-symbol-coverage-low; recent-symbol-coverage-low; recent-20d-coverage-low; latest-date-stale |
| cs_main_flow_rank_1d | flow | short-history-family; base-recent-symbol-coverage-low; recent-symbol-coverage-low; recent-20d-coverage-low; latest-date-stale |
| cs_main_flow_rank_5d | flow | short-history-family; base-recent-symbol-coverage-low; recent-symbol-coverage-low; recent-20d-coverage-low; latest-date-stale |
| cs_small_flow_rank_1d | flow | short-history-family; base-recent-symbol-coverage-low; recent-symbol-coverage-low; recent-20d-coverage-low; latest-date-stale |
| cs_super_flow_rank_1d | flow | short-history-family; base-recent-symbol-coverage-low; recent-symbol-coverage-low; recent-20d-coverage-low; latest-date-stale |
| cs_flow_reversal_5d | flow | short-history-family; base-recent-symbol-coverage-low; recent-symbol-coverage-low; recent-20d-coverage-low; latest-date-stale |
| cs_margin_balance_change_5d | margin | base-recent-symbol-coverage-low; recent-symbol-coverage-low; recent-20d-coverage-low; latest-date-stale |
| cs_rzrq_ratio_rank | margin | base-recent-symbol-coverage-low; recent-symbol-coverage-low; recent-20d-coverage-low; latest-date-stale |