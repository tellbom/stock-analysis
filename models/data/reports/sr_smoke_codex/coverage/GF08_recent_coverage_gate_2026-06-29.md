# Feature Coverage Gate

Generated: 2026-07-09T14:49:19

- Features checked: 57
- Base model allowed: 35
- Recent model allowed: 40
- Prediction-only: 17

## Rejections

| Feature | Family | Reason |
|---|---|---|
| fund_revenue | fundamental | base-missing>30%; base-recent-coverage-low; recent-20d-coverage-low |
| fund_net_profit | fundamental | base-missing>30%; base-recent-coverage-low; recent-20d-coverage-low |
| fund_eps | fundamental | base-missing>30%; base-recent-coverage-low; recent-20d-coverage-low |
| fund_roe | fundamental | base-missing>30%; base-recent-coverage-low; recent-20d-coverage-low |
| fund_lag_days | fundamental | base-missing>30%; base-recent-coverage-low; recent-20d-coverage-low |
| pe_ttm | raw_aux | base-recent-coverage-low; recent-20d-coverage-low |
| pb | raw_aux | base-recent-coverage-low; recent-20d-coverage-low |
| turnover_pct | raw_aux | base-recent-coverage-low; recent-20d-coverage-low |
| cs_log_float_mcap | valuation | base-recent-coverage-low; recent-20d-coverage-low |
| cs_pe_ttm_rank | valuation | base-recent-coverage-low; recent-20d-coverage-low |
| cs_pb_rank | valuation | base-recent-coverage-low; recent-20d-coverage-low |
| cs_turnover_rank | valuation | base-recent-coverage-low; recent-20d-coverage-low |
| cs_log_mcap_rank | valuation | base-recent-coverage-low; recent-20d-coverage-low |
| pe_momentum_5d | valuation | base-recent-coverage-low; recent-20d-coverage-low |
| ind_rank_turnover | industry | base-recent-coverage-low; recent-20d-coverage-low |
| cs_main_flow_rank_1d | flow | short-history-family |
| cs_main_flow_rank_5d | flow | short-history-family |
| cs_small_flow_rank_1d | flow | short-history-family |
| cs_super_flow_rank_1d | flow | short-history-family |
| cs_flow_reversal_5d | flow | short-history-family |
| cs_margin_balance_change_5d | margin | base-recent-coverage-low; recent-20d-coverage-low; latest-date-stale |
| cs_rzrq_ratio_rank | margin | base-recent-coverage-low; recent-20d-coverage-low; latest-date-stale |