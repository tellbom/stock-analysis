# Feature Coverage Gate

Generated: 2026-07-09T00:33:44

- Features checked: 57
- Base model allowed: 46
- Recent model allowed: 42
- Prediction-only: 6

## Rejections

| Feature | Family | Reason |
|---|---|---|
| fund_revenue | fundamental | base-missing>30%; base-recent-coverage-low; recent-20d-coverage-low |
| fund_net_profit | fundamental | base-missing>30%; base-recent-coverage-low; recent-20d-coverage-low |
| fund_eps | fundamental | base-missing>30%; base-recent-coverage-low; recent-20d-coverage-low |
| fund_roe | fundamental | base-missing>30%; base-recent-coverage-low; recent-20d-coverage-low |
| fund_lag_days | fundamental | base-missing>30%; base-recent-coverage-low; recent-20d-coverage-low |
| pe_ttm | raw_aux | latest-date-stale |
| pb | raw_aux | latest-date-stale |
| turnover_pct | raw_aux | latest-date-stale |
| cs_log_float_mcap | valuation | latest-date-stale |
| cs_pe_ttm_rank | valuation | base-recent-coverage-low; recent-20d-coverage-low; latest-date-stale |
| cs_pb_rank | valuation | latest-date-stale |
| cs_turnover_rank | valuation | latest-date-stale |
| cs_log_mcap_rank | valuation | latest-date-stale |
| pe_momentum_5d | valuation | latest-date-stale |
| ind_rank_turnover | industry | latest-date-stale |
| cs_main_flow_rank_1d | flow | short-history-family |
| cs_main_flow_rank_5d | flow | short-history-family |
| cs_small_flow_rank_1d | flow | short-history-family |
| cs_super_flow_rank_1d | flow | short-history-family |
| cs_flow_reversal_5d | flow | short-history-family |