# Feature Coverage Gate

Generated: 2026-07-08T22:46:02

- Features checked: 52
- Base model allowed: 46
- Recent model allowed: 37
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