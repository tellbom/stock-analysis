# Feature Coverage Gate

Generated: 2026-07-07T10:51:12

- Features checked: 69
- Base model allowed: 35
- Recent model allowed: 54
- Prediction-only: 15

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
| announcement_count_3d | announcement_events | short-history-family |
| announcement_count_5d | announcement_events | short-history-family |
| announcement_count_10d | announcement_events | short-history-family |
| has_announcement_3d | announcement_events | short-history-family |
| has_major_event_10d | announcement_events | short-history-family |
| has_risk_announcement_5d | announcement_events | short-history-family |
| has_financial_report_30d | announcement_events | short-history-family |
| has_reduction_notice_30d | announcement_events | short-history-family |
| has_dragon_tiger_5d | dragon_tiger | short-history-family |
| dragon_tiger_count_10d | dragon_tiger | short-history-family |
| dragon_tiger_net_buy_5d | dragon_tiger | short-history-family |
| dragon_tiger_net_buy_rank_5d | dragon_tiger | short-history-family |
| institution_net_buy_5d | dragon_tiger | short-history-family |
| institution_net_buy_rank_5d | dragon_tiger | short-history-family |
| block_trade_count_20d | block_trade | short-history-family |
| block_trade_amount_20d | block_trade | short-history-family |
| block_trade_amount_rank_20d | block_trade | short-history-family |
| block_trade_discount_mean_20d | block_trade | short-history-family |
| has_large_discount_block_trade_20d | block_trade | short-history-family |