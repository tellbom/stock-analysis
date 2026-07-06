# Fund Flow Staleness Debug Report

Generated: 2026-07-06T23:24:03

## 1. Current Silver Coverage

- fund_flow parquet files: **10**
- covered symbols: **10 / 300**
- global latest fund_flow date after debug backfill attempt: **2026-07-06**
- latest date before this debug attempt was **2026-06-25** across the original 6 parquet files.
- original stale symbols still maxing at 2026-06-25: `000001, 000002, 000063, 000100, 300498, 300502`
- successful symbols in this debug backfill attempt: `002236, 002241, 002304, 688041`

Artifacts:

- `models/data/reports/fund_flow_symbol_date_summary.csv`
- `models/data/reports/fund_flow_date_distribution.csv`
- `models/data/reports/fund_flow_success_symbols.csv`
- `models/data/reports/fund_flow_failed_symbols.csv`

Recent date distribution tail:

| trade_date | symbol_count |
|---|---:|
| 2026-05-25 | 10 |
| 2026-05-26 | 10 |
| 2026-05-27 | 10 |
| 2026-05-28 | 10 |
| 2026-05-29 | 10 |
| 2026-06-01 | 10 |
| 2026-06-02 | 10 |
| 2026-06-03 | 10 |
| 2026-06-04 | 10 |
| 2026-06-05 | 10 |
| 2026-06-08 | 10 |
| 2026-06-09 | 10 |
| 2026-06-10 | 10 |
| 2026-06-11 | 10 |
| 2026-06-12 | 10 |
| 2026-06-15 | 10 |
| 2026-06-16 | 10 |
| 2026-06-17 | 10 |
| 2026-06-18 | 10 |
| 2026-06-22 | 10 |
| 2026-06-23 | 10 |
| 2026-06-24 | 10 |
| 2026-06-25 | 10 |
| 2026-06-26 | 4 |
| 2026-06-29 | 4 |
| 2026-06-30 | 4 |
| 2026-07-01 | 4 |
| 2026-07-02 | 4 |
| 2026-07-03 | 4 |
| 2026-07-06 | 4 |

## 2. Direct Cause of 2026-06-25 Staleness

The direct cause is **collector/backfill incompleteness plus silent failure handling**, not a data source date ceiling:

- Before this debug pass, `models/data/silver/fund_flow` physically contained only 6 parquet files: `000001`, `000002`, `000063`, `000100`, `300498`, `300502`.
- All six had 120 rows and stopped at `2026-06-25`.
- A direct Eastmoney `push2his` probe earlier in this session returned data up to `2026-07-06` for `600000`, `000001`, and `300750`.
- Therefore the source is capable of returning dates after 2026-06-25.
- The collector previously returned `0` for empty/error responses and only logged low-level failures at debug level inside `_fetch_push2his`, so failed symbols could look like harmless no-new-data skips.

## 3. Data Source vs Collector vs Write Path

- Data source only returns to 2026-06-25: **No**. Direct Eastmoney tests returned `2026-07-06` for several symbols.
- Collector did not continue collecting all CSI300 symbols historically: **Yes**. Only 6 parquet files existed before this debug pass.
- Current collection fails for most symbols: **Yes**. The fast full-universe diagnostic succeeded for 4 symbols and failed for 296.
- Write path failure: **Not the primary cause**. The 4 successful symbols were written/upserted correctly with `source`, `raw_update_time`, and `fetched_at` columns.

## 4. Single-Symbol Interface Tests

Direct `_fetch_push2his` results observed before rate/SSL degradation:

| Symbol | Market class | Rows | Max date | Result |
|---|---|---:|---|---|
| 600000 | Shanghai main board | 120 | 2026-07-06 | success |
| 000001 | Shenzhen main board | 120 | 2026-07-06 | success |
| 300750 | ChiNext | 120 | 2026-07-06 | success |
| 688981 | STAR / 科创板 | 0 | n/a | empty response in that probe |

AKShare `stock_individual_fund_flow(stock, market)` tests for `600000/sh`, `000001/sz`, `300750/sz`, `688981/sh` all failed in this environment with `SSLError: [SSL] record layer failure`. The project-native `requests` implementation succeeded briefly for some symbols, then also degraded into SSL/timeout/proxy errors.

## 5. Failed Symbols

- failed symbols CSV: `models/data/reports/fund_flow_failed_symbols.csv`
- failure count: **296**
- error type counts: `{"SSLError": 205, "ReadTimeout": 70, "ProxyError": 21}`

Top failure modes are SSL record layer failures, read timeouts, and proxy remote-disconnect errors from `push2his.eastmoney.com`.

## 6. Sample / Limit Mode Check

- No `--limit`, `--sample`, `--max-symbols`, `dry_run`, or `symbols[:6]` restriction was found in `FundFlowCollector.run` or the CLI enrich path.
- The six-file state was not caused by an explicit sample flag in the active collector code.

## 7. Symbol to Market Mapping Check

Mapping has been made explicit in `quant_platform/ingest/flow_collector.py`:

- `600/601/603/605/688` -> Eastmoney market `1` / Shanghai
- `000/001/002/003/300/301` -> Eastmoney market `0` / Shenzhen
- `43/83/87/88/92` -> Eastmoney market `2` / Beijing

No evidence was found that the main failure is caused by incorrect market mapping. The successful 600/000/300 probes used the same mapping.

## 8. Silent Exception Handling

Yes, the old code could silently hide the problem:

- `_fetch_push2his` caught exceptions, logged at debug level, and returned an empty DataFrame.
- `_collect_one` returned 0 on empty DataFrame.
- `run` treated 0 as no new rows rather than a failed collection.

This has been fixed so empty/error fetches raise and are written to `fund_flow_failed_symbols.csv`.

## 9. Fixes Applied

- Added `_normalise_symbol` for `600000.SH`, `sh600000`, and 6-digit symbols.
- Added explicit `_market_code` mapping for SH/SZ/BJ prefixes.
- Added retry-aware `_fetch_push2his_result` with structured error messages for HTTP, JSON, empty klines, SSL, timeout, and parse failures.
- Changed collector behavior so empty/error responses are failures, not silent 0-row skips.
- Added `fund_flow_failed_symbols.csv` report with `symbol`, `error_type`, `error_message`, `retry_count`.
- Added `source`, `raw_update_time`, and `fetched_at` metadata to newly written fund_flow rows.

## 10. Backfill Attempt After Fix

- Existing fund_flow data was backed up to `models/data/backup_fund_flow_before_debug_20260706_225808` before writes.
- A normal CLI-only fund_flow enrich was started and immediately showed SSL/timeout failures. It was interrupted to avoid a multi-hour doomed run.
- A fast full-universe diagnostic/upsert pass was then run with short timeout and no retries.
- Result: **4 success**, **296 failed**.
- Latest fund_flow date after successful upserts: **2026-07-06**.
- Coverage on actual D3 Gate as-of date `2026-07-03`: **4/300**.
- Recent 20 trading-date average coverage through `2026-07-03`: **8.20/300**.
- Recent model coverage gate requirement is approximately `recent_symbol_coverage >= 250/300`, `recent_20d_avg_symbol_coverage >= 250/300`, `available_trading_days >= 80`, and latest not stale.
- Gate status: **FAIL** (`4/300` latest coverage and `8.20/300` recent-20d average).

## 11. Repair Recommendations

1. Keep the new failure reporting in place; do not revert to silent 0-row skips.
2. Retry the full fund_flow backfill from a stable network or after Eastmoney SSL/proxy failures clear.
3. Consider adding an AKShare fallback only if its SSL failures are solved in this environment; current AKShare calls hit the same host and failed similarly.
4. Add a nightly post-collector assertion that fails the job when latest-date coverage is below threshold, so this cannot silently reach model training.
5. Do not admit fund_flow into recent model until coverage gate passes; current coverage remains far below threshold.