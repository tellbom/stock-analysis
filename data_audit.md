# OHLCV Data Audit

Audit time: 2026-06-25  
Store root: `models/data`  
Scope: read-only inspection of existing OHLCV, calendar, universe, and index files. No data was deleted, rewritten, or re-collected.

## Executive Summary

Final judgment: **需要局部补齐**.

The existing stock OHLCV data is largely reusable: all 300 current CSI300 universe symbols have OHLCV files, required fields are present, null rate is 0, duplicate dates are absent, and all dates are on the trading calendar. It does not need a full re-collection.

Local fixes are still needed before Phase 4 evaluation is considered complete:

- Add CSI300 index data: `silver/index_ohlcv/000300.parquet` is missing.
- Fill small trading-day gaps in 29 stock OHLCV files, totaling 276 missing trading dates inside each symbol's own stored date range.
- Be aware that `silver/ohlcv` contains one extra symbol, `600895`, which is not in the current CSI300 universe but is present in the `hs100` universe. This is not harmful if downstream code filters by universe.

## Store Layout Checked

- Stock OHLCV directory: `models/data/silver/ohlcv`
- Index OHLCV directory: `models/data/silver/index_ohlcv`
- CSI300 universe: `models/data/universe/csi300/membership.parquet`
- Trading calendar: `models/data/calendar/trading_calendar.parquet`

## Date Coverage

Stock OHLCV:

- Files found: 301
- Total rows: 249,293
- Global date range: 2023-01-03 to 2026-06-18
- Unique OHLCV dates: 836
- All 301 files end on: 2026-06-18

Start-date distribution:

| Start date | File count |
|---|---:|
| 2023-01-03 | 294 |
| 2023-01-06 | 1 |
| 2023-02-01 | 1 |
| 2023-06-09 | 1 |
| 2023-08-08 | 1 |
| 2024-12-30 | 1 |
| 2025-07-16 | 1 |
| 2025-12-03 | 1 |

Late-start files:

| Symbol | Rows | Start | End |
|---|---:|---|---|
| 688506 | 833 | 2023-01-06 | 2026-06-18 |
| 601059 | 800 | 2023-02-01 | 2026-06-18 |
| 688472 | 732 | 2023-06-09 | 2026-06-18 |
| 603296 | 692 | 2023-08-08 | 2026-06-18 |
| 001391 | 354 | 2024-12-30 | 2026-06-18 |
| 600930 | 224 | 2025-07-16 | 2026-06-18 |
| 001280 | 130 | 2025-12-03 | 2026-06-18 |

## Symbol Coverage

CSI300 universe:

- Universe rows: 300
- Current members: 300
- Current CSI300 members with OHLCV files: 300 / 300
- Missing CSI300 OHLCV symbols: none

Extra OHLCV symbols:

- `600895` exists under `silver/ohlcv` but is not in current CSI300 membership.
- `600895` is present in `models/data/universe/hs100/membership.parquet`, so this appears to be cross-universe residue rather than corrupt data.

## Field Completeness

All 301 OHLCV files share the same column set:

```text
date, open, high, low, close, volume, amount, outstanding_share, turnover, symbol
```

Required canonical fields are present in every file:

```text
symbol, date, open, high, low, close, volume
```

Null counts:

| Field | Null count | Null rate |
|---|---:|---:|
| symbol | 0 | 0.00% |
| date | 0 | 0.00% |
| open | 0 | 0.00% |
| high | 0 | 0.00% |
| low | 0 | 0.00% |
| close | 0 | 0.00% |
| volume | 0 | 0.00% |
| amount | 0 | 0.00% |
| turnover | 0 | 0.00% |

No symbol-column mismatches were found: each file's internal `symbol` value matches its filename.

## Adjustment / Price Basis

The collector implementation uses `adjust="qfq"` for both AKShare endpoints:

- `ak.stock_zh_a_daily(..., adjust="qfq")`
- `ak.stock_zh_a_hist(..., adjust="qfq")`

The schema comments also state prices are forward-adjusted by default. Therefore the intended current OHLCV basis is **qfq / forward-adjusted**.

Important caveat: the Parquet files themselves do not store an explicit adjustment metadata flag, raw-price columns, or adjustment factors. `silver/adj_factor` currently has 0 files. If strict auditability of adjustment basis is required, add per-file metadata or retain raw prices plus adjustment factors in a future ingestion pass.

## Missing Rate

Field-level missing rate:

- Required OHLCV fields: 0 missing values.
- Optional observed fields `amount` and `turnover`: 0 missing values.

Trading-day continuity missing rate:

- Expected rows within each symbol's own stored start/end range: 249,569
- Actual rows: 249,293
- Missing trading-day rows: 276
- Temporal missing rate: 0.11%

## Duplicate Dates

No duplicate dates were found:

- Duplicate `(symbol, date)` pairs: 0
- Duplicate dates within individual symbol files: 0

## Trading-Day Continuity

The trading calendar has 4,128 trading days from 2010-01-04 to 2026-12-31. For the OHLCV global range 2023-01-03 to 2026-06-18, the calendar expects 836 trading days.

Continuity results:

- Fully continuous files: 272 / 301
- Files with missing trading days: 29 / 301
- Total missing trading days: 276
- Off-calendar OHLCV dates: 0

Top missing symbols:

| Symbol | Rows | Start | End | Missing trading days |
|---|---:|---|---|---:|
| 601059 | 800 | 2023-02-01 | 2026-06-18 | 20 |
| 601995 | 816 | 2023-01-03 | 2026-06-18 | 20 |
| 601211 | 817 | 2023-01-03 | 2026-06-18 | 19 |
| 600150 | 822 | 2023-01-03 | 2026-06-18 | 14 |
| 688041 | 826 | 2023-01-03 | 2026-06-18 | 10 |
| 301269 | 826 | 2023-01-03 | 2026-06-18 | 10 |
| 600027 | 826 | 2023-01-03 | 2026-06-18 | 10 |
| 302132 | 826 | 2023-01-03 | 2026-06-18 | 10 |
| 002736 | 826 | 2023-01-03 | 2026-06-18 | 10 |
| 002049 | 826 | 2023-01-03 | 2026-06-18 | 10 |
| 600438 | 826 | 2023-01-03 | 2026-06-18 | 10 |
| 600482 | 826 | 2023-01-03 | 2026-06-18 | 10 |
| 600958 | 826 | 2023-01-03 | 2026-06-18 | 10 |
| 601088 | 826 | 2023-01-03 | 2026-06-18 | 10 |
| 601456 | 826 | 2023-01-03 | 2026-06-18 | 10 |
| 603019 | 826 | 2023-01-03 | 2026-06-18 | 10 |
| 688126 | 826 | 2023-01-03 | 2026-06-18 | 10 |
| 688521 | 826 | 2023-01-03 | 2026-06-18 | 10 |
| 000657 | 826 | 2023-01-03 | 2026-06-18 | 10 |

Examples of exact missing ranges:

- `601059`: missing 2025-11-20 through 2025-12-17 trading dates, 20 rows.
- `601995`: missing 2025-11-20 through 2025-12-17 trading dates, 20 rows.
- `000408`: missing 2025-01-10 through 2025-01-16 trading dates, 5 rows.
- `600066`: missing 2023-04-17, 1 row.

## CSI300 Index Data

No existing CSI300 index OHLCV was found:

- `models/data/silver/index_ohlcv` does not exist.
- `models/data/silver/index_ohlcv/000300.parquet` does not exist.
- `models/data/silver/ohlcv/000300.parquet` does not exist.

This blocks workflows that require `excess_vs_csi300_*` labels unless the index is collected or supplied separately.

## Final Decision

**需要局部补齐**.

Recommended action:

1. Reuse the existing stock OHLCV data as the base dataset.
2. Incrementally fill the 276 missing stock trading-day rows for the 29 affected symbols.
3. Add CSI300 index OHLCV as `silver/index_ohlcv/000300.parquet`.
4. Do not full-recollect all stock OHLCV unless later checks reveal adjustment-basis inconsistency or broader source corruption.
5. Keep filtering by the selected universe during feature/model assembly so the extra `600895` file does not enter CSI300 runs.
