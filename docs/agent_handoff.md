# Agent Handoff

Date: 2026-06-21

## Current State

- HS100 full run completed once under `models/data`.
- OHLCV completed: 100/100 symbols.
- Fundamentals completed: 63/100 symbols in the full run; 37 symbols were missing fundamentals.
- Features completed: 100/100 symbols, latest full-run feature set was `8acd93aa`.
- Labels completed: 100/100 symbols.
- Original quality report failed because `yjyg_em` forecast rows can have `announce_date < period_end`.

## Code Changes Already Made

- `quant_platform/store/quality_report.py`
  - `yjyg_em` forecast rows are allowed to have `announce_date < period_end`.
  - Formal/non-forecast rows still fail if `announce_date < period_end`.
  - Invalid or empty PIT dates still fail.

- `quant_platform/features/fundamental.py`
  - Formal `fund_*` features only use rows containing formal metric values: `revenue`, `net_profit`, `eps`, or `roe`.
  - Forecast-only `yjyg_em` rows no longer overwrite the latest formal metrics.

- `models/data/logs/hs100_batch_runner.py`
  - Resume mode: existing OHLCV/fundamentals parquet files are skipped.
  - Missing fundamentals are the only rows that trigger fresh network requests.
  - Quality report now checks all usable symbols, not only the first 10.

## Verification Already Run

- Targeted tests:
  - `tests/test_quality_report.py`
  - `tests/test_p1.py`
  - `tests/test_fundamentals_collector.py`
  - Result: 60 passed.

- Full regression:
  - Command: `.venv/bin/python -m pytest tests -q`
  - Result: 164 passed, 42 warnings.

- Local full quality check after code fix:
  - `has_errors=False`
  - `fundamentals_forecast_pre_period_rows=8`

## Stopped Resume Run

- A resume run was started and then stopped at user request.
- It was stopped with normal `TERM`.
- No data cleanup was performed.
- Logs and stale pid files under `models/data/logs` were later deleted by user request.
- The scripts `hs100_batch_runner.py` and `launch_hs100_daemon.py` were intentionally kept.

## Next Recommended Work

1. Re-run HS100 resume collection only when the user asks.
   - Use `models/data/logs/launch_hs100_daemon.py`.
   - The runner will skip existing OHLCV/fundamentals files and request only missing fundamentals.

2. After resume finishes, inspect:
   - `models/data/quality_report.txt`
   - final `[DONE]` log line
   - count of `models/data/silver/fundamentals/*.parquet`

3. If quality has no errors, proceed to P2:
   - build panel from features and labels
   - run purged CV / OOF predictions
   - run metrics, baselines, backtest, robustness, alpha verdict

4. Only after P2 alpha verdict is acceptable, proceed to P3:
   - HPO
   - model zoo comparison
   - leaderboard / research ledger
   - explainability
   - registry / promotion

## Important Constraints

- Do not add fallback fundamentals values.
- Do not fabricate or estimate `announce_date`.
- Keep `announce_date` as the PIT as-of join key.
- `yjyg_em` is forecast data, not formal financial statement data.
- Formal metrics must not be overwritten by forecast-only rows.
