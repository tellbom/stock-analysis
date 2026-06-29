# P4B Enrich Merge Report

Generated for the cleaned merge path, replacing the invalid Phase 3 experiment
report.

## Summary

- The old Phase 3 model report is invalid because `cmd_model` constructed
  `FeaturePipeline` without `include_valuation` / `include_industry`; LightGBM
  therefore trained on the P4A baseline feature set only.
- `cmd_model` now accepts `--include-valuation` and `--include-industry` and
  passes both flags into `FeaturePipeline`.
- Training now prints a feature audit before LightGBM: final feature list,
  family counts, all-NaN exclusions, and fail-fast checks for requested P4B
  families.

## Reproducible Backfill

Formal entry points:

- `quant_platform.ingest.valuation_collector.ValuationCollector.backfill_history`
- `quant_platform.ingest.industry_collector.IndustryCollector.backfill_history`
- `tools/enrich/backfill_valuation.py`
- `tools/enrich/backfill_industry.py`

Both tools are parameterized by `store_root`, `universe`, `start_date`,
`end_date`, success-rate threshold, and dry-run/write mode.  The default mode is
dry-run; silver files are written only with `--write`.

## Coverage Snapshot

Current local data snapshot after the experiment:

- Valuation: 300 symbol files.  Sampled symbols (`000001`, `600519`, `601857`)
  each cover `2023-01-03` to `2026-06-18` with 836 rows and non-null
  `pe_ttm`, `pb`, `total_mcap_yi`, `float_mcap_yi`, `turnover_pct`.
- Industry SCD: `industry_map.parquet` contains 1686 SCD rows for 300 symbols
  with `effective_date`, `out_date`, and `source`.  PIT lookup remains
  `effective_date <= as_of AND (out_date IS NULL OR out_date > as_of)`.
- Source audit: 1677 rows are `cninfo_stock_industry_change`; 9 rows are
  `eastmoney_static_fallback` and are current/static supplements, not complete
  historical SCD.

## Feature Audit Expectations

Baseline command, no P4B flags:

```bash
python -m quant_platform.cli model --store-root models/data
```

Observed on a 20-symbol local audit: 32 features.

P4B command:

```bash
python -m quant_platform.cli model --store-root models/data \
  --include-valuation --include-industry
```

Observed on a 20-symbol local audit: 44 features:

- technical: 23
- cross_sectional: 8
- raw_aux: 4 (`volume`, `pe_ttm`, `pb`, `turnover_pct`)
- valuation: 6
- industry: 3

`ind_rank_main_flow` was excluded as `all-NaN` because fund_flow was not
enabled, and the audit printed the exclusion reason.

## Tests Added

- Valuation `stock_value_em` schema mapping by column name.
- CNINFO industry events to SCD out_date/source and PIT lookup.
- `cmd_model` include flags parsing and propagation to `FeaturePipeline`.
- Feature audit fail-fast and all-NaN exclusion logging.

## Merge Recommendation

Merge is reasonable after the focused tests pass and invalid experiment
artifacts are excluded from the commit.  Do not merge old Phase 3 reports,
MLflow runtime files, backup files, or `temp/` validation artifacts.
