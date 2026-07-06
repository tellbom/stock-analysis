# FundFlowProvider Routing Implementation Report

Generated: 2026-07-06T23:41:29

## Implemented Provider Route

Priority order: `adata -> qstock -> native_eastmoney -> akshare`.

Provider modules:

- `quant_platform/ingest/fund_flow_providers.py`
- `ADataFundFlowProvider`: optional dependency `adata`; documented API shape `adata.stock.market.get_capital_flow`; same-source risk noted because public docs state Eastmoney capital-flow source.
- `QStockFundFlowProvider`: optional dependency `qstock`; qstock aggregates public web sources including Eastmoney/THS/Sina; same-source risk noted.
- `NativeEastmoneyFundFlowProvider`: existing native Eastmoney push2his implementation.
- `AKShareFundFlowProvider`: AKShare `stock_individual_fund_flow`; same Eastmoney push2his host in current installed implementation.

## Canonical Schema

Providers return:

`symbol, trade_date, main_net, main_net_rate, super_net, super_net_rate, large_net, large_net_rate, medium_net, medium_net_rate, small_net, small_net_rate, source, raw_update_time, fetched_at`

Fields that a provider cannot supply are left missing/NA and surfaced in provider result/reporting. They are not fabricated.

## Collector Changes

- `FundFlowCollector` now tries providers in priority order per symbol.
- Provider failures are recorded with `symbol, provider, error_type, error_message, retry_count, latest_success_provider`.
- All-provider failure no longer returns fake success/0 silently.
- Silver write keeps both `trade_date` and legacy `date` alias for downstream compatibility.

## Sector/Concept Proxy Flow

- Added `quant_platform/ingest/sector_fund_flow_collector.py`.
- Added silver paths `silver/sector_fund_flow/` and `silver/concept_fund_flow/`.
- Added `quant_platform/features/sector_flow.py` with proxy features:
  - `sector_main_flow_rank`
  - `sector_flow_momentum_3d`
  - `sector_flow_momentum_5d`
  - `stock_industry_flow_strength`
- These features are family `sector_flow`, not stock-level `flow`, and are not written as `main_net`.

## Smoke Reports

- `models/data/reports/fund_flow_provider_smoke_report.md`
- `models/data/reports/sector_flow_provider_smoke_report.md`

Current smoke status in this environment:

- `adata`: dependency missing.
- `qstock`: dependency missing.
- `native_eastmoney`: SSL record layer failures to `push2his.eastmoney.com`.
- `akshare`: same Eastmoney host, SSL record layer failures.
- sector/concept AKShare endpoints: Eastmoney host / 502 / JSON decode failures during smoke.

## Coverage Gate Position

No coverage gate bypass was added. True stock-level `fund_flow` still must satisfy:

- `recent_symbol_coverage >= 250/300`
- `recent_20d_avg_symbol_coverage >= 250/300`
- `available_trading_days >= 80`
- latest available date no later than T-1/T-2

`sector_flow` is a separate short-history proxy family and can only enter recent model if its own coverage passes. It must be disclosed as proxy flow, not main-force stock flow.

## Optional Dependencies

Added optional notes to `requirements.txt`:

- `adata>=2.8.0`
- `qstock>=1.3.7`