# Quant Platform — Phase 4 Task Breakdown

*Companion to `plan.md`. Tasks are ordered by execution sequence within each phase. Dependencies are stated explicitly. No implementation begins until this document is reviewed.*

---

## Task ID Convention

`P4{A/B/C}-{nn}` where A = Fix Instrument, B = Data Enrichment, C = Factor Diagnostics.
Tasks marked `[PREREQUISITE]` must be completed before any downstream task in their phase is started.

---

## Phase 4A — Fix the Instrument

---

### P4A-01: Fix robustness embargo inconsistency

| Field | Detail |
|---|---|
| **Task ID** | P4A-01 |
| **Task Name** | Robustness Suite Embargo Fix |
| **Priority** | P0 — Must be first; all ablation results are currently misleading |
| **Implementation Order** | 1 |

**Objective**

The `run_robustness_tests()` function in `evaluation/robustness.py` defaults `embargo=5` while the main pipeline uses `embargo=horizon` (defaulting to 20 for a 20-day label). Ablation re-fits inside the robustness suite therefore compare an embargo-20 baseline against embargo-5 variants. The embargo gap difference directly affects how much label autocorrelation bleeds across fold boundaries, making ablation deltas apples-to-oranges and slightly overstating the ablation models' apparent IC.

Fix: change the `embargo` default argument in `run_robustness_tests()` from `embargo: int = 5` to `embargo: int | None = None`, with `None` resolving to `horizon` inside the function body — matching the `PurgedKFold` default logic.

**Related Modules / Files**
- `evaluation/robustness.py` — `run_robustness_tests()`, line ~96
- `training/splitter.py` — `PurgedKFold` (reference for the correct default)
- Any call sites that invoke `run_robustness_tests` with the old default

**Dependencies**
- None (standalone fix)

**Expected Deliverables**
- Modified `run_robustness_tests` signature with corrected default
- Updated docstring clarifying the embargo=horizon default
- Re-run of the existing ablation suite to produce corrected deltas
- Brief comparison table: old ablation deltas vs corrected ablation deltas

**Validation Criteria**
- `PurgedKFold(horizon=h).embargo == h` and `run_robustness_tests(..., horizon=h).embargo == h` (no explicit override needed)
- Unit test: `run_robustness_tests` called with `horizon=20` and no explicit `embargo` uses `embargo=20` internally
- Ablation deltas are non-trivially different from the old ones (confirms the bug was real)
- Label-shuffle null test still passes (Rank IC collapses to ~0), canary still inflates IC — the fix does not break the robustness machinery

---

### P4A-02: Surface subperiod stability report

| Field | Detail |
|---|---|
| **Task ID** | P4A-02 |
| **Task Name** | Subperiod Stability Report |
| **Priority** | P0 — Free information already computed; must be read before any modelling decisions |
| **Implementation Order** | 2 |

**Objective**

`RobustnessReport` already stores `first_half_ric` and `second_half_ric`. These numbers are the most direct indicator of regime stability for the in-period signal, but they are not surfaced in the alpha verdict synthesizer or in any persistent report. A stable signal (both halves share sign and rough magnitude) implies the lockbox result is likely noise; divergent signs imply regime sensitivity and explain the OOF/lockbox gap.

Fix: (a) ensure the alpha verdict synthesizer reads and interprets the subperiod stability numbers; (b) add a `subperiod_ic_ratio` derived metric (min(|first|, |second|) / max(|first|, |second|)) as a stability index; (c) log a clear interpretation statement ("signal is regime-stable" / "signal shows regime sensitivity — further investigation required") that propagates into the alpha verdict.

**Related Modules / Files**
- `evaluation/robustness.py` — `RobustnessReport`, `run_robustness_tests`
- `evaluation/alpha_verdict.py` — alpha verdict synthesizer
- `evaluation/research_ledger.py` — should persist the stability index

**Dependencies**
- P4A-01 (so the ablation in the robustness suite is consistent before reading results)

**Expected Deliverables**
- `subperiod_ic_ratio` field added to `RobustnessReport`
- Alpha verdict synthesizer consumes and interprets this field
- Research ledger entry includes stability index
- Written interpretation of the current `first_half_ric` / `second_half_ric` values

**Validation Criteria**
- `subperiod_ic_ratio` is 1.0 when both halves are identical, 0.0 when one is zero
- Alpha verdict output contains a readable stability statement
- If `first_half_ric` and `second_half_ric` have opposite signs, the verdict flags "regime-sensitive signal"

---

### P4A-03: Walk-forward / rolling OOS evaluator

| Field | Detail |
|---|---|
| **Task ID** | P4A-03 |
| **Task Name** | Walk-Forward OOS Evaluator |
| **Priority** | P1 — Blocks all downstream evaluation |
| **Implementation Order** | 3 |

**Objective**

Replace the single static 12-month lockbox with a walk-forward evaluator that produces sequential, non-overlapping out-of-sample windows. At a 5d label horizon (see P4A-04), five windows of 12 months each produce ~250 independent 5d periods — enough to detect a 0.04 forward Rank IC with reasonable statistical power.

Design requirements:
- `WalkForwardEvaluator` class in a new `evaluation/walk_forward.py` module
- Configurable: `n_windows` (default 5), `window_months` (default 12), `step_months` (default 12 for non-overlapping), `horizon` (matches label horizon)
- Each window: train on all data preceding the window minus a purge gap, predict on the window, yield `(window_id, pred_series, label_series, dates)`
- Aggregate output: per-window Rank IC, IC decay curve (IC at 1d, 2d, 3d, 5d, 10d lags), IC sign stability across windows, forward Rank ICIR computed across all windows
- The old `make_lockbox_split` is **deprecated** but not deleted (kept for backward compatibility with existing tests)
- MLflow logging: each window is a child run; aggregate walk-forward metrics are the parent run

**Related Modules / Files**
- New: `evaluation/walk_forward.py`
- `training/splitter.py` — reference for purge/embargo logic (do not duplicate; call `PurgedKFold` logic)
- `evaluation/metrics.py` — reuse `evaluate()` for per-window metrics
- `evaluation/backtest.py` — apply `run_backtest()` to concatenated OOS predictions
- `evaluation/alpha_verdict.py` — update to consume walk-forward results
- `training/tracking.py` — MLflow child run structure

**Dependencies**
- P4A-01 (embargo fix must be in place)

**Expected Deliverables**
- `WalkForwardEvaluator` class with `fit_predict(panel, feature_cols, label_col, model)` interface
- Per-window metrics table (Rank IC, ICIR, Sharpe, n_periods)
- IC decay curve plot data (IC vs lag, averaged across windows)
- Window-level IC series stored in MLflow artifacts
- `make_lockbox_split` marked deprecated in docstring

**Validation Criteria**
- With a synthetic panel of known forward signal, the evaluator's aggregate Rank IC is within 20% of the true signal
- With a shuffled-label panel (null), aggregate Rank IC < 0.01 and ICIR < 0.3
- The number of independent forward periods equals `n_windows × window_months × 252/12 / horizon`
- Each window's train set does not overlap with its test set (verified by date-range assertion)
- Unit test: `n_windows=3, window_months=6, horizon=5` produces exactly 3 folds with correct date boundaries
- MLflow run structure: parent run contains aggregate metrics; child runs contain per-window metrics

---

### P4A-04: Multi-horizon label extension (5d, 10d)

| Field | Detail |
|---|---|
| **Task ID** | P4A-04 |
| **Task Name** | Multi-Horizon Label Extension |
| **Priority** | P1 — Near-zero cost; directly increases evaluation power |
| **Implementation Order** | 4 |

**Objective**

The label builder already supports `horizons` as a parameter. The default `DEFAULT_HORIZONS = [1, 5, 20]` already includes 5d, but the existing pipeline may not be consistently running with all three horizons populated and verified. This task ensures: (a) all horizons `[1, 5, 10, 20]` are built and stored; (b) the leakage harness verifies all horizons; (c) the primary evaluation horizon is changed from 20d to 5d throughout the platform; (d) downstream modules (feature pipeline, model training, backtest, alpha verdict) are updated to operate on the 5d label by default while retaining 20d as a secondary label.

**Related Modules / Files**
- `labels/builder.py` — `DEFAULT_HORIZONS` constant; `build_labels()`
- `labels/leakage_harness.py` — run harness on all horizons
- `features/pipeline.py` — ensure feature panel is joined to all label horizons
- `evaluation/metrics.py`, `evaluation/backtest.py` — `label_col` default parameter
- `evaluation/alpha_verdict.py` — primary label constant
- `training/lgbm_model.py`, `training/model_zoo.py` — `fit_oof()` default label

**Dependencies**
- P4A-01, P4A-02 (embargo/stability fix)

**Expected Deliverables**
- `DEFAULT_HORIZONS = [1, 5, 10, 20]` updated and documented
- Label Parquet files for all four horizons for the full universe
- Leakage harness passing for all four horizons
- Platform-wide default label changed to `ret_fwd_5d`
- `CLAUDE.md` or equivalent documentation note: "primary evaluation horizon is 5d; 20d is secondary"
- A comparison table: OOF Rank IC for the same feature set at 1d, 5d, 10d, 20d horizons (expected: monotonically decreasing; any exception is noteworthy)

**Validation Criteria**
- `build_labels(horizons=[1,5,10,20])` produces correctly shaped Parquet with no NaN except at series-end embargo zone
- Canary test (inject `close.shift(-1)` feature): Rank IC inflates for all four horizons
- Shuffle test: Rank IC collapses for all four horizons
- For 5d horizon: the number of independent OOS periods in walk-forward is ≥50 per 12-month window
- 20d horizon is retained as `secondary_label` parameter throughout

---

### P4A-05: Excess-vs-CSI300 label

| Field | Detail |
|---|---|
| **Task ID** | P4A-05 |
| **Task Name** | Market-Excess Return Label |
| **Priority** | P1 — No new data needed; directly strips market beta from training target |
| **Implementation Order** | 5 |

**Objective**

For each date T and horizon h, compute `excess_vs_csi300_{h}d = ret_fwd_{h}d(stock) - ret_fwd_{h}d(CSI300_index)`. This strips market beta from the label so the model predicts cross-sectional relative performance rather than direction. This is the recommended primary label change from the R&D analysis.

Implementation: (a) collect CSI 300 index OHLCV via `tencent_quote(["000300"])` or equivalent — it is a daily series, not a stock, so it joins on date without PIT complexity; (b) in `labels/builder.py`, after computing raw `ret_fwd_{h}d` for each symbol, merge the index return for the same window and subtract; (c) add `excess_vs_csi300_{h}d` to the label schema and registry.

**Related Modules / Files**
- `labels/builder.py` — `_build_symbol_labels()`, `build_label_panel()`
- `store/schemas.py` — add `excess_vs_csi300` column family
- New: `ingest/index_collector.py` — collect daily CSI 300 OHLCV (small, near-zero maintenance cost)
- `labels/leakage_harness.py` — verify the excess label is also leakage-free

**Dependencies**
- P4A-04 (multi-horizon labels must exist first)

**Expected Deliverables**
- CSI 300 index OHLCV in the lake (silver table: `silver/index_ohlcv/000300.parquet`)
- `excess_vs_csi300_{1,5,10,20}d` columns in the label panel
- Leakage harness verification for excess labels
- Comparison: OOF Rank IC on `ret_fwd_5d` vs `excess_vs_csi300_5d` (expected: excess label shows higher or equivalent ICIR because market-beta noise is removed)

**Validation Criteria**
- For any date, `excess_vs_csi300_5d.mean()` across the universe ≈ 0 (market-neutral by construction)
- The excess label does not introduce future information: verify that the index return used for subtraction uses the same `close(T+1+h)/close(T+1) − 1` window
- Leakage harness canary test passes on excess labels
- Unit test: for a date where CSI 300 gained 2% and a stock gained 5%, `excess_vs_csi300_5d = 0.03` (±floating point)

---

### P4A-06: Ridge / linear diagnostic baseline

| Field | Detail |
|---|---|
| **Task ID** | P4A-06 |
| **Task Name** | Ridge Linear Baseline |
| **Priority** | P2 — Diagnostic; detects GBM overfitting |
| **Implementation Order** | 6 |

**Objective**

On Alpha158-style features, Ridge regression reaches ~0.046 Rank IC versus LightGBM's ~0.048 — the signal is largely linear. If the platform's GBM substantially exceeds Ridge OOF, the GBM is likely capturing nonlinear overfitting artefacts, which would explain the OOF/lockbox gap. Adding a Ridge baseline costs one day and produces a permanent diagnostic.

Add `RidgeModel` to the model zoo: a `sklearn.Pipeline` of `StandardScaler → Ridge` (or `Lasso`) wrapped in the same interface as `LightGBMModel`. Run it through `fit_oof` with the same PurgedKFold settings as the GBM baseline. Log both to MLflow under the same parent run.

Decision rule: if `GBM_OOF_RankIC / Ridge_OOF_RankIC > 1.3`, flag as potential GBM overfitting and recommend reducing `num_leaves` or `n_estimators`.

**Related Modules / Files**
- `training/model_zoo.py` — add `RidgeModel` class
- `training/lgbm_model.py` — `fit_oof()` (reuse without change)
- `evaluation/baselines.py` — record Ridge as a baseline alongside the trivial baselines
- `evaluation/leaderboard.py` — Ridge entry in the leaderboard

**Dependencies**
- P4A-03 (walk-forward evaluator; Ridge should also run in walk-forward mode)
- P4A-04 (5d label is primary)

**Expected Deliverables**
- `RidgeModel` class in model zoo with `fit(X, y)` and `predict(X)` interface
- OOF Rank IC comparison table: Ridge vs LightGBM vs trivial baselines
- Walk-forward comparison: Ridge vs LightGBM per window
- MLflow: Ridge run linked to the same experiment as LightGBM

**Validation Criteria**
- Ridge OOF Rank IC is reproducible (fixed `alpha` hyperparameter, no tuning)
- GBM/Ridge IC ratio computed and logged
- If GBM/Ridge > 1.3: a warning is emitted and documented in the research ledger
- Unit test: Ridge on shuffled labels produces OOF Rank IC < 0.01

---

## Phase 4B — Data Enrichment

---

### P4B-01: Valuation and size collector

| Field | Detail |
|---|---|
| **Task ID** | P4B-01 |
| **Task Name** | Valuation / Size / Turnover Ingest |
| **Priority** | P2 — Highest generalization upside; cheapest new data |
| **Implementation Order** | 7 |

**Objective**

Collect daily PE_TTM, PB, total market cap, float market cap, and turnover rate for all CSI 300 constituents via the Tencent Finance API (`tencent_quote`). This is the lowest-risk new data source: not IP-banned, batch-safe (all 300 stocks in one HTTP call), returns same-day data, and requires no PIT announcement-lag handling.

Design: `ingest/valuation_collector.py` following the same catalog-driven incremental pattern as `ingest/ohlcv_collector.py`. Daily batch pull at market close. Silver table: `silver/valuation/{symbol}.parquet` with columns `(symbol, date, pe_ttm, pb, total_mcap_yi, float_mcap_yi, turnover_pct)`.

Rate-limiting: Tencent API accepts batch requests (comma-separated codes). A single call for 300 stocks is feasible. No rate limiting issue.

**Related Modules / Files**
- New: `ingest/valuation_collector.py`
- `store/lake.py` — add `valuation_path()`, `valuation_dir()`
- `store/schemas.py` — add `VALUATION_SCHEMA`
- `ingest/catalog.py` — extend catalog to track valuation collection state
- `store/quality_report.py` — add valuation coverage check

**Dependencies**
- P4A-04 (5d label as primary — so valuation IC is evaluated at the right horizon)

**Expected Deliverables**
- `ValuationCollector` class with `run(symbols, date_range)` method
- Silver Parquet files for all CSI 300 symbols
- Catalog entries updated after each successful collection run
- Data quality report: coverage %, missing dates per symbol, PE outlier detection (PE < 0 or PE > 500 flagged)

**Validation Criteria**
- Tencent API response parsed correctly: PE_TTM in `vals[39]`, PB in `vals[46]` (not 43 — documented API trap)
- For a spot-check date: PE_TTM matches a public financial data source within 1%
- Incremental run: second collection run only fetches dates after the last collected date
- Crash-safe: interrupted collection leaves no corrupted Parquet files

---

### P4B-02: Valuation feature builder

| Field | Detail |
|---|---|
| **Task ID** | P4B-02 |
| **Task Name** | Valuation / Size Feature Engineering |
| **Priority** | P2 |
| **Implementation Order** | 8 |

**Objective**

Build cross-sectional valuation and size features from the silver valuation table. All features must be cross-sectionally normalised (rank or z-score within each date) so they are comparable across stocks at different price levels.

New feature columns:
- `cs_log_float_mcap` — log of float market cap, then cross-sectional z-score (size factor; large values = large-cap)
- `cs_pe_ttm_rank` — cross-sectional rank of PE_TTM (winsorised at 1st/99th percentile; negative PE set to 999 before ranking)
- `cs_pb_rank` — cross-sectional rank of PB
- `cs_turnover_rank` — cross-sectional rank of turnover rate (attention/liquidity proxy)
- `cs_log_mcap_rank` — cross-sectional rank of log(float_mcap)

Additional derived features:
- `pe_momentum_5d` — change in PE_TTM over the last 5 trading days (valuation expansion/compression signal, cross-sectionally ranked)

PIT safety: all fields are derived from the closing price of the same day T. No announcement-date lag is needed. Join on `(symbol, date)` directly.

**Related Modules / Files**
- New: `features/valuation.py` — `build_valuation_features(panel, valuation_df)`
- `features/registry.py` — add `VALUATION_SPECS` to the feature registry
- `features/pipeline.py` — add valuation builder to the pipeline orchestration
- `features/data_dictionary.py` — document new feature columns

**Dependencies**
- P4B-01 (silver valuation table must exist)

**Expected Deliverables**
- `build_valuation_features()` function
- 6 new feature columns added to the feature panel
- Data dictionary entries for each new feature
- Single-factor IC diagnostic run: report Rank IC of each valuation feature vs `ret_fwd_5d` and `excess_vs_csi300_5d`

**Validation Criteria**
- On any given date, `cs_pe_ttm_rank` is uniformly distributed in [0, 1] across the universe
- Features are leakage-free: the leakage harness canary test confirms valuation features respond to the canary injection
- Unit test: `cs_log_float_mcap` for a large-cap stock (e.g., 600519 Kweichow Moutai) ranks in the top decile on any historical date
- Negative PE is handled: stocks with PE < 0 are ranked at the bottom (rank = 0) not excluded

---

### P4B-03: Industry classification collector and mapper

| Field | Detail |
|---|---|
| **Task ID** | P4B-03 |
| **Task Name** | Industry Classification Ingest |
| **Priority** | P2 — Enables industry-neutral labels; unlocks sector-relative features |
| **Implementation Order** | 9 |

**Objective**

Collect and store the industry classification for each CSI 300 constituent as a point-in-time membership table. Industry classification in A-shares is slow-moving (changes are rare) but must be stored with `effective_date` and `out_date` in the same pattern as the universe membership table — because using today's industry classification for a 2021 date would contaminate historical industry-neutral labels.

Sources:
- Eastmoney `eastmoney_stock_info()` returns `industry` field (primary industry name)
- Eastmoney `eastmoney_concept_blocks()` returns all board/concept tags per stock (for multi-level classification)

Silver table schema: `silver/industry_map.parquet` with columns `(symbol, industry_code, industry_name, concept_tags, effective_date, out_date)`.

Collection strategy: pull current classification on first run; on subsequent runs, detect changes by comparing to the last stored classification and insert a new row with a new `effective_date` when a change is detected. This produces a slowly-changing-dimension table.

**Related Modules / Files**
- New: `ingest/industry_collector.py`
- `store/lake.py` — add `industry_map_path()`
- `store/schemas.py` — add `INDUSTRY_SCHEMA`
- `core/universe.py` — cross-reference: industry map uses same symbol universe
- `ingest/catalog.py` — add industry map to catalog

**Dependencies**
- P4B-01 (order is flexible; both are standalone collectors)

**Expected Deliverables**
- `IndustryCollector` class with monthly refresh schedule
- `silver/industry_map.parquet` covering all CSI 300 constituents
- Eastmoney rate limiting respected: `em_get()` used for all requests; no concurrent calls
- Point-in-time query function: `get_industry_as_of(symbol, date) → industry_code`

**Validation Criteria**
- Coverage: every symbol in the current CSI 300 universe has at least one industry record
- Historical correctness: a stock that changed industries retains its old industry code for all dates before the change
- Unit test: `get_industry_as_of("600519", "2020-01-01")` returns the correct historical industry for Kweichow Moutai
- Rate limit compliance: batch collection of 300 symbols at 1 req/sec does not trigger Eastmoney banning (verified by successful completion without 403/429)

---

### P4B-04: Industry-relative features and sector momentum

| Field | Detail |
|---|---|
| **Task ID** | P4B-04 |
| **Task Name** | Industry-Relative Feature Builder |
| **Priority** | P2 |
| **Implementation Order** | 10 |

**Objective**

Build cross-sectional features that are industry-relative (within-industry rank/z-score) rather than market-wide. Industry-relative signals remove sector beta and tend to generalise better in A-share markets where sector rotation is dominant.

New feature columns:
- `ind_rank_rsi_6` — rank of RSI-6 within the stock's industry on each date (within-industry momentum)
- `ind_rank_turnover` — rank of turnover rate within industry (within-industry attention/liquidity)
- `ind_rank_cs_main_flow` — rank of main-force net inflow within industry (after P4B-06)
- `ind_return_5d` — industry average 5d forward return (used for excess-vs-industry label; see P4B-05)
- `sector_momentum_10d` — industry average 10d past return (sector trend feature)

Implementation: `groupby(["date", "industry_code"])` operations, mirroring the existing `groupby("date")` pattern in `features/cross_sectional.py`. Join the industry map as-of each date before groupby.

**Related Modules / Files**
- New: `features/industry.py` — `build_industry_features(panel, industry_map)`
- `features/cross_sectional.py` — reference implementation for groupby pattern
- `features/registry.py` — add `INDUSTRY_SPECS`
- `features/pipeline.py` — add industry builder

**Dependencies**
- P4B-03 (industry map must exist)
- P4B-02 (valuation features must exist for within-industry rank of PE)

**Expected Deliverables**
- `build_industry_features()` function
- 5 new feature columns in the feature panel
- Single-factor IC diagnostics for all new industry features
- `ind_return_5d` available as a join-ready series for label construction (P4B-05)

**Validation Criteria**
- `ind_rank_rsi_6` is uniformly distributed in [0, 1] within each industry on any date
- Industry-relative features do not look ahead: confirmed by leakage harness
- `sector_momentum_10d` uses only past 10 days, not future
- For a date with exactly N stocks in one industry, `ind_rank_turnover` takes N distinct values

---

### P4B-05: Excess-vs-industry label

| Field | Detail |
|---|---|
| **Task ID** | P4B-05 |
| **Task Name** | Industry-Excess Return Label |
| **Priority** | P3 — Highest-payoff label after industry integration |
| **Implementation Order** | 11 |

**Objective**

Add `excess_vs_industry_{h}d = ret_fwd_{h}d(stock) - ret_fwd_{h}d(industry_average)` as a primary label. This strips sector beta from the training target. In A-shares where sector rotation is violent, this is typically the single biggest label improvement: a model trained on this label predicts within-sector relative performance rather than absolute direction, which is both more predictable and more practically useful for a long-short strategy.

`ret_fwd_{h}d(industry_average)` is computed for each (industry, date) as the equal-weighted average forward return of all universe stocks in that industry over the same T+1…T+1+h window. This uses only forward prices for stocks that are in the universe at date T — no lookahead.

**Related Modules / Files**
- `labels/builder.py` — `build_label_panel()` extension
- `features/industry.py` — `ind_return_5d` series (from P4B-04) is the raw material
- `store/schemas.py` — add `excess_vs_industry` column family
- `labels/leakage_harness.py` — verify industry-excess label

**Dependencies**
- P4B-04 (`ind_return_5d` must exist)
- P4A-05 (excess-vs-CSI300 pattern to follow)

**Expected Deliverables**
- `excess_vs_industry_{1,5,10,20}d` columns in the label panel
- Leakage harness verification
- Three-way comparison: OOF Rank IC on `ret_fwd_5d`, `excess_vs_csi300_5d`, `excess_vs_industry_5d` (expected: industry-excess highest ICIR)

**Validation Criteria**
- For any date, `excess_vs_industry_5d.groupby(industry).mean() ≈ 0` (industry-neutral by construction)
- The industry average uses only the T+1…T+1+h window, not prices before T
- Unit test: for a stock in an industry where the industry gained 3% and the stock gained 5%, `excess_vs_industry_5d = 0.02`
- Does not overfit during construction: the industry average includes the stock being labelled (acceptable — this matches professional factor construction practice)

---

### P4B-06: Capital flow collector

| Field | Detail |
|---|---|
| **Task ID** | P4B-06 |
| **Task Name** | Capital Flow Ingest (Daily 120-day history) |
| **Priority** | P3 — Primary short-horizon orthogonal signal |
| **Implementation Order** | 12 |

**Objective**

Collect daily fund flow data (main/super-large/large/medium/small net inflow) for all CSI 300 constituents via Eastmoney `push2his` (`stock_fund_flow_120d`). This is the most important new data source for a short-horizon strategy: capital flow is orthogonal to price-derived technicals and has been shown to carry short-horizon predictive content in A-share markets.

Rate limiting: Eastmoney push2his is rate-limited but less aggressively than the datacenter API. Use `em_get()` with `EM_MIN_INTERVAL = 1.0s`. For 300 stocks, a full refresh takes ~5 minutes — acceptable for a daily batch job.

Silver table: `silver/fund_flow/{symbol}.parquet` with columns `(symbol, date, main_net, small_net, mid_net, large_net, super_net)`. All values in yuan (元), not 万元.

Important: this endpoint returns the most recent 120 calendar days. The incremental catalog update must track the last collection date and only trigger re-collection when the oldest available date in the silver table is >60 days old (to maintain continuous history).

**Related Modules / Files**
- New: `ingest/flow_collector.py`
- `store/lake.py` — add `fund_flow_path()`
- `store/schemas.py` — add `FUND_FLOW_SCHEMA`
- `ingest/catalog.py` — extend catalog for fund flow
- Reference: SKILL.md §4.5 `stock_fund_flow_120d()` function (use verbatim or adapt)

**Dependencies**
- P4B-01 (valuation collector establishes the collection pattern)

**Expected Deliverables**
- `FundFlowCollector` class
- Silver Parquet files for all CSI 300 symbols
- Continuous history for at least 12 months (requires monthly collection runs)
- Rate limiting compliance: no Eastmoney IP banning during collection

**Validation Criteria**
- `main_net` values are in yuan (unit check: for a liquid large-cap, daily main net flow should be in the hundreds of millions of yuan range)
- No duplicate dates per symbol
- Incremental run: second collection only extends the tail, does not overwrite existing history
- For a date where the stock was a top buyer target, `main_net > 0` and `super_net > 0` (directional sanity check)

---

### P4B-07: Capital flow feature builder

| Field | Detail |
|---|---|
| **Task ID** | P4B-07 |
| **Task Name** | Capital Flow Feature Engineering |
| **Priority** | P3 |
| **Implementation Order** | 13 |

**Objective**

Build cross-sectional capital flow features from the silver fund flow table. All flow features are normalised by float market cap to be cross-sectionally comparable (a 100亿 inflow into a 500亿 stock is very different from the same into a 5000亿 stock).

New feature columns:
- `cs_main_flow_rank_1d` — cross-sectional rank of `main_net(T) / float_mcap` (1-day main-force flow intensity)
- `cs_main_flow_rank_5d` — rank of 5-day cumulative `main_net / float_mcap`
- `cs_small_flow_rank_1d` — rank of `small_net(T) / float_mcap` (retail flow; contrarian signal)
- `cs_flow_reversal_5d` — `cs_main_flow_rank_1d - cs_main_flow_rank_5d` (short-term reversal within flow momentum)
- `cs_super_flow_rank_1d` — rank of super-large order net inflow / float_mcap (institutional signal)

PIT safety: all features use flow data through date T only. The 1-day flow uses `T`'s data (released after market close); the 5-day cumulative uses `T-4` through `T`.

Float market cap for normalisation comes from the valuation silver table (P4B-01) — join on `(symbol, date)`.

**Related Modules / Files**
- New: `features/flow.py` — `build_flow_features(panel, flow_df, valuation_df)`
- `features/registry.py` — add `FLOW_SPECS`
- `features/pipeline.py` — add flow builder
- `features/data_dictionary.py` — document flow features

**Dependencies**
- P4B-06 (flow silver table)
- P4B-01 (valuation silver table for float_mcap normalisation)

**Expected Deliverables**
- `build_flow_features()` function
- 5 new feature columns in the feature panel
- Single-factor IC diagnostic: each flow feature vs `ret_fwd_1d`, `ret_fwd_5d`, `excess_vs_csi300_5d`
- IC decay curve: flow features are expected to show highest IC at 1d and decay quickly

**Validation Criteria**
- `cs_main_flow_rank_1d` is uniformly distributed in [0, 1] on any given date
- Flow normalised by float_mcap: unit test confirms `cs_main_flow_rank_1d` varies across stocks of different sizes
- Leakage harness: flow features at date T do not use data from T+1 or later
- Expected signal direction: `cs_main_flow_rank_1d` should show positive single-factor Rank IC vs `ret_fwd_1d` (main inflow predicts next-day positive return)

---

### P4B-08: Margin trading collector and features

| Field | Detail |
|---|---|
| **Task ID** | P4B-08 |
| **Task Name** | Margin Trading Ingest and Features |
| **Priority** | P4 — Secondary short-horizon signal; lower priority than flow |
| **Implementation Order** | 14 |

**Objective**

Collect daily margin trading data (融资余额, 融资买入额, 融券余额) and build leverage sentiment features. Margin balance changes are a lagging-but-useful leverage sentiment signal: rapidly increasing margin balance indicates over-leveraged retail momentum; margin balance contraction can signal selling pressure.

Source: Eastmoney `datacenter-web` `RPTA_WEB_RZRQ_GGMX` endpoint (SKILL.md §4.1 `margin_trading()`). Rate limited; use `em_get()`.

New features:
- `cs_margin_balance_change_5d` — 5-day rate of change in 融资余额 / float_mcap (cross-sectional rank)
- `cs_rzrq_ratio_rank` — rank of total 融资融券余额 / float_mcap (leverage level)

Note: margin trading data released by Eastmoney is typically 1-business-day delayed. Use a 1-day lag when joining to the feature panel.

**Related Modules / Files**
- New: `ingest/margin_collector.py`
- `store/lake.py` — add `margin_path()`
- New: `features/margin.py`
- `features/registry.py` — add `MARGIN_SPECS`

**Dependencies**
- P4B-01 (valuation table for float_mcap normalisation)
- P4B-06 (flow collector establishes the Eastmoney collection pattern)

**Expected Deliverables**
- `MarginCollector` class
- Silver Parquet files per symbol
- 2 new cross-sectional features
- Single-factor IC diagnostic

**Validation Criteria**
- 1-day lag is applied: feature at date T uses margin data through date T-1
- Coverage: margin data only available for margin-eligible stocks (not all CSI 300 stocks); missing stocks are handled gracefully (NaN, not error)
- Rate limit compliance with `em_get()`

---

## Phase 4C — Factor Diagnostics and Refinements

---

### P4C-01: Single-factor IC diagnostic framework

| Field | Detail |
|---|---|
| **Task ID** | P4C-01 |
| **Task Name** | Per-Feature IC Diagnostic |
| **Priority** | P3 — Must run after each new data block; ongoing hygiene |
| **Implementation Order** | 15 (and re-run after each P4B task) |

**Objective**

For each feature in the feature registry, compute: (a) mean daily Rank IC vs each label horizon (1d, 5d, 10d, 20d); (b) IC t-statistic; (c) IC decay curve (feature → label IC at lags 1, 2, 3, 5, 10, 20 days); (d) pairwise Spearman correlation matrix across all features. Output: a ranked feature summary table persisted to the lake and visualised.

This framework is a permanent diagnostic that runs after every data integration and before any feature pruning decision.

**Related Modules / Files**
- New: `evaluation/feature_ic.py` — `FeatureICReport`
- `evaluation/metrics.py` — reuse `evaluate()` for per-feature IC
- `evaluation/decay_monitor.py` — extend with per-feature IC decay
- `features/registry.py` — iterate over all registered features
- MLflow — log feature IC report as an artifact

**Dependencies**
- P4A-03 (walk-forward evaluator; IC decay should be computed on OOS folds)
- P4A-04 (multi-horizon labels)

**Expected Deliverables**
- `compute_feature_ic_report(panel, feature_cols, label_cols)` function
- Sorted table: feature | IC_1d | IC_5d | IC_20d | t_stat | decay_halflife
- Collinearity heatmap (Spearman correlation matrix for all features)
- CSV exported to lake: `evaluation/feature_ic_report_{run_date}.csv`
- MLflow artifact: feature IC report for each experiment run

**Validation Criteria**
- IC t-statistic formula: `mean_IC * sqrt(n_dates) / std_IC` (standard formula)
- Decay half-life estimated as the lag where IC falls below 50% of its peak value
- A deliberately injected canary feature (`close.shift(-1)`) ranks first in IC at 1d horizon
- A random noise feature ranks near zero IC (t-stat < 1) across all horizons
- Pairwise correlation matrix is symmetric and diagonal = 1

---

### P4C-02: Technical feature collinearity pruning

| Field | Detail |
|---|---|
| **Task ID** | P4C-02 |
| **Task Name** | Technical Feature Pruning |
| **Priority** | P4 — Reduces overfitting surface; do after IC diagnostic |
| **Implementation Order** | 16 |

**Objective**

Use the collinearity analysis from P4C-01 to identify groups of technical features with pairwise Spearman correlation > 0.85. For each group, retain only the feature with the highest single-factor Rank IC at the primary label horizon (5d). Expected outcome: the 27 normalised technical indicators reduce to ~10–12 genuinely independent signals.

This is a structured pruning decision, not an ad-hoc one: document each pruned feature, its correlation with the retained feature, and the IC comparison. Record in the research ledger.

**Related Modules / Files**
- `features/registry.py` — mark pruned features as `active=False` (do not delete, preserve history)
- `features/technical.py` — pruned features still computed but not passed to the model
- `evaluation/feature_ic.py` — pruning decision criteria
- `evaluation/research_ledger.py` — record pruning as a research decision

**Dependencies**
- P4C-01 (IC diagnostic must have run)
- P4B-07 (flow features integrated; prune after all new features are in so collinearity with new features is visible)

**Expected Deliverables**
- Pruning decision table: feature | correlation group | IC_5d | pruned? | reason
- Updated `TECHNICAL_SPECS` with `active` flags
- Re-run of OOF evaluation with pruned feature set: confirm IC does not decrease (expected: slight improvement or no change, with smaller model)
- Research ledger entry documenting the pruning decision

**Validation Criteria**
- No retained feature pair has Spearman correlation > 0.85 after pruning
- Pruned features are retained in Parquet but excluded from training panel (verifiable by inspecting `feature_cols` list passed to `fit_oof`)
- OOF Rank IC on pruned feature set ≥ 95% of unpruned set IC (pruning should not hurt significantly)

---

### P4C-03: Lockup expiry feature

| Field | Detail |
|---|---|
| **Task ID** | P4C-03 |
| **Task Name** | Lockup Expiry Event Feature |
| **Priority** | P4 — PIT-safe event feature; low-cost, moderate signal |
| **Implementation Order** | 17 |

**Objective**

Upcoming lockup expiry dates are public knowledge at all prior dates (announced at IPO or placement). This makes them legitimately PIT-safe for any lookforward. Implement two features:
- `days_to_next_unlock` — integer: number of calendar days until the next lockup expiry for this stock (or 999 if no unlock in the next 180 days)
- `unlock_size_ratio` — float: number of shares unlocking / total float shares (proxy for supply pressure magnitude; 0 if no unlock in the next 30 days)

Source: Eastmoney `lockup_expiry()` (SKILL.md §3.6).

These features are expected to carry short-horizon signal around unlock events: large unlocks create supply pressure, particularly in the 5–10 trading days before the unlock date.

**Related Modules / Files**
- New: `ingest/lockup_collector.py`
- New: `features/event.py` — `build_lockup_features(panel, lockup_df)`
- `store/lake.py` — add `lockup_path()`
- `features/registry.py` — add `LOCKUP_SPECS`

**Dependencies**
- P4B-03 (establishes the Eastmoney rate-limited collection pattern)

**Expected Deliverables**
- `LockupCollector` with forward-looking collection: for each symbol, collect next 180 days of upcoming unlocks
- Silver Parquet: `silver/lockup/{symbol}.parquet` with columns `(symbol, unlock_date, type, shares, ratio)`
- `days_to_next_unlock` and `unlock_size_ratio` added to feature panel
- Single-factor IC diagnostic: expected to show negative IC at 1d–5d horizon for large unlocks

**Validation Criteria**
- `days_to_next_unlock` is strictly positive for all dates T < unlock date (never zero or negative)
- For dates after the unlock, that unlock record is no longer the "next" event
- PIT correctness: feature at date T only uses unlock dates >= T+1 (the unlock itself is in the future)
- Sparsity handled: most stocks have no upcoming unlock on any given day; `days_to_next_unlock = 999` default

---

### P4C-04: Residualised return label

| Field | Detail |
|---|---|
| **Task ID** | P4C-04 |
| **Task Name** | Residualised Return Label |
| **Priority** | P5 — Professional-grade label; high value, medium implementation cost |
| **Implementation Order** | 18 |

**Objective**

The residualised return is the standard label used by professional factor desks. For each date T and horizon h, run a cross-sectional regression of `ret_fwd_{h}d` on `[1, market_return, industry_dummies, log_float_mcap]`. The residual is the label: it represents the component of return unexplained by market, sector, and size exposure. A model trained on this label predicts pure idiosyncratic return.

This requires industry features (P4B-04) and size features (P4B-02) to be in place. The regression runs per-date using only the cross-section of stocks on that date — no across-time data leakage.

**Related Modules / Files**
- `labels/builder.py` — `build_label_panel()` extension
- New helper: `labels/residualiser.py` — `residualise_returns(panel, label_col, industry_col, mcap_col)`
- `store/schemas.py` — add `residual_ret` column family

**Dependencies**
- P4B-04 (industry features)
- P4B-02 (size features)
- P4A-05 (excess-vs-CSI300 as reference)
- P4B-05 (excess-vs-industry as intermediate step)

**Expected Deliverables**
- `residualise_returns()` function using `sklearn.LinearRegression` per-date
- `residual_ret_{1,5,10,20}d` columns in the label panel
- Comparison: OOF Rank IC on all five label variants (raw, excess_csi300, excess_industry, residual) at 5d horizon

**Validation Criteria**
- For any date, `residual_ret_5d.mean() ≈ 0` and `residual_ret_5d.corr(market_ret_5d) ≈ 0` (by construction)
- Residual labels have higher ICIR than raw labels (expected; the residual is cleaner)
- Regression uses only within-date data (no across-time leakage)
- Unit test: for a date with 100 stocks, the regression has 100 observations and 2 + n_industries regressors

---

### P4C-05: Walk-forward regime analysis and stability report

| Field | Detail |
|---|---|
| **Task ID** | P4C-05 |
| **Task Name** | Walk-Forward Regime Analysis |
| **Priority** | P5 — Strategic insight; run after all features are integrated |
| **Implementation Order** | 19 |

**Objective**

Use the walk-forward evaluator (P4A-03) with the full enriched feature set to compute per-window Rank IC broken down by feature group (technical, valuation, industry, flow). Map IC over time to identify: (a) which feature groups are regime-stable vs regime-sensitive; (b) which calendar windows are "hard regimes" (low IC across all features); (c) whether the ensemble performance degrades uniformly or in feature-group-specific patterns.

Output: a regime analysis report with IC timelines per feature group, hard regime identification (IC < 0.01 for more than 2 consecutive windows), and recommendations for regime-conditional weighting.

**Related Modules / Files**
- `evaluation/walk_forward.py` — extend with per-feature-group breakdown
- New: `evaluation/regime_analysis.py`
- `evaluation/research_ledger.py` — log regime findings

**Dependencies**
- P4A-03 (walk-forward evaluator)
- P4B-07 (flow features integrated)
- P4C-01 (feature IC diagnostic framework)

**Expected Deliverables**
- Per-window IC table: `(window_id, date_range, technical_IC, valuation_IC, industry_IC, flow_IC, ensemble_IC)`
- Hard regime identification: dates flagged as hard regimes with explanatory notes
- IC timeline plot data (stored as CSV; plotting is external)
- Research ledger entry with regime analysis findings

**Validation Criteria**
- IC breakdown sums are consistent: ensemble IC is not less than the best individual group IC (up to correlation effects)
- Hard regime windows have ensemble IC < 0.01 for at least 2 consecutive windows
- Regime report is reproducible with fixed seeds and window parameters

---

## Cross-Cutting Tasks

---

### P4X-01: Documentation and CLAUDE.md update

| Field | Detail |
|---|---|
| **Task ID** | P4X-01 |
| **Task Name** | Documentation Update for Phase 4 |
| **Priority** | Run alongside each completed task |
| **Implementation Order** | Parallel |

**Objective**

Update `CLAUDE.md` and the data dictionary to reflect: (a) new primary label (`ret_fwd_5d` / `excess_vs_csi300_5d`); (b) new modules and their interfaces; (c) the walk-forward evaluation as the primary evaluation method; (d) the feature registry state (which features are active, which are pruned). Maintain the architectural principles section.

---

### P4X-02: Test suite extension

| Field | Detail |
|---|---|
| **Task ID** | P4X-02 |
| **Task Name** | Test Suite Extension for Phase 4 Modules |
| **Priority** | Required for each new module |
| **Implementation Order** | Parallel (after each module) |

**Objective**

Every new module introduced in Phase 4 must have unit tests covering: (a) the happy path; (b) edge cases (empty data, single-stock universe, missing dates); (c) PIT safety (the leakage harness is the integration test, but unit-level temporal checks are needed); (d) rate limiting (mock `em_get()` to verify throttle is called). The existing 159-test suite must continue to pass with zero regressions.

**Validation Criteria**
- All new modules have ≥3 unit tests
- Total test count increases monotonically with each task
- `pytest` runs clean (zero failures, zero warnings that are not pre-existing)

---

## Summary Table

| Task ID | Name | Priority | Order | Phase | Estimated Days |
|---|---|---|---|---|---|
| P4A-01 | Robustness embargo fix | P0 | 1 | 4A | 0.5 |
| P4A-02 | Subperiod stability report | P0 | 2 | 4A | 0.5 |
| P4A-03 | Walk-forward OOS evaluator | P1 | 3 | 4A | 5 |
| P4A-04 | Multi-horizon labels (5d, 10d) | P1 | 4 | 4A | 1 |
| P4A-05 | Excess-vs-CSI300 label | P1 | 5 | 4A | 1.5 |
| P4A-06 | Ridge linear baseline | P2 | 6 | 4A | 2 |
| P4B-01 | Valuation/size collector | P2 | 7 | 4B | 2 |
| P4B-02 | Valuation feature builder | P2 | 8 | 4B | 2 |
| P4B-03 | Industry classification collector | P2 | 9 | 4B | 3 |
| P4B-04 | Industry-relative feature builder | P2 | 10 | 4B | 2 |
| P4B-05 | Excess-vs-industry label | P3 | 11 | 4B | 1.5 |
| P4B-06 | Capital flow collector | P3 | 12 | 4B | 3 |
| P4B-07 | Capital flow feature builder | P3 | 13 | 4B | 2 |
| P4B-08 | Margin trading collector/features | P4 | 14 | 4B | 2 |
| P4C-01 | Single-factor IC diagnostic | P3 | 15 | 4C | 3 |
| P4C-02 | Technical feature pruning | P4 | 16 | 4C | 1.5 |
| P4C-03 | Lockup expiry feature | P4 | 17 | 4C | 2 |
| P4C-04 | Residualised return label | P5 | 18 | 4C | 3 |
| P4C-05 | Walk-forward regime analysis | P5 | 19 | 4C | 3 |
| P4X-01 | Documentation update | Parallel | — | All | Ongoing |
| P4X-02 | Test suite extension | Parallel | — | All | Ongoing |

**Total estimated implementation time:** Phase 4A ≈ 10 days; Phase 4B ≈ 16 days; Phase 4C ≈ 13 days. Total ≈ 8–10 weeks including documentation and testing, assuming one developer working part-time on this project alongside other work.

---

## Critical Path

```
P4A-01 → P4A-02 → P4A-03 → P4A-04 → P4A-05 → P4A-06
                                │
                                ▼
                           P4B-01 ──► P4B-02 ──► (IC diagnostic via P4C-01)
                                │
                                ▼
                           P4B-03 ──► P4B-04 ──► P4B-05
                                │
                                ▼
                           P4B-06 ──► P4B-07 ──► (IC diagnostic)
                                                  │
                                                  ▼
                                            P4C-01 ──► P4C-02
                                                       │
                                                       ▼
                                                  P4C-04 ──► P4C-05
```

The walk-forward evaluator (P4A-03) is the true critical path dependency. No subsequent evaluation of new data blocks is meaningful until it exists.

---

## Pre-Registration Checklist (before running any evaluation)

For each new data block integration, fill in this checklist before running any code:

```
Data Block: _______________
Date pre-registered: _______________

Hypothesis: "This data block will improve walk-forward OOF Rank IC at 5d horizon by ≥ X bps"
X = _______________

Primary label for this test: [ ] ret_fwd_5d  [ ] excess_vs_csi300_5d  [ ] excess_vs_industry_5d
Secondary label: _______________

Feature group to add: _______________
Features excluded from model while testing this block alone: _______________

Promotion criterion: challenger ICIR >= champion ICIR + 0.1, Wilcoxon p ≤ 0.05
Walk-forward configuration: n_windows=5, window_months=12, step_months=12  [FIXED, not changed]

Result (filled after run):
  Walk-forward OOF Rank IC: _______________
  ICIR: _______________
  Wilcoxon p-value vs champion: _______________
  Conclusion: [ ] promoted  [ ] rejected  [ ] needs investigation

Ledger entry ID: _______________
```

This checklist enforces the pre-registration principle and is the primary defence against the multiple-comparisons problem as more data sources are integrated.
