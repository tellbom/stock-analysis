# Quant Platform — Phase 4 Strategic Roadmap

*Prepared after full review of `rd_roadmap_analysis.md`, the complete P0–P3 codebase, and the SKILL.md data-source toolkit. This document is the foundation for the next development phase. No code is modified here.*

---

## 1. Current Project Assessment

### What has been built and what works

The platform has completed four substantial phases and possesses a more methodologically sound evaluation stack than most academic or early-commercial implementations. Specific strengths:

**Evaluation engine (genuine asset):**
- `PurgedKFold` splits by *unique trading date* across the panel — the correct behaviour for an equity cross-section, and a mistake many published notebooks get wrong.
- Embargo defaults to `horizon`, ensuring no label-window overlap across fold boundaries.
- Backtest correctly sub-samples to non-overlapping rebalance dates and annualises with `periods_per_year = 252 / horizon`. The catastrophic Sharpe inflation bug is fixed.
- Labels use strict T+1 execution (`close(T+1+h)/close(T+1) − 1`), matching Qlib's Alpha158 convention.
- Cross-sectional features are computed `groupby("date")`, preventing across-time leakage.
- Absolute-price features are normalised into dimensionless ratios (close/MA, %B, MACD/close, ATR/close, OBV z-score) so cross-sectional ranking is not corrupted by price level.

**Infrastructure:**
- Parquet medallion lake (bronze/silver/gold) with crash-safe catalog.
- MLflow tracking, Optuna HPO, model zoo (LightGBM/XGBoost/CatBoost), SHAP explainability.
- Research ledger with deflated ICIR (Bailey–López de Prado) to address the multiple-comparisons problem.
- Pre-registered champion/challenger promotion criteria (Wilcoxon tests) to prevent p-hacking.
- DVC-based reproducibility hardening, leakage harness with canary detection.

### Headline metrics and their honest interpretation

| Metric | Value | Interpretation |
|---|---|---|
| OOF Rank IC | ~0.091 | Strong *in-period*; regime-flattered |
| OOF ICIR | ~0.65 | Modest; consistent with daily IC std ≈ 0.14 |
| Lockbox Rank IC | 0.004 | Statistically indistinguishable from zero (SE ≈ 0.039) |
| Lockbox Sharpe | −1.82 | ~12 independent observations; t-stat < 2; not significant |
| True forward estimate | ~0.02–0.05 | Consistent with Qlib benchmark on Alpha158 (0.048 RankIC) |

The negative lockbox result is **not a refutation** of the approach. At a 20-day horizon, 12 months of lockbox yields only ~12 independent forward periods. The 95% confidence interval on the lockbox Rank IC is approximately [−0.07, +0.08] — equally consistent with zero and with a healthy 0.05 signal. The evaluation instrument is underpowered; no meaningful verdict is possible in either direction until this is fixed.

### Known defects

1. **`evaluation_robustness.py` defaults `embargo=5` while `horizon=20`.** Ablation re-fits compare an embargo-20 baseline against embargo-5 variants — apples to oranges, slightly under-purged, and producing misleading ablation deltas. This must be fixed for internal consistency before ablation results are acted upon.

2. **Static 12-month lockbox is structurally insufficient** for a 20-day horizon. Walk-forward / rolling out-of-sample evaluation with 5–10 sequential test windows is required to accumulate enough independent forward periods for a reliable verdict.

3. **Feature set is information-poor.** All 27 technical indicators are transforms of the same OHLCV series — a one-dimensional information space. The empirical ceiling for price-only models on CSI 300 is ~0.048 forward Rank IC (Qlib's 158-feature Alpha158). No amount of additional indicators or better models can push substantially past this ceiling.

---

## 2. Architecture Understanding

### Data flow (current state)

```
AKShare / TDX / Tencent / Eastmoney / Sina
        │
        ▼
ingest/ ──── catalog + incremental fetch
        │        (ohlcv_collector, fundamentals_collector, universe_service)
        ▼
store/  ──── Parquet medallion lake (bronze / silver / gold)
        │        + DuckDB views
        ▼
features/ ── technical (27 indicators, normalised)
        │    + cross-sectional (rank / z-score per date)
        │    + fundamental (PIT join on announce_date)
        │    → versioned feature sets (hash-based feature_set_id)
        ▼
labels/ ──── forward return (1d / 5d / 20d) + risk labels
        │    + leakage harness (canary injection, temporal checks)
        ▼
training/ ── LightGBM baseline (sklearn Pipeline)
        │    + Optuna HPO inside purged CV
        │    + model zoo (LightGBM/XGBoost/CatBoost)
        │    + MLflow tracking + DVC reproducibility
        ▼
evaluation/ ─ IC / Rank IC / ICIR / quantile metrics
             + cost-aware backtest
             + robustness null tests (shuffle, canary, subperiod, ablation)
             + alpha verdict synthesizer
             + SHAP explainability + PSI drift monitor
             + champion/challenger protocol (pre-registered Wilcoxon)
             + research ledger (deflated ICIR)
```

### Key architectural strengths

- **Datasets are the product; models are consumers.** This principle is enforced structurally: the feature pipeline is hash-versioned, labels are separately built and verified, and model training is gated behind a leakage harness.
- The PIT fundamentals store using `announce_date` as the join key is the hardest correctness problem in financial ML, and it is implemented correctly.
- The PurgedKFold + embargo-equals-horizon design eliminates the most common label-leakage path in cross-validation.
- Survivorship bias is addressed via effective-date universe membership (`in_date`, `out_date`).

### Short-horizon focus and architectural implications

The project's core objective is **short-horizon quantitative trading (fast-entry / fast-exit)**, not long-term factor investing. This is a fundamental framing choice that changes every subsequent decision:

| Dimension | Long-horizon (20d+) | Short-horizon (1–5d) target |
|---|---|---|
| Primary signals | Fundamental quality, value, size | Momentum, flow, sentiment, event-driven |
| Label noise | Low (large cumulative return) | High (small signal-to-noise ratio) |
| Transaction cost sensitivity | Low | **High — cost management is alpha** |
| Evaluation power | Low (few independent periods) | **High (many independent periods)** |
| Feature decay | Slow | **Fast — decay monitoring is critical** |
| Information sources | Quarterly fundamentals | **Daily flow, intraday, event calendars** |
| Leakage risk | Moderate (PIT join timing) | **Extreme (intraday timing discipline required)** |

This framing makes the `rd_roadmap_analysis.md` priority order even more compelling: **fix the measuring instrument → enrich the data → refine labels.** A 5-day horizon yields ~50 independent forward periods per year versus ~12 at 20 days — this is not a minor improvement, it is a 4× amplification of statistical power that changes what can and cannot be measured.

---

## 3. Key Bottlenecks

### Bottleneck 1 — Evaluation instrument (most urgent)

The current evaluation cannot produce a reliable verdict. Two complementary fixes are needed:

1. **Walk-forward / rolling OOS replaces the static lockbox.** Five to ten sequential test windows accumulate 60–120 independent forward periods. This is the minimum needed to detect a forward Rank IC of ~0.04 with reasonable power.
2. **Shorter horizons (5d, 10d).** Adding these costs almost nothing (the label builder already supports multi-horizon), and they immediately multiply the number of independent periods. A 5-day horizon makes the evaluation approximately 4× more powerful per unit of history.

Until both are done, the platform is optimising blind. Every change to features or labels that "improves" the lockbox might be fitting the twelve available observations.

### Bottleneck 2 — Information ceiling (the real upside)

The 27 current technical indicators reach the practical ceiling for price-volume signal on CSI 300. The `rd_roadmap_analysis.md` benchmark is unambiguous: 158 carefully-engineered price/volume features reach only ~0.048 forward Rank IC. The next signal increment requires genuinely orthogonal data. In priority order (per the analysis, confirmed by short-horizon focus):

1. **Valuation / size / turnover + industry classification** — cheap, same-day safe, unlocks industry neutralisation.
2. **Capital flow (main/super-large/large/medium/small net inflow)** — explicitly shorter-horizon, orthogonal to price.
3. **Margin trading data (融资融券)** — leverage sentiment, daily frequency.
4. **Fundamentals via PIT** — higher cost, higher payoff for quality/value factors.

The SKILL.md toolkit provides working, rate-limit-aware code for all four of these data blocks that the existing ingestion architecture (bronze/silver/gold lake, PIT join on `announce_date`, catalog-driven incremental fetch) can absorb without structural change.

### Bottleneck 3 — Label design

The current label is `ret_fwd_20d` — market-beta-inclusive, broad-horizon. Three additions are high value at low cost:

1. **Excess return vs CSI 300** — strips market beta, makes the label cross-sectionally neutral.
2. **Excess return vs industry** — strips sector beta; in A-shares, sector rotation is violent and dominates raw returns. This is the single most effective label improvement available after industry data is integrated.
3. **Residualised return** (market + industry + size regressed out per day) — the professional-grade factor target; medium implementation cost, highest generalization payoff.

### Bottleneck 4 — Robustness suite inconsistency

The `evaluation_robustness.py` ablation refits use `embargo=5` while the baseline uses `embargo=20` (horizon). This produces misleading ablation deltas and could lead to wrong feature-pruning decisions. This is a one-line parameter fix, but it must be done before acting on any ablation result.

### Bottleneck 5 — No linear diagnostic baseline

On Alpha158-style features, linear/ridge models reach ~0.046 Rank IC versus LightGBM's ~0.048. If the platform's GBM substantially exceeds the linear baseline on OOF, the excess is overfitting — an overfitting red flag that explains the OOF/lockbox gap. Adding a Ridge baseline is a diagnostic, not a performance play, and takes negligible effort.

---

## 4. Research Conclusions

Drawing from `rd_roadmap_analysis.md` and validated against the codebase:

1. **The evaluation engine is trustworthy for in-period signal measurement.** The purge/embargo logic, label construction, backtest annualisation, and cross-sectional feature design are all correct. The problem is not bugs; it is that the static 20-day-horizon lockbox is structurally underpowered.

2. **OOF ≈ 0.09 is regime-flattered in-period signal.** The honest forward estimate is in the 0.02–0.05 range. This is consistent with Qlib's benchmark and represents a real, modest, regime-sensitive signal — not a refutation of the approach.

3. **More technical indicators, algorithm swaps, and hyperparameter tuning are all low-value.** The OHLCV-derived feature set is near its forward ceiling. Fancier models provide ~±0.01 Rank IC on this feature set — inside lockbox noise. HPO on an information-poor feature set against an underpowered lockbox simply overfits the evaluation instrument.

4. **Shorter horizons are a multiplicative force multiplier.** For a short-horizon strategy, operating at 5d labels is not a concession; it is the natural home of the approach. Capital flow, volume momentum, sector rotation signals, and event-driven signals all decay quickly and are most powerful at 1–5 day horizons. Moving the primary horizon from 20d to 5d simultaneously increases evaluation power and matches the strategy's intended trading frequency.

5. **Capital flow data from SKILL.md is the highest-priority new data source for a short-horizon strategy.** Fund flow signals (main/super-large/large/medium/small net inflow) are orthogonal to price-derived technicals, are available at daily and minute frequency, have been shown to carry short-horizon predictive content in A-share markets, and require only modest integration work given the existing lake architecture.

6. **Industry classification is an enabler, not a standalone factor.** Its primary value is unlocking industry-neutral labels and industry-relative cross-sectional features — both of which disproportionately improve generalization in A-shares where sector rotation is a dominant regime-switching driver.

7. **The subperiod stability test is free information that has not been read.** The robustness suite already computes `first_half_ric` and `second_half_ric`. If both halves share sign and rough magnitude, the in-period signal is regime-stable; if they diverge in sign, it is regime-fragile. This is the single most informative number for understanding the OOF/lockbox gap.

---

## 5. Recommended Roadmap

The roadmap is organised into three phases, each independently valuable. The ordering is dictated by the measurement-first principle: no data enrichment is meaningful until the instrument can detect whether it helps.

---

### Phase 4A — Fix the Instrument (prerequisite for everything)

**Duration estimate:** 2–3 weeks of focused development.

**Goal:** Replace the underpowered static lockbox with a walk-forward evaluation system that can actually measure ~0.04 forward Rank IC, add shorter-horizon labels, and fix the robustness suite embargo inconsistency. At the end of this phase, the platform can for the first time produce a *reliable* out-of-sample verdict.

**Components:**

**4A-1: Walk-forward / rolling OOS evaluator**
Replace `make_lockbox_split` with a `WalkForwardEvaluator` that walks a test window through the available history in sequential non-overlapping steps. Each step produces an OOS prediction set; combined, they yield 60–120 independent forward periods. The evaluator should produce: per-window Rank IC, forward IC decay curve, regime stability indicator (IC sign stability across windows), and the existing backtest metrics applied to the concatenated OOS predictions. This module replaces, not augments, the static lockbox.

**4A-2: 5d and 10d labels (cost: near zero)**
The label builder already handles multi-horizon. Add `horizons=[1, 5, 10, 20]` as the default. Run the leakage harness verification on all horizons. The 5d label immediately becomes the primary evaluation horizon — it yields ~50 independent periods per year versus ~12 at 20d.

**4A-3: Fix robustness embargo inconsistency**
Change the default `embargo` parameter in `run_robustness_tests` from `embargo=5` to `embargo=horizon` (matching the PurgedKFold default). Re-run ablation to get clean deltas. Read and report the existing `first_half_ric` / `second_half_ric` numbers — they are already computed but not surfaced in the alpha verdict.

**4A-4: Ridge / linear baseline**
Add a `sklearn.linear_model.Ridge` baseline wrapped in the same Pipeline/PurgedKFold harness. Compare its OOF Rank IC against LightGBM. If GBM substantially exceeds Ridge, flag as potential overfitting. If they are close, the signal is genuinely linear and the GBM is not manufacturing nonlinear noise.

---

### Phase 4B — Data Enrichment (the real signal upside)

**Duration estimate:** 5–8 weeks, with each data block independently deliverable.

**Goal:** Integrate three orthogonal data blocks from SKILL.md using the existing lake architecture. Each block is a new ingestion collector → silver table → feature builder → feature registry entries → validation via single-factor IC diagnostics.

**4B-1: Valuation / Size / Turnover (highest priority)**

*Source:* Tencent Finance API (`tencent_quote`) — not susceptible to IP banning, returns PE_TTM, PB, total market cap (亿), float market cap (亿), and turnover rate (换手率%). All fields are daily, post-close safe, same-day.

*New features (cross-sectional):*
- `cs_log_mcap` — log of float market cap (size factor)
- `cs_pe_ttm_rank` — cross-sectional rank of PE_TTM (value)
- `cs_pb_rank` — cross-sectional rank of PB
- `cs_turnover_rank` — cross-sectional rank of turnover rate (liquidity/attention)
- `cs_pe_ttm_zscore`, `cs_pb_zscore`, `cs_turnover_zscore`

*PIT safety:* Market cap and turnover are same-day price-derived — no announcement lag. PE_TTM uses trailing earnings already disclosed; no forward leakage. Join on daily date.

*Integration:* New `ingest/valuation_collector.py` using `tencent_quote` batch pull across the universe. Silver table: `silver/valuation/{symbol}.parquet`. New `features/valuation.py` builder. Entries in `FeatureRegistry`.

**4B-2: Industry Classification (enables industry-neutral labels)**

*Source:* Eastmoney `eastmoney_concept_blocks()` (individual stock → all board/concept tags) + `eastmoney_stock_info()` (industry field). Near-static — monthly refresh is sufficient.

*New features:*
- `industry_code` — Eastmoney industry code (integer, used for groupby)
- `industry_name` — human-readable label
- `sector_return_fwd_5d` — forward return of the industry ETF / average over the window (for excess-vs-industry label construction)

*Integration:* `ingest/industry_collector.py`. Silver table: `silver/industry_map.parquet` (universe-level, one row per (symbol, effective_date)). Feature join is a simple merge.

*Label unlock:* Once industry is available, add `excess_vs_industry_5d = ret_fwd_5d - sector_ret_fwd_5d` as a primary label. This strips sector beta from the training target and is the single most impactful label change available for A-share cross-sectional models.

**4B-3: Capital Flow (primary short-horizon signal)**

*Source:* Eastmoney `stock_fund_flow_120d()` (daily 120-day history: main/super-large/large/medium/small net inflow) via `push2his`. Rate-limited to ~1 request/second via existing `em_get()` pattern.

*New features (all cross-sectional, dimensionless):*
- `cs_main_flow_rank_5d` — rank of 5-day cumulative main-force net inflow (normalised by float market cap)
- `cs_main_flow_rank_1d` — rank of prior-day main net inflow / float mcap
- `cs_small_flow_rank_1d` — rank of retail (small order) net inflow (contrarian signal)
- `cs_flow_reversal_5d` — 1-day main inflow rank minus 5-day rank (short-term reversal within flow)

*PIT safety:* Flow data is released after market close for the same day. Features at date T use flow data through date T only.

*Integration:* Extend `ingest/ohlcv_collector.py` or create `ingest/flow_collector.py`. Silver table: `silver/fund_flow/{symbol}.parquet`. New `features/flow.py` builder. Single-factor IC diagnostic before adding to the joint model.

**4B-4: Margin Trading (leverage sentiment, secondary)**

*Source:* Eastmoney `margin_trading()` daily: 融资余额 (margin balance), 融资买入额 (margin buy), 融券余额 (short balance).

*New features:*
- `cs_margin_balance_change_5d` — 5-day rate of change in margin balance / float mcap (leverage momentum)
- `cs_rzrq_ratio_rank` — cross-sectional rank of total margin balance / float mcap (leverage level)

*PIT safety:* Released next trading day; use with a 1-day lag.

**4B-5: Excess-vs-CSI300 label (cheap, high value)**

*Source:* CSI 300 index forward return — already collectible via `tencent_quote(["000300"])`. No additional endpoint needed.

*Implementation:* In `labels/builder.py`, after computing `ret_fwd_{h}d`, subtract the cross-sectional median (approximate index return) or the actual index return to get `excess_vs_csi300_{h}d`. This is the recommended primary label going forward; it strips market beta from the training target.

---

### Phase 4C — Factor Diagnostics and Short-Horizon Refinements

**Duration estimate:** 2–3 weeks, can be parallelised with 4B.

**Goal:** Rigorous single-factor IC analysis of all existing and new features, pruning of collinear technicals, and refinements specifically targeting short-horizon (1–5d) performance.

**4C-1: Single-factor IC diagnostic framework**
For each feature in the registry, compute: (a) mean daily Rank IC vs 1d, 5d, and 20d labels; (b) IC t-statistic; (c) IC decay curve (IC vs 1d, 2d, 3d, 5d, 10d, 20d horizons); (d) pairwise Spearman correlation matrix for collinearity clustering. Output: a ranked feature summary table. Features with |mean Rank IC| < 0.01 and t-stat < 1.5 at all horizons are candidates for pruning. This diagnostic runs after each new data block is integrated.

**4C-2: Collinearity pruning for the 27 technicals**
Run the collinearity clustering on the existing 27 technical indicators. Group features with pairwise rank correlation > 0.85 and retain one representative per cluster. Expected outcome: the 27 reduce to ~10–12 genuinely independent signals, reducing overfitting surface without signal loss.

**4C-3: Residualised return label**
After industry integration, compute the daily residual return by regressing each day's cross-section of returns against market return, industry dummies, and log-size. The residual is the label used by professional factor desks. This requires industry features to exist (4B-2 must complete first). Medium implementation effort, highest generalization payoff.

**4C-4: Lockup expiry feature (PIT-safe short-horizon event)**
From SKILL.md `lockup_expiry()`: upcoming lockup expiry dates are public knowledge at all prior dates (they are announced at IPO/placement). A feature `days_to_next_unlock` and `unlock_size_ratio` (shares unlocking / float shares) is legitimately PIT-safe and carries event-driven short-horizon signal (supply pressure). This is the one "deferred" data source from the analysis that is worth collecting cheaply.

**4C-5: Walk-forward regime analysis**
Use the walk-forward evaluator from 4A-1 to compute Rank IC per rolling window for each feature group independently. Map IC over time to identify: (a) which features are regime-stable vs regime-sensitive; (b) which windows are "hard regimes" (low IC across features); (c) whether ensemble performance degrades uniformly or in feature-group-specific patterns. This informs future feature selection and model architecture decisions.

---

## 6. SKILL.md Data Source Integration Map

The SKILL.md toolkit provides 28 verified endpoints organised in 7 layers. The following table maps each to its integration priority for this project:

| Layer | Data | Short-horizon value | PIT safety | Integration phase | Notes |
|---|---|---|---|---|---|
| Layer 1: 腾讯财经 | PE/PB/市值/换手率 | High | Easy (same-day) | **4B-1 (priority 1)** | Not IP-banned; batch-safe |
| Layer 3: 东财 slist | Industry/concept classification | High (enabler) | Easy (slow-moving) | **4B-2 (priority 1)** | Rate-limited; monthly refresh |
| Layer 4: push2his fund flow | Main/super/large/small net flow | **Very high** | Easy (post-close) | **4B-3 (priority 1)** | 120-day history; key short-horizon signal |
| Layer 4: 融资融券 | Margin balance/buy/short | Moderate | 1-day lag | **4B-4 (priority 2)** | Daily, rate-limited |
| Layer 3: 限售解禁 | Lockup expiry calendar | Moderate (event-driven) | PIT-safe (dates public) | **4C-4 (priority 2)** | Forward-looking dates are safe |
| Layer 3: 行业板块排名 | Sector-level daily return/momentum | Moderate | Same-day | **4B-2 supplement** | Sector rotation signal |
| Layer 3: 龙虎榜 | Dragon-tiger board records | Low-moderate | Post-close | **Defer** | Sparse, event-driven; low coverage |
| Layer 6: mootdx finance | Quarterly financial snapshot (ROE/EPS) | Low (slow) | Announce date join | **Defer to fundamentals phase** | PIT discipline required |
| Layer 6: 新浪三表 | Balance sheet / P&L / cash flow | Low (slow) | Announce date join | **Defer to fundamentals phase** | Use existing fundamentals collector |
| Layer 2: 东财研报 | Research report metadata/ratings | Low (uncertain at 5d) | Announce date | **Defer** | High integration cost |
| Layer 2: 同花顺热点 | Hot stock reasons/tags (text) | Uncertain | Same-day | **Defer** | NLP required for features |
| Layer 3: 北向资金 | Northbound flow (HGT/SGT) | Low-moderate | Same-day | **Defer** | Upstream断供 since 2024-08 |
| Layer 5: News | Individual stock news | Uncertain (NLP) | High risk | **Defer indefinitely** | Highest leakage risk |

---

## 7. Priority Ranking

| Priority | Action | Rationale | Unlocks |
|---|---|---|---|
| **P0** | Fix `embargo=5→horizon` in robustness suite | One-line fix; currently produces misleading ablation deltas | Trustworthy feature pruning |
| **P0** | Read and surface subperiod stability numbers | Already computed; free information about regime stability | Understand OOF/lockbox gap |
| **P1** | Walk-forward OOS evaluator (4A-1) | Platform cannot measure improvement without this | All subsequent changes measurable |
| **P1** | Add 5d / 10d labels (4A-2) | Near-zero cost; 4× evaluation power; natural short-horizon labels | 50 independent periods/year at 5d |
| **P1** | Excess-vs-CSI300 label (4B-5) | Strips market beta; cheap; no new data needed | Cleaner cross-sectional signal |
| **P2** | Valuation + industry integration (4B-1, 4B-2) | Highest generalization upside; enables industry-neutral labels | Industry-excess label, size/value factors |
| **P2** | Ridge diagnostic baseline (4A-4) | Detects whether GBM is overfitting nonlinear noise | Trustworthy model comparison |
| **P3** | Capital flow integration (4B-3) | Primary short-horizon orthogonal signal | New information axis |
| **P3** | Single-factor IC diagnostic (4C-1) | Required before feature expansion increases collinearity | Principled feature selection |
| **P4** | Collinearity pruning of 27 technicals (4C-2) | Reduces overfitting surface without signal loss | Cleaner model |
| **P4** | Margin trading data (4B-4) | Secondary short-horizon signal | Leverage sentiment axis |
| **P5** | Excess-vs-industry label (after 4B-2) | Strips sector beta; single biggest label improvement after industry | Most robust cross-sectional target |
| **P5** | Lockup expiry feature (4C-4) | PIT-safe event feature; supply-pressure signal | Event-driven short-horizon alpha |
| **P5** | Residualised return label (4C-3) | Professional-grade factor target | Highest generalisation |
| **Defer** | HPO (Optuna), deep models, more technicals | Information-poor feature set; evaluation instrument insufficient | — |
| **Defer** | News/NLP, dragon-tiger, margin/block trades | High cost, uncertain short-horizon payoff, or sparse coverage | — |

---

## 8. Risks and Assumptions

### Data source risks

**AKShare instability (ongoing):** AKShare is a wrapper over public endpoints that added anti-scraping in early 2026. The SKILL.md toolkit addresses this by moving to direct HTTP connections (`em_get()`, `tencent_quote()`, `tdx_client()`). The platform should progressively migrate new collectors to the SKILL.md endpoint patterns rather than AKShare wrappers.

**Eastmoney rate limiting:** All eastmoney.com endpoints have documented thresholds (>5 req/sec triggers banning; 1-minute total ≥200 triggers banning). Batch collection of 300 stocks at 1 req/second takes ~5 minutes per endpoint — manageable, but concurrent requests must be avoided. The `em_get()` throttle must be respected in all new collectors.

**mootdx overseas restriction:** TCP 7709 to TDX servers is blocked from non-mainland IPs. If the platform runs on cloud infrastructure outside mainland China, mootdx-based collection will fail. Fallback to Tencent Finance API (not IP-restricted) must be in place for all OHLCV data.

**Northbound fund flow historical gap:** Eastmoney's northbound flow data has been returning NaN/0 for net amounts since August 2024 (upstream断供). The SKILL.md self-cache mechanism captures current data but historical series are sparse. This source should not be relied upon for backtesting features that require historical depth.

### Methodology risks

**Walk-forward overfitting:** Sequential walk-forward windows share the same training procedure. If feature selection or model hyperparameters are tuned against aggregated walk-forward metrics, the walk-forward result is no longer out-of-sample. The pre-registration principle must extend to walk-forward configuration (window size, step size) — these parameters must be fixed before the first run and not changed based on results.

**Regime non-stationarity:** The in-period signal (OOF ~0.09) sits in a specific macro regime (the train/val span). A-share markets have experienced regime shifts (regulatory changes, circuit breakers, margin rule changes, registration-based IPO reform). Short-horizon signals are more vulnerable to regime shifts than fundamental signals. The walk-forward decay analysis (4C-5) is the primary tool for detecting regime-driven IC collapse.

**Industry-neutral label construction:** Computing `sector_ret_fwd_5d` requires knowing which stocks were in each industry on each historical date. If the industry classification is scraped as a point-in-time snapshot rather than a historical series, the industry membership used for historical labels will be contaminated by forward information. The industry collector must store `(symbol, industry_code, effective_date, out_date)` in the same pattern as the universe membership table.

**Cost model underestimation:** The current backtest uses 10 bps one-way. For a 5-day holding period with a portfolio rebalance, realistic A-share costs (stamp duty 0.1% sell, commission ~0.03%, market impact ~0.05–0.2% depending on liquidity) can easily exceed 0.3–0.5% round-trip. At a 5d horizon with gross spread of ~0.3–0.5%, cost is the dominant P&L driver. The cost model must be calibrated more carefully as the horizon shortens.

**Multiple comparisons at 4B data integration:** Each new data block added to the model is an implicit hypothesis test. The research ledger's deflated ICIR (Bailey–López de Prado) is the correct instrument for tracking the effective number of trials. The pre-registration protocol must be applied to each new data block integration: define the hypothesis (this data block will improve forward Rank IC by ≥X), run once, record.

### Assumptions

- The CSI 300 universe remains the primary evaluation universe. Any expansion (CSI 500, ChiNext) requires revisiting the survivorship bias controls and universe service.
- The Parquet medallion lake can absorb new data blocks as additional silver tables without structural change.
- The `announce_date` PIT discipline applied to fundamentals can be extended to all new data sources using the same join pattern.
- Walk-forward windows of 6–12 months are computationally feasible with the current LightGBM training times.

---

## 9. Suggested Evolution Path

### Near-term (1–3 months): Close the measurement gap

The immediate goal is to reach a state where the platform can for the first time answer "does this feature set have forward predictive power?" with a statistically meaningful answer. This requires only Phase 4A plus the two cheap label additions. No new data sources are needed; no model changes are needed.

Expected state: walk-forward evaluation with 60+ independent 5d forward periods, Ridge and LightGBM baselines, subperiod stability report, excess-vs-CSI300 label active.

### Medium-term (3–6 months): Build the information base

With a reliable measuring instrument, integrate the three priority data blocks (valuation/size/turnover, industry, capital flow). Run single-factor IC diagnostics before each integration. Prune the collinear technical features. Activate excess-vs-industry labels. Each data integration is a structured research experiment: pre-register the hypothesis, integrate, measure, record in the ledger.

Expected state: 40–60 feature dimensions across 4 orthogonal information axes (technical, valuation, industry, flow), industry-neutral labels, walk-forward forward Rank IC expected to reach 0.04–0.07 at 5d horizon.

### Long-term (6–12 months): Signal stability and strategy readiness

With a stable information base and reliable evaluation, shift attention to: (a) cost-efficient execution simulation for 1–5d strategies; (b) regime-conditional model variants (separate models for bull/sideways/bear market regimes identified from the walk-forward analysis); (c) residualised return as the primary label; (d) feature drift monitoring and automated retraining triggers.

The platform is, by design, a learning and research tool rather than a trading system. The evolution path above is calibrated to produce the most rigorous understanding of A-share short-horizon predictability — which is itself the core objective.

---

*This document should be treated as a living specification. Each completed task in the task.md should trigger a review of the assumptions and priority ranking here.*
