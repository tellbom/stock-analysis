# From Report Generator to Quant Research Platform — Architecture Evaluation

*A review of the existing AKShare-based stock-analysis project and a roadmap for evolving it into a learning-oriented quantitative research and prediction system. Goal: understand how market data becomes features, how predictive models are trained and evaluated, and how probabilities, risk scores, and explainable insights are produced — not automated trading.*

---

## 1. Where you are today

The current project is a well-built **single-stock data-to-HTML pipeline** in three parts:

- **`stock_full_report.py`** — a Phase-1 collector. For one ticker it pulls ~13 categories of data from AKShare (K-line, intraday, fund flow, dragon-tiger list, three financial statements, earnings pre-announcements, shareholder structure, dividends/unlocks, announcements/news/research, ratings/fund holdings, margin trading), computes technical indicators inline, and dumps everything to `output/data_{code}.json`.
- **`technical_indicators.py`** — clean, dependency-light functions for MA / MACD / KDJ / RSI / BOLL, plus a `get_latest_signals()` rule layer and a `format_indicators_for_json()` serializer.
- **`update_stock_report.py`** — an incremental updater that re-fetches K-line, diffs it against the data embedded in an existing HTML, and patches the K-line array, hero quote, and the chapter-9 technical block.

It's a solid foundation. The data-source knowledge encoded here (which AKShare endpoint returns what, how to filter market-wide tables down to one code, retry/back-off discipline) is the genuinely hard-won part and it transfers directly. What has to change is everything downstream of "fetch one stock": the **storage model, the unit of work, and the absence of a feature/label/model/evaluation loop.**

### The three structural limits to fix

1. **Unit of work is one stock.** A predictive model needs a *universe* — hundreds to thousands of stocks, collected on a schedule, kept consistent.
2. **Storage is JSON-per-stock.** Fine for rendering one page; unworkable for "load 5 years × 300 stocks × 40 features into a training matrix." You need a columnar store.
3. **The pipeline ends at presentation.** There is no feature layer, no labels, no dataset assembly, no model, no evaluation. Those are the new layers.

---

## 2. The conceptual shift (and the concepts that actually matter)

Before tooling, the mental model. A research platform reframes the problem from *"describe this stock"* to *"given everything knowable at time T, predict something about T+h, then measure whether the prediction had signal."* Most of the difficulty — and most of the learning — is in doing that **without fooling yourself.** These five concepts are what separate a real quant pipeline from a backtest that looks amazing and means nothing:

- **Lookahead bias / data leakage.** Any feature at time T that secretly contains information from after T will inflate results and vanish in reality. The most common sources: using a financial figure on its *report-period* date instead of its *announcement* date, computing a normalization (z-score, min/max) over the whole dataset before splitting, and label/feature misalignment.
- **Point-in-time (PIT) correctness.** Fundamentals are known to the market only when *announced*, often weeks after the period they describe. Your store needs both a `period` and a `known_at` timestamp, and features must be built using `known_at`.
- **Survivorship bias.** If your universe is "stocks that exist today," you've silently deleted every company that delisted or blew up. Predictions trained on survivors are optimistic. You need a universe that includes historical constituents and delisted names.
- **Cross-sectional vs. time-series framing.** Classic equity ML is *cross-sectional*: on each day, rank all stocks by predicted forward return, and measure whether the ranking is right. This is a very different (and usually more robust) target than predicting one stock's price level.
- **The right metric.** Accuracy is nearly useless here. The field standard is the **Information Coefficient (IC)** — the correlation between predicted scores and realized forward returns, measured per-day and averaged — plus **Rank IC**, **ICIR** (IC mean / IC std), and **quantile spread** (return of top-decile minus bottom-decile predictions). A model with 52% directional accuracy but a stable positive Rank IC can be genuinely useful; a model with 70% accuracy concentrated in a few days may be noise.

If you internalize only one thing from this document, make it this section. Every tool below is in service of getting these right.

---

## 3. Target architecture

A layered pipeline, each layer independently runnable and testable, data flowing one direction:

```
                         ┌─────────────────────────────────────────────┐
   AKShare / sources ──▶ │ 1. INGESTION    batch collector + scheduler  │
                         │                 (your collector, generalized) │
                         └───────────────────────┬─────────────────────┘
                                                 ▼
                         ┌─────────────────────────────────────────────┐
                         │ 2. STORAGE      Parquet lake + DuckDB        │
                         │                 partitioned by date/symbol,   │
                         │                 PIT-aware, raw + curated      │
                         └───────────────────────┬─────────────────────┘
                                                 ▼
                         ┌─────────────────────────────────────────────┐
                         │ 3. FEATURES     technical + cross-sectional + │
                         │                 fundamental factors           │
                         └───────────────────────┬─────────────────────┘
                                                 ▼
                         ┌─────────────────────────────────────────────┐
                         │ 4. LABELS       forward returns, triple-      │
                         │                 barrier, quantile buckets      │
                         └───────────────────────┬─────────────────────┘
                                                 ▼
                         ┌─────────────────────────────────────────────┐
                         │ 5. DATASET      feature/label join, time-     │
                         │                 based splits, no leakage       │
                         └───────────────────────┬─────────────────────┘
                                                 ▼
                  ┌──────────────┬───────────────┴───────────────┬──────────────┐
                  ▼              ▼                                ▼              ▼
            6. TRAINING   7. EVALUATION                    8. EXPLAIN     9. BACKTEST
            LightGBM +    IC / RankIC / ICIR /             SHAP           vectorbt /
            Optuna HPO    quantile spread / calibration                   Qlib backtester
                  │              │                                │              │
                  └──────────────┴────────────┬───────────────────┴──────────────┘
                                              ▼
                              10. TRACKING + REGISTRY (MLflow)
                              experiments, metrics, artifacts, model versions
                                              ▼
                              11. PREDICTION + ANALYSIS
                              daily scores, probabilities, risk flags,
                              per-stock explanations → your existing HTML report
```

The closing loop is the nice part: **layer 11 feeds back into the report generator you already have.** The HTML report stops being a static description and becomes "here is the model's score for this stock, its probability of outperforming over the next 20 days, its risk flags, and the three features driving the call." That's the continuity bridge between the old project and the new one.

---

## 4. Layer-by-layer build, with tooling

### Layer 1 — Ingestion (batch collection)

Generalize the existing collector from one code to a **universe**. Add:

- A **universe definition** — start with an index constituent list (CSI 300 / CSI 500 via `ak.index_stock_cons_csindex` or similar), stored with effective dates so you can reconstruct *who was in the index on a past date* (the survivorship fix).
- A **driver loop** with rate limiting and a **manifest/catalog** recording, per symbol per source, the last successful fetch and the latest date retrieved — so re-runs are incremental, exactly like `update_stock_report.py` does for one stock, generalized.
- **Caching and idempotency** so a crashed run resumes cheaply. AKShare endpoints are flaky; your `_safe_call` back-off pattern is the right primitive — lift it into the driver.
- Keep **raw responses** separately from curated/normalized tables. When you later discover a parsing bug you'll want to reprocess without re-hitting the network.

Orchestration: start with a `Makefile` or plain scripts. When the DAG grows (ingest → validate → featurize → label → train), graduate to **Prefect** or **Dagster** for retries, scheduling, and lineage. Don't reach for them on day one.

### Layer 2 — Storage

Replace JSON-per-stock with a **Parquet data lake** partitioned by `date` and/or `symbol`, queried with **DuckDB**. This combination is the modern default for single-machine quant research: Parquet is columnar and compresses well; DuckDB runs SQL directly over Parquet files with no server, and loads a multi-year, multi-stock training matrix in seconds. Schema discipline to adopt now:

- One row per `(symbol, date)` for price/volume and derived features.
- Separate **PIT tables** for fundamentals carrying both `period_end` and `announce_date`.
- A **corporate-actions / adjustment** table so you can switch between adjusted and raw prices deliberately rather than by accident.

(If you later adopt Qlib, it expects its own binary format; you'd add a one-time exporter from Parquet to Qlib bins. More on that in §5.)

### Layer 3 — Feature engineering

Three families, in increasing sophistication:

1. **Technical indicators** — your `technical_indicators.py` already does MA/MACD/KDJ/RSI/BOLL correctly and transparently; **keep it.** For breadth (ATR, OBV, ADX, Stochastics, Ichimoku, 150+ others) add a library rather than hand-rolling each:
   - **TA-Lib** — C-backed, fast, the industry reference; install friction (needs the C library compiled) is the only downside.
   - **pandas-ta-classic** — the actively maintained successor to the original `pandas-ta` (whose upstream repo was pulled from GitHub); pure-Python, easy install, DataFrame-native, releases ongoing into 2026. Good default if you want zero compilation.
2. **Cross-sectional features** — on each date, rank/z-score every stock's raw features against the universe (e.g., "this stock's 20-day momentum percentile today"). This is where most equity predictive signal lives and it's cheap to compute with a DuckDB window query or pandas `groupby('date')`.
3. **Automated / fundamental features** — **tsfresh** (automatic extraction of hundreds of time-series features with built-in significance filtering) or **TSFEL** for systematic feature mining; plus ratios and growth rates from the PIT fundamentals you're already collecting.

A note worth holding onto: Qlib's **Alpha158 / Alpha360** feature sets are open, documented, expression-based factor definitions — even if you don't adopt Qlib, reading how they're constructed is one of the best free curricula in practical factor design.

### Layer 4 — Label construction

The target variable. Build several and treat the choice as a first-class experiment:

- **Forward returns** over horizons (1/5/20 days), computed strictly from *future* prices and aligned so no label leaks into its own features.
- **Quantile / rank labels** — bucket each day's forward returns into deciles; predict the bucket. Pairs naturally with cross-sectional features and IC evaluation.
- **Triple-barrier labeling** (from de Prado's *Advances in Financial Machine Learning*) — label by which barrier a path hits first: profit-take, stop-loss, or time limit. More realistic than fixed-horizon returns and a great learning exercise. Reference implementation: **mlfinlab** (note its open-source version is partially frozen/commercialized, but the book + code teach the method).
- **Risk labels** — forward realized volatility or max drawdown, for the "risk score" output you want.

### Layer 5 — Dataset assembly & splits

Where leakage is won or lost:

- **Time-based splits only.** Train on the past, validate on the middle, test on the most recent untouched slice. Never random k-fold across time.
- **Purged k-fold with embargo** for cross-validation: drop training samples whose label windows overlap the validation period, and embargo a gap after each fold so autocorrelation doesn't leak. This is the de Prado technique and it's the single biggest credibility upgrade over naive CV.
- **Fit all transforms on train only** — scalers, quantile bins, imputation values — then apply to validation/test. Leaking the normalization is the most common subtle bug.
- Handle **delisted symbols and NaNs** explicitly; a stock that stops trading is information, not a row to silently drop.

### Layer 6 — Model training

Start boring and strong: **gradient-boosted trees** are the workhorse of tabular financial ML.

- **LightGBM** (fast, handles missing values, the common default), **XGBoost**, or **CatBoost**. Begin with LightGBM as a ranking or regression objective on cross-sectional features.
- Wrap everything in **scikit-learn** `Pipeline`/`ColumnTransformer` so preprocessing travels with the model and can't leak.
- Defer deep learning. When you're ready, **darts**, **neuralforecast**, and **PyTorch Forecasting** cover sequence models, and Qlib ships LSTM/GRU/Transformer/TFT/TRA implementations — but a clean GBM baseline beats a sloppy neural net almost every time, and you'll learn more by understanding why the baseline works first.

### Layer 7 — Evaluation

Build a dedicated metrics module — this is the heart of "understanding how predictive models are evaluated":

- **IC, Rank IC, ICIR** per the §2 definitions.
- **Quantile spread returns** — group predictions into deciles, plot mean forward return per decile; a monotone staircase is the goal.
- **Precision@k / hit rate** on the top bucket — closest to "did the calls I'd actually act on work."
- **Calibration** — bin predicted probabilities and check whether predicted 60% actually realizes ~60%. Use `sklearn.calibration` (reliability curves, `CalibratedClassifierCV`). This is what turns raw model scores into trustworthy probabilities for your report.

### Layer 8 — Explainability

- **SHAP** is the standard. `TreeExplainer` on a LightGBM/XGBoost model is fast and exact, giving global feature importance *and* per-prediction attributions — exactly the "why did the model flag this stock" you want surfaced in the report. Add **ELI5** or permutation importance as a sanity cross-check.

### Layer 9 — Backtesting

To answer "does this signal actually have predictive value over time," not to trade:

- **vectorbt** (open-source) — vectorized, NumPy/Numba-fast, made for sweeping thousands of parameter/signal combinations in seconds; ideal for research-grade signal evaluation. (A paid "PRO" successor exists with more realistic fills; the free version is plenty to learn on.)
- **backtesting.py** — minimal, single-asset, define a strategy in two methods and get a report; best for quick prototypes and teaching the mechanics.
- **Qlib's built-in backtester** — natively cross-sectional and A-share aware; the most natural fit for ranking-model evaluation if you go the Qlib route.
- Always include **transaction costs and a realistic delay** (trade on T+1 open, not T close). A signal that only works with zero costs and instant fills isn't a signal.

### Layer 10 — Experiment tracking, HPO, model registry

- **MLflow** — open-source, self-hostable; logs parameters, metrics, and artifacts, and its model registry versions your models. The backbone of "continuous learning and model iteration." (**Weights & Biases** is a hosted alternative with a generous free tier; **Aim** is another OSS option.)
- **Optuna** — the modern hyperparameter optimizer: define a search space, let it prune bad trials early, persist studies to resume later. Integrates cleanly with LightGBM and MLflow. (**Ray Tune** if you outgrow a single machine.)
- **DVC** — version your *datasets* alongside your code, so an experiment is reproducible end to end.

### Layer 11 — Prediction & analysis (closing the loop)

The daily inference job: load the latest data, build features with the *same* pipeline used in training (reuse, never re-implement — leakage hides in the gap), score the universe, and emit per-stock **score, calibrated probability, risk flags, and top SHAP drivers.** Feed that straight into your existing HTML report so each stock page gains a model-opinion section. This is where the old and new systems become one product.

---

## 5. The big decision: build modular, or adopt Qlib?

Microsoft's **Qlib** (MIT-licensed, ~37k★, actively developed into 2026, now paired with **RD-Agent** for automated research) is almost exactly the system you're describing: a full ML pipeline — data → features → model → backtest — covering alpha seeking, with first-class **A-share support**, a 40+ model zoo, the Alpha158/360 feature sets, and `qrun` one-command workflows. Independent assessments are consistent: it is the premier *research* platform for ML-driven quant, explicitly **not** production trading infrastructure — which is a perfect match for your stated learning (not trading) goal.

So why not just adopt it wholesale? Because the learning objective changes the calculus:

| | Build modular (your stack) | Adopt Qlib |
|---|---|---|
| **You learn** | how each layer works internally — the most valuable outcome for your stated goal | how a mature platform is *organized*; less about the internals |
| **Effort** | higher; you write the dataset/eval/training glue | lower to a first result; higher to bend it to custom data |
| **Flexibility** | total | high within its conventions, friction outside them |
| **Data fit** | your AKShare data flows in directly | needs a one-time export to Qlib's binary format |
| **A-share readiness** | you build it | native |

**Recommended path: a hybrid.** Build the modular pipeline (§4) for everything through evaluation — that's where the understanding lives, and your existing code slots in. In parallel, **install Qlib and run its A-share LightGBM example end to end as a reference implementation and benchmark.** Read its dataset handlers and Alpha158 definitions as a curriculum. Later, optionally add a Parquet→Qlib exporter so you can borrow its backtester and model zoo. You get the deep understanding *and* a battle-tested yardstick to check your own numbers against. If your numbers wildly beat Qlib's, that's usually a leakage bug, not genius — having the benchmark is itself a safeguard.

---

## 6. What to reuse from the current codebase

| Existing asset | Verdict | Role in the new platform |
|---|---|---|
| `technical_indicators.py` (all indicator fns) | **Reuse as-is** | Core of the technical-feature family; transparent and correct |
| `get_latest_signals()` | **Reuse, reframe** | Rule-based features *and* an interpretable baseline to beat |
| `format_indicators_for_json()` / `_df_to_records()` | **Reuse** | Serialization for the report layer (layer 11) |
| `detect_market()` | **Reuse** | Symbol/market normalization, needed everywhere |
| `_safe_call()` retry/back-off | **Reuse, promote** | Lift into the batch ingestion driver as the standard fetch wrapper |
| `fetch_latest_kline()` + incremental diff logic | **Reuse, generalize** | The per-symbol incremental-update pattern becomes the universe-wide catalog logic |
| AKShare endpoint knowledge (the 13 blocks) | **Reuse — this is the crown jewel** | The map of which source yields which data; hardest to rebuild |
| `output/data_{code}.json` storage | **Retire** | Replace with the Parquet/DuckDB lake |
| HTML embedding / `update_html_*` | **Keep, downstream** | Becomes the presentation layer fed by predictions (layer 11) |

You are reusing more than you're discarding. The collector's *knowledge* survives; only its *storage and unit of work* change.

---

## 7. Pitfalls that will quietly ruin results

A consolidated checklist — pin this somewhere:

1. **Normalize after splitting, never before.** Fit scalers/bins on train only.
2. **Use `announce_date`, not `period_end`,** for every fundamental feature.
3. **Trade on T+1, with costs.** Signal-on-close/trade-on-close is fantasy.
4. **Reconstruct historical universes** to dodge survivorship bias; keep delisted names.
5. **Align labels to features** so a 20-day-forward label never overlaps its own feature window without purging.
6. **Purge + embargo** your cross-validation.
7. **One feature pipeline, shared** between training and live prediction — re-implementing it twice is how leakage sneaks back in.
8. **Benchmark against a trivial baseline** (yesterday's return, sector mean). Beating nothing is not signal.
9. **Watch capacity/turnover** — a signal that requires churning the whole book daily is fragile even if IC is high.

---

## 8. Phased roadmap

**Phase 0 — Foundation (storage + universe).** Define a CSI 300 universe with effective dates; stand up the Parquet lake + DuckDB; port the collector to batch with the catalog/incremental logic. Outcome: 3–5 years of clean OHLCV for the universe, queryable in seconds.

**Phase 1 — Features + labels.** Wire `technical_indicators.py` into a feature builder; add cross-sectional ranks and a TA library for breadth; build forward-return and quantile labels with strict alignment. Outcome: a leakage-checked `(symbol, date, features…, label)` table.

**Phase 2 — First model + honest evaluation.** LightGBM baseline; the IC/RankIC/quantile/calibration metrics module; purged-CV; MLflow tracking. Run Qlib's A-share example as a benchmark in parallel. Outcome: a number you trust, checked against an external yardstick.

**Phase 3 — Iteration loop.** Optuna HPO; SHAP explanations; vectorbt signal backtest with costs and T+1. Outcome: a repeatable train→evaluate→explain→backtest cycle with versioned experiments.

**Phase 4 — Close the loop.** Daily prediction job → scores, calibrated probabilities, risk flags, SHAP drivers → injected into the existing HTML report. Outcome: the report generator reborn as a prediction-analysis surface.

**Phase 5 — Depth (optional).** Triple-barrier/meta-labeling; sequence models (darts / Qlib's neural zoo); Prefect/Dagster orchestration; DVC dataset versioning.

---

## 9. Quick tool reference

| Need | Primary pick | Alternatives / notes |
|---|---|---|
| Full reference platform | **Qlib** (MIT, A-share native) | benchmark + curriculum; pair with RD-Agent later |
| Columnar storage / query | **Parquet + DuckDB** | the single-machine research default |
| Technical indicators | **your `technical_indicators.py`** + **TA-Lib** | **pandas-ta-classic** for no-compile install |
| Automated features | **tsfresh** | TSFEL; featuretools |
| ML model | **LightGBM** | XGBoost, CatBoost; scikit-learn pipelines |
| Sequence/DL (later) | **darts** | neuralforecast, PyTorch Forecasting, Qlib zoo |
| Labeling techniques | **mlfinlab** (+ de Prado's book) | triple-barrier, meta-labeling |
| Evaluation metrics | **custom IC/RankIC module** | sklearn.calibration for probabilities |
| Explainability | **SHAP** (TreeExplainer) | ELI5, permutation importance |
| Experiment tracking | **MLflow** | Weights & Biases, Aim |
| Hyperparameter search | **Optuna** | Ray Tune at scale |
| Data versioning | **DVC** | — |
| Backtesting | **vectorbt** | backtesting.py (simple); Qlib backtester (cross-sectional) |
| Orchestration (later) | **Prefect / Dagster** | start with Make/scripts |

---

## 10. Recommended first concrete step

Don't start by training a model. Start by **standing up Phase 0 and proving you can load a leakage-free `(symbol, date, MA/MACD/RSI…, forward_return_20d)` matrix for the full CSI 300 across several years in one DuckDB query** — built entirely from your existing collector and `technical_indicators.py`. Everything else (models, HPO, SHAP, backtests) is comparatively easy and well-supported once that clean, correctly-aligned dataset exists. The dataset is the product; the model is a consumer of it.

---

*A closing note on expectations, since the goal is genuine understanding: public data plus simple models can demonstrably produce real, measurable signal — that's exactly what makes this a great learning vehicle — but that signal is typically small, unstable, and easily destroyed by costs and overfitting. The skill you're building is less "find the model that predicts the market" and more "construct an evaluation honest enough that you can tell signal from self-deception." That discipline is the transferable asset.*
