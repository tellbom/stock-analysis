# P0 — Data Foundation & P1 — Features & Labels: Plan and Tasks

*Execution plan for the first two phases of the quant research platform. Companion to the architecture evaluation. No implementation here — this is a plan for review. Guiding principle, which we share: **the dataset is the product; the model is a consumer.** Therefore P0 and P1 carry most of the project's real value and most of its risk.*

---

## Orientation: what these two phases must produce

By the end of P1 you should be able to run one DuckDB query and get back a clean panel:

```
(symbol, date, feature_1 … feature_n, label_1 … label_k)
```

…for a defined universe across several years, where **every feature at date T uses only information knowable at T, and every label uses only the future** — verified by an automated leakage harness, not by hope. P2 (models) becomes almost mechanical once this panel exists and is trustworthy. If P0/P1 are wrong, every model number downstream is fiction.

Two reference points shaped the design decisions below, both worth internalizing:

- **Qlib's data layout** is a proven blueprint we mirror without adopting Qlib wholesale: an `instruments` registry where each symbol carries a `start_date/end_date` membership span (the survivorship mechanism), a master `calendar`, and a **separate point-in-time (PIT) store** for fundamentals keyed by `date` (announce date) and `period` (reporting quarter) — you cannot query PIT data as a plain daily field, you convert it as-of. We adopt the same shapes in Parquet.
- **Qlib's label convention.** Alpha158 labels with `Ref($close,-2)/Ref($close,-1) − 1` — the return from **T+1 to T+2**, not from T's close — precisely because when you observe T's close in an A-share, the earliest you can buy is T+1 and sell T+2. This T+1 execution assumption is now a first-class rule in P1 labels.

---

## Phase P0 — Data Foundation

### Goal / Definition of Done

A **point-in-time-correct, survivorship-aware Parquet + DuckDB data lake** for an initial universe (CSI 300), fed by an **incremental, idempotent batch collector** that reuses the existing AKShare endpoint knowledge. Done when:

1. You can query adjusted **and** raw OHLCV for the full universe over ≥3 years in seconds via DuckDB.
2. The universe is reconstructable *as of any past date* (delisted/removed constituents included).
3. Fundamentals are stored with both `period_end` and `announce_date`.
4. A second collector run only fetches what changed (incremental), and is safe to crash and resume (idempotent).

### Key design decisions (for review)

- **Storage = Parquet lake + DuckDB, medallion layout.** Three zones: **bronze** (raw API responses, immutable, reprocessable), **silver** (normalized, typed, deduped tables), **gold** (curated, analysis-ready panels). DuckDB runs SQL directly over the Parquet files — no server. Partition price data by `symbol` and year. This replaces today's `output/data_{code}.json` entirely.
- **Universe as a membership table with effective dates** (`symbol, in_date, out_date`), mirroring Qlib's `instruments` model. Sourced from index-constituent history. *This is the survivorship-bias fix and it must exist before any cross-sectional feature in P1.*
- **A master trading-calendar table.** Needed for gap detection, forward-return shifting, and as-of joins. Don't derive dates from whatever happened to trade.
- **Adjustment handled explicitly.** Store **raw prices + adjustment factors** separately and derive qfq/hfq on demand. Today's code bakes in `adjust="qfq"`; that's convenient but lossy and the qfq series silently rewrites history every time a new dividend occurs. Storing factors makes adjustment a deliberate, reproducible choice.
- **PIT fundamentals store**, mirroring Qlib's `(date, period, value)` shape: `announce_date` is the row's timestamp, `period_end` identifies the quarter. Features in P1 join to this **as-of `announce_date`**, never `period_end`. Single most important correctness decision in the whole project.
- **Catalog/manifest-driven incremental collection.** A small table recording, per `(symbol, source)`, the last successful date and run status. The collector consults it and fetches only the tail — a generalization of the diff logic already in `update_stock_report.py`.
- **Source strategy: AKShare primary, fallback-ready.** Keep reusing the endpoint knowledge in `stock_full_report.py`, but design the fetch wrapper so a second source (e.g., Tushare) can slot in. Context worth noting: AKShare is a wrapper over EM/Sina/THS public endpoints and inherits their instability (some endpoints added anti-scraping in early 2026; pandas-version breakages occur). Concurrency should stay modest (≈8 workers) with backoff — your `_safe_call` pattern is already the right primitive.

### Task breakdown

| ID | Task | Reuses from current code | Reference | Done when |
|---|---|---|---|---|
| **T0.1** | Restructure flat scripts into a package (`core/`, `ingest/`, `store/`); lift shared helpers into a `core` module | `detect_market()`, `_safe_call()`, `_df_to_records()`, `_filter_by_code()` | — | Helpers importable, single source of truth, no duplication |
| **T0.2** | **Universe service** — build & persist CSI 300 membership with effective dates, including exited constituents | — | `ak.index_stock_cons_csindex`; Qlib `instruments` model | Can list "constituents as of 2022-06-30" and it differs from today |
| **T0.3** | **Trading-calendar table** | — | `ak.tool_trade_date_hist_sina`; Qlib `calendars/day.txt` | Calendar spans the full history; used by collector for gap checks |
| **T0.4** | **Storage layer** — Parquet schema, partitioning, DuckDB views; bronze/silver/gold zones | replaces `data_{code}.json` writer | DuckDB-over-Parquet patterns | A documented schema; `SELECT … FROM ohlcv` returns universe-wide data |
| **T0.5** | **Batch OHLCV collector** — generalize single-stock fetch across the universe; store raw + adjustment factors | `fetch_latest_kline()` (incl. retry), `detect_market()` | `ThreadPoolExecutor`(~8) batch pattern; `stock_zh_a_daily` (Sina, current) vs `stock_zh_a_hist` (EM, adds turnover) | Full-universe OHLCV in the lake, raw + factors |
| **T0.6** | **Catalog + incremental update** — manifest table; collector fetches only the tail; crash-safe | the HTML-diff incremental logic in `update_stock_report.py`, generalized | — | Second run fetches only new dates; interrupt/resume leaves no corruption |
| **T0.7** | **PIT fundamentals** — collect 3 statements + abstract; store with `announce_date` + `period_end` | the 13-block collection logic in `stock_full_report.py` (statements, abstract) | Qlib PIT `(date, period, value)` design | Fundamentals queryable as-of any date by announce timestamp |
| **T0.8** | **Data-quality validation** — calendar-gap, dedupe, NaN audit, adjustment sanity, type checks | — | assertions or a light validation lib | A validation report runs each load and flags anomalies |
| **T0.9** | *(Optional)* **Qlib export bridge** — dump lake → Qlib `.bin` so the CSI 300 LightGBM example runs as a benchmark | — | Qlib `scripts/dump_bin.py`, `parse_instruments` | Qlib's example trains on your data; gives an external yardstick |

### Phase-specific pitfalls

- **Survivorship** is decided here, not later: if T0.2 only captures *current* members, every downstream result is biased and it's expensive to retrofit. Do it first.
- **Adjustment drift** — qfq series are not stable over time; storing factors + the snapshot date is what makes a backtest reproducible months later.
- **Announce vs. period date** — collecting fundamentals without `announce_date` quietly destroys PIT correctness; there's no way to reconstruct it afterward.
- **AKShare flakiness** — budget for endpoint failures and schema drift; the catalog (T0.6) is what makes re-runs cheap when an endpoint breaks for a day.

### Deliverables
The lake (bronze/silver/gold), universe + calendar tables, the batch collector + catalog, a validation report, and optionally the Qlib `.bin` export.

---

## Phase P1 — Features & Labels

### Goal / Definition of Done

A **reproducible, leakage-checked feature store and label store** built on the P0 lake. Done when:

1. A feature panel `(symbol, date, features…)` exists for the universe, each feature traceable to a declarative spec.
2. Label tables `(symbol, date, label…)` exist for multiple horizons, all honoring **T+1 execution**.
3. An **automated leakage harness** passes — no feature sees the future; no label overlaps its own feature window without purging.
4. No global normalization has been applied yet (scaler fitting is deliberately deferred to P2's train split); only per-day cross-sectional transforms, which are not leaky, are computed here.

### Key design decisions (for review)

- **Four feature families, layered.**
  1. **Technical** — reuse `technical_indicators.py` as-is (MA/MACD/KDJ/RSI/BOLL); add breadth (ATR, OBV, ADX, Stochastics…) via **TA-Lib** or **pandas-ta-classic** (the maintained successor to the original pandas-ta).
  2. **Cross-sectional** — per-day rank / z-score / percentile of raw features across the universe, via DuckDB window functions. Most equity signal lives here, and these are computed *within a single date* so they don't leak across time.
  3. **Fundamental** — ratios and growth from the PIT store, as-of-joined on `announce_date`.
  4. **Automated** *(deferred / optional)* — tsfresh-style extraction once the core panel works.
- **Declarative feature-spec registry.** Each feature defined as `{name, inputs, window, transform}` in config, hashed into a **feature-set version**. Reproducibility and versioning from day one; conceptually the same idea as Qlib's expression-based Alpha158 definitions, which are worth reading as a template.
- **Label conventions with T+1 execution baked in.** Forward returns at 1/5/20-day horizons, computed as return **from T+1 to T+1+h**, never from T's close — matching Qlib's Alpha158 reasoning. Variants: continuous return, **cross-sectional quantile bucket** (per day), binary outperform-vs-universe-median, and a **forward realized-volatility / max-drawdown risk label** to feed the eventual "risk score." Triple-barrier labeling is noted for later.
- **Strict alignment & no premature normalization.** Feature at T uses only `known_at ≤ T`; label uses only future prices; overlapping windows are purged. Defer all fit-on-data transforms (scalers, quantile bins) to P2 so the train/valid/test boundary fits them correctly — the single most common subtle leak.
- **Storage.** Feature Parquet keyed `(symbol, date)`, label tables separate, joined lazily by DuckDB at P2 assembly time. Feature computation is idempotent and incremental, same discipline as P0.

### Task breakdown

| ID | Task | Reuses from current code | Reference | Done when |
|---|---|---|---|---|
| **T1.1** | **Feature-pipeline scaffold** — runner reads gold OHLCV, applies builders per symbol, writes feature Parquet (idempotent/incremental) | `calculate_all_indicators()` orchestration shape | — | Runner produces a feature panel for the universe |
| **T1.2** | **Technical adapter** — apply `technical_indicators.py` over the panel (`groupby(symbol)`); add TA-Lib / pandas-ta-classic breadth; mask warm-up NaNs | **all of `technical_indicators.py`** | TA-Lib; pandas-ta-classic | Technical features present, warm-up window masked (see pitfalls) |
| **T1.3** | **Cross-sectional features** — per-date rank/z-score/percentile across the universe | — | DuckDB window functions; Qlib `$rank()` filters | Each stock has its daily percentile features |
| **T1.4** | **Fundamental builder** — as-of join PIT fundamentals on `announce_date`; ratios/growth | the statement/abstract collection knowledge | Qlib `P()` as-of operator | Fundamental features aligned by announce date, no period-date leak |
| **T1.5** | **Feature-spec registry + versioning** — declarative defs, hashed feature-set version | — | Qlib Alpha158 expression style | Re-running a version reproduces identical features |
| **T1.6** | **Label builder** — forward returns (1/5/20d) with T+1 execution; quantile / binary / risk variants | — | Qlib Alpha158 `Ref($close,-2)/Ref($close,-1)−1` | Labels exist, T+1 verified, multiple horizons |
| **T1.7** | **Leakage test harness** — automated: no-future-in-features, label/feature non-overlap, a canary test that breaks if a future value leaks | — | de Prado purge/embargo concepts | Harness runs in CI and fails loudly on injected leakage |
| **T1.8** | **Feature/label catalog & data dictionary** — every column: formula, source, `known_at` semantics | — | — | A reviewer can read what each column means and trust its timing |
| **T1.9** | *(Optional)* **Alpha158 parity check** — reproduce a few Qlib features and confirm values match | — | Qlib Alpha158 handler | A handful of features match Qlib within tolerance |

### Phase-specific pitfalls

- **Normalization before the split.** Do *not* fit scalers or global quantile bins here; that leaks test-set statistics into training. Per-day cross-sectional ranks are fine (computed within one date); anything fit across dates waits for P2.
- **Warm-up instability.** The current indicators use `min_periods=1`, so e.g. MA5 on a stock's first day is just its close — unreliable values that will quietly degrade a model if fed in raw. Mask the warm-up window per feature (T1.2).
- **Cross-sectional leakage via the universe.** Compute per-day cross-sectional features against the **point-in-time membership** from P0/T0.2, not today's members — otherwise the ranking secretly "knows" which firms survived.
- **Period-vs-announce, again.** The single place this bites in P1 is T1.4; an as-of join on the wrong column invalidates every fundamental feature.
- **Label/feature window overlap.** A 20-day forward label must not overlap a feature that itself looks 20 days ahead without purging; the harness (T1.7) is what catches this.

### Deliverables
The feature store, the label store, the feature-spec registry, the leakage harness + report, and the data dictionary.

---

## Sequencing & dependencies

```
T0.1 ─┬─ T0.2 (universe) ───────────────┐
      ├─ T0.3 (calendar) ──┐            │
      └─ T0.4 (storage) ───┼─ T0.5 ─ T0.6 (OHLCV + incremental)
                           └─ T0.7 (PIT fundamentals)
                                  │
        T0.8 (validation) ◀───────┘        T0.9 (Qlib bridge, optional)
                                  │
                                  ▼
   P1:  T1.1 ─ T1.2 (technical) ─┐
              T1.3 (cross-sec) ──┼─ T1.5 (registry) ─┐
              T1.4 (fundamental) ┘                   │
                          T1.6 (labels) ─────────────┼─ T1.7 (leakage harness) ─ T1.8 (dictionary)
                                                      └─ T1.9 (parity, optional)
```

**Critical path:** `T0.1 → T0.2 → T0.4 → T0.5 → T0.7 → T1.1/1.4 → T1.6 → T1.7`. The universe (T0.2) and PIT store (T0.7) gate correctness for everything after them, so they're worth doing carefully even though they feel like setup.

## Explicitly deferred to P2+ (so scope here stays honest)

Train/valid/test **time splits**, scaler/quantile **fitting**, purged-CV with embargo, the model itself (LightGBM), IC/Rank-IC **evaluation**, MLflow tracking, Optuna, SHAP, and backtesting. None of these belong in P0/P1 — and trying to pull them forward is the usual way the foundation ends up rushed.

## Reference projects worth a look before building

- **microsoft/qlib** — its `scripts/data_collector` (CSI300 `parse_instruments`), `dump_bin.py`, PIT docs, and the Alpha158 handler are the closest thing to a reference implementation for every P0/P1 decision above.
- **akfamily/akshare** — the source-of-truth endpoint docs; note `stock_zh_a_hist` (EM, includes turnover) alongside the `stock_zh_a_daily` (Sina) your code currently uses.
- A-share batch-collector repos on GitHub show the practical patterns (thread-pool concurrency, TTL caching, dual-source Tushare/AKShare fallback, proxy rotation for heavy pulls) — useful for T0.5/T0.6, though most store to MongoDB rather than a Parquet lake.

---

*One framing note for the review: P0 and P1 will feel slow relative to "train a model," and that asymmetry is correct. The leakage harness (T1.7), the point-in-time membership (T0.2), and the PIT fundamentals (T0.7) are the parts that determine whether anything you build later is real. They're also the parts no tutorial bothers with — which is exactly why building them yourself is where the understanding comes from.*
