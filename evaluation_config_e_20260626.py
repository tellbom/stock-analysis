"""
Config E Forward Test Validation (2026-06-26 -> 2026-07-03)
===========================================================
Sina-only version (Eastmoney blocked by SSL in this environment).

Data sources:
  - Prediction: forward_test_config_e_20260626.csv
  - OHLCV cache: silver/ohlcv/ (entry prices)
  - Live fetch: akshare stock_zh_a_daily (Sina source)
  - Index: silver/index_ohlcv/000300.parquet + Sina fetch
"""

from __future__ import annotations

import datetime as dt
import io
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from quant_platform.ingest.industry_collector import load_industry_map, get_industry_as_of
from quant_platform.selection.config import SelectionConfig

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ROOT = Path("E:/stock-analysis")
STORE_ROOT = ROOT / "models/data"  # silver/... lives under here (see store.lake)
PRED_CSV = ROOT / "models/data/reports/forward_test_config_e_20260626.csv"
SILVER_OHLCV = ROOT / "models/data/silver/ohlcv"
INDEX_FILE = ROOT / "models/data/silver/index_ohlcv/000300.parquet"
OUTPUT_DIR = ROOT / "models/data/reports"

# T1.1: SelectionConfig is the single source of truth for the "unknown
# industry" sentinel — do not hardcode "_UNKNOWN" separately here.
_SELECTION_CONFIG = SelectionConfig()

PRED_DATE = dt.date(2026, 6, 26)
EVAL_DATE = dt.date(2026, 7, 3)
FETCH_START = "20260627"
FETCH_END = "20260704"
CHECKPOINT_PATH = OUTPUT_DIR / "config_e_fetch_checkpoint.csv"

import akshare as ak

# Market prefix detection (Sina API needs sh/sz prefix)
def _get_market_prefix(symbol: str) -> str:
    """Return prefixed symbol for Sina API (e.g. sh600519, sz000001)."""
    code = symbol.strip()
    # Shanghai: 6xxxxx, 688xxx, 689xxx
    # Shenzhen: 0xxxxx, 2xxxxx, 3xxxxx
    # Beijing: 8xxxxx (but rarely in CSI 300)
    if code.startswith(('6', '689')):
        return f"sh{code}"
    elif code.startswith(('0', '2', '3')):
        return f"sz{code}"
    elif code.startswith('8'):
        return f"bj{code}"
    return f"sh{code}"  # fallback

# ---------------------------------------------------------------------------
# Step 1: Load predictions
# ---------------------------------------------------------------------------
print("=" * 70)
print("STEP 1: Loading prediction data")
print("=" * 70, flush=True)

pred_df = pd.read_csv(PRED_CSV)
pred_df["symbol"] = pred_df["symbol"].astype(str).str.zfill(6)
print(f"  Prediction date: {PRED_DATE}")
print(f"  Stocks: {len(pred_df)}", flush=True)

# --- T1.1: PIT industry_code join -----------------------------------------
# The prediction CSV only carries a free-text `industry_name`, which is not
# a stable join/group key for IndustryNeutralRanker (which requires
# `industry_code`). Look up the as-of `industry_code` from the PIT SCD
# table (silver/industry_map.parquet) for the prediction date, rather than
# fabricating a code from the name string.
_industry_map = load_industry_map(STORE_ROOT)
if _industry_map.empty:
    print("  WARNING: industry_map.parquet not found/empty — all stocks "
          f"fall back to unknown_industry_label={_SELECTION_CONFIG.unknown_industry_label!r}",
          flush=True)

_industry_codes = []
for sym in pred_df["symbol"]:
    rec = get_industry_as_of(_industry_map, sym, PRED_DATE) if not _industry_map.empty else {}
    code = rec.get("industry_code", "") if rec else ""
    _industry_codes.append(code if code else _SELECTION_CONFIG.unknown_industry_label)
pred_df["industry_code"] = _industry_codes

_known_frac = (pred_df["industry_code"] != _SELECTION_CONFIG.unknown_industry_label).mean()
print(f"  industry_code coverage: {_known_frac*100:.1f}% "
      f"({(pred_df['industry_code'] != _SELECTION_CONFIG.unknown_industry_label).sum()}/{len(pred_df)} known, "
      f"unknowns kept as {_SELECTION_CONFIG.unknown_industry_label!r}, not dropped)", flush=True)
if _known_frac < 0.95:
    print(f"  WARNING: industry_code coverage {_known_frac*100:.1f}% is below the "
          f"95% target set in T1.1's verify step", flush=True)

# ---------------------------------------------------------------------------
# Step 2: Entry prices from local OHLCV
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 2: Loading entry prices from local OHLCV")
print("=" * 70, flush=True)

entry_prices = {}
for _, row in pred_df.iterrows():
    sym = row["symbol"]
    fpath = SILVER_OHLCV / f"{sym}.parquet"
    if fpath.exists():
        df_s = pd.read_parquet(fpath)
        df_s["date"] = pd.to_datetime(df_s["date"]).dt.date
        mask = df_s["date"] == PRED_DATE
        entry_prices[sym] = float(df_s.loc[mask, "close"].iloc[0]) if mask.any() else float(df_s["close"].iloc[-1])
    else:
        entry_prices[sym] = float(row["entry_close"])
print(f"  Entry prices: {len(entry_prices)} symbols", flush=True)

# ---------------------------------------------------------------------------
# Step 3: Fetch latest prices via Sina API
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 3: Fetching post-prediction prices (Sina API)")
print("=" * 70, flush=True)

# Check for checkpoint (resume from partial fetch)
price_history: dict[str, dict[dt.date, float]] = {}
if CHECKPOINT_PATH.exists():
    cp = pd.read_csv(CHECKPOINT_PATH)
    for _, r in cp.iterrows():
        sym = str(r["symbol"]).zfill(6)
        d = dt.date.fromisoformat(r["date"])
        price_history.setdefault(sym, {})[d] = float(r["close"])
    print(f"  Loaded checkpoint: {len(price_history)} symbols", flush=True)

symbols_to_fetch = pred_df["symbol"].tolist()
total = len(symbols_to_fetch)
fetched = 0
errors = 0

for i, sym in enumerate(symbols_to_fetch):
    # Skip if already in price_history with data past PRED_DATE
    if sym in price_history:
        dates_after = [d for d in price_history[sym] if d > PRED_DATE]
        if len(dates_after) >= 3:  # enough data
            continue

    if (i + 1) % 25 == 0 or i == 0:
        print(f"  [{i+1}/{total}] Fetching... (ok={fetched}, err={errors})", flush=True)

    # Delay between requests
    if i > 0:
        time.sleep(2.5 + np.random.uniform(0, 1.5))

    try:
        prefixed = _get_market_prefix(sym)
        df = ak.stock_zh_a_daily(
            symbol=prefixed,
            start_date=FETCH_START,
            end_date=FETCH_END,
            adjust="qfq",
        )
        if df is not None and not df.empty:
            df = df.reset_index() if df.index.name == "date" or isinstance(df.index, pd.DatetimeIndex) else df
            rename_map = {
                "日期": "date", "开盘": "open", "最高": "high",
                "最低": "low", "收盘": "close", "成交量": "volume",
            }
            df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
            df["date"] = pd.to_datetime(df["date"]).dt.date

            sym_hist = {}
            for _, r in df.iterrows():
                sym_hist[r["date"]] = float(r["close"])
            price_history[sym] = sym_hist
            fetched += 1

            # Save checkpoint every 20 successful fetches
            if fetched % 20 == 0:
                cp_rows = []
                for s, hist in price_history.items():
                    for d, c in hist.items():
                        cp_rows.append({"symbol": s, "date": str(d), "close": c})
                pd.DataFrame(cp_rows).to_csv(CHECKPOINT_PATH, index=False)
        else:
            errors += 1
    except Exception as e:
        errors += 1
        # Longer wait after error
        time.sleep(3.0)

# Final checkpoint save
cp_rows = []
for s, hist in price_history.items():
    for d, c in hist.items():
        cp_rows.append({"symbol": s, "date": str(d), "close": c})
pd.DataFrame(cp_rows).to_csv(CHECKPOINT_PATH, index=False)

print(f"\n  Fetch complete: ok={fetched}, errors={errors}, total_history={len(price_history)}", flush=True)

# ---------------------------------------------------------------------------
# Step 4: Benchmark (CSI 300)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 4: Loading benchmark (CSI 300)")
print("=" * 70, flush=True)

benchmark_entry = float(pred_df["benchmark_close"].iloc[0])
benchmark_latest = None

# Try local cache
if INDEX_FILE.exists():
    idx_df = pd.read_parquet(INDEX_FILE)
    idx_df["date"] = pd.to_datetime(idx_df["date"]).dt.date
    idx_latest = idx_df[idx_df["date"] <= EVAL_DATE]
    if not idx_latest.empty:
        benchmark_latest = float(idx_latest["close"].iloc[-1])
        print(f"  CSI 300 (local): {benchmark_entry:.2f} -> {benchmark_latest:.2f}")

# Try fetching full-period index data via stock_zh_index_daily
if benchmark_latest is None:
    time.sleep(1.0)
    try:
        df_idx = ak.stock_zh_index_daily(symbol="sh000300")
        if df_idx is not None and not df_idx.empty:
            df_idx["date"] = pd.to_datetime(df_idx["date"]).dt.date
            # Get entry (Jun 26) and latest (Jul 3)
            entry_row = df_idx[df_idx["date"] == PRED_DATE]
            exit_row = df_idx[df_idx["date"] <= EVAL_DATE].iloc[-1]
            if not entry_row.empty:
                benchmark_entry = float(entry_row["close"].iloc[0])
                benchmark_latest = float(exit_row["close"])
                print(f"  CSI 300 (stock_zh_index_daily): {benchmark_entry:.2f} -> {benchmark_latest:.2f}")
                print(f"  CSI 300 dates: {entry_row['date'].iloc[0]} -> {exit_row['date']}")
    except Exception as e:
        print(f"  stock_zh_index_daily error: {e}")

if benchmark_latest is None:
    benchmark_latest = benchmark_entry
    print("  WARNING: Using flat benchmark")
benchmark_ret = (benchmark_latest / benchmark_entry) - 1.0
print(f"  Benchmark return: {benchmark_ret*100:.4f}%", flush=True)

# ---------------------------------------------------------------------------
# Step 5: Compute forward returns
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 5: Computing forward returns")
print("=" * 70, flush=True)

results = []
excluded = []

for _, row in pred_df.iterrows():
    sym = row["symbol"]
    rank = int(row["rank"])
    score = float(row["model_score"])
    name = row["stock_name"]
    industry = row["industry_name"]
    industry_code = row["industry_code"]  # T1.1: PIT-joined, not from CSV
    entry_close = entry_prices.get(sym, float(row["entry_close"]))

    if sym not in price_history or not price_history[sym]:
        excluded.append({
            "rank": rank, "symbol": sym, "stock_name": name,
            "model_score": score, "entry_close": entry_close,
            "status": "no_data", "error": "No post-prediction price data"
        })
        continue

    sym_hist = price_history[sym]
    dates_after = sorted([d for d in sym_hist if d > PRED_DATE])

    if not dates_after:
        excluded.append({
            "rank": rank, "symbol": sym, "stock_name": name,
            "model_score": score, "entry_close": entry_close,
            "status": "no_data", "error": "All data dates <= prediction date"
        })
        continue

    exit_date = dates_after[-1]
    exit_close = sym_hist[exit_date]

    rec = {
        "rank": rank, "symbol": sym, "stock_name": name,
        "industry_name": industry, "industry_code": industry_code,
        "model_score": score,
        "entry_close": entry_close, "exit_close": exit_close,
        "exit_date": str(exit_date),
        "total_return": (exit_close / entry_close) - 1.0,
        "benchmark_return": benchmark_ret,
        "excess_vs_benchmark": (exit_close / entry_close) - 1.0 - benchmark_ret,
        "status": "ok",
    }

    # D+1, D+3, D+5
    for horizon, label in [(1, "d1"), (3, "d3"), (5, "d5")]:
        if len(dates_after) >= horizon:
            td = dates_after[horizon - 1]
            rec[f"{label}_return"] = (sym_hist[td] / entry_close) - 1.0
            rec[f"{label}_date"] = str(td)
        else:
            rec[f"{label}_return"] = None
            rec[f"{label}_date"] = None

    results.append(rec)

df = pd.DataFrame(results)
print(f"  Valid: {len(df)}, Excluded: {len(excluded)}", flush=True)

if excluded:
    print("  Excluded:")
    for exc in excluded[:10]:
        print(f"    {exc['symbol']} {exc['stock_name']}: {exc['error']}")

if df.empty:
    print("ERROR: No valid results!")
    sys.exit(1)

df = df.sort_values("rank").reset_index(drop=True)
N = len(df)

# ---------------------------------------------------------------------------
# Step 5b (T1.2): Industry-neutral ranking/selection
# ---------------------------------------------------------------------------
# Route the scored panel through IndustryNeutralRanker instead of trusting
# the naive global sort_values("rank")/head(N) below to pick "the" Top-N.
# Ranker does NOT retrain/rescale the model — it only consumes model_score.
print("\n" + "=" * 70)
print("STEP 5b: Industry-neutral ranking (T1.2)")
print("=" * 70, flush=True)

from quant_platform.selection.ranker import IndustryNeutralRanker

_ranker = IndustryNeutralRanker(
    _SELECTION_CONFIG,
    industry_col="industry_code",
    name_col="industry_name",
    score_col="model_score",
    symbol_col="symbol",
)
df = _ranker.run(df)  # adds industry_rank, industry_neutral_score, industry_size,
                       # global_score, selected, selection_reason, exposure_flag
n_selected = int(df["selected"].sum())
print(f"  strategy={_SELECTION_CONFIG.strategy.value}, top_k={_SELECTION_CONFIG.top_k}, "
      f"max_total={_SELECTION_CONFIG.max_total} -> selected {n_selected}/{N} stocks", flush=True)

# T1.3: surface ExposureMonitor flags/concentration for this run
from quant_platform.selection.exposure import ExposureMonitor

_run_exposure_flag = (
    df.loc[df["selected"], "exposure_flag"].iloc[0] if n_selected else "not_selected"
)
_concentration_report = ExposureMonitor.concentration_report(
    df, industry_col="industry_code", name_col="industry_name",
    selected_col="selected", symbol_col="symbol",
)
print(f"  run-level exposure_flag: {_run_exposure_flag}")
print("  per-industry concentration (selected set):")
for _ind_name, _stats in _concentration_report.items():
    print(f"    {_ind_name or '(unnamed)'}: count={_stats['count']}, "
          f"fraction={_stats['fraction']*100:.1f}%")

# ---------------------------------------------------------------------------
# Step 6: ALL metrics
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 6: Computing evaluation metrics")
print("=" * 70, flush=True)

total_return_col = "total_return"

# 6a: Whole sample
mean_ret = df[total_return_col].mean()
median_ret = df[total_return_col].median()
std_ret = df[total_return_col].std()
hit_rate = (df[total_return_col] > 0).mean()
print(f"  N={N}, mean={mean_ret*100:.4f}%, median={median_ret*100:.4f}%, "
      f"std={std_ret*100:.4f}%, hit_rate={hit_rate*100:.1f}%")

# 6b: Top-N — NAIVE global-rank slices, kept only as a diagnostic baseline
# for comparison against the industry-neutral selection below (T1.4 uses
# this contrast across strategies). These are NOT the operative selection.
top_groups = {"Top1": 1, "Top3": 3, "Top5": 5, "Top10": 10, "Top20": 20, "Top50": 50, "Top100": 100}
top_metrics = {}
for label, n in top_groups.items():
    if n > N:
        continue
    sub = df.head(n)
    top_metrics[label] = {
        "n": n, "mean_return": sub[total_return_col].mean(),
        "median_return": sub[total_return_col].median(),
        "hit_rate": (sub[total_return_col] > 0).mean(),
        "min_return": sub[total_return_col].min(),
        "max_return": sub[total_return_col].max(),
    }
    print(f"  {label} (naive global rank): mean={top_metrics[label]['mean_return']*100:.4f}%, "
          f"hit_rate={top_metrics[label]['hit_rate']*100:.1f}%")

# 6b'(T1.2): the operative, industry-neutral selected set (ranker output)
selected_df = df[df["selected"]]
industry_neutral_metrics = {
    "n": len(selected_df),
    "mean_return": selected_df[total_return_col].mean(),
    "median_return": selected_df[total_return_col].median(),
    "hit_rate": (selected_df[total_return_col] > 0).mean(),
}
if len(selected_df):
    _counts = selected_df["industry_code"].value_counts()
    _max_industry_fraction = float(_counts.max() / len(selected_df))
else:
    _max_industry_fraction = 0.0
print(f"  Industry-neutral selected ({_SELECTION_CONFIG.strategy.value}, n={industry_neutral_metrics['n']}): "
      f"mean={industry_neutral_metrics['mean_return']*100:.4f}%, "
      f"hit_rate={industry_neutral_metrics['hit_rate']*100:.1f}%, "
      f"max_single_industry_fraction={_max_industry_fraction*100:.1f}% "
      f"(warning threshold={_SELECTION_CONFIG.exposure_warning_threshold*100:.0f}%)", flush=True)

# 6c: Top vs all
print(f"\n  Top5 vs All: {top_metrics['Top5']['mean_return']*100:.4f}% vs {mean_ret*100:.4f}% "
      f"(excess: {(top_metrics['Top5']['mean_return']-mean_ret)*100:.4f}%)")
print(f"  Top10 vs All: {top_metrics['Top10']['mean_return']*100:.4f}% vs {mean_ret*100:.4f}% "
      f"(excess: {(top_metrics['Top10']['mean_return']-mean_ret)*100:.4f}%)")

# 6d: Top vs benchmark
print(f"\n  Benchmark (CSI 300): {benchmark_ret*100:.4f}%")
for label in ["Top5", "Top10", "Top20"]:
    if label in top_metrics:
        ex = top_metrics[label]["mean_return"] - benchmark_ret
        print(f"  {label} vs CSI 300: excess={ex*100:.4f}%")

# 6e: Correlations
print(f"\n  --- Correlations ---")
spearman_r = spearman_p = kendall_r = kendall_p = pearson_r = pearson_p = sr2 = sp2 = None
if N >= 5:
    spearman_r, spearman_p = stats.spearmanr(df["model_score"], df[total_return_col])
    kendall_r, kendall_p = stats.kendalltau(df["model_score"], df[total_return_col])
    pearson_r, pearson_p = stats.pearsonr(df["model_score"], df[total_return_col])
    print(f"  Spearman (score vs ret): r={spearman_r:.4f}, p={spearman_p:.4f}")
    print(f"  Kendall tau: {kendall_r:.4f}, p={kendall_p:.4f}")
    print(f"  Pearson r: {pearson_r:.4f}, p={pearson_p:.4f}")
    df_c = df.copy()
    df_c["actual_rank"] = df_c[total_return_col].rank(ascending=False)
    sr2, sp2 = stats.spearmanr(df_c["rank"], df_c["actual_rank"])
    print(f"  Spearman (pred_rank vs actual_rank): r={sr2:.4f}, p={sp2:.4f}")

# 6f: Decile analysis
df_s = df.sort_values("model_score", ascending=False).reset_index(drop=True)
try:
    n_q = min(10, N)
    df_s["decile"] = pd.qcut(df_s["model_score"], q=n_q,
                              labels=[f"D{i}" for i in range(1, n_q+1)], duplicates="drop")
    decile_stats = df_s.groupby("decile", observed=False).agg(
        mean_return=(total_return_col, "mean"),
        median_return=(total_return_col, "median"),
        hit_rate=(total_return_col, lambda x: (x > 0).mean()),
        count=(total_return_col, "count"),
        score_mean=("model_score", "mean"),
    ).sort_index(ascending=False)
    print(f"\n  --- Decile Returns ---")
    print(decile_stats.to_string())
    dm = decile_stats["mean_return"].values
    monotonic = all(dm[i] >= dm[i+1] for i in range(len(dm)-1))
    monotonic_reverse = all(dm[i] <= dm[i+1] for i in range(len(dm)-1))
    print(f"  Monotonic (desc): {monotonic}, Monotonic (asc): {monotonic_reverse}")
except Exception as e:
    print(f"  Decile error: {e}")
    decile_stats = pd.DataFrame()
    monotonic = False
    monotonic_reverse = False

# 6g: IC
ic = df["model_score"].corr(df[total_return_col])
rank_ic = df["model_score"].corr(df[total_return_col], method="spearman")
print(f"\n  Pearson IC: {ic:.4f}, Spearman IC: {rank_ic:.4f}")

# 6h: Failure cases
top20_cutoff = df["model_score"].quantile(0.80)
bot20_cutoff = df["model_score"].quantile(0.20)
high_score_losers = df[(df["model_score"] >= top20_cutoff) & (df[total_return_col] < 0)].sort_values("model_score", ascending=False)
low_score_winners = df[(df["model_score"] <= bot20_cutoff) & (df[total_return_col] > 0)].sort_values("model_score", ascending=True)
print(f"\n  High-score losers: {len(high_score_losers)}, Low-score winners: {len(low_score_winners)}")

# 6i: Horizon
for horizon, label in [(1, "D+1"), (3, "D+3"), (5, "D+5")]:
    col = f"d{horizon}_return"
    if col in df.columns:
        hd = df[col].dropna()
        if len(hd) > 0:
            hv = df.dropna(subset=[col])
            h_ic = hv["model_score"].corr(hv[col], method="spearman") if len(hv) >= 5 else None
            ic_s = f", IC={h_ic:.4f}" if h_ic is not None else ""
            print(f"  {label}: N={len(hd)}, mean={hd.mean()*100:.4f}%, hit={(hd>0).mean()*100:.1f}%{ic_s}")

# 6j: Worst
worst_5 = df.nsmallest(5, total_return_col)
print(f"\n  Worst 5:")
for _, r in worst_5.iterrows():
    print(f"    R{r['rank']:3.0f}: {r['symbol']} {r['stock_name']} "
          f"score={r['model_score']:.4f} ret={r[total_return_col]*100:.3f}%")

# ---------------------------------------------------------------------------
# Step 7: Report
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 7: Generating Markdown Report")
print("=" * 70, flush=True)

report_path = OUTPUT_DIR / "config_e_validation_20260703.md"

def fp(val, d=2):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    return f"{val*100:.{d}f}%"

def fr(val, d=4):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    return f"{val:.{d}f}"

# Verdict
verdict = ""
if spearman_r is not None:
    if spearman_r > 0.1:
        verdict += "**正向排序有效** -- 模型评分与真实收益呈正相关, 高分股整体跑赢低分股."
    elif spearman_r > 0.03:
        verdict += "**弱正向** -- 相关性存在但较弱, 排序能力有限."
    elif spearman_r > -0.03:
        verdict += "**无显著排序能力** -- 模型评分与真实收益接近随机."
    else:
        verdict += "**反向** -- 模型评分与真实收益呈负相关, 高分股反而跑输."
if monotonic:
    verdict += " 分层收益呈单调递减趋势."
elif monotonic_reverse:
    verdict += " 分层收益呈反向单调."
else:
    verdict += " 分层收益不完全单调."

report = f"""# Config E Forward Test 验证报告

## 基本信息

| 项目 | 内容 |
|------|------|
| **预测模型** | Config E |
| **预测日期** | 2026-06-26 (周五) |
| **对照收盘日期** | 2026-07-03 (周五, 最新交易日, T+5) |
| **交易天数** | 5 个交易日 (Jun 29, 30, Jul 1, 2, 3) |
| **预测样本数** | {N + len(excluded)} |
| **有效样本数** | {N} |
| **排除样本数** | {len(excluded)} |
| **Benchmark** | CSI 300 (沪深300) |
| **Benchmark 预测日收盘** | {benchmark_entry:.2f} |
| **Benchmark 最新收盘** | {benchmark_latest:.2f} |
| **Benchmark 区间收益** | {fp(benchmark_ret)} |
| **数据来源 (行情)** | akshare stock_zh_a_daily (Sina 数据源, 前复权) |
| **数据来源 (预测)** | `forward_test_config_e_20260626.csv` |
| **获取方式** | Sina daily API, 每请求间隔 2.5-4s, 共计 {N + len(excluded)} 只 |

---

## 1. 全样本统计

| 指标 | 数值 |
|------|------|
| 样本数 | {N} |
| 平均收益 | {fp(mean_ret)} |
| 中位数收益 | {fp(median_ret)} |
| 标准差 | {fp(std_ret)} |
| 命中率 (收益>0) | {fp(hit_rate, 1)} |
| 最佳个股 | {df.loc[df[total_return_col].idxmax(), 'stock_name']} ({df.loc[df[total_return_col].idxmax(), 'symbol']}) {fp(df[total_return_col].max())} |
| 最差个股 | {df.loc[df[total_return_col].idxmin(), 'stock_name']} ({df.loc[df[total_return_col].idxmin(), 'symbol']}) {fp(df[total_return_col].min())} |

---

## 2. Top-N 选股表现

| 分组 | N | 平均收益 | 中位数收益 | 命中率 | 超额(CSI 300) | 超额(全样本) |
|------|---|----------|------------|--------|---------------|-------------|"""

for label in ["Top1", "Top3", "Top5", "Top10", "Top20", "Top50", "Top100"]:
    if label in top_metrics:
        tm = top_metrics[label]
        report += (f"\n| {label} | {tm['n']} | {fp(tm['mean_return'])} | {fp(tm['median_return'])} | "
                   f"{fp(tm['hit_rate'], 1)} | {fp(tm['mean_return']-benchmark_ret)} | "
                   f"{fp(tm['mean_return']-mean_ret)} |")

report += f"""

---

## 3. Top 10 持仓明细（行业中性选股, T1.2 IndustryNeutralRanker）

| 排名 | 代码 | 名称 | 行业 | 模型评分 | 入场价 | 最新价 | 实际收益 | D+5收益 |
|------|------|------|------|----------|--------|--------|----------|--------|"""

# T1.2: pull the report table from the industry-neutral selected set
# (sorted by original global rank), not a naive head(10) of the whole panel.
for _, r in selected_df.sort_values("rank").head(10).iterrows():
    d5_s = fp(r.get("d5_return")) if r.get("d5_return") is not None else "N/A"
    ex_s = f"{r.get('exit_close', 0):.2f}" if r.get('exit_close') is not None else "N/A"
    report += f"\n| {int(r['rank'])} | {r['symbol']} | {r['stock_name']} | {r['industry_name']} | {r['model_score']:.4f} | {r['entry_close']:.2f} | {ex_s} | {fp(r['total_return'])} | {d5_s} |"

# T1.3: industry exposure section — makes concentration visible per run
report += f"""

---

## 3b. 行业集中度 (Industry Exposure Monitor — T1.3)

**Run-level flag:** `{_run_exposure_flag}`
(thresholds: overweight > {_SELECTION_CONFIG.exposure_warning_threshold*100:.0f}%, diversified < {_SELECTION_CONFIG.exposure_diversified_threshold*100:.0f}%)

| 行业 | 数量 | 占比 |
|------|------|------|"""
for _ind_name, _stats in _concentration_report.items():
    report += f"\n| {_ind_name or '(unnamed)'} | {_stats['count']} | {fp(_stats['fraction'], 1)} |"

report += f"""

---

## 4. 排序有效性

### 4.1 相关性分析

| 指标 | 数值 | p-value |
|------|------|---------|
| Spearman (评分 vs 收益) | {fr(spearman_r)} | {fr(spearman_p)} |
| Kendall tau | {fr(kendall_r)} | {fr(kendall_p)} |
| Pearson r | {fr(pearson_r)} | {fr(pearson_p)} |
| Spearman (预测排名 vs 实际排名) | {fr(sr2)} | {fr(sp2)} |
| Pearson IC | {fr(ic)} | -- |
| Spearman IC (Rank IC) | {fr(rank_ic)} | -- |

### 4.2 分层收益 (按模型评分 Decile)

| Decile | 数量 | 评分均值 | 平均收益 | 中位数收益 | 命中率 |
|--------|------|----------|----------|------------|--------|"""

if not decile_stats.empty:
    for idx, row_data in decile_stats.iterrows():
        report += f"\n| {idx} | {int(row_data['count'])} | {row_data['score_mean']:.4f} | {fp(row_data['mean_return'])} | {fp(row_data['median_return'])} | {fp(row_data['hit_rate'], 1)} |"

mono_status = "单调递减" if monotonic else ("反向单调" if monotonic_reverse else "非单调")
report += f"""

**单调性**: {mono_status}

### 4.3 不同时间窗口

| 窗口 | 有效样本 | 平均收益 | 命中率 | Spearman IC |
|------|----------|----------|--------|-------------|"""

for horizon, label in [(1, "D+1"), (3, "D+3"), (5, "D+5")]:
    col = f"d{horizon}_return"
    if col in df.columns:
        hd = df[col].dropna()
        if len(hd) > 0:
            hv = df.dropna(subset=[col])
            h_ic = hv["model_score"].corr(hv[col], method="spearman") if len(hv) >= 5 else None
            report += f"\n| {label} | {len(hd)} | {fp(hd.mean())} | {fp((hd>0).mean(), 1)} | {fr(h_ic)} |"

report += f"""

---

## 5. 失败案例分析

### 5.1 高分低收益 (Top 20% score, 负收益)

共 {len(high_score_losers)} 只:

| 排名 | 代码 | 名称 | 行业 | 评分 | 收益 |
|------|------|------|------|------|------|"""

for _, r in high_score_losers.head(15).iterrows():
    report += f"\n| {int(r['rank'])} | {r['symbol']} | {r['stock_name']} | {r['industry_name']} | {r['model_score']:.4f} | {fp(r['total_return'])} |"

report += f"""

### 5.2 低分高收益 (Bottom 20% score, 正收益)

共 {len(low_score_winners)} 只:

| 排名 | 代码 | 名称 | 行业 | 评分 | 收益 |
|------|------|------|------|------|------|"""

for _, r in low_score_winners.head(15).iterrows():
    report += f"\n| {int(r['rank'])} | {r['symbol']} | {r['stock_name']} | {r['industry_name']} | {r['model_score']:.4f} | {fp(r['total_return'])} |"

report += f"""

### 5.3 最大回撤 (Worst 5)

| 排名 | 代码 | 名称 | 行业 | 评分 | 收益 |
|------|------|------|------|------|------|"""

for _, r in worst_5.iterrows():
    report += f"\n| {int(r['rank'])} | {r['symbol']} | {r['stock_name']} | {r['industry_name']} | {r['model_score']:.4f} | {fp(r['total_return'])} |"""

if excluded:
    report += f"""

---

## 6. 排除样本

以下 {len(excluded)} 只股票因无法获取 2026-06-26 之后的行情数据被排除:

| 排名 | 代码 | 名称 | 原因 |
|------|------|------|------|"""
    for exc in excluded:
        report += f"\n| {exc['rank']} | {exc['symbol']} | {exc['stock_name']} | {exc.get('error', exc.get('status'))} |"

report += f"""

---

## 7. 综合结论

### 7.1 排序有效性: {verdict}

### 7.2 关键发现

"""

if "Top5" in top_metrics:
    t5m = top_metrics["Top5"]["mean_return"]
    if t5m > mean_ret:
        report += f"- **Top 5 选股有效**: Top 5 平均 {fp(t5m)}, 超额全样本 {fp(t5m-mean_ret)}.\n"
    else:
        report += f"- **Top 5 选股不足**: Top 5 平均 {fp(t5m)}, 低于全样本 {fp(mean_ret)}, 差 {fp(mean_ret-t5m)}.\n"

    t5bm = t5m - benchmark_ret
    if t5bm > 0:
        report += f"- **跑赢大盘**: Top 5 相对 CSI 300 超额 {fp(t5bm)}.\n"
    else:
        report += f"- **跑输大盘**: Top 5 跑输 CSI 300 {fp(abs(t5bm))}.\n"

if spearman_r is not None:
    if spearman_r > 0.05:
        sig = "统计显著" if spearman_p < 0.05 else "统计不显著"
        report += f"- **正向排序**: Spearman rho={spearman_r:.4f} (p={spearman_p:.4f}), {sig}.\n"
    else:
        report += f"- **排序弱**: Spearman rho={spearman_r:.4f}, 短期波动主导个股差异.\n"

if monotonic:
    report += "- **分层有效**: Decile 分组收益单调递减, 模型组间区分度良好.\n"

if len(high_score_losers) > 0:
    report += f"- **高分失败**: {len(high_score_losers)}/{N} ({100*len(high_score_losers)/max(N,1):.1f}%) Top 20% 评分股票实际亏损.\n"

if len(low_score_winners) > 0:
    report += f"- **低分逆袭**: {len(low_score_winners)}/{N} ({100*len(low_score_winners)/max(N,1):.1f}%) Bottom 20% 评分股票实现正收益.\n"

# Check if Top D+5 hit rate is good
d5_col = "d5_return"
if d5_col in df.columns:
    d5_top5 = df.head(5)[d5_col].dropna()
    if len(d5_top5) > 0:
        d5_top5_mean = d5_top5.mean()
        d5_all_mean = df[d5_col].dropna().mean()
        report += f"- **D+5 表现**: Top 5 在 D+5 窗口平均 {fp(d5_top5_mean)}, 全样本 {fp(d5_all_mean)}. 模型以 ret_fwd_5d 为标签, 此窗口为最直接验证.\n"

report += f"""
### 7.3 局限性说明

1. **时间窗口极短**: 仅 5 个交易日 (1 个日历周), 统计结论高度受短期市场波动影响.
2. **单次截面验证**: 仅评估一个预测日的排序, 不代表模型在其他日期或其他市场状态下的表现.
3. **数据源**: Eastmoney API 在本环境存在 SSL 兼容性问题, 改用 Sina API (akshare stock_zh_a_daily). 两家数据源的复权方式可能存在细微差异.
4. **排除样本**: {len(excluded)} 只股票因 API 获取失败被排除, 可能引入 survivor bias.
5. **价格复权**: 使用前复权 (qfq), 与模型训练时一致.
6. **样本范围**: CSI 300 成分股, 结论不适用于小盘股.

### 7.4 后续改进建议

1. **滚动验证**: 建议连续验证 20+ 个交易日的预测截面, 计算平均 IC、IC_IR、超额收益稳定性.
2. **行业中性化**: Top 排序偏重半导体/电子行业, 增加行业中性约束后可评估 pure alpha.
3. **多窗口验证**: 以 D+5 (模型训练标签) 为主要验证窗口, D+1/D+3 为辅助参考.
4. **失败归因**: 对高分低收益个股做信号归因, 识别失效因子的共同特征.
5. **市场状态分层**: 按波动率/趋势强度分层评估, 识别模型的优势/劣势市况.
6. **本地数据库**: 建立定期增量更新的 OHLCV 数据库, 减少对外部 API 的实时依赖.
7. **换手成本**: 若 Top 选股日间换手率高, 需扣除交易成本后评估净超额.

---

*报告生成: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
*评估脚本: `evaluation_config_e_20260626.py`*
*数据获取: akshare Sina daily API*
"""

with open(report_path, "w", encoding="utf-8") as f:
    f.write(report)
print(f"  Report: {report_path}")

# Save results
results_csv = OUTPUT_DIR / "config_e_validation_results_20260703.csv"
df.to_csv(results_csv, index=False, encoding="utf-8-sig")
print(f"  Results: {results_csv}")

if excluded:
    excl_csv = OUTPUT_DIR / "config_e_validation_excluded_20260703.csv"
    pd.DataFrame(excluded).to_csv(excl_csv, index=False, encoding="utf-8-sig")
    print(f"  Excluded: {excl_csv}")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
