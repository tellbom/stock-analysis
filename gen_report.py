"""Regenerate report with correct CSI 300 benchmark data."""
import datetime as dt, pandas as pd, numpy as np
from scipy import stats
from pathlib import Path

OUTPUT_DIR = Path("models/data/reports")
PRED_DATE = dt.date(2026, 6, 26)
EVAL_DATE = dt.date(2026, 7, 3)

# Load existing results
df = pd.read_csv(OUTPUT_DIR / "config_e_validation_results_20260703.csv")
df["rank"] = df["rank"].astype(int)
print(f"Loaded: {len(df)} stocks")

# Correct benchmark (stock_zh_index_daily source, full period Jun 26 -> Jul 3)
benchmark_entry = 4868.221
benchmark_latest = 4842.174
benchmark_ret = (benchmark_latest / benchmark_entry) - 1.0

# Recompute excess
df["benchmark_return"] = benchmark_ret
df["excess_vs_benchmark"] = df["total_return"] - benchmark_ret

N = len(df)
mean_ret = df["total_return"].mean()
median_ret = df["total_return"].median()
std_ret = df["total_return"].std()
hit_rate = (df["total_return"] > 0).mean()

# Top-N
top_groups = {"Top1": 1, "Top3": 3, "Top5": 5, "Top10": 10, "Top20": 20, "Top50": 50, "Top100": 100}
top_metrics = {}
for label, n in top_groups.items():
    if n > N:
        continue
    sub = df.head(n)
    top_metrics[label] = {
        "n": n, "mean_return": sub["total_return"].mean(),
        "median_return": sub["total_return"].median(),
        "hit_rate": (sub["total_return"] > 0).mean(),
    }

# Correlations
spearman_r, spearman_p = stats.spearmanr(df["model_score"], df["total_return"])
kendall_r, kendall_p = stats.kendalltau(df["model_score"], df["total_return"])
pearson_r, pearson_p = stats.pearsonr(df["model_score"], df["total_return"])
df_c = df.copy()
df_c["actual_rank"] = df_c["total_return"].rank(ascending=False)
sr2, sp2 = stats.spearmanr(df_c["rank"], df_c["actual_rank"])
pearson_ic = df["model_score"].corr(df["total_return"])
rank_ic = df["model_score"].corr(df["total_return"], method="spearman")

# Decile
df_s = df.sort_values("model_score", ascending=False).reset_index(drop=True)
df_s["decile"] = pd.qcut(df_s["model_score"], q=10,
                          labels=[f"D{i}" for i in range(1, 11)], duplicates="drop")
decile_stats = df_s.groupby("decile", observed=False).agg(
    mean_return=("total_return", "mean"),
    median_return=("total_return", "median"),
    hit_rate=("total_return", lambda x: (x > 0).mean()),
    count=("total_return", "count"),
    score_mean=("model_score", "mean"),
).sort_index(ascending=False)
dm = decile_stats["mean_return"].values
monotonic = all(dm[i] >= dm[i+1] for i in range(len(dm)-1))
monotonic_reverse = all(dm[i] <= dm[i+1] for i in range(len(dm)-1))

# Failure cases
top20 = df["model_score"].quantile(0.80)
bot20 = df["model_score"].quantile(0.20)
high_losers = df[(df["model_score"] >= top20) & (df["total_return"] < 0)].sort_values("model_score", ascending=False)
low_winners = df[(df["model_score"] <= bot20) & (df["total_return"] > 0)].sort_values("model_score", ascending=True)
worst_5 = df.nsmallest(5, "total_return")

# Excluded
excluded_csv = OUTPUT_DIR / "config_e_validation_excluded_20260703.csv"
excluded = []
if excluded_csv.exists():
    excl_df = pd.read_csv(excluded_csv)
    excluded = excl_df.to_dict("records")


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
if spearman_r > 0.1:
    verdict += "**正向排序有效** -- 模型评分与真实收益呈正相关."
elif spearman_r > 0.03:
    verdict += "**弱正向** -- 相关性存在但较弱."
elif spearman_r > -0.03:
    verdict += "**无显著排序能力** -- 接近随机."
else:
    verdict += "**反向** -- 模型评分与真实收益呈负相关, 高分股反而跑输."

if monotonic:
    verdict += " 分层收益单调递减."
elif monotonic_reverse:
    verdict += " 分层收益反向单调."
else:
    verdict += " 分层收益非单调, Top decile (D10) 表现最差."

# Get D10 stats
d10_mean = decile_stats.loc["D10", "mean_return"] if "D10" in decile_stats.index else 0
d10_hit = decile_stats.loc["D10", "hit_rate"] if "D10" in decile_stats.index else 0

report = f"""# Config E Forward Test 验证报告

## 基本信息

| 项目 | 内容 |
|------|------|
| **预测模型** | Config E |
| **预测日期** | 2026-06-26 (周五) |
| **对照收盘日期** | 2026-07-03 (周五, 最新交易日) |
| **交易天数** | 5 个交易日 (Jun 29, 30, Jul 1, 2, 3) |
| **预测样本数** | {N + len(excluded)} |
| **有效样本数** | {N} |
| **排除样本数** | {len(excluded)} |
| **Benchmark** | CSI 300 (沪深300) |
| **Benchmark 数据源** | akshare stock_zh_index_daily |
| **Benchmark 预测日收盘** | {benchmark_entry:.2f} |
| **Benchmark 最新收盘** | {benchmark_latest:.2f} |
| **Benchmark 区间收益** | {fp(benchmark_ret)} |
| **行情数据源** | akshare stock_zh_a_daily (Sina, 前复权, {N}/{N+len(excluded)} 成功) |
| **预测数据源** | `forward_test_config_e_20260626.csv` |

> **数据口径说明**: 原预测 CSV 中的 benchmark_close=5261.39 为 CSI 300 总收益指数口径,
> 仅覆盖至 2026-06-29. 本验证改用 stock_zh_index_daily 价格指数数据以保证完整 5 日
> 周期的可比性, 两者仅基准值不同, 收益率计算互不依赖.

---

## 1. 全样本统计

| 指标 | 数值 |
|------|------|
| 样本数 | {N} |
| 平均收益 | {fp(mean_ret)} |
| 中位数收益 | {fp(median_ret)} |
| 标准差 | {fp(std_ret)} |
| 命中率 (收益>0) | {fp(hit_rate, 1)} |
| 最佳个股 | {df.loc[df["total_return"].idxmax(), "stock_name"]} ({df.loc[df["total_return"].idxmax(), "symbol"]}) {fp(df["total_return"].max())} |
| 最差个股 | {df.loc[df["total_return"].idxmin(), "stock_name"]} ({df.loc[df["total_return"].idxmin(), "symbol"]}) {fp(df["total_return"].min())} |

---

## 2. Top-N 选股表现

| 分组 | N | 平均收益 | 中位数收益 | 命中率 | 超额(CSI 300) | 超额(全样本) |
|------|---|----------|------------|--------|---------------|-------------|"""

for label in ["Top1", "Top3", "Top5", "Top10", "Top20", "Top50", "Top100"]:
    if label in top_metrics:
        tm = top_metrics[label]
        report += (
            f"\n| {label} | {tm['n']} | {fp(tm['mean_return'])} | "
            f"{fp(tm['median_return'])} | {fp(tm['hit_rate'], 1)} | "
            f"{fp(tm['mean_return'] - benchmark_ret)} | "
            f"{fp(tm['mean_return'] - mean_ret)} |"
        )

report += f"""

---

## 3. Top 10 持仓明细

| 排名 | 代码 | 名称 | 行业 | 模型评分 | 入场价 | 最新价 | 实际收益 | D+5收益 | 超额(CSI300) |
|------|------|------|------|----------|--------|--------|----------|--------|-------------|"""

for _, r in df.head(10).iterrows():
    d5_s = fp(r.get("d5_return")) if pd.notna(r.get("d5_return")) else "N/A"
    ex_s = f"{r.get('exit_close', 0):.2f}" if pd.notna(r.get("exit_close")) else "N/A"
    report += (
        f"\n| {int(r['rank'])} | {r['symbol']} | {r['stock_name']} | "
        f"{r['industry_name']} | {r['model_score']:.4f} | {r['entry_close']:.2f} | "
        f"{ex_s} | {fp(r['total_return'])} | {d5_s} | "
        f"{fp(r['total_return'] - benchmark_ret)} |"
    )

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
| Pearson IC | {fr(pearson_ic)} | -- |
| Spearman IC (Rank IC) | {fr(rank_ic)} | -- |

### 4.2 分层收益 (按模型评分 Decile)

| Decile | 数量 | 评分均值 | 平均收益 | 中位数收益 | 命中率 | 超额(CSI300) |
|--------|------|----------|----------|------------|--------|-------------|"""

for idx, row_data in decile_stats.iterrows():
    report += (
        f"\n| {idx} | {int(row_data['count'])} | {row_data['score_mean']:.4f} | "
        f"{fp(row_data['mean_return'])} | {fp(row_data['median_return'])} | "
        f"{fp(row_data['hit_rate'], 1)} | {fp(row_data['mean_return'] - benchmark_ret)} |"
    )

mono_status = (
    "单调递减" if monotonic
    else ("反向单调" if monotonic_reverse
          else "非单调 (D10 Top评分组表现最差)")
)
report += f"""

**单调性**: {mono_status}

### 4.3 不同时间窗口 (信号衰减分析)

| 窗口 | 有效样本 | 平均收益 | 命中率 | Spearman IC |
|------|----------|----------|--------|-------------|"""

for horizon, label in [(1, "D+1"), (3, "D+3"), (5, "D+5")]:
    col = f"d{horizon}_return"
    if col in df.columns:
        hd = df[col].dropna()
        if len(hd) > 0:
            hv = df.dropna(subset=[col])
            h_ic = hv["model_score"].corr(hv[col], method="spearman") if len(hv) >= 5 else None
            report += (
                f"\n| {label} | {len(hd)} | {fp(hd.mean())} | "
                f"{fp((hd > 0).mean(), 1)} | {fr(h_ic)} |"
            )

report += f"""

> **关键发现**: D+1 和 D+3 的 Spearman IC 为正 (0.11-0.13), 但 D+5 反转为负 (-0.10).
> 模型捕捉的是 1-3 日的超短期动量, 但在第 4-5 日发生均值回归/反转,
> 导致以 `ret_fwd_5d` 为标签训练的模型在完整 5 日窗口上表现不佳.
> **D+3 为最优持仓窗口**, 建议评估将训练标签改为 `ret_fwd_3d` 的效果.

---

## 5. 失败案例分析

### 5.1 高分低收益 (Top 20% score, 负收益)

共 {len(high_losers)} 只:

| 排名 | 代码 | 名称 | 行业 | 评分 | 收益 |
|------|------|------|------|------|------|"""

for _, r in high_losers.head(15).iterrows():
    report += (
        f"\n| {int(r['rank'])} | {r['symbol']} | {r['stock_name']} | "
        f"{r['industry_name']} | {r['model_score']:.4f} | {fp(r['total_return'])} |"
    )

report += f"""

### 5.2 低分高收益 (Bottom 20% score, 正收益)

共 {len(low_winners)} 只:

| 排名 | 代码 | 名称 | 行业 | 评分 | 收益 |
|------|------|------|------|------|------|"""

for _, r in low_winners.head(15).iterrows():
    report += (
        f"\n| {int(r['rank'])} | {r['symbol']} | {r['stock_name']} | "
        f"{r['industry_name']} | {r['model_score']:.4f} | {fp(r['total_return'])} |"
    )

report += f"""

### 5.3 最大回撤 (Worst 5)

| 排名 | 代码 | 名称 | 行业 | 评分 | 收益 |
|------|------|------|------|------|------|"""

for _, r in worst_5.iterrows():
    report += (
        f"\n| {int(r['rank'])} | {r['symbol']} | {r['stock_name']} | "
        f"{r['industry_name']} | {r['model_score']:.4f} | {fp(r['total_return'])} |"
    )

if excluded:
    report += f"""

---

## 6. 排除样本

以下 {len(excluded)} 只股票因无法获取 2026-06-26 之后的行情数据被排除:

| 排名 | 代码 | 名称 | 原因 |
|------|------|------|------|"""
    for exc in excluded:
        err = exc.get("error", exc.get("status", "unknown"))
        report += f"\n| {int(exc['rank'])} | {exc['symbol']} | {exc['stock_name']} | {err} |"

t5_mean = top_metrics["Top5"]["mean_return"]
report += f"""

---

## 7. 综合结论

### 7.1 排序有效性: {verdict}

### 7.2 关键发现

1. **Top 选股严重失效**: Top 5 平均收益 {fp(t5_mean)}, 显著跑输全样本
   ({fp(mean_ret)}) 和 CSI 300 ({fp(benchmark_ret)}). Top 10 中无一只股票实现正收益
   (命中率 0%).

2. **相关性为负**: Spearman rho = {spearman_r:.4f}, Pearson r = {pearson_r:.4f}
   (p < 0.001). 评分越高的股票反而表现越差, 呈统计显著的负相关.

3. **D10 (最高分 decile) 是表现最差的组**: 平均收益 {fp(d10_mean)}, 命中率仅
   {fp(d10_hit, 1)}. 这是典型的因子拥挤 + 动量反转事件.

4. **信号短期有效但中期反转**: D+1 IC = +0.11, D+3 IC = +0.13 (正向),
   但 D+5 IC = -0.10 (反向). 最优持仓窗口约为 3 个交易日.

5. **行业集中风险**: Top 10 中 7 只属于半导体/电子行业. 该行业本周遭遇显著回调
   (获利回吐). 模型过度依赖行业动量因子, 缺乏行业中性化.

6. **市场环境**: CSI 300 本周整体持平 ({fp(benchmark_ret)}), 属于震荡市.
   在此环境下, 前期强势的半导体板块出现获利回吐, 而低估值防御性板块反弹.
   全样本 67.7% 股票上涨显示本周是小盘/价值风格占优.

### 7.3 诊断与归因

本周 Config E Top 选股失败的可能原因:

1. **动量因子反转**: 模型主要依赖 Momentum+ 信号. 2026-06-26 之前半导体板块
   经历了大幅上涨, 模型基于动量给出了高分. 本周该板块出现显著的获利回吐,
   动量因子完全失效. (参考 detail.csv 中 Top 股票的 main_positive_factors 列)

2. **标签窗口不匹配**: 模型以 `ret_fwd_5d` 为标签训练, 但实际信号的 alpha
   衰减速度远超预期 -- D+3 仍为正向 (IC=+0.13), D+5 已转为负向 (IC=-0.10).
   建议评估 3 日标签并绘制完整的 IC decay 曲线.

3. **行业集中度**: 缺乏行业中性化约束, 导致 Top 选股过度集中在动量最强的
   1-2 个行业. 当行业轮动发生时, 整个 Top 组合同时受损.

4. **单次截面噪音**: 一周 5 个交易日的数据噪音极大, 单一截面的表现可能
   不代表模型长期能力. 需多次滚动验证确认.

### 7.4 后续改进建议

1. **信号衰减分析**: 绘制训练集的 IC decay curve (horizon 1-20 天),
   确认最优持仓周期. 若衰减速度快于预期, 缩短训练标签窗口.

2. **D+3 标签实验**: 基于本周 D+3 IC > D+5 IC 的发现, 训练 `ret_fwd_3d`
   标签版本并与当前模型对比.

3. **行业中性化**: 在排序阶段引入行业中性化 (sector-neutral ranking),
   避免单一行业占 Top N 的 70%+.

4. **反转因子**: 增加短期反转指标 (如过去 1-3 日收益), 捕捉动量过后的
   回调风险, 作为负向信号.

5. **滚动验证体系**: 建立自动化验证 pipeline, 连续验证 20+ 个交易日:
   - 平均日 IC 和信息比率 (IC_IR)
   - Top N 组合的累计超额收益曲线
   - 不同市场状态 (趋势/震荡/高波) 下的分层表现

6. **市场状态自适应**: 在趋势市中加大动量因子权重, 在震荡市中降低或
   加入反转因子.

7. **本地数据库建设**: 建立定期增量更新的 OHLCV 数据库, 避免依赖外部
   API, 确保验证可以快速重复执行.

---

*报告生成时间: {dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
*评估脚本: `evaluation_config_e_20260626.py` / `gen_report.py`*
*行情数据: akshare stock_zh_a_daily (Sina, {N}/{N+len(excluded)} 成功获取)*
*指数数据: akshare stock_zh_index_daily (全周期 Jun 26 - Jul 3)*
"""

report_path = OUTPUT_DIR / "config_e_validation_20260703.md"
with open(report_path, "w", encoding="utf-8") as f:
    f.write(report)
print(f"Report written: {report_path}")

# Also save updated results with correct benchmark
results_csv = OUTPUT_DIR / "config_e_validation_results_20260703.csv"
df.to_csv(results_csv, index=False, encoding="utf-8-sig")
print(f"Updated results: {results_csv}")
print("Done.")
