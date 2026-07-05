# Next-Week D+3 Stock Recommendation Report

**Generated:** 2026-07-04T17:09:51.175395
**Status:** PRODUCTION MODEL INFERENCE — Real-market prediction, NOT backtest

---

## A. Next-Week D+3 Prediction Results

| Parameter | Value |
|-----------|-------|
| Prediction date (T) | **2026-07-03** (Friday) |
| Earliest execution (T+1) | **2026-07-06** (Monday) |
| D+3 target date | **2026-07-08** (Wednesday) |
| D+5 comparison date | 2026-07-10 (Friday) |
| Horizon | 3 trading days |
| Model | Ridge Regression (production, saved to E:\stock-analysis\models\production\production_model_ridge.pkl) |
| Feature set | `d02a4ebf` + reversal_3d + industry |
| Feature count | 26 |
| Selection strategy | IndustryNeutralRanker (EqualTopK=3, max_total=50) |
| Universe | CSI 300 (300 symbols) |
| Scored symbols | 122 |

### Top 50 Industry-Neutral Stock Picks

- **Selected:** 50 stocks across 37 industries
- **Exposure flag:** diversified
- **Max single industry:** 6.0%

| # | Symbol | Industry | Model Score | Industry Rank |
|---|--------|----------|-------------|---------------|
| 1 | 000100 | 光学光电子 | +0.0103 | 1 |
| 2 | 000625 | 乘用车 | +0.0098 | 2 |
| 3 | 000657 | 稀有金属 | +0.0239 | 1 |
| 4 | 000776 | 证券公司 | +0.0093 | 1 |
| 5 | 000858 | 酒 | +0.0093 | 1 |
| 6 | 000975 | 贵金属 | +0.0126 | 1 |
| 7 | 000977 | 电子终端及组件 | +0.0130 | 1 |
| 8 | 000988 | 其他电子 | +0.0167 | 1 |
| 9 | 002049 | 集成电路 | +0.0141 | 2 |
| 10 | 002074 | 储能设备 | +0.0177 | 1 |
| 11 | 002179 | 航空航天 | +0.0139 | 1 |
| 12 | 002241 | 电子终端及组件 | +0.0118 | 2 |
| 13 | 002353 | 油气开采与油田服务 | +0.0091 | 1 |
| 14 | 002371 | 半导体材料与设备 | +0.0152 | 1 |
| 15 | 002384 | 电子元件 | +0.0201 | 3 |
| 16 | 002460 | 稀有金属 | +0.0165 | 1 |
| 17 | 002463 | 电子元器件 | +0.0203 | 1 |
| 18 | 002466 | 稀有金属 | +0.0153 | 2 |
| 19 | 002475 | 电子终端及组件 | +0.0104 | 3 |
| 20 | 002532 | 工业金属 | +0.0131 | 1 |
| 21 | 002558 | 文化娱乐 | +0.0160 | 1 |
| 22 | 002602 | 文化娱乐 | +0.0127 | 2 |
| 23 | 002648 | 化学原料 | +0.0140 | 2 |
| 24 | 002837 | 专用机械 | +0.0239 | 1 |
| 25 | 002938 | 电子元件 | +0.0241 | 1 |
| 26 | 300033 | 计算机软件 | +0.0142 | 2 |
| 27 | 300251 | 文化娱乐 | +0.0099 | 1 |
| 28 | 300274 | 发电设备 | +0.0234 | 1 |
| 29 | 300308 | 通信设备 | +0.0170 | 2 |
| 30 | 300394 | 通信设备 | +0.0258 | 1 |
| 31 | 300408 | 电子元件 | +0.0266 | 1 |
| 32 | 300413 | 数字媒体 | +0.0133 | 1 |
| 33 | 300418 | 数字媒体 | +0.0102 | 1 |
| 34 | 300433 | 电子终端及组件 | +0.0223 | 1 |
| 35 | 300450 | 专用机械 | +0.0206 | 1 |
| 36 | 300502 | 通信设备 | +0.0160 | 3 |
| 37 | 300661 | 集成电路 | +0.0157 | 1 |
| 38 | 300803 | 计算机软件 | +0.0205 | 1 |
| 39 | 300896 | 医疗器械 | +0.0085 | 1 |
| 40 | 301269 | 软件和信息技术服务业 | +0.0205 | 1 |
| 41 | 600009 | 交通基本设施 | +0.0093 | 1 |
| 42 | 600104 | 乘用车 | +0.0116 | 1 |
| 43 | 600115 | 运输业 | +0.0113 | 1 |
| 44 | 600160 | 化学制品 | +0.0175 | 1 |
| 45 | 600176 | 其他非金属材料 | +0.0162 | 1 |
| 46 | 600183 | 电子元件 | +0.0230 | 2 |
| 47 | 600219 | 工业金属 | +0.0103 | 2 |
| 48 | 600309 | 化学制品 | +0.0101 | 1 |
| 49 | 600346 | 化学原料 | +0.0185 | 1 |
| 50 | 600362 | 工业金属 | +0.0097 | 1 |

### Industry Distribution

| Industry | Count | Fraction | Top Symbols |
|----------|-------|----------|-------------|
| 电子元件 | 3 | 6.0% | 002384, 002938, 600183 |
| 电子终端及组件 | 3 | 6.0% | 002241, 002475, 300433 |
| 通信设备 | 3 | 6.0% | 300308, 300394, 300502 |
| 乘用车 | 2 | 4.0% | 000625, 600104 |
| 化学原料 | 2 | 4.0% | 002648, 600346 |
| 计算机软件 | 2 | 4.0% | 300033, 300803 |
| 文化娱乐 (游戏) | 2 | 4.0% | 002558, 002602 |
| 工业金属 (铝) | 2 | 4.0% | 002532, 600219 |
| 稀有金属 (锂) | 2 | 4.0% | 002460, 002466 |
| 集成电路 | 2 | 4.0% | 002049, 300661 |
| 软件和信息技术服务业 | 1 | 2.0% | 301269 |
| 数字媒体 | 1 | 2.0% | 300418 |
| 发电设备 | 1 | 2.0% | 300274 |
| 专用机械 | 1 | 2.0% | 300450 |
| 其他电子 | 1 | 2.0% | 000988 |
| 医疗器械 | 1 | 2.0% | 300896 |
| 半导体材料与设备 | 1 | 2.0% | 002371 |
| 文化娱乐 | 1 | 2.0% | 300251 |
| 交通基本设施 | 1 | 2.0% | 600009 |
| 专用机械 (楼宇设备) | 1 | 2.0% | 002837 |
| 化学制品 | 1 | 2.0% | 600160 |
| 油气开采与油田服务 | 1 | 2.0% | 002353 |
| 其他非金属材料 | 1 | 2.0% | 600176 |
| 电子元器件 | 1 | 2.0% | 002463 |
| 储能设备 | 1 | 2.0% | 002074 |
| 电子终端及组件 (电脑与外围设备) | 1 | 2.0% | 000977 |
| 酒 | 1 | 2.0% | 000858 |
| 化学制品 (聚氨酯) | 1 | 2.0% | 600309 |
| 航空航天 | 1 | 2.0% | 002179 |
| 运输业 | 1 | 2.0% | 600115 |
| 电子元件 (被动元件) | 1 | 2.0% | 300408 |
| 数字媒体 (视频媒体) | 1 | 2.0% | 300413 |
| 证券公司 | 1 | 2.0% | 000776 |
| 稀有金属 | 1 | 2.0% | 000657 |
| 工业金属 | 1 | 2.0% | 600362 |
| 光学光电子 | 1 | 2.0% | 000100 |
| 贵金属 | 1 | 2.0% | 000975 |

## B. Baseline Comparison

### Global Top-50 (Naive, NOT industry-neutral)

**WARNING:** This is a CONTROL BASELINE only — NOT the recommended strategy.

- **Max single industry:** 8.0% (industry-neutral: 6.0%)
- **Top 3 industries by concentration:**
  - 电子元件: 4 stocks (8.0%)
  - 电子终端及组件: 3 stocks (6.0%)
  - 通信设备: 3 stocks (6.0%)

| # | Symbol | Industry | Score |
|---|--------|----------|-------|
| 1 | 300408 | 电子元件 | +0.0266 |
| 2 | 300394 | 通信设备 | +0.0258 |
| 3 | 002938 | 电子元件 | +0.0241 |
| 4 | 002837 | 专用机械 | +0.0239 |
| 5 | 000657 | 稀有金属 | +0.0239 |
| 6 | 300274 | 发电设备 | +0.0234 |
| 7 | 600183 | 电子元件 | +0.0230 |
| 8 | 300433 | 电子终端及组件 | +0.0223 |
| 9 | 300450 | 专用机械 | +0.0206 |
| 10 | 300803 | 计算机软件 | +0.0205 |
| 11 | 301269 | 软件和信息技术服务业 | +0.0205 |
| 12 | 002463 | 电子元器件 | +0.0203 |
| 13 | 002384 | 电子元件 | +0.0201 |
| 14 | 600346 | 化学原料 | +0.0185 |
| 15 | 002074 | 储能设备 | +0.0177 |
| 16 | 600160 | 化学制品 | +0.0175 |
| 17 | 300308 | 通信设备 | +0.0170 |
| 18 | 000988 | 其他电子 | +0.0167 |
| 19 | 002460 | 稀有金属 | +0.0165 |
| 20 | 600176 | 其他非金属材料 | +0.0162 |
| 21 | 002558 | 文化娱乐 | +0.0160 |
| 22 | 300502 | 通信设备 | +0.0160 |
| 23 | 300661 | 集成电路 | +0.0157 |
| 24 | 002466 | 稀有金属 | +0.0153 |
| 25 | 002371 | 半导体材料与设备 | +0.0152 |
| 26 | 300033 | 计算机软件 | +0.0142 |
| 27 | 002049 | 集成电路 | +0.0141 |
| 28 | 002648 | 化学原料 | +0.0140 |
| 29 | 002179 | 航空航天 | +0.0139 |
| 30 | 300476 | 电子元件 | +0.0136 |
| ... | ... | ... | ... |

### Overlap with Global Top-50
- **Overlap:** 49/50 symbols appear in both lists
- **Industry-neutral only:** 1 unique picks
- **Global only:** 1 unique picks

## C. Risk Checks

| Check | Result | Threshold | Status |
|-------|--------|-----------|--------|
| Industry concentration (max) | 6.0% | 30.0% | PASS |
| Exposure flag | diversified | diversified/balanced | PASS |
| Industry-neutral constraint | Yes (EqualTopK=3) | Required | PASS |
| Universe constraint | CSI 300 only | Required | PASS |
| Model type | Ridge (linear) | Production | PASS |

All risk checks PASSED. No exposure warnings triggered.

## D. Wednesday Review Plan (2026-07-08)

### Data Required (D+1 -> D+3)

Fetch complete OHLCV covering: **2026-07-06 (Mon) -> 2026-07-08 (Wed)**

```python
# Fetch command (run on Wednesday after market close):
cd E:/stock-analysis && PYTHONPATH=. python -c "
import akshare as ak
for sym in <selected_50_symbols>:
    df = ak.stock_zh_a_daily(symbol=prefix(sym), start_date='20260706', end_date='20260709', adjust='qfq')
    # ... compute realized returns
"
```

### Metrics to Compute

| Metric | Formula | Target |
|--------|---------|--------|
| D+3 Realized Return (per stock) | close(07-08) / close(07-06) - 1 | — |
| D+3 Rank IC | Spearman corr(model_score, realized_ret_3d) across all scored symbols | > 0 |
| Selected-set D+3 Mean Return | Mean of realized_ret_3d for industry-neutral Top 50 | Positive |
| Hit Rate | Fraction of Top 50 with positive D+3 return | > 50% |
| Global Top-50 D+3 Mean Return | Mean of realized_ret_3d for naive global Top 50 | For comparison |
| Industry-neutral vs Global | Difference in mean returns | Industry-neutral > Global |

### Pass Criteria

1. **D+3 Rank IC > 0** (or stable positive across multiple weeks)
2. **Single industry <= 30%** of selected set
3. **Industry-neutral strategy outperforms global baseline** on D+3 realized returns
4. **Hit rate > 50%** (more than half of Top 50 have positive D+3 returns)

### Validation Script

Run on Wednesday (2026-07-08) after market close:
```bash
cd E:/stock-analysis && PYTHONPATH=. python scripts/d3_wednesday_review.py
```

---

## IMPORTANT DISCLAIMER

- **Prediction date:** 2026-07-03 (this is a FORWARD-LOOKING prediction)
- **D+3 verification date:** 2026-07-08 — market data NOT YET available
- **Current status:** PREDICTION PHASE — realized returns are UNKNOWN
- **Next action:** Run Wednesday review on 2026-07-08 after market close (after 15:00 CST)
- All metrics in this report are MODEL OUTPUTS, not realized P&L
- Do NOT execute trades based solely on this prediction without Wednesday review