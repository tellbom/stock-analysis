# Next-Week D+3 Stock Recommendation Report

**Generated:** 2026-07-05T17:10:32.041137
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
| Model | Ridge Regression (registered: `quant_platform_d3_model` v4, run e88c8002) |
| Feature set | `d02a4ebf` + reversal_3d + industry |
| Feature count | 25 |
| Selection strategy | IndustryNeutralRanker (EqualTopK=3, max_total=50) |
| Universe | CSI 300 (300 symbols) |
| Scored symbols | 298 |

### Top 50 Industry-Neutral Stock Picks

- **Selected:** 50 stocks across 36 industries
- **Exposure flag:** diversified
- **Max single industry:** 6.0%

| # | Symbol | Industry | Model Score | Industry Rank |
|---|--------|----------|-------------|---------------|
| 1 | 000617 | 其他金融服务 | +0.0084 | 1 |
| 2 | 000657 | 稀有金属 | +0.0088 | 1 |
| 3 | 000807 | 工业金属 | +0.0107 | 2 |
| 4 | 000975 | 贵金属 | +0.0203 | 1 |
| 5 | 001979 | 商业地产开发和管理 | +0.0086 | 1 |
| 6 | 002001 | 化学药 | +0.0088 | 1 |
| 7 | 002028 | 电网设备 | +0.0092 | 1 |
| 8 | 002074 | 储能设备 | +0.0082 | 1 |
| 9 | 002384 | 电子元件 | +0.0122 | 2 |
| 10 | 002460 | 稀有金属 | +0.0082 | 1 |
| 11 | 002475 | 电子终端及组件 | +0.0100 | 1 |
| 12 | 002493 | 化学原料 | +0.0097 | 3 |
| 13 | 002532 | 工业金属 | +0.0114 | 1 |
| 14 | 002648 | 化学原料 | +0.0118 | 1 |
| 15 | 002837 | 专用机械 | +0.0094 | 1 |
| 16 | 002920 | 汽车零部件与轮胎 | +0.0111 | 1 |
| 17 | 002938 | 电子元件 | +0.0130 | 1 |
| 18 | 300274 | 发电设备 | +0.0083 | 2 |
| 19 | 300316 | 发电设备 | +0.0096 | 1 |
| 20 | 300394 | 通信设备 | +0.0088 | 2 |
| 21 | 300413 | 数字媒体 | +0.0089 | 1 |
| 22 | 300433 | 电子终端及组件 | +0.0091 | 2 |
| 23 | 300450 | 专用机械 | +0.0154 | 1 |
| 24 | 300803 | 计算机软件 | +0.0120 | 1 |
| 25 | 301308 | 数字芯片设计 | +0.0086 | 1 |
| 26 | 600026 | 运输业 | +0.0126 | 1 |
| 27 | 600066 | 交通运输设备 | +0.0096 | 1 |
| 28 | 600104 | 乘用车 | +0.0093 | 1 |
| 29 | 600188 | 煤炭 | +0.0090 | 2 |
| 30 | 600426 | 化学原料 | +0.0111 | 2 |
| 31 | 600460 | 分立器件 | +0.0099 | 1 |
| 32 | 600489 | 贵金属 | +0.0085 | 3 |
| 33 | 600522 | 通信设备 | +0.0118 | 1 |
| 34 | 600547 | 贵金属 | +0.0094 | 2 |
| 35 | 600584 | 集成电路 | +0.0102 | 1 |
| 36 | 600845 | 软件开发 | +0.0087 | 1 |
| 37 | 600893 | 航空航天 | +0.0084 | 1 |
| 38 | 600938 | 石油与天然气 | +0.0086 | 1 |
| 39 | 600989 | 化学原料 | +0.0083 | 1 |
| 40 | 601066 | 证券公司 | +0.0098 | 1 |
| 41 | 601127 | 乘用车 | +0.0087 | 3 |
| 42 | 601238 | 乘用车 | +0.0091 | 2 |
| 43 | 601600 | 工业金属 | +0.0103 | 3 |
| 44 | 601872 | 运输业 | +0.0096 | 2 |
| 45 | 601898 | 煤炭 | +0.0113 | 1 |
| 46 | 603392 | 医疗器械 | +0.0132 | 1 |
| 47 | 603986 | 集成电路 | +0.0120 | 1 |
| 48 | 688082 | 半导体材料与设备 | +0.0104 | 1 |
| 49 | 688126 | 半导体材料与设备 | +0.0091 | 1 |
| 50 | 688396 | 集成电路 | +0.0097 | 1 |

### Industry Distribution

| Industry | Count | Fraction | Top Symbols |
|----------|-------|----------|-------------|
| 乘用车 | 3 | 6.0% | 600104, 601127, 601238 |
| 化学原料 | 3 | 6.0% | 002493, 002648, 600426 |
| 工业金属 | 3 | 6.0% | 000807, 002532, 601600 |
| 贵金属 | 3 | 6.0% | 000975, 600489, 600547 |
| 发电设备 | 2 | 4.0% | 300274, 300316 |
| 电子元件 | 2 | 4.0% | 002384, 002938 |
| 电子终端及组件 | 2 | 4.0% | 002475, 300433 |
| 煤炭 | 2 | 4.0% | 600188, 601898 |
| 运输业 | 2 | 4.0% | 600026, 601872 |
| 通信设备 | 2 | 4.0% | 300394, 600522 |
| 数字芯片设计 | 1 | 2.0% | 301308 |
| 医疗器械 | 1 | 2.0% | 603392 |
| 专用机械 | 1 | 2.0% | 300450 |
| 其他金融服务 | 1 | 2.0% | 000617 |
| 分立器件 | 1 | 2.0% | 600460 |
| 化学原料 (化学原料) | 1 | 2.0% | 600989 |
| 半导体材料与设备 | 1 | 2.0% | 688126 |
| 半导体材料与设备 (半导体设备) | 1 | 2.0% | 688082 |
| 化学药 | 1 | 2.0% | 002001 |
| 商业地产开发和管理 | 1 | 2.0% | 001979 |
| 交通运输设备 | 1 | 2.0% | 600066 |
| 计算机软件 | 1 | 2.0% | 300803 |
| 专用机械 (楼宇设备) | 1 | 2.0% | 002837 |
| 汽车零部件与轮胎 | 1 | 2.0% | 002920 |
| 石油与天然气 | 1 | 2.0% | 600938 |
| 储能设备 | 1 | 2.0% | 002074 |
| 航空航天 | 1 | 2.0% | 600893 |
| 软件开发 | 1 | 2.0% | 600845 |
| 数字媒体 | 1 | 2.0% | 300413 |
| 证券公司 | 1 | 2.0% | 601066 |
| 电网设备 | 1 | 2.0% | 002028 |
| 稀有金属 | 1 | 2.0% | 000657 |
| 稀有金属 (锂) | 1 | 2.0% | 002460 |
| 集成电路 | 1 | 2.0% | 688396 |
| 集成电路 (集成电路封测) | 1 | 2.0% | 600584 |
| 集成电路 (集成电路设计) | 1 | 2.0% | 603986 |

## B. Baseline Comparison

### Global Top-50 (Naive, NOT industry-neutral)

**WARNING:** This is a CONTROL BASELINE only — NOT the recommended strategy.

- **Max single industry:** 8.0% (industry-neutral: 6.0%)
- **Top 3 industries by concentration:**
  - 化学原料: 4 stocks (8.0%)
  - 工业金属: 4 stocks (8.0%)
  - 贵金属: 3 stocks (6.0%)

| # | Symbol | Industry | Score |
|---|--------|----------|-------|
| 1 | 000975 | 贵金属 | +0.0203 |
| 2 | 300450 | 专用机械 | +0.0154 |
| 3 | 603392 | 医疗器械 | +0.0132 |
| 4 | 002938 | 电子元件 | +0.0130 |
| 5 | 600026 | 运输业 | +0.0126 |
| 6 | 002384 | 电子元件 | +0.0122 |
| 7 | 603986 | 集成电路 | +0.0120 |
| 8 | 300803 | 计算机软件 | +0.0120 |
| 9 | 600522 | 通信设备 | +0.0118 |
| 10 | 002648 | 化学原料 | +0.0118 |
| 11 | 002532 | 工业金属 | +0.0114 |
| 12 | 601898 | 煤炭 | +0.0113 |
| 13 | 600426 | 化学原料 | +0.0111 |
| 14 | 002920 | 汽车零部件与轮胎 | +0.0111 |
| 15 | 000807 | 工业金属 | +0.0107 |
| 16 | 688082 | 半导体材料与设备 | +0.0104 |
| 17 | 601600 | 工业金属 | +0.0103 |
| 18 | 600584 | 集成电路 | +0.0102 |
| 19 | 002475 | 电子终端及组件 | +0.0100 |
| 20 | 600460 | 分立器件 | +0.0099 |
| 21 | 601066 | 证券公司 | +0.0098 |
| 22 | 002493 | 化学原料 | +0.0097 |
| 23 | 688396 | 集成电路 | +0.0097 |
| 24 | 601872 | 运输业 | +0.0096 |
| 25 | 300316 | 发电设备 | +0.0096 |
| 26 | 600066 | 交通运输设备 | +0.0096 |
| 27 | 600547 | 贵金属 | +0.0094 |
| 28 | 002837 | 专用机械 | +0.0094 |
| 29 | 600104 | 乘用车 | +0.0093 |
| 30 | 002028 | 电网设备 | +0.0092 |
| ... | ... | ... | ... |

### Overlap with Global Top-50
- **Overlap:** 48/50 symbols appear in both lists
- **Industry-neutral only:** 2 unique picks
- **Global only:** 2 unique picks

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