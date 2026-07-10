# Next-Week D+3 Stock Recommendation Report

**Generated:** 2026-07-09T19:32:34.342457
**Status:** PRODUCTION MODEL INFERENCE — Real-market prediction, NOT backtest

---

## A. Next-Week D+3 Prediction Results

| Parameter | Value |
|-----------|-------|
| Prediction date (T) | **2026-07-03** (Friday) |
| Earliest execution (T+1) | **2026-07-06** (Monday) |
| D+3 target date | **2026-07-09** (Thursday) |
| Comparison date | 2026-07-13 (Monday) |
| Horizon | 3 trading days |
| Model | Ridge Regression (registered: `quant_platform_d3_model` v7, run af8504e3) |
| Feature set | `80fd2338` (registered DEFAULT_SPECS, includes reversal_3d) + industry |
| Feature count | 32 |
| Selection strategy | IndustryNeutralRanker (EqualTopK=3, max_total=50) |
| Universe | CSI 300 (300 symbols) |
| Scored symbols | 299 |

### Top 50 Industry-Neutral Stock Picks

- **Selected:** 50 stocks across 38 industries
- **Exposure flag:** diversified
- **Max single industry:** 6.0%

| # | Symbol | Industry | Model Score | Industry Rank |
|---|--------|----------|-------------|---------------|
| 1 | 000425 | 专用机械 | +0.0091 | 1 |
| 2 | 000617 | 其他金融服务 | +0.0089 | 1 |
| 3 | 000807 | 工业金属 | +0.0104 | 2 |
| 4 | 000975 | 贵金属 | +0.0215 | 1 |
| 5 | 001391 | 跨境物流 | +0.0086 | 1 |
| 6 | 001979 | 商业地产开发和管理 | +0.0096 | 1 |
| 7 | 002001 | 化学药 | +0.0091 | 1 |
| 8 | 002028 | 电网设备 | +0.0083 | 1 |
| 9 | 002074 | 储能设备 | +0.0086 | 1 |
| 10 | 002384 | 电子元件 | +0.0111 | 2 |
| 11 | 002475 | 电子终端及组件 | +0.0094 | 1 |
| 12 | 002493 | 化学原料 | +0.0100 | 3 |
| 13 | 002532 | 工业金属 | +0.0116 | 1 |
| 14 | 002648 | 化学原料 | +0.0115 | 1 |
| 15 | 002837 | 专用机械 | +0.0087 | 1 |
| 16 | 002920 | 汽车零部件与轮胎 | +0.0107 | 1 |
| 17 | 002938 | 电子元件 | +0.0121 | 1 |
| 18 | 300316 | 发电设备 | +0.0082 | 2 |
| 19 | 300413 | 数字媒体 | +0.0099 | 1 |
| 20 | 300433 | 电子终端及组件 | +0.0091 | 2 |
| 21 | 300450 | 专用机械 | +0.0154 | 1 |
| 22 | 300803 | 计算机软件 | +0.0110 | 1 |
| 23 | 600026 | 运输业 | +0.0131 | 1 |
| 24 | 600066 | 交通运输设备 | +0.0108 | 1 |
| 25 | 600104 | 乘用车 | +0.0094 | 2 |
| 26 | 600160 | 化学制品 | +0.0083 | 1 |
| 27 | 600176 | 其他非金属材料 | +0.0082 | 1 |
| 28 | 600188 | 煤炭 | +0.0093 | 2 |
| 29 | 600372 | 航空航天 | +0.0086 | 1 |
| 30 | 600426 | 化学原料 | +0.0108 | 2 |
| 31 | 600460 | 分立器件 | +0.0092 | 1 |
| 32 | 600489 | 贵金属 | +0.0100 | 3 |
| 33 | 600522 | 通信设备 | +0.0098 | 1 |
| 34 | 600547 | 贵金属 | +0.0109 | 2 |
| 35 | 600584 | 集成电路 | +0.0088 | 1 |
| 36 | 600803 | 燃气 | +0.0082 | 1 |
| 37 | 600845 | 软件开发 | +0.0095 | 1 |
| 38 | 600938 | 石油与天然气 | +0.0087 | 1 |
| 39 | 600989 | 化学原料 | +0.0087 | 1 |
| 40 | 601066 | 证券公司 | +0.0090 | 1 |
| 41 | 601238 | 乘用车 | +0.0095 | 1 |
| 42 | 601600 | 工业金属 | +0.0099 | 3 |
| 43 | 601872 | 运输业 | +0.0088 | 2 |
| 44 | 601898 | 煤炭 | +0.0120 | 1 |
| 45 | 603392 | 医疗器械 | +0.0135 | 1 |
| 46 | 603986 | 集成电路 | +0.0101 | 1 |
| 47 | 688082 | 半导体材料与设备 | +0.0094 | 1 |
| 48 | 688126 | 半导体材料与设备 | +0.0092 | 1 |
| 49 | 688223 | 发电设备 | +0.0084 | 1 |
| 50 | 688396 | 集成电路 | +0.0090 | 1 |

### Industry Distribution

| Industry | Count | Fraction | Top Symbols |
|----------|-------|----------|-------------|
| 化学原料 | 3 | 6.0% | 002493, 002648, 600426 |
| 工业金属 | 3 | 6.0% | 000807, 002532, 601600 |
| 贵金属 | 3 | 6.0% | 000975, 600489, 600547 |
| 乘用车 | 2 | 4.0% | 600104, 601238 |
| 发电设备 | 2 | 4.0% | 300316, 688223 |
| 电子元件 | 2 | 4.0% | 002384, 002938 |
| 电子终端及组件 | 2 | 4.0% | 002475, 300433 |
| 煤炭 | 2 | 4.0% | 600188, 601898 |
| 运输业 | 2 | 4.0% | 600026, 601872 |
| 跨境物流 | 1 | 2.0% | 001391 |
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
| 专用机械 (工程机械) | 1 | 2.0% | 000425 |
| 计算机软件 | 1 | 2.0% | 300803 |
| 专用机械 (楼宇设备) | 1 | 2.0% | 002837 |
| 化学制品 | 1 | 2.0% | 600160 |
| 汽车零部件与轮胎 | 1 | 2.0% | 002920 |
| 燃气 | 1 | 2.0% | 600803 |
| 石油与天然气 | 1 | 2.0% | 600938 |
| 其他非金属材料 | 1 | 2.0% | 600176 |
| 储能设备 | 1 | 2.0% | 002074 |
| 航空航天 | 1 | 2.0% | 600372 |
| 软件开发 | 1 | 2.0% | 600845 |
| 数字媒体 | 1 | 2.0% | 300413 |
| 证券公司 | 1 | 2.0% | 601066 |
| 通信设备 | 1 | 2.0% | 600522 |
| 电网设备 | 1 | 2.0% | 002028 |
| 集成电路 | 1 | 2.0% | 688396 |
| 集成电路 (集成电路封测) | 1 | 2.0% | 600584 |
| 集成电路 (集成电路设计) | 1 | 2.0% | 603986 |

## B. Baseline Comparison

### Global Top-50 (Naive, NOT industry-neutral)

**WARNING:** This is a CONTROL BASELINE only — NOT the recommended strategy.

- **Max single industry:** 8.0% (industry-neutral: 6.0%)
- **Top 3 industries by concentration:**
  - 工业金属: 4 stocks (8.0%)
  - 化学原料: 4 stocks (8.0%)
  - 贵金属: 3 stocks (6.0%)

| # | Symbol | Industry | Score |
|---|--------|----------|-------|
| 1 | 000975 | 贵金属 | +0.0215 |
| 2 | 300450 | 专用机械 | +0.0154 |
| 3 | 603392 | 医疗器械 | +0.0135 |
| 4 | 600026 | 运输业 | +0.0131 |
| 5 | 002938 | 电子元件 | +0.0121 |
| 6 | 601898 | 煤炭 | +0.0120 |
| 7 | 002532 | 工业金属 | +0.0116 |
| 8 | 002648 | 化学原料 | +0.0115 |
| 9 | 002384 | 电子元件 | +0.0111 |
| 10 | 300803 | 计算机软件 | +0.0110 |
| 11 | 600547 | 贵金属 | +0.0109 |
| 12 | 600066 | 交通运输设备 | +0.0108 |
| 13 | 600426 | 化学原料 | +0.0108 |
| 14 | 002920 | 汽车零部件与轮胎 | +0.0107 |
| 15 | 000807 | 工业金属 | +0.0104 |
| 16 | 603986 | 集成电路 | +0.0101 |
| 17 | 600489 | 贵金属 | +0.0100 |
| 18 | 002493 | 化学原料 | +0.0100 |
| 19 | 601600 | 工业金属 | +0.0099 |
| 20 | 300413 | 数字媒体 | +0.0099 |
| 21 | 600522 | 通信设备 | +0.0098 |
| 22 | 001979 | 商业地产开发和管理 | +0.0096 |
| 23 | 601238 | 乘用车 | +0.0095 |
| 24 | 600845 | 软件开发 | +0.0095 |
| 25 | 688082 | 半导体材料与设备 | +0.0094 |
| 26 | 002475 | 电子终端及组件 | +0.0094 |
| 27 | 600104 | 乘用车 | +0.0094 |
| 28 | 600346 | 化学原料 | +0.0094 |
| 29 | 600188 | 煤炭 | +0.0093 |
| 30 | 600460 | 分立器件 | +0.0092 |
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

## D. Review Plan (target ~2026-07-09)

### Data Required (T+1 -> T+1+3)

Fetch complete OHLCV covering: **2026-07-06 -> 2026-07-09** (exact realized dates are derived from real data by the review).

### Metrics to Compute

| Metric | Formula | Target |
|--------|---------|--------|
| D+3 Realized Return (per stock) | close(T+1+3) / close(T+1) - 1 | — |
| D+3 Rank IC | Spearman corr(model_score, realized_ret) across all scored symbols | > 0 |
| Selected-set Mean Return | Mean of realized_ret for industry-neutral Top 50 | Positive |
| Hit Rate | Fraction of Top 50 with positive D+3 return | > 50% |
| Global Top-50 Mean Return | Mean of realized_ret for naive global Top 50 | For comparison |
| Industry-neutral vs Global | Difference in mean returns | Industry-neutral > Global |

### Pass Criteria

1. **D+3 Rank IC > 0** (or stable positive across multiple weeks)
2. **Single industry <= 30%** of selected set
3. **Industry-neutral strategy outperforms global baseline** on realized returns
4. **Hit rate > 50%** (more than half of Top 50 have positive returns)

### Validation Script

Run on/after 2026-07-09 after market close:
```bash
cd E:/stock-analysis && PYTHONPATH=. python scripts/d3_wednesday_review.py --horizon 3 --prediction-date 2026-07-03
```

---

## IMPORTANT DISCLAIMER

- **Prediction date:** 2026-07-03 (this is a FORWARD-LOOKING prediction)
- **D+3 verification date:** 2026-07-09 — market data now available
- **Current status:** VERIFIABLE — realized returns can now be computed
- **Next action:** Run review for 2026-07-09 to replace model outputs with realized P&L
- All metrics in this report are MODEL OUTPUTS, not realized P&L
- Do NOT execute trades based solely on this prediction without Wednesday review