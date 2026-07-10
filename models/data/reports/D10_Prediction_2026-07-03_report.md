# Next-Week D+10 Stock Recommendation Report

**Generated:** 2026-07-09T19:24:50.708994
**Status:** PRODUCTION MODEL INFERENCE — Real-market prediction, NOT backtest

---

## A. Next-Week D+10 Prediction Results

| Parameter | Value |
|-----------|-------|
| Prediction date (T) | **2026-07-03** (Friday) |
| Earliest execution (T+1) | **2026-07-06** (Monday) |
| D+10 target date | **2026-07-20** (Monday) |
| Comparison date | 2026-07-22 (Wednesday) |
| Horizon | 10 trading days |
| Model | Ridge Regression (registered: `quant_platform_d10_model` v1, run 62a48658) |
| Feature set | `80fd2338` (registered DEFAULT_SPECS, includes reversal_3d) + industry |
| Feature count | 32 |
| Selection strategy | IndustryNeutralRanker (EqualTopK=3, max_total=50) |
| Universe | CSI 300 (300 symbols) |
| Scored symbols | 299 |

### Top 50 Industry-Neutral Stock Picks

- **Selected:** 50 stocks across 36 industries
- **Exposure flag:** diversified
- **Max single industry:** 6.0%

| # | Symbol | Industry | Model Score | Industry Rank |
|---|--------|----------|-------------|---------------|
| 1 | 000338 | 交通运输设备 | +0.0250 | 1 |
| 2 | 000617 | 其他金融服务 | +0.0258 | 1 |
| 3 | 000807 | 工业金属 | +0.0328 | 3 |
| 4 | 000975 | 贵金属 | +0.0394 | 1 |
| 5 | 001979 | 商业地产开发和管理 | +0.0288 | 1 |
| 6 | 002028 | 电网设备 | +0.0267 | 1 |
| 7 | 002074 | 储能设备 | +0.0289 | 1 |
| 8 | 002460 | 稀有金属 | +0.0269 | 1 |
| 9 | 002532 | 工业金属 | +0.0380 | 1 |
| 10 | 002938 | 电子元件 | +0.0253 | 2 |
| 11 | 300122 | 生物药品 | +0.0266 | 1 |
| 12 | 300274 | 发电设备 | +0.0393 | 1 |
| 13 | 300394 | 通信设备 | +0.0401 | 1 |
| 14 | 300408 | 电子元件 | +0.0261 | 1 |
| 15 | 300413 | 数字媒体 | +0.0282 | 1 |
| 16 | 300418 | 数字媒体 | +0.0256 | 1 |
| 17 | 300433 | 电子终端及组件 | +0.0248 | 1 |
| 18 | 300450 | 专用机械 | +0.0321 | 1 |
| 19 | 300476 | 电子元件 | +0.0285 | 1 |
| 20 | 301308 | 数字芯片设计 | +0.0298 | 1 |
| 21 | 600026 | 运输业 | +0.0317 | 1 |
| 22 | 600104 | 乘用车 | +0.0255 | 3 |
| 23 | 600176 | 其他非金属材料 | +0.0304 | 1 |
| 24 | 600188 | 煤炭 | +0.0282 | 2 |
| 25 | 600346 | 化学原料 | +0.0260 | 2 |
| 26 | 600415 | 商业服务与用品 | +0.0265 | 1 |
| 27 | 600426 | 化学原料 | +0.0321 | 1 |
| 28 | 600438 | 农牧渔产品 | +0.0299 | 1 |
| 29 | 600460 | 分立器件 | +0.0246 | 1 |
| 30 | 600489 | 贵金属 | +0.0263 | 3 |
| 31 | 600547 | 贵金属 | +0.0300 | 2 |
| 32 | 600845 | 软件开发 | +0.0292 | 1 |
| 33 | 600875 | 发电设备 | +0.0257 | 1 |
| 34 | 600989 | 化学原料 | +0.0256 | 1 |
| 35 | 601127 | 乘用车 | +0.0259 | 2 |
| 36 | 601238 | 乘用车 | +0.0275 | 1 |
| 37 | 601600 | 工业金属 | +0.0357 | 2 |
| 38 | 601898 | 煤炭 | +0.0319 | 1 |
| 39 | 603392 | 医疗器械 | +0.0382 | 1 |
| 40 | 603799 | 工业金属 | +0.0276 | 1 |
| 41 | 603986 | 集成电路 | +0.0324 | 1 |
| 42 | 688008 | 集成电路 | +0.0282 | 3 |
| 43 | 688012 | 半导体材料与设备 | +0.0284 | 1 |
| 44 | 688082 | 半导体材料与设备 | +0.0278 | 2 |
| 45 | 688126 | 半导体材料与设备 | +0.0265 | 1 |
| 46 | 688223 | 发电设备 | +0.0358 | 2 |
| 47 | 688256 | 集成电路 | +0.0289 | 2 |
| 48 | 688396 | 集成电路 | +0.0271 | 1 |
| 49 | 688472 | 光伏设备 | +0.0261 | 3 |
| 50 | 688506 | 生物制品 | +0.0267 | 1 |

### Industry Distribution

| Industry | Count | Fraction | Top Symbols |
|----------|-------|----------|-------------|
| 乘用车 | 3 | 6.0% | 600104, 601127, 601238 |
| 发电设备 | 3 | 6.0% | 300274, 688223, 688472 |
| 工业金属 (铝) | 3 | 6.0% | 000807, 002532, 601600 |
| 集成电路 (集成电路设计) | 3 | 6.0% | 603986, 688008, 688256 |
| 贵金属 | 3 | 6.0% | 000975, 600489, 600547 |
| 化学原料 | 2 | 4.0% | 600346, 600426 |
| 半导体材料与设备 (半导体设备) | 2 | 4.0% | 688012, 688082 |
| 电子元件 | 2 | 4.0% | 002938, 300476 |
| 煤炭 | 2 | 4.0% | 600188, 601898 |
| 生物制品 | 1 | 2.0% | 688506 |
| 数字芯片设计 | 1 | 2.0% | 301308 |
| 数字媒体 | 1 | 2.0% | 300418 |
| 医疗器械 | 1 | 2.0% | 603392 |
| 专用机械 | 1 | 2.0% | 300450 |
| 农牧渔产品 | 1 | 2.0% | 600438 |
| 发电设备 (其他发电设备) | 1 | 2.0% | 600875 |
| 其他金融服务 | 1 | 2.0% | 000617 |
| 分立器件 | 1 | 2.0% | 600460 |
| 化学原料 (化学原料) | 1 | 2.0% | 600989 |
| 半导体材料与设备 | 1 | 2.0% | 688126 |
| 商业地产开发和管理 | 1 | 2.0% | 001979 |
| 交通运输设备 | 1 | 2.0% | 000338 |
| 商业服务与用品 | 1 | 2.0% | 600415 |
| 电子终端及组件 | 1 | 2.0% | 300433 |
| 其他非金属材料 | 1 | 2.0% | 600176 |
| 储能设备 | 1 | 2.0% | 002074 |
| 生物药品 | 1 | 2.0% | 300122 |
| 运输业 | 1 | 2.0% | 600026 |
| 软件开发 | 1 | 2.0% | 600845 |
| 电子元件 (被动元件) | 1 | 2.0% | 300408 |
| 数字媒体 (视频媒体) | 1 | 2.0% | 300413 |
| 通信设备 | 1 | 2.0% | 300394 |
| 电网设备 | 1 | 2.0% | 002028 |
| 工业金属 | 1 | 2.0% | 603799 |
| 稀有金属 | 1 | 2.0% | 002460 |
| 集成电路 | 1 | 2.0% | 688396 |

## B. Baseline Comparison

### Global Top-50 (Naive, NOT industry-neutral)

**WARNING:** This is a CONTROL BASELINE only — NOT the recommended strategy.

- **Max single industry:** 10.0% (industry-neutral: 6.0%)
- **Top 3 industries by concentration:**
  - 集成电路: 5 stocks (10.0%)
  - 工业金属: 4 stocks (8.0%)
  - 乘用车: 3 stocks (6.0%)

| # | Symbol | Industry | Score |
|---|--------|----------|-------|
| 1 | 300394 | 通信设备 | +0.0401 |
| 2 | 000975 | 贵金属 | +0.0394 |
| 3 | 300274 | 发电设备 | +0.0393 |
| 4 | 603392 | 医疗器械 | +0.0382 |
| 5 | 002532 | 工业金属 | +0.0380 |
| 6 | 688223 | 发电设备 | +0.0358 |
| 7 | 601600 | 工业金属 | +0.0357 |
| 8 | 000807 | 工业金属 | +0.0328 |
| 9 | 603986 | 集成电路 | +0.0324 |
| 10 | 600426 | 化学原料 | +0.0321 |
| 11 | 300450 | 专用机械 | +0.0321 |
| 12 | 601898 | 煤炭 | +0.0319 |
| 13 | 600026 | 运输业 | +0.0317 |
| 14 | 600176 | 其他非金属材料 | +0.0304 |
| 15 | 600547 | 贵金属 | +0.0300 |
| 16 | 600438 | 农牧渔产品 | +0.0299 |
| 17 | 301308 | 数字芯片设计 | +0.0298 |
| 18 | 600845 | 软件开发 | +0.0292 |
| 19 | 002074 | 储能设备 | +0.0289 |
| 20 | 688256 | 集成电路 | +0.0289 |
| 21 | 001979 | 商业地产开发和管理 | +0.0288 |
| 22 | 300476 | 电子元件 | +0.0285 |
| 23 | 688012 | 半导体材料与设备 | +0.0284 |
| 24 | 600219 | 工业金属 | +0.0283 |
| 25 | 300413 | 数字媒体 | +0.0282 |
| 26 | 600188 | 煤炭 | +0.0282 |
| 27 | 688008 | 集成电路 | +0.0282 |
| 28 | 688082 | 半导体材料与设备 | +0.0278 |
| 29 | 603799 | 工业金属 | +0.0276 |
| 30 | 601238 | 乘用车 | +0.0275 |
| ... | ... | ... | ... |

### Overlap with Global Top-50
- **Overlap:** 47/50 symbols appear in both lists
- **Industry-neutral only:** 3 unique picks
- **Global only:** 3 unique picks

## C. Risk Checks

| Check | Result | Threshold | Status |
|-------|--------|-----------|--------|
| Industry concentration (max) | 6.0% | 30.0% | PASS |
| Exposure flag | diversified | diversified/balanced | PASS |
| Industry-neutral constraint | Yes (EqualTopK=3) | Required | PASS |
| Universe constraint | CSI 300 only | Required | PASS |
| Model type | Ridge (linear) | Production | PASS |

All risk checks PASSED. No exposure warnings triggered.

## D. Review Plan (target ~2026-07-20)

### Data Required (T+1 -> T+1+10)

Fetch complete OHLCV covering: **2026-07-06 -> 2026-07-20** (exact realized dates are derived from real data by the review).

### Metrics to Compute

| Metric | Formula | Target |
|--------|---------|--------|
| D+10 Realized Return (per stock) | close(T+1+10) / close(T+1) - 1 | — |
| D+10 Rank IC | Spearman corr(model_score, realized_ret) across all scored symbols | > 0 |
| Selected-set Mean Return | Mean of realized_ret for industry-neutral Top 50 | Positive |
| Hit Rate | Fraction of Top 50 with positive D+10 return | > 50% |
| Global Top-50 Mean Return | Mean of realized_ret for naive global Top 50 | For comparison |
| Industry-neutral vs Global | Difference in mean returns | Industry-neutral > Global |

### Pass Criteria

1. **D+10 Rank IC > 0** (or stable positive across multiple weeks)
2. **Single industry <= 30%** of selected set
3. **Industry-neutral strategy outperforms global baseline** on realized returns
4. **Hit rate > 50%** (more than half of Top 50 have positive returns)

### Validation Script

Run on/after 2026-07-20 after market close:
```bash
cd E:/stock-analysis && PYTHONPATH=. python scripts/d3_wednesday_review.py --horizon 10 --prediction-date 2026-07-03
```

---

## IMPORTANT DISCLAIMER

- **Prediction date:** 2026-07-03 (this is a FORWARD-LOOKING prediction)
- **D+10 verification date:** 2026-07-20 — market data NOT YET available
- **Current status:** PREDICTION PHASE — realized returns are UNKNOWN
- **Next action:** Run review on/after 2026-07-20 after market close (after 15:00 CST)
- All metrics in this report are MODEL OUTPUTS, not realized P&L
- Do NOT execute trades based solely on this prediction without Wednesday review