# Next-Week D+3 Stock Recommendation Report

**Generated:** 2026-07-05T16:56:23.719957
**Status:** PRODUCTION MODEL INFERENCE — Real-market prediction, NOT backtest

---

## A. Next-Week D+3 Prediction Results

| Parameter | Value |
|-----------|-------|
| Prediction date (T) | **2026-07-02** (Thursday — PIT: data <= T close) |
| Earliest execution (T+1) | **2026-07-03** (Monday) |
| D+3 target date | **2026-07-08** (Wednesday) |
| D+5 comparison date | 2026-07-10 (Friday) |
| Horizon | 3 trading days |
| Model | Ridge Regression (registered: `quant_platform_d3_model` v3, run b8d48766) |
| Feature set | `d02a4ebf` + reversal_3d + industry |
| Feature count | 25 |
| Selection strategy | IndustryNeutralRanker (EqualTopK=3, max_total=50) |
| Universe | CSI 300 (300 symbols) |
| Scored symbols | 298 |

### Top 50 Industry-Neutral Stock Picks

- **Selected:** 50 stocks across 35 industries
- **Exposure flag:** diversified
- **Max single industry:** 6.0%

| # | Symbol | Industry | Model Score | Industry Rank |
|---|--------|----------|-------------|---------------|
| 1 | 000338 | 交通运输设备 | +0.0104 | 1 |
| 2 | 000425 | 专用机械 | +0.0088 | 1 |
| 3 | 000657 | 稀有金属 | +0.0112 | 1 |
| 4 | 000807 | 工业金属 | +0.0104 | 2 |
| 5 | 000975 | 贵金属 | +0.0214 | 1 |
| 6 | 001979 | 商业地产开发和管理 | +0.0097 | 1 |
| 7 | 002001 | 化学药 | +0.0089 | 1 |
| 8 | 002028 | 电网设备 | +0.0109 | 1 |
| 9 | 002074 | 储能设备 | +0.0098 | 1 |
| 10 | 002384 | 电子元件 | +0.0112 | 1 |
| 11 | 002460 | 稀有金属 | +0.0097 | 1 |
| 12 | 002532 | 工业金属 | +0.0124 | 1 |
| 13 | 002837 | 专用机械 | +0.0092 | 1 |
| 14 | 002920 | 汽车零部件与轮胎 | +0.0105 | 1 |
| 15 | 002938 | 电子元件 | +0.0110 | 2 |
| 16 | 300251 | 文化娱乐 | +0.0089 | 1 |
| 17 | 300274 | 发电设备 | +0.0093 | 3 |
| 18 | 300316 | 发电设备 | +0.0108 | 1 |
| 19 | 300394 | 通信设备 | +0.0108 | 2 |
| 20 | 300408 | 电子元件 | +0.0111 | 1 |
| 21 | 300413 | 数字媒体 | +0.0088 | 1 |
| 22 | 300418 | 数字媒体 | +0.0096 | 1 |
| 23 | 300433 | 电子终端及组件 | +0.0130 | 1 |
| 24 | 300450 | 专用机械 | +0.0139 | 1 |
| 25 | 300803 | 计算机软件 | +0.0103 | 1 |
| 26 | 301308 | 数字芯片设计 | +0.0098 | 1 |
| 27 | 600026 | 运输业 | +0.0126 | 1 |
| 28 | 600104 | 乘用车 | +0.0107 | 1 |
| 29 | 600188 | 煤炭 | +0.0088 | 2 |
| 30 | 600426 | 化学原料 | +0.0117 | 1 |
| 31 | 600482 | 交通运输设备 | +0.0103 | 1 |
| 32 | 600489 | 贵金属 | +0.0103 | 3 |
| 33 | 600522 | 通信设备 | +0.0148 | 1 |
| 34 | 600547 | 贵金属 | +0.0111 | 2 |
| 35 | 600584 | 集成电路 | +0.0107 | 1 |
| 36 | 600845 | 软件开发 | +0.0094 | 1 |
| 37 | 600875 | 发电设备 | +0.0088 | 1 |
| 38 | 601127 | 乘用车 | +0.0088 | 2 |
| 39 | 601138 | 电子终端及组件 | +0.0096 | 2 |
| 40 | 601600 | 工业金属 | +0.0094 | 3 |
| 41 | 601872 | 运输业 | +0.0111 | 2 |
| 42 | 601898 | 煤炭 | +0.0101 | 1 |
| 43 | 603392 | 医疗器械 | +0.0146 | 1 |
| 44 | 603799 | 工业金属 | +0.0093 | 1 |
| 45 | 603986 | 集成电路 | +0.0149 | 1 |
| 46 | 688041 | 数字芯片设计 | +0.0094 | 2 |
| 47 | 688082 | 半导体材料与设备 | +0.0135 | 1 |
| 48 | 688223 | 发电设备 | +0.0099 | 2 |
| 49 | 688256 | 集成电路 | +0.0115 | 2 |
| 50 | 688521 | 集成电路 | +0.0098 | 3 |

### Industry Distribution

| Industry | Count | Fraction | Top Symbols |
|----------|-------|----------|-------------|
| 发电设备 | 3 | 6.0% | 300274, 300316, 688223 |
| 工业金属 (铝) | 3 | 6.0% | 000807, 002532, 601600 |
| 集成电路 (集成电路设计) | 3 | 6.0% | 603986, 688256, 688521 |
| 贵金属 | 3 | 6.0% | 000975, 600489, 600547 |
| 数字芯片设计 | 2 | 4.0% | 301308, 688041 |
| 乘用车 | 2 | 4.0% | 600104, 601127 |
| 电子元件 | 2 | 4.0% | 002384, 002938 |
| 电子终端及组件 | 2 | 4.0% | 300433, 601138 |
| 煤炭 | 2 | 4.0% | 600188, 601898 |
| 运输业 | 2 | 4.0% | 600026, 601872 |
| 通信设备 | 2 | 4.0% | 300394, 600522 |
| 数字媒体 | 1 | 2.0% | 300418 |
| 医疗器械 | 1 | 2.0% | 603392 |
| 专用机械 | 1 | 2.0% | 300450 |
| 化学原料 | 1 | 2.0% | 600426 |
| 发电设备 (其他发电设备) | 1 | 2.0% | 600875 |
| 半导体材料与设备 | 1 | 2.0% | 688082 |
| 化学药 | 1 | 2.0% | 002001 |
| 商业地产开发和管理 | 1 | 2.0% | 001979 |
| 交通运输设备 | 1 | 2.0% | 000338 |
| 专用机械 (工程机械) | 1 | 2.0% | 000425 |
| 计算机软件 | 1 | 2.0% | 300803 |
| 文化娱乐 | 1 | 2.0% | 300251 |
| 专用机械 (楼宇设备) | 1 | 2.0% | 002837 |
| 汽车零部件与轮胎 | 1 | 2.0% | 002920 |
| 储能设备 | 1 | 2.0% | 002074 |
| 交通运输设备 (船舶及其他航运设备) | 1 | 2.0% | 600482 |
| 软件开发 | 1 | 2.0% | 600845 |
| 电子元件 (被动元件) | 1 | 2.0% | 300408 |
| 数字媒体 (视频媒体) | 1 | 2.0% | 300413 |
| 电网设备 | 1 | 2.0% | 002028 |
| 稀有金属 | 1 | 2.0% | 000657 |
| 工业金属 | 1 | 2.0% | 603799 |
| 稀有金属 (锂) | 1 | 2.0% | 002460 |
| 集成电路 | 1 | 2.0% | 600584 |

## B. Baseline Comparison

### Global Top-50 (Naive, NOT industry-neutral)

**WARNING:** This is a CONTROL BASELINE only — NOT the recommended strategy.

- **Max single industry:** 8.0% (industry-neutral: 6.0%)
- **Top 3 industries by concentration:**
  - 集成电路: 4 stocks (8.0%)
  - 工业金属: 3 stocks (6.0%)
  - 发电设备: 3 stocks (6.0%)

| # | Symbol | Industry | Score |
|---|--------|----------|-------|
| 1 | 000975 | 贵金属 | +0.0214 |
| 2 | 603986 | 集成电路 | +0.0149 |
| 3 | 600522 | 通信设备 | +0.0148 |
| 4 | 603392 | 医疗器械 | +0.0146 |
| 5 | 300450 | 专用机械 | +0.0139 |
| 6 | 688082 | 半导体材料与设备 | +0.0135 |
| 7 | 300433 | 电子终端及组件 | +0.0130 |
| 8 | 600026 | 运输业 | +0.0126 |
| 9 | 002532 | 工业金属 | +0.0124 |
| 10 | 600426 | 化学原料 | +0.0117 |
| 11 | 688256 | 集成电路 | +0.0115 |
| 12 | 002384 | 电子元件 | +0.0112 |
| 13 | 000657 | 稀有金属 | +0.0112 |
| 14 | 601872 | 运输业 | +0.0111 |
| 15 | 300408 | 电子元件 | +0.0111 |
| 16 | 600547 | 贵金属 | +0.0111 |
| 17 | 002938 | 电子元件 | +0.0110 |
| 18 | 002028 | 电网设备 | +0.0109 |
| 19 | 300316 | 发电设备 | +0.0108 |
| 20 | 300394 | 通信设备 | +0.0108 |
| 21 | 600104 | 乘用车 | +0.0107 |
| 22 | 600584 | 集成电路 | +0.0107 |
| 23 | 002920 | 汽车零部件与轮胎 | +0.0105 |
| 24 | 000338 | 交通运输设备 | +0.0104 |
| 25 | 000807 | 工业金属 | +0.0104 |
| 26 | 600482 | 交通运输设备 | +0.0103 |
| 27 | 600489 | 贵金属 | +0.0103 |
| 28 | 300803 | 计算机软件 | +0.0103 |
| 29 | 601898 | 煤炭 | +0.0101 |
| 30 | 688223 | 发电设备 | +0.0099 |
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

- **Prediction date:** 2026-07-02 (this is a FORWARD-LOOKING prediction)
- **D+3 verification date:** 2026-07-08 — market data NOT YET available
- **Current status:** PREDICTION PHASE — realized returns are UNKNOWN
- **Next action:** Run Wednesday review on 2026-07-08 after market close (after 15:00 CST)
- All metrics in this report are MODEL OUTPUTS, not realized P&L
- Do NOT execute trades based solely on this prediction without Wednesday review