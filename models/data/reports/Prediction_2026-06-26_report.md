# Current-Week Prediction & Selection Report

**Generated:** 2026-07-04T16:23:24.689960
**Status:** ⚠️ PREDICTION / PREPARATION PHASE — D+3 full verification has NOT occurred

---

## 1. Prediction Configuration

| Parameter | Value |
|-----------|-------|
| Prediction date (T) | **2026-06-26** (Friday) |
| Earliest execution (T+1) | 2026-06-29 (Monday) |
| D+3 target date | 2026-07-02 |
| D+5 target date | 2026-07-06 |
| Primary label column | `ret_fwd_5d` |
| Primary horizon | 5 trading days |
| Feature set ID | `d02a4ebf` |
| Feature count | 26 |
| reversal_3d in features | True |
| industry_code in panel | True |
| ret_fwd_3d labels | Appended via `append_horizon_labels` |
| Model | Ridge Regression (sklearn) |
| Universe | CSI 300 (300 symbols) |

## 2. Verification Schedule

| Milestone | Date | Status |
|-----------|------|--------|
| Prediction generated | 2026-06-26 | ✅ Done |
| D+1 data available (T+1) | 2026-06-29 | ✅ Data in checkpoint |
| D+3 data available | 2026-07-02 | ⚠️ Partial (checkpoint, 263 symbols) |
| D+5 data available | 2026-07-06 | ❌ Not yet available |
| Full D+3 verification | After complete OHLCV through 2026-07-02 | Pending |
| Full D+5 verification | After complete OHLCV through 2026-07-06 | Pending |

## 3. Historical Walk-Forward Results (ret_fwd_5d)

**Configuration:** 4 windows, 6-month each, horizon=5d

| Metric | Value |
|--------|-------|
| Aggregated ICIR (OOS) | +0.2706 |
| IC sign stability | 1.00 |
| Status | ⚠️ HISTORICAL BACKTEST ONLY — NOT live P&L |

| Window | Test Period | Rank IC | ICIR | Sharpe | IndepPd |
|--------|-------------|---------|------|--------|---------|
| 0 | 2024-06-11 - 2024-12-10 | +0.0230 | +0.1465 | -1.4812 | 24 |
| 1 | 2024-12-11 - 2025-06-10 | +0.0538 | +0.4899 | +1.0296 | 23 |
| 2 | 2025-06-11 - 2025-12-10 | +0.0094 | +0.0707 | -1.0664 | 25 |
| 3 | 2025-12-11 - 2026-06-10 | +0.0556 | +0.5404 | +5.1238 | 23 |

**IC Decay (cross-horizon):**
| Horizon | Rank IC |
|---------|---------|
| 1 | +nan |
| 2 | +0.0088 |
| 3 | +0.0158 |
| 5 | +0.0310 |
| 10 | +0.0415 |
| 20 | +0.0398 |


## 4. Industry-Neutral Stock Selection

### Strategy: `industry_neutral_equal_top3`

- **Selected:** 50 stocks
- **Exposure flag:** diversified

| # | Symbol | Industry | Score | Industry Rank |
|---|--------|----------|-------|---------------|
| 1 | 000002 | 房地产开发与园区 | +0.0160 | 1 |
| 2 | 000338 | 交通运输设备 | +0.0155 | 1 |
| 3 | 000617 | 其他金融服务 | +0.0212 | 1 |
| 4 | 000625 | 乘用车 | +0.0171 | 3 |
| 5 | 000630 | 工业金属 | +0.0153 | 2 |
| 6 | 000657 | 稀有金属 | +0.0160 | 2 |
| 7 | 001979 | 商业地产开发和管理 | +0.0176 | 1 |
| 8 | 002074 | 储能设备 | +0.0241 | 1 |
| 9 | 002460 | 稀有金属 | +0.0193 | 1 |
| 10 | 002466 | 稀有金属 | +0.0160 | 2 |
| 11 | 002532 | 工业金属 | +0.0245 | 1 |
| 12 | 002625 | 国防装备 | +0.0170 | 1 |
| 13 | 300014 | 电气部件与设备 | +0.0157 | 1 |
| 14 | 300122 | 生物药品 | +0.0176 | 1 |
| 15 | 300251 | 文化娱乐 | +0.0201 | 1 |
| 16 | 300408 | 电子元件 | +0.0159 | 1 |
| 17 | 300413 | 数字媒体 | +0.0220 | 1 |
| 18 | 300418 | 数字媒体 | +0.0199 | 1 |
| 19 | 300433 | 电子终端及组件 | +0.0155 | 1 |
| 20 | 300442 | 专用机械 | +0.0154 | 1 |
| 21 | 300498 | 养殖 | +0.0154 | 1 |
| 22 | 300661 | 集成电路 | +0.0168 | 1 |
| 23 | 300896 | 医疗器械 | +0.0185 | 1 |
| 24 | 600104 | 乘用车 | +0.0200 | 1 |
| 25 | 600183 | 电子元件 | +0.0182 | 1 |
| 26 | 600219 | 工业金属 | +0.0206 | 2 |
| 27 | 600362 | 工业金属 | +0.0193 | 1 |
| 28 | 600415 | 商业服务与用品 | +0.0183 | 1 |
| 29 | 600426 | 化学原料 | +0.0202 | 1 |
| 30 | 600482 | 交通运输设备 | +0.0209 | 1 |
| 31 | 600489 | 贵金属 | +0.0233 | 2 |
| 32 | 600522 | 通信设备 | +0.0158 | 1 |
| 33 | 600547 | 贵金属 | +0.0288 | 1 |
| 34 | 600588 | 信息技术服务 | +0.0178 | 1 |
| 35 | 600875 | 发电设备 | +0.0199 | 1 |
| 36 | 600893 | 航空航天 | +0.0160 | 1 |
| 37 | 600989 | 化学原料 | +0.0170 | 1 |
| 38 | 601238 | 乘用车 | +0.0185 | 2 |
| 39 | 601360 | 软件开发 | +0.0186 | 1 |
| 40 | 601600 | 工业金属 | +0.0202 | 3 |
| ... | ... | ... | ... | ... |
| Total | 50 symbols | | | |

## 5. Industry Exposure Concentration

### `industry_neutral_equal_top3`

| Industry | Count | Fraction | Top Symbols |
|----------|-------|----------|-------------|
| 乘用车 | 3 | 6.0% | 000625, 600104, 601238 |
| 工业金属 | 3 | 6.0% | 002532, 600219, 601600 |
| 贵金属 | 3 | 6.0% | 600489, 600547, 601899 |
| 发电设备 | 2 | 4.0% | 688223, 688472 |
| 软件开发 | 2 | 4.0% | 601360, 688111 |
| 稀有金属 | 2 | 4.0% | 000657, 603993 |
| 工业金属 | 2 | 4.0% | 000630, 600362 |
| 稀有金属 | 2 | 4.0% | 002460, 002466 |
| 信息技术服务 | 1 | 2.0% | 600588 |
| 数字媒体 | 1 | 2.0% | 300418 |
| 医疗器械 | 1 | 2.0% | 603392 |
| 化学原料 | 1 | 2.0% | 600426 |
| 发电设备 | 1 | 2.0% | 600875 |
| 其他金融服务 | 1 | 2.0% | 000617 |
| 化学原料 | 1 | 2.0% | 600989 |
| 医疗器械 | 1 | 2.0% | 300896 |
| 半导体材料与设备 | 1 | 2.0% | 688082 |
| 电子元件 | 1 | 2.0% | 600183 |
| 专用机械 | 1 | 2.0% | 300442 |
| 商业地产开发和管理 | 1 | 2.0% | 001979 |
| 交通运输设备 | 1 | 2.0% | 000338 |
| 国防装备 | 1 | 2.0% | 002625 |
| 商业服务与用品 | 1 | 2.0% | 600415 |
| 文化娱乐 | 1 | 2.0% | 300251 |
| 房地产开发与园区 | 1 | 2.0% | 000002 |
| 电子终端及组件 | 1 | 2.0% | 300433 |
| 煤炭 | 1 | 2.0% | 601898 |
| 电气部件与设备 | 1 | 2.0% | 300014 |
| 储能设备 | 1 | 2.0% | 002074 |
| 养殖 | 1 | 2.0% | 300498 |
| 生物药品 | 1 | 2.0% | 300122 |
| 航空航天 | 1 | 2.0% | 600893 |
| 交通运输设备 | 1 | 2.0% | 600482 |
| 电子元件 | 1 | 2.0% | 300408 |
| 数字媒体 | 1 | 2.0% | 300413 |
| 软饮料 | 1 | 2.0% | 605499 |
| 通信设备 | 1 | 2.0% | 600522 |
| 工业金属 | 1 | 2.0% | 603799 |
| 集成电路 | 1 | 2.0% | 300661 |

## 6. Naive Global Top-N Baseline (Comparison)

- **Top N:** 50
- **Selection method:** Global score ranking, NO industry neutralization
- **WARNING:** This is a CONTROL BASELINE, NOT the recommended selection method.
- **Max single industry fraction:** 8.0%

| # | Symbol | Industry | Score |
|---|--------|----------|-------|
| 1 | 600547 | 贵金属 | +0.0288 |
| 2 | 603392 | 医疗器械 | +0.0277 |
| 3 | 002532 | 工业金属 | +0.0245 |
| 4 | 002074 | 储能设备 | +0.0241 |
| 5 | 600489 | 贵金属 | +0.0233 |
| 6 | 688223 | 发电设备 | +0.0227 |
| 7 | 300413 | 数字媒体 | +0.0220 |
| 8 | 000617 | 其他金融服务 | +0.0212 |
| 9 | 600482 | 交通运输设备 | +0.0209 |
| 10 | 603799 | 工业金属 | +0.0208 |
| 11 | 600219 | 工业金属 | +0.0206 |
| 12 | 600426 | 化学原料 | +0.0202 |
| 13 | 601600 | 工业金属 | +0.0202 |
| 14 | 300251 | 文化娱乐 | +0.0201 |
| 15 | 600104 | 乘用车 | +0.0200 |
| 16 | 600875 | 发电设备 | +0.0199 |
| 17 | 300418 | 数字媒体 | +0.0199 |
| 18 | 688472 | 光伏设备 | +0.0195 |
| 19 | 600362 | 工业金属 | +0.0193 |
| 20 | 002460 | 稀有金属 | +0.0193 |

## 7. D+3 Partial Verification (Config E Checkpoint)

⚠️ **THIS IS A PARTIAL CHECK ONLY. Full D+3 verification has NOT been completed.**

- **Status:** PARTIAL — only 263/267 symbols in checkpoint
- **Symbols available:** 263
- **D+3 Rank IC (partial):** +0.0596

| Selection Method | D+3 Mean Return | N | Note |
|-------------------|-----------------|---|------|
| Industry-Neutral (EqualTopK=3) | +0.1083% | 50 | Primary selection |
| Naive Global Top-50 | +0.0385% | 50 | CONTROL BASELINE only |

- **Note:** Based on Config E checkpoint (close prices only, 263 symbols). Full D+3 verification requires complete OHLCV data for all 300 symbols.

## 8. D+3 / D+5 Full Verification Plan

### When data becomes available:

1. **Fetch complete OHLCV** for all 300 CSI 300 symbols covering June 27 – July 8, 2026
2. **Compute realized returns:**
   - `realized_ret_3d(T=2026-06-26)` = close(2026-07-02) / close(2026-06-29) - 1
   - `realized_ret_5d(T=2026-06-26)` = close(2026-07-06) / close(2026-06-29) - 1
3. **Compute D+3 metrics:**
   - Rank IC (prediction vs realized_ret_3d)
   - Industry-neutral selected set mean return
   - Naive global Top-N mean return
   - Industry exposure performance breakdown
4. **Compare D+3 vs D+5:**
   - IC decay from D+3 to D+5
   - Selected set performance at both horizons
   - Determine if D+3 is a better short-cycle target (must be data-driven)

### Key constraint:
> Do NOT claim D+3 or D+5 verification is complete until the corresponding
> market data has been fetched and verified. The current report is
> PREDICTION-PHASE only; realized P&L is NOT yet confirmed.

## 9. Feature Details

- **Feature count:** 26
- **reversal_3d included:** True
  - `reversal_3d` = -(close / close.shift(3) - 1), dimensionless
  - Short-term reversal counter-signal to momentum
- **Technical spec ID (with reversal_3d):** 8cc13114
- **Note:** prediction uses existing feature set `d02a4ebf` plus on-the-fly `reversal_3d`; no new feature parquet set was written.
- **3-day labels:** ret_fwd_3d, vol_fwd_3d, mdd_fwd_3d appended to all 300 symbols

---

## ⚠️ IMPORTANT DISCLAIMER

**THIS IS A PREDICTION / PREPARATION REPORT. D+3 AND D+5 REAL-MARKET VERIFICATION HAS NOT OCCURRED.**

- All historical metrics (walk-forward IC, backtest results) are **in-sample historical backtests**.
- The prediction scores for 2026-06-26 are **model outputs**, NOT realized returns.
- The D+3 partial check uses Config E checkpoint data only — it is **not** a full verification.
- Do NOT interpret any metric in this report as confirmed live P&L.

**Next action:** Run full D+3 verification once complete OHLCV through 2026-07-02 is available.
**Next action:** Run full D+5 verification once complete OHLCV through 2026-07-06 is available.