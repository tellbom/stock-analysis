# D3 Prediction 2026-07-03 — Interim Review Memo

**Status: INTERIM — not a model verdict.** Written 2026-07-09, before the
correct-horizon realized data is fully available.

## Scope

Weekly D+3 prediction for signal date **T = 2026-07-03** (model:
`ret_fwd_3d`, Ridge(alpha=1.0), industry-neutral Top-50 + global Top-50).
This memo records what the first realized window shows and, more importantly,
what it does **not** license us to conclude.

## What is validated (production loop)

- **Data closed the loop.** OHLCV backfilled 07-06/07/08 at 299/300 (688072 is
  a single-symbol source gap — Sina empty, Eastmoney unreachable; the collector
  correctly wrote no synthetic data). Wednesday review runs on 298 names.
- **SHAP drivers fixed.** `run_d3_gate_once.py` was passing the full sklearn
  `Pipeline` to `TreeExplainer`, which silently returned all-zero drivers. Now
  unwrapped to the final LGBM estimator → reco cards 55/55 carry non-zero SHAP
  drivers.
- **Review horizon fixed.** `d3_wednesday_review.py` hardcoded the exit date as
  07-08 (a 2-trading-day window) while the model's label `ret_fwd_3d` targets
  `close(T+1+3)/close(T+1)` → exit = **07-09**. The review is now horizon-driven
  (entry/exit derived from the trading calendar + `HORIZON`), with a coverage
  guard that aborts on partially-backfilled exit days.

## First realized window — evidence

Rank IC of `model_score` vs realized return, over candidate exit dates
(entry = T+1 = 2026-07-06):

| Exit date | Hold (trading days) | n | Rank IC | mean_ret |
|---|---|---|---|---|
| 07-07 | 1 | 298 | -0.2379 | -1.56% |
| 07-08 (old hardcoded review) | 2 | 298 | -0.3538 | -2.30% |
| **07-09 (correct `ret_fwd_3d` endpoint)** | 3 | 298 | **-0.2507** | −1.60% / −1.71% (neut / global Top50) |

The original reported **-0.3538 / hit rate 10% / -3.44%** was computed on the
**wrong 2-day window**. The definitive **correct-horizon (07-09, 298/298)**
result is **Rank IC -0.2507**, industry-neutral Top-50 mean −1.60% (hit 32%,
16/50), global Top-50 mean −1.71% (hit 32%). Fixing the horizon pulled the IC
from −0.35 toward −0.25 but did **not** change the sign.

## Verdict

1. **This is a genuine adverse window, not a pipeline artifact.** The IC is
   negative across *every* endpoint (07-07 −0.24, 07-08 −0.35, correct 07-09
   −0.25). A symbol-misalignment or sign bug would more likely give IC ≈ 0; a
   consistent negative sign across horizons is the signature of a real
   cross-sectional **factor reversal** that week (both the industry-neutral set
   and the naive global-Top-50 underperformed, −1.60% vs −1.71%). The one
   "PASS" — industry-neutral beats global baseline — is a wafer-thin 11bp edge
   and not meaningful on n=1.
2. **It does NOT support "the model failed."** n = 1 window. One factor-reversal
   week is expected to happen to any factor model periodically. A single weekly
   review is not the platform's model-judgment instrument — `WalkForwardEvaluator`
   (multi-window OOS) is (see CLAUDE.md, Phase 4 primary evaluation).
3. **Turnover-aware (SR-07) is unaffected and remains `reject`** — separate track,
   not revisited here.

## Next steps

1. **DONE — correct-horizon review recorded above** (07-09, 298/298, IC
   −0.2507). 688072 remains unfetchable (Sina empty / Eastmoney SSL) but does
   not affect the 298-name review.
2. **Do not promote/reject the D3 model on this window.** Accumulate several
   weekly windows and/or run `WalkForwardEvaluator` before any decision. This
   is the first live negative sample, not a verdict.
3. Deferred polish (not blocking): strict SHAP consistency — `explainability.py`
   explains raw features via `fillna(0)`, bypassing the trained imputer+scaler,
   so driver magnitudes are directionally usable but not fully faithful.

## Diagnostic (2026-07-09): the horizon is the problem, not the algorithm

Ran `WalkForwardEvaluator` (4 windows × 6m OOS, Ridge, feature_set 80fd2338,
~250k rows, 2023-01 .. 2026-07) on the SAME panel across horizons
(`scripts/diag_d3_horizon.py`):

| horizon | agg Rank IC | ICIR | sign stability | net Sharpe (10bps) |
|---|---|---|---|---|
| **3d (current D3 target)** | **-0.0026** | -0.017 | 0.50 (coin flip) | -0.35 |
| 5d | +0.0063 | +0.041 | 0.75 | -0.00 |
| **10d** | **+0.0183** | **+0.120** | **1.00 (4/4)** | **+0.46** |
| 20d | +0.0156 | +0.112 | 0.75 | +0.43 |

All four runs' independent IC-decay curves peak at 10d (~0.023). Conclusion:

- **`ret_fwd_3d` has no out-of-sample alpha** (agg IC ≈ 0, coin-flip sign). The
  branding-driven switch to 3d (never validated — see
  `next_week_d3_prediction.py:266-274`) is confirmed a dud. The single 07-03
  live window (−0.25) is just noise from a ~0-IC target.
- The feature set's alpha lives at **~10 trading days** (IC ~7× the 3d, 100%
  sign stability, positive net Sharpe). Modest but real.

**Direction: re-anchor the line from 3d to a ~10d horizon** — this is upstream
of both "tune the algorithm" and "fix the strategy": neither helps a zero-signal
target. Algorithm/strategy refinement comes *after* the horizon is fixed.

**Decision (2026-07-09): lock 10d, keep 3d live in parallel.**
- Stand up a **parallel 10d line** (`ret_fwd_10d`, best IC/ICIR + 100% sign
  stability + positive net Sharpe).
- **Keep the existing 3d line live** as the current short-term recommendation —
  do NOT retire it yet; cut over only after the 10d line has a few weeks of live
  track record.
- Algorithm tuning deferred (low EV until a real-signal horizon is in place;
  3d's noise is a horizon problem, not a model problem).
- SR-07 turnover work becomes relevant once the 10d line is the candidate.
