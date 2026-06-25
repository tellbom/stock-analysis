# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

---

## Phase 4 Architecture Reference

*Updated after Phase 4A/4B/4C implementation.  Keep this section current.*

### Primary evaluation method (updated P4A-03)

Walk-forward OOS evaluation replaces the static 12-month lockbox:

```python
from quant_platform.evaluation.walk_forward import WalkForwardEvaluator
wf = WalkForwardEvaluator(n_windows=5, window_months=12, horizon=5)
result = wf.run(panel, feature_cols, label_col="ret_fwd_5d")
```

The legacy `make_lockbox_split()` in `training/splitter.py` is deprecated
but retained for backward compatibility.

### Primary label (updated P4A-04)

**Primary:** `ret_fwd_5d` — 5-day forward return with T+1 execution convention.
**Secondary:** `ret_fwd_20d` — retained for comparison only.
**Market-neutral:** `excess_vs_csi300_5d` — strips market beta (P4A-05).
**Sector-neutral:** `excess_vs_industry_5d` — strips sector beta (P4B-05).
**Professional-grade:** `residual_ret_5d` — strips market + sector + size (P4C-04).

Constants in `labels/builder.py`:
```python
DEFAULT_HORIZONS = [1, 5, 10, 20]   # P4A-04
PRIMARY_LABEL_HORIZON = 5
PRIMARY_LABEL_COL = "ret_fwd_5d"
```

### Feature families (updated P4B + P4C)

| Family | Module | Specs constant | Source |
|---|---|---|---|
| technical | `features/technical.py` | `TECHNICAL_SPECS` | OHLCV |
| cross_sectional | `features/cross_sectional.py` | `CROSS_SECTIONAL_SPECS` | derived |
| valuation | `features/valuation.py` | `VALUATION_SPECS` | Tencent Finance |
| industry | `features/industry.py` | `INDUSTRY_SPECS` | Eastmoney |
| flow | `features/flow.py` | `FLOW_SPECS` | Eastmoney push2his |
| margin | `features/margin.py` | `MARGIN_SPECS` | Eastmoney datacenter |
| event | `features/event.py` | `LOCKUP_SPECS` | Eastmoney datacenter |

Full spec set: `from quant_platform.features.registry import FULL_SPECS`

Active feature cols (excludes pruned): use `FeaturePruner.get_active_feature_cols()`.

### Evaluation instrument

| Instrument | Module | Status |
|---|---|---|
| Walk-forward OOS | `evaluation/walk_forward.py` | **Primary** (P4A-03) |
| Single-factor IC | `evaluation/feature_ic.py` | **Ongoing hygiene** (P4C-01) |
| Regime analysis | `evaluation/regime_analysis.py` | **Strategic** (P4C-05) |
| Static lockbox | `training/splitter.make_lockbox_split` | *Deprecated* |
| OOF PurgedKFold | `training/splitter.PurgedKFold` | Retained for HPO |

### Robustness suite (fixed P4A-01)

`embargo` defaults to `None → horizon` in `run_robustness_tests()`.
Never pass `embargo=5` explicitly for a horizon=20 model — this was the
bug that caused misleading ablation deltas before P4A-01.

### Data lake layout (P4A/B/C additions)

```
silver/
  ohlcv/           per-symbol daily OHLCV
  index_ohlcv/     P4A-05: CSI 300 index OHLCV
  valuation/       P4B-01: PE/PB/mcap/turnover (Tencent Finance)
  industry_map     P4B-03: SCD industry classification (Eastmoney)
  fund_flow/       P4B-06: capital flow (Eastmoney push2his)
  margin/          P4B-08: margin trading (Eastmoney datacenter)
  lockup/          P4C-03: lockup expiry calendar (Eastmoney datacenter)
evaluation/
  feature_ic_*.csv       P4C-01 reports
  feature_pruning_log    P4C-02 pruning history
  regime_analysis_*.csv  P4C-05 reports
```

### Collinearity pruning (P4C-02)

Pruned features are recorded in `evaluation/feature_pruning_log.parquet`.
They are still *computed* by builders but excluded from the training list:

```python
from quant_platform.features.pruning import FeaturePruner
pruner = FeaturePruner(store_root)
active_cols = pruner.get_active_feature_cols(all_feature_cols)
```

### PIT discipline

All features at date T use only data available before market close on T.
Margin data uses a 1-day lag (`features/margin.py`).
Industry classification uses an SCD PIT join (`ingest/industry_collector.py`).
Lockup features only look at unlock_date **strictly after T** — the unlock
event itself (unlock_date == T) is conservatively treated as T+1.

### Eastmoney rate limiting

All Eastmoney requests go through the module-level `_em_get()` throttle
(≥1s between calls + random jitter).  Never run multiple Eastmoney
collectors concurrently.  The full CSI 300 universe takes ~5 minutes per
collector.
