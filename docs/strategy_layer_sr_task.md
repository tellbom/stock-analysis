# SR — Strategy & Recommendation Layer — Task Breakdown (Awaiting Review)

> **中文导语.** GF-08 bake-off 已判定门控融合（`gate_first`）**不晋级**：它与平滑融合各臂在 20 个 OOS 日期上**统计不可区分**（Wilcoxon p=0.2025），模型侧边际收益见底。本计划把重心从"再调融合"转向**策略/推荐层**——因为产品定位是**把排序结果作为推荐方案下发给用户**（不接实盘），真正未开采的价值在"如何把 score 变成用户可信、可执行、低换手的推荐"。
>
> 本文档只做**计划**。实现交由 Sonnet 5，最终由 Opus 4.8 + GPT-5.5 依据本文 §Promotion Rule 做**联合评测**。任务节点顺序、验收口径、预注册评测规则均已冻结在本文中——实现期间不得改动评测规则或日期集（多重比较纪律）。

*Companion to `docs/gate_fusion_task.md` and `docs/gate_fusion_gf08_plan.md`. Follows the same task-ID / priority / blocker / validation conventions. Anchored files below reflect the current tree.*

---

## 0. What GF-08 actually told us (read before scoping)

From `models/data/reports/gf08/gf08_aggregate_metrics.csv` (20 dates, CSI 300, `ret_fwd_3d`):

| Arm | Rank IC | ICIR | Top20 net | Prec@20 | Turnover(top20) |
|---|---:|---:|---:|---:|---:|
| **rrf** | **0.1066** | 2.677 | 0.0274 | 0.603 | 0.653 |
| **recent_only** | 0.0955 | 2.549 | **0.0301** | **0.633** | **0.582** |
| fixed_weight_0.7 | 0.0913 | 2.726 | 0.0280 | 0.615 | 0.684 |
| gate_first | 0.0985 | 2.591 | 0.0242 | 0.588 | 0.686 |

Facts that drive this plan:
1. **The `fixed_weight_0.7` vs `gate_first` gap the team fixated on is noise** — `wilcoxon_p_gate_vs_best_smooth = 0.2025`, single-date Rank-IC std ≈ 0.15. Do **not** keep tuning fusion variants; that is fitting noise.
2. **Neither fixated arm is the winner.** `rrf` has the strongest ranking signal; `recent_only` has the best net return, best precision, **and the lowest turnover (0.58)** — the metric that matters most for a recommendation product users cannot rebalance daily.
3. **Turnover ~0.65–0.69 one-way per rebalance is the biggest untapped lever.** For a user-facing recommendation, churning ~2/3 of the list each period destroys realizable net return and trust. No task in GF-08 touched this.
4. GF-08 §6.2 already pre-registered the fallback when gate ties a smooth arm: **the hybrid — gate veto/observe as a hard filter wrapping a smooth combined score.** gate_first was rejected ⇒ the hybrid is the sanctioned default. SR-01 implements exactly this.

**Product frame.** The deliverable is a *recommendation* (`{prefix}_ranked.csv` + report), not an order stream. So the layer's job is: turn a fused score into a **stable, explained, confidence-annotated, low-churn** shortlist with holding guidance. Ranking quality is necessary but no longer the bottleneck.

---

## 1. Anchored code surfaces (all exist today)

| Concern | File | Current state |
|---|---|---|
| Fused score (base+recent) | `quant_platform/selection/gate_fusion.py` | **FROZEN** (GF-01..07). Emits A/B `_ranked.csv`, observe/veto `_observe.csv`. |
| Ind-neutral ranking + select + monitor | `quant_platform/selection/ranker.py` | `IndustryNeutralRanker.run()` = rank→select→monitor. Consumes a single `score_col="model_score"`. |
| Selection strategies | `quant_platform/selection/strategies.py` | `EqualTopK`, `ProportionalTopK`, `Hybrid`. **No turnover awareness.** |
| Exposure / concentration | `quant_platform/selection/exposure.py` | `ExposureMonitor.flag/concentration_report`. Static, single-date. |
| Config | `quant_platform/selection/config.py` | `SelectionConfig` (top_k, max_total, hybrid_weight, exposure thresholds). |
| Per-symbol SHAP drivers | `quant_platform/evaluation/explainability.py` | `build_explainability_report` → `per_symbol_drivers`. **Not wired into the recommendation.** |
| Regime analysis | `quant_platform/evaluation/regime_analysis.py` | Exists (P4C-05). Not consulted by the recommender. |
| L/S decile backtest | `quant_platform/evaluation/backtest.py` | `run_backtest` — arbiter only. |
| Production recommendation | `scripts/next_week_d3_prediction.py:~404–445` | Builds `D3_Prediction_*_ranked.csv` via `IndustryNeutralRanker` on `model_score`. **Ships forward-label columns (`ret_fwd_*`) in the user CSV** — see SR-06. |
| Gate bake-off harness (reuse) | `scripts/gf08_bakeoff.py` | Date selection, RRF/fixed-weight helpers, metric row, turnover, Wilcoxon. **Reuse verbatim for SR-07.** |

**Two separate surfaces exist today** and this plan does *not* merge them blindly:
`next_week_d3_prediction.py` (ranker on `model_score`) vs `run_d3_gate_once.py` (gate fusion base+recent). SR-01 defines the single score contract so the strategy layer consumes the sanctioned hybrid, not an ad-hoc `model_score`.

---

## 2. Task ID convention

`SR-{nn}` = Strategy & Recommendation. Priority mirrors GF conventions:
`P1` = correctness / core value (do first), `P2` = user-facing quality, `P3` = hygiene/robustness.
`[BLOCKER]` = must land before the SR-07 evaluation is run, else the eval measures a moving/partial target.

---

## 3. Execution order & dependencies

```
SR-01 (P1  score contract: hybrid = gate filter × smooth score)   ← FIRST; everything downstream ranks off this
   ├─ SR-02 (P1  turnover-aware construction / hysteresis)   [BLOCKER]
   ├─ SR-03 (P2  confidence & regime gating)
   ├─ SR-04 (P2  per-pick explainability cards)
   └─ SR-05 (P2  holding-period & sizing guidance)
SR-06 (P3  recommendation output hygiene — drop forward-label leakage)   ← independent, do early
SR-07 (VERIFY  strategy bake-off + pre-registered promotion rule)   ← LAST; needs SR-01, SR-02 [+ SR-03 if landed]
```

---

## P1 — Core value

### SR-01: Lock the score contract — hybrid (gate filter × smooth fusion) `[BLOCKER]`

| Field | Detail |
|---|---|
| **Priority** | P1 — resolves the `fixed_weight` vs `gate_first` question by adopting GF-08 §6.2 |
| **Order** | 1 |

**Problem.** The strategy layer ranks off a single `score_col`. GF-08 proved gate_first does not beat smooth fusion, and the fixated arms are statistically tied. There is no defined, reviewed contract for *which* score feeds selection, and the gate's veto/observe filter is not applied on the ranker path (`next_week_d3_prediction.py`).

**Fix (pre-registered as GF-08 §6.2 hybrid).**
1. Ranking score = a **smooth fusion** over `(base_pct, recent_pct)` — the harness helpers `add_rrf_rank` / `add_fixed_weight_rank` in `scripts/gf08_bakeoff.py` are the reference implementations; promote the chosen one into a small library fn (do **not** touch frozen `gate_fusion.py`). Default candidate = **`rrf`** (best IC, weight-free, nothing to overfit); **`recent_only`** and `fixed_weight_0.7` are the pre-registered alternates decided in SR-07, not now.
2. Gate tiers act as a **hard filter only**: `RISK_VETO` / `E_REJECT` / `D_OBSERVE` are excluded from the actionable pool (they already route to `_observe.csv`); the smooth score ranks **within** the A/B pool.
3. Expose the result as the `score_col` the `IndustryNeutralRanker` consumes, so ranker/strategies/exposure are unchanged.

**Deliverables.** A `fusion_score()` helper (new small module under `selection/`, e.g. `selection/fusion_score.py`) + wiring so the recommendation path ranks the A/B pool by the hybrid score. Frozen gate code untouched.

**Validation.**
- Unit: on a synthetic frame, veto/observe/reject symbols never appear in the actionable output; within-A/B order equals the smooth-score order.
- Parity: on one real date, actionable set ⊆ gate `_ranked.csv` (A/B) set.
- No import of, or edit to, `gate_fusion.py` internals.

---

### SR-02: Turnover-aware construction (hysteresis / no-churn band) `[BLOCKER]`

| Field | Detail |
|---|---|
| **Priority** | P1 — highest ROI per §0.3 |
| **Order** | 2 (after SR-01) |
| **Depends on** | SR-01 |

**Problem.** `strategies.py` re-selects from scratch each date; measured one-way top-20 turnover ≈ 0.65–0.69. For a recommendation users act on manually, that is churn they cannot follow and cost they silently eat. `recent_only`'s 0.58 shows headroom without a signal change.

**Fix.** Add a stateful construction step that takes *(today's ranked A/B pool, yesterday's held set)* and applies **hysteresis**: keep an incumbent unless it falls out of a wider `keep_rank` band; admit a challenger only if it clears a tighter `enter_rank` band (enter/exit asymmetry = the no-churn band). Cap per-rebalance turnover at a configured ceiling. Prefer a new `TurnoverAwareStrategy` implementing the existing `SelectionStrategy` ABC (so it plugs into `_STRATEGY_REGISTRY` in `ranker.py`), plus config fields `enter_rank`, `keep_rank`, `max_turnover`.

**Deliverables.** `TurnoverAwareStrategy` + `StrategyType.TURNOVER_AWARE` + `SelectionConfig` fields. Prior-holdings passed in explicitly (stateless call, caller supplies previous set) — no hidden global state.

**Validation.**
- On the SR-07 date panel, one-way top-20 turnover drops to **≤ 0.50** (target) with net Top-20 not materially worse (see SR-07 rule).
- Unit: with an empty prior set it reduces to plain Top-K; with prior = current it produces zero turnover.
- Determinism: same inputs → same selection (no `Date.now`/random tie-breaks).

---

## P2 — User-facing quality

### SR-03: Recommendation confidence & regime gating

| Field | Detail |
|---|---|
| **Priority** | P2 |
| **Depends on** | SR-01 |

**Problem.** The recommender always emits a full list at full conviction. GF-08 shows per-date Rank IC swings sign (std ≈ 0.15). A recommendation product must be able to say "signal is weak this period."

**Fix.** Compute a per-date **confidence scalar** from already-available inputs (e.g. cross-sectional score dispersion, base/recent agreement, and `regime_analysis.py` regime label). Map it to: (a) a shortlist-size / conviction annotation, and (b) a soft "low-confidence — reduce exposure / observe" banner in the report. **No new alpha, no forward data** — confidence is derived from T-available features only.

**Deliverables.** `selection/confidence.py` (pure fn: panel@T → confidence ∈ [0,1] + label), consumed by the recommendation writer. Regime read via existing `regime_analysis` API.

**Validation.** Unit: degenerate flat-score panel → low confidence; strongly separated panel → high. Assert confidence uses no `ret_fwd_*` column (PIT canary).

---

### SR-04: Per-pick explainability cards

| Field | Detail |
|---|---|
| **Priority** | P2 — trust is the recommendation product's moat |
| **Depends on** | SR-01 |

**Problem.** `explainability.py` already computes `per_symbol_drivers` (top SHAP features per symbol) but it is **not wired into the recommendation**. Users get a ranked list with no "why."

**Fix.** For each recommended stock, attach a compact **why-card**: top-N SHAP drivers (from `build_explainability_report`) grouped by feature *family* (technical / flow / valuation / industry / margin / event — families already exist in `features/registry.py`), plus its gate tier and within-industry rank. Emit as a per-symbol JSON/columnar block alongside the ranked output. Reuse the existing SHAP path; do not recompute a second explainer.

**Deliverables.** A `build_reco_cards()` glue fn (in `selection/` or `evaluation/`) that joins `per_symbol_drivers` + family map + rank fields to the actionable set; written next to `{prefix}_ranked.csv`.

**Validation.** Every recommended symbol has a non-empty card; family attribution sums to the driver set; snapshot test on a small panel.

---

### SR-05: Holding-period & position-sizing guidance

| Field | Detail |
|---|---|
| **Priority** | P2 |
| **Depends on** | SR-01 |

**Problem.** The recommendation gives *what* but not *how long / how much*. The label horizon (3d prod / 5d primary) implies a holding period the user is never told.

**Fix.** Emit, per pick and per list: the intended **holding horizon** (from the label convention) and a **suggested relative weight** (equal-weight default; optional confidence- or score-tilt, capped, sector-aware via existing `ExposureMonitor`). Sizing is *advisory* metadata in the payload — no leverage, no optimizer. Keep it minimal.

**Deliverables.** Horizon + `suggested_weight` columns in the recommendation payload; weights sum to 1 within the actionable set; respect `exposure_warning_threshold`.

**Validation.** Weights normalized; no single-industry weight exceeds the exposure threshold when the tilt is on; unit test on a toy panel.

---

## P3 — Hygiene / robustness

### SR-06: Recommendation output hygiene — drop forward-label leakage from the user CSV

| Field | Detail |
|---|---|
| **Priority** | P3 — but do EARLY (correctness of the delivered artifact) |
| **Order** | independent |

**Problem.** `D3_Prediction_*_ranked.csv` currently ships forward-return label columns — `ret_fwd_1d`, `ret_fwd_3d`, `ret_fwd_5d`, `ret_fwd_10d`, `ret_fwd_20d` and their `_cs` / `_bin` variants — in the **user-facing** file (confirmed in the real 2026-07-03 / 07-02 headers). These are future data. They are (almost certainly) not model inputs, but shipping them in a recommendation is a presentation-layer leak and an audit red flag.

**Fix.** Define an explicit **recommendation schema** (symbol, name, industry, fused score, gate tier, within-industry rank, selected/reason, exposure flag, + SR-03/04/05 fields) and write only that. Move raw feature/label columns to an internal debug artifact if still needed offline. Do not change model training.

**Validation.** No column matching `ret_fwd_*` / `*_fwd_*` / `_bin` in the user CSV; a test asserts the shipped schema is a fixed allow-list.

---

## Verification

### SR-07: Strategy bake-off + pre-registered promotion rule `[gate for the joint eval]`

| Field | Detail |
|---|---|
| **Priority** | VERIFY — required before any "adopt this recommender" claim |
| **Order** | LAST |
| **Depends on** | SR-01, SR-02 (SR-03 if landed) |

**Problem.** A single cross-section proves plumbing, not that the strategy layer improves the *net, user-facing* outcome — same lesson as Config E and GF-08.

**Fix.** Extend `scripts/gf08_bakeoff.py` (harness-only) into a strategy bake-off over the **same pre-registered 20 dates** (do not reselect — reuse for comparability). Arms:
- `baseline` = SR-01 hybrid score + plain Top-K (no turnover awareness).
- `turnover_aware` = SR-01 hybrid + SR-02 hysteresis.
- (optional) `+confidence` = also apply SR-03 gating.
Metric row per arm: **net Top-K return (K∈{20,50}, cost_bps=10)**, **one-way turnover**, **precision@K / hit-rate**, **Rank IC + ICIR**, and **regime-conditional** net return (weak vs strong regime split). Reuse the existing metric fns verbatim.

**Promotion rule (pre-registered — frozen before the first run):** adopt `turnover_aware` as the default recommender **iff**
1. one-way top-20 turnover ≤ **0.50** (materially below the ~0.65 baseline), **and**
2. net Top-20 ≥ baseline − **0.002** (≈ within one date's noise; i.e. turnover cut does **not** cost real net return), **and**
3. no worse in the weak-regime split than baseline on net Top-20.
If (1) holds but (2) fails, report the turnover/return trade-off curve and **defer to reviewers** — do not silently pick.

**Deliverables.** Per-date detail + aggregated summary + a results `.md` (mirror `gf08_results.md`) under `models/data/reports/sr/`, with the promotion decision recorded.

**Validation.** PIT asserted in-harness (features ≤ T, labels from T+1); the date list and promotion rule are unchanged from this document.

---

## Guardrails

- **Do NOT** modify frozen gate logic (`gate_fusion.py`, `coverage_gate.py`, `registry.py`) — SR code is additive (new `selection/` modules + harness). If a gate change seems needed, record it as a finding, don't fold it in.
- **Do NOT** retrain models, change labels, or touch feature builders. This layer is strictly post-inference.
- **PIT:** every SR input at date T uses only T-available data; the only future series is the realized label, used solely for SR-07 scoring, never in the recommender.
- **Simplicity (CLAUDE.md §2):** no optimizer, no leverage, no configurability beyond the named config fields. If a task grows past ~a screen of logic, stop and flag it.
- **Multiple-comparisons discipline:** the SR-07 date list, arms, and promotion rule are frozen by this document. No post-hoc changes after the first SR-07 run.

---

## Open questions for reviewers (Opus 4.8 + GPT-5.5)

1. **Horizon.** Match production (`ret_fwd_3d`) for continuity with GF-08, or move the recommendation to `ret_fwd_5d` (CLAUDE.md primary, more realistic for a manually-followed recommendation and kinder to turnover economics)? Affects SR-02/05/07.
2. **Smooth arm for SR-01.** Default `rrf` (best IC, weight-free). But `recent_only` had the best net + lowest turnover at 3d. Decide by SR-07, or pre-commit `rrf` now and treat `recent_only` as the alternate arm?
3. **Turnover target.** Is ≤0.50 the right ceiling, or should SR-07 sweep the enter/keep band and report the frontier instead of a single threshold?
4. **Confidence gating (SR-03) scope.** Ship as a *report banner only* (safe), or also let it shrink the shortlist size (changes the eval arm)?

---

## Handoff

- **Implements:** Sonnet 5, task order SR-01 → SR-02 → SR-06 (early) → SR-03/04/05 → SR-07. Land unit tests with each task (repo convention: see `tests/test_gate_first.py`, `tests/test_gf08_bakeoff.py`).
- **Joint evaluation:** Opus 4.8 + GPT-5.5, against the **frozen SR-07 promotion rule** above, once SR-01/02 (+ optionally SR-03) land and the SR-07 bake-off has run on the pre-registered dates.
- **Commit convention:** land SR code + `models/data/reports/sr/` results separately, mirroring GF-08.

```
Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
```
