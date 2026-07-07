# GF-08 — Bake-off Execution Plan (for the next agent)

*Self-contained spec for the §7 multi-date OOS bake-off. Written after GF-01…GF-07 + GF-04b + wiring + provenance-assertion landed. The executing agent should run this WITHOUT changing the gate logic (GF-01…GF-04b are frozen and reviewed). Do a **smoke** pass first (Phase A), then the **authoritative** bake-off (Phase B).*

---

## 0. Preconditions (all DONE — do not re-do)

Frozen and unit-tested (gate suite 19/19; full suite 319+ passed, only pre-existing `optuna`/MLflow-Windows env failures):

- **GF-01/02** exhaustive tier tiling; `RISK_VETO` strictly last.
- **GF-03** `write_gate_fusion_outputs` emits actionable A/B to `{prefix}_ranked.csv`, everything else to `{prefix}_observe.csv` (2-tuple return preserved).
- **GF-04 / GF-04b** typed PIT registry (`registry.FeatureMetadata`, `feature_metadata_lookup()`). base **and** recent-alpha require registered + `pit_safe=True` + non-empty `known_at`; leak tokens (`future/fwd/label/target/pit_risk`) block both; **event families never drive `recent_pct`** (risk channel only). Provenance is asserted at build time (`build_feature_metadata` raises on an unregistered family).
- **GF-05** structured flag schema `name[:detail]@known_at`; veto/downgrade fire only for a **registered code carrying an explicit `@time`**.
- **GF-06** universe-ratio coverage thresholds (`recent_symbol_ratio`/`recent_20d_symbol_ratio` = 0.83); trailing-trading-day window; 20d-avg coverage gate.
- **GF-07** tier-dependent within-tier sort (B/D lead with `recent_pct`).
- Typed metadata is wired into **all** gate call sites: `cli.py:209`, `run_d3_gate_once.py:355,366`, `run_d3_gate_event3_once.py:428,439`, `next_week_d3_prediction.py:245`.

**Do NOT** modify `coverage_gate.py`, `gate_fusion.py`, or `registry.py` gate logic during GF-08. If the bake-off suggests a logic change, record it as a finding — do not fold it into the bake-off run (that would measure a moving target).

---

## 1. Objective

Decide whether **gate-first fusion** should become the default ranking, by comparing it against alternatives over a multi-date out-of-sample panel. Gate-first must not be crowned default unless it demonstrably dominates (see §6 promotion rule). A hybrid (hard veto/observe filter wrapping a smooth combined score) is an expected candidate.

Horizon: **3 trading days** (`LABEL_COL = ret_fwd_3d`, matching the production D3 path). Report 5d (`ret_fwd_5d`) as a secondary cross-check.

---

## 2. Arms to compare

Each arm produces, per as-of date, a ranking of the universe from the **same** `(base_pct, recent_pct, base_rank, recent_rank, risk_flags, event_flags)` inputs. Only the combination rule differs.

| Arm | Ranking rule | Status |
|---|---|---|
| `base_only` | sort by `base_pct` desc (≡ `base_rank`) | exists (base_scored) |
| `recent_only` | sort by `recent_pct` desc | exists (recent_scored) |
| `gate_first` | `gate_first_fusion(...)` current tiering + `final_rank` | **exists — the arm under test** |
| `rrf` | Reciprocal-Rank Fusion: `score = Σ 1/(k + rank_i)` over base_rank, recent_rank (k≈60) | **NEW small fn to add** |
| `fixed_weight` | `w·base_pct + (1−w)·recent_pct`; sweep `w∈{0.3,0.5,0.7}` | **NEW small fn to add** |
| `recent_heavy` | fixed_weight with `w=0.3` (recent carries more than confirm/veto), OR a gate variant where recent can promote | **NEW — per GF-08 note: fast signals may be under-used at a 3d horizon** |
| `gate_first_noveto` | gate-first with the veto/downgrade channel disabled (ablation) | **derive: run gate_first with `risk_flags`/`event_flags` blanked** |

`±veto` is the pair (`gate_first` vs `gate_first_noveto`) — used to measure the veto channel's realized value (§4).

The two NEW fusion functions (`rrf`, `fixed_weight`) are pure ranking helpers over the merged frame. Put them in the bake-off harness module (NOT in `gate_fusion.py`) so the frozen gate code is untouched.

---

## 3. Data, dates, PIT discipline

- **Universe:** CSI 300 (`models/data/universe/csi300`), as used by the D3 scripts.
- **Dates:** ≥ **20** distinct as-of dates (trading days), spaced ≥ 3 trading days apart to keep 3d-horizon periods non-overlapping. Prefer a contiguous recent span (e.g. the last ~5 months of available OHLCV) so coverage is realistic. **Pre-register the exact date list before running** (§7 checklist) and do not change it mid-run.
- **Per date T**, build inputs exactly as the production gate does (reuse `run_d3_gate_once.py` internals):
  - base model trained on `date < T`, scored on `T` → `base_pct/base_rank` (`_score_model`, `LABEL_COL`).
  - recent model likewise → `recent_pct/recent_rank`.
  - `risk_flags`/`event_flags` from `_risk_flags(recent_panel, T)` (already emit timed structured flags).
  - features selected through the gate **with `feature_metadata=feature_metadata_lookup()`** (already wired).
- **Realized labels:** join `models/data/labels/forward_returns/{symbol}.parquet` → `ret_fwd_3d` at date T (already PIT: forward return is measured from T+1). Secondary: `ret_fwd_3d_cs` (cross-sectional demeaned) for a market-neutral read; `ret_fwd_5d` for horizon robustness.
- **PIT invariants to assert in the harness:** no feature column at T uses data dated > T; the realized label is the ONLY future-looking series and is used solely for scoring, never as a feature. Reuse the leakage-harness canary if convenient.

---

## 4. Metric row (per arm, aggregated across dates)

For each arm compute, per date then aggregated (mean, and ICIR-style mean/std·√n):

1. **Rank IC** of the arm's score vs realized `ret_fwd_3d` (Spearman, per date) → mean Rank IC, **ICIR**. Use `evaluation/metrics.evaluate` where possible.
2. **Top-K portfolio forward return**, equal-weight, K ∈ {20, 50}, **net of `cost_bps=10`** (round-trip proxy); report gross and net.
3. **Precision@K / hit-rate**: fraction of Top-K with positive realized excess (vs universe mean).
4. **Market-neutral cross-check**: repeat (1)/(2) against `ret_fwd_3d_cs`.
5. **Turnover** across consecutive dates (context for the cost drag).

**Veto-specific (±veto arm):**
6. **Veto realized-excess**: mean realized `ret_fwd_3d` of names the gate **vetoed/downgraded**, vs the universe mean. A working veto shows **negative** excess (it removed losers). Report count of vetoed names per date and the distribution — a veto that fires on ~nothing or on random names is a red flag.

Output one CSV row per (arm, metric) plus a summary markdown table.

---

## 5. Phase A — SMOKE (plumbing; run first)

Goal: prove routing/classification/veto/output plumbing on ONE date. No alpha claim.

1. Run `python scripts/run_d3_gate_once.py` (uses `REQUESTED_AS_OF_DATE`; adjust the constant or add a CLI/env override for the smoke date — a **harness-only** change, not gate logic).
2. Verify:
   - base/recent/gate outputs written; no crash.
   - coverage-gate reports (`D3_base_coverage_gate_*`, `D3_recent_coverage_gate_*`) show the typed reasons where expected: `unregistered`, `not-pit-safe`, `known-at-missing`, `event-family-risk-channel-only`, `untagged-fail-closed`.
   - `D3_gate_fused_*_ranked.csv` contains **only** `A_MAIN`/`B_SHORT_BOOST`; `D3_gate_fused_*_observe.csv` contains the rest; `RISK_VETO` has the max `final_rank` within the observe pool.
   - at least one veto/downgrade fires and every firing flag carries an `@` time (grep the fused frame's `risk_flags`/`event_flags`); confirm an **untimed** synthetic flag does NOT veto (already unit-tested — spot check only).
   - event-family features do NOT appear in the recent feature list (`D3_recent_*_X_train_columns_*`).
3. **Smoke acceptance:** all of the above hold. Then proceed to Phase B.

---

## 6. Phase B — AUTHORITATIVE bake-off

1. Build a harness `scripts/gf08_bakeoff.py` (NEW, harness-only) that:
   - loops the pre-registered date list, produces the per-date merged input (reusing `run_d3_gate_once` scoring functions — import, don't fork the logic),
   - applies every arm (§2), joins realized labels (§3), computes the metric row (§4),
   - writes per-date detail + an aggregated summary to `models/data/reports/gf08/`.
2. **Promotion rule (pre-registered):** adopt gate-first as default **only if**:
   - `gate_first` ICIR ≥ max(alternative ICIRs) − 0.02 (not materially worse), **and**
   - veto realized-excess ≤ 0 (veto avoids losers), **and**
   - `gate_first` net Top-K return ≥ the best smooth arm (`rrf`/`fixed_weight`) OR the gap is within noise (Wilcoxon signed-rank p > 0.05 across dates ⇒ tie ⇒ prefer the simpler/ smoother arm).
   - If gate-first ties a smooth arm, the recommended default is the **hybrid**: gate veto/observe filter + smooth combined score within the A/B pool.
3. Record the decision (promote / hybrid / reject) in a results doc with the metric tables.

---

## 7. Pre-registration checklist (fill BEFORE running Phase B)

```
Date list (≥20):            ____________________
Horizon:                    ret_fwd_3d  (secondary: ret_fwd_5d)
Universe:                   CSI300 (n=____)
Arms:                       base_only, recent_only, gate_first, rrf,
                            fixed_weight{0.3,0.5,0.7}, recent_heavy, gate_first_noveto
K:                          {20, 50}     cost_bps: 10
Coverage config:            CoverageGateConfig() defaults (ratio 0.83)
Promotion rule:             as §6 (frozen before run)
Result location:            models/data/reports/gf08/
```

---

## 8. Guardrails

- **Do not** modify gate logic (`coverage_gate.py`, `gate_fusion.py`, `registry.py`); bake-off code is additive harness only.
- **Do not** refactor the training flow or data collection.
- **Do not** change the pre-registered date list, arms, or promotion rule after the first Phase-B run (multiple-comparisons discipline).
- Keep PIT: features ≤ T, labels from T+1; assert it in the harness.
- Gate-first is **not** the default until the promotion rule is met.

---

## 9. Risks / open questions for the runner

- **3d horizon vs long-history base:** fast flow/event alpha decays quickly; the `recent_heavy` arm exists to test whether the current design under-uses the fastest signals. Report it prominently.
- **Recent-model availability:** on some dates the recent model may not train (see `recent_status != "trained"` fallback → `BASE_ONLY`). Exclude such dates from fusion-arm comparison or handle explicitly; log how many.
- **Event flags today** only `high_unlock`(veto)/`unlock`(downgrade)/`risk_warning`(veto) are actionable codes; `major_announcement`/`dragon_tiger`/`large_discount_block_trade` are informational (not in veto/downgrade token sets). If the bake-off wants them actionable, that is a **config change to `GateFusionConfig` tokens** — a separate decision, not part of this run.
- Small remaining Info items from the impl review (`_ranked.csv` now A/B-only; `recent_symbol_ratio` naming) — non-blocking.

---

## 10. Handoff — uncommitted changeset to commit

The following is staged in the working tree (GF-01…GF-07 + GF-04b + wiring + provenance assert). Recommended commit BEFORE starting GF-08 so the bake-off runs against a committed baseline:

```
 M quant_platform/cli.py                      (+2)    feature_metadata wired into _apply_coverage_gate
 M quant_platform/evaluation/coverage_gate.py (+160)  typed PIT gating (base+recent), event exclusion, universe ratio, 20d-avg gate
 M quant_platform/features/registry.py        (+139)  FeatureMetadata + provenance + build-time assertion
 M quant_platform/selection/gate_fusion.py    (+99)   tier tiling, veto-last, tier-dependent sort, structured @timed flags, A/B output split
 M scripts/next_week_d3_prediction.py         (+3)    feature_metadata wiring
 M scripts/run_d3_gate_event3_once.py         (+20)   feature_metadata wiring + timed flags
 M scripts/run_d3_gate_once.py                (+17)   feature_metadata wiring + timed flags
 M tests/test_gate_first.py                   (+427)  19 gate tests (all green)
?? docs/gate_fusion_review.md                         original code-grounded review
?? docs/gate_fusion_task.md                           task breakdown + status
?? docs/gate_fusion_impl_review.md                    read-only impl review (all items resolved)
?? docs/gate_fusion_gf08_plan.md                      THIS plan
```

Suggested commit message subject: `gate-fusion: GF-01..07 + GF-04b typed PIT gating, structured flags, output split`.
New bake-off harness (`scripts/gf08_bakeoff.py`) and results (`models/data/reports/gf08/`) are produced during GF-08 and committed separately.
```
Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
```
