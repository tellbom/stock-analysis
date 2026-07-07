# Gate-First Fusion — Task Breakdown (Awaiting Review)

*Companion to `docs/gate_fusion_review.md` (code-grounded review). Task nodes are ordered by the review's §11 recommended sequence. No implementation begins until this document is reviewed and approved.*

*Anchored files: `quant_platform/selection/gate_fusion.py`, `quant_platform/evaluation/coverage_gate.py`, `tests/test_gate_first.py`. Line numbers reflect the current tree.*

---

## Task ID Convention

`GF-{nn}` where GF = Gate-First Fusion. Priority mirrors the review: `P1` = ranking-correctness (do first), `P2` = output safety / isolation, `P3` = measurement/robustness, plus event-regularization and the bake-off.

`[BLOCKER]` tasks must land before the §7 bake-off (GF-08) is run, otherwise the bake-off measures a distorted ranking.

---

## Execution Order & Dependencies

```
GF-01 (P1 tier tiling)  ┐
GF-02 (P1 veto-last)    ┘─ ranking correctness  ← FIRST; every downstream metric distorted until fixed
        └─ GF-03 (P2 output pool separation)
                └─ GF-04 (P2 fail-closed default + typed PIT registry)
                        └─ GF-08 (§7 bake-off, ≥20-date OOS)   [needs GF-01..04]
GF-05 (event regularization → veto use)   ← parallel with GF-03/04
GF-06 (P3 measurement/threshold fixes)    ← independent, interleavable
GF-07 (P3 tier-dependent within-tier sort)
```

---

## P1 — Ranking Correctness (do first)

### GF-01: Make the tier map exhaustive (fix `UNCLASSIFIED` holes)

| Field | Detail |
|---|---|
| **Task ID** | GF-01 |
| **Priority** | P1 — Highest-impact correctness fix `[BLOCKER]` |
| **Order** | 1 |

**Problem.** The tier conditions in `gate_fusion.py:112–129` do not tile the `(base_pct, recent_pct)` space. Two regions fall through to `UNCLASSIFIED`:
- `base_pct ≥ 0.80` with `recent_pct ∈ [0.35, 0.60)` — a top-10% base name with median recent confirmation.
- `base_pct ∈ [0.60, 0.80)` with `recent_pct ∈ [0.35, 0.85)` — a top-25% base name, neutral recent.

`UNCLASSIFIED` has `_TIER_PRIORITY = 7` (`gate_fusion.py:52`) — last, below `E_REJECT (5)` and `RISK_VETO (6)`. Sample scale: `UNCLASSIFIED = 85` (28%) in the fund-flow run, `91` (30%) in Event3 — decent-base names dumped to the bottom. The current test hides this: stock F (`base=0.70, recent=0.80`) would be `UNCLASSIFIED` but the `high_unlock` flag reroutes it to `RISK_VETO` (`tests/test_gate_first.py:87, 98`), so the hole is never exercised.

**Fix.** Make the tier map exhaustive — every `(base_pct, recent_pct)` maps to a defined, sensibly-ranked tier. At minimum, rank `UNCLASSIFIED` by `base_pct` in a middle position rather than last.

**Deliverables.** Revised tier logic; updated `_TIER_PRIORITY`; tests covering both hole regions.

**Validation.**
- Test with `base=0.85, recent=0.45` and `base=0.70, recent=0.50` (no risk flags): each is assigned a defined tier and does **not** rank below `E_REJECT`/`RISK_VETO`.
- No `(base_pct, recent_pct)` on a fine grid maps to a last-place bucket.

---

### GF-02: Rank `RISK_VETO` strictly last

| Field | Detail |
|---|---|
| **Task ID** | GF-02 |
| **Priority** | P1 `[BLOCKER]` (answers review Q5) |
| **Order** | 2 |

**Problem.** `_TIER_PRIORITY`: `RISK_VETO = 6`, `UNCLASSIFIED = 7` (`gate_fusion.py:51–52`). A risk-vetoed stock outranks an unclassified one. Semantically a veto should be strictly last. Untested — no assertion orders `RISK_VETO` last.

**Fix.** Reorder `_TIER_PRIORITY` so `RISK_VETO` is the lowest-priority tier (strictly last). Coordinate with GF-01's new `UNCLASSIFIED` placement.

**Validation.** Assert a `RISK_VETO` row has the maximum `final_rank` in the fused frame.

---

## P2 — Output Safety & Isolation

### GF-03: Separate veto / reject / observe from the recommendation pool

| Field | Detail |
|---|---|
| **Task ID** | GF-03 |
| **Priority** | P2 `[BLOCKER]` (review §10 risk #3) |
| **Order** | 3 |
| **Depends on** | GF-01, GF-02 |

**Problem.** `write_gate_fusion_outputs` writes the entire ranked frame to one `{prefix}_ranked.csv` (`gate_fusion.py:155`); `RISK_VETO`, `E_REJECT`, `D_OBSERVE` are distinguished only by the `gate_tier` column and a larger `final_rank`. A consumer taking "top-K by rank" ingests lower tiers whenever the A/B pool is smaller than K.

**Fix.** Emit only actionable tiers (A/B) to the main CSV; route observe/veto/reject to a separate file (or a clearly delimited section), so "rank" can never be misread as "buyable."

**Validation.** Main output contains no `RISK_VETO`/`E_REJECT`/`D_OBSERVE` rows; a separate file/section contains them; test asserts the split.

---

### GF-04: Fail-closed isolation + typed PIT feature registry

| Field | Detail |
|---|---|
| **Task ID** | GF-04 |
| **Priority** | P2 `[BLOCKER]` — highest-value hardening (review Q8) |
| **Order** | 4 |
| **Depends on** | (independent of GF-01..03, but sequence before GF-08) |

**Problem A (fail-open default).** `family = family_by_col.get(col, "raw_aux")` (`coverage_gate.py:165`), and `raw_aux ∈ STABLE_FAMILIES` (`coverage_gate.py:33–41`). An untagged feature is base-eligible by default. A full-coverage, PIT-risky feature with an innocuous name passes into base undetected.

**Problem B (name-substring PIT detection).** `has_future_field` / `has_pit_risk` come from `_has_token(col, …)` against `FUTURE_FIELD_TOKENS` / `PIT_RISK_TOKENS` (`coverage_gate.py:191–192`). This catches only self-declaring leaks — a leaky feature named `momentum_signal` is undetectable by design.

**Fix.** Replace name-token + convention with a typed per-feature registry record: `family`, `source`, `pit_safe: bool`, `known_at` rule (post-close-T / announce_date / …), `history_start`, `coverage`. The gate reads metadata; **untagged ⇒ fail-closed (prediction-only, locked out of base)**.

**Validation.** Test asserts a mis-tagged/untagged feature **cannot** reach base. Existing tests in `tests/test_gate_first.py` still pass.

**Note.** Problems A and B share the same registry direction — implement as one hardening effort.

---

## P3 — Measurement / Robustness

### GF-06: Fix threshold and coverage-metric semantics

| Field | Detail |
|---|---|
| **Task ID** | GF-06 |
| **Priority** | P3 (review §9 Q3 + P3 findings) |
| **Order** | interleavable |

Bundle of measurement-fidelity fixes in `coverage_gate.py`:

1. **`available_days` is lifetime, gated as recent.** `available_days = non_null["_gate_date"].nunique()` over the full panel (`coverage_gate.py:170`), checked as `≥ min_recent_trading_days` in `recent_allowed` (`coverage_gate.py:224`). Measure recent-window trading days instead.
2. **`recent_symbol_coverage` is a single-day snapshot** (`coverage_gate.py:174–178`) used as a hard gate. Prefer the 20-day average `recent_20d_avg` (already computed) or a median.
3. **`recent_window_days = 120` is natural days** (`coverage_gate.py:155`) — inconsistent with `min_recent_trading_days` (trading days); switch to trading days.
4. **`250/300` coverage thresholds are absolute counts** — express as a fraction of the actual universe so they survive a universe change.

**Validation.** Unit tests for each: a long-history/recent-sparse feature no longer passes the recent floor on lifetime count; a single glitchy as-of day no longer flips admit/reject; thresholds computed from universe size.

---

### GF-07: Tier-dependent within-tier sort keys

| Field | Detail |
|---|---|
| **Task ID** | GF-07 |
| **Priority** | P3 (review §4) |
| **Order** | interleavable |

**Problem.** The final sort is `["_tier_priority", "base_pct", "recent_pct", "short_boost"]` for every tier (`gate_fusion.py:135–137`). For `D_OBSERVE` (recent strong, base weak) this ranks by `base_pct` — near-random; for `B_SHORT_BOOST` it partly defeats "recent boosts."

**Fix.** Make within-tier sort keys tier-dependent — B and D lead with `recent_pct`.

**Validation.** Test asserts a `D_OBSERVE` / `B_SHORT_BOOST` block is ordered by `recent_pct`, not `base_pct`.

---

### GF-05: Regularize event features toward veto use

| Field | Detail |
|---|---|
| **Task ID** | GF-05 |
| **Priority** | P3 (review §9 Q4) |
| **Order** | parallel |

**Problem.** Sparse binary/count event features over a 6-month window let the recent model split on rare coincidences. Event3 shows the event model contributes ~nothing to Top20 and everything to the 10 vetoes.

**Also (P3, brittle risk tokens).** `veto/downgrade` use `_text_has_any` substring matching over `f"{risk_flags};{event_flags}"` (`gate_fusion.py:100–102, 56–58`). A flag containing `"coverage_ok"` trips the `"coverage"` downgrade token; any `"unlock"` substring downgrades. Prefer structured flag codes matched exactly.

**Fix.** Treat events primarily as rule-based veto/downgrade flags, not alpha inputs; if used as features, add minimum-support/regularization. Replace substring risk-token matching with exact structured codes. Don't let sparse events drive `recent_pct`.

**Validation.** Test that `event_flags="coverage_ok"` does **not** trigger a downgrade; event-driven contribution to `recent_pct` is bounded/absent per config.

---

## Verification

### GF-08: §7 bake-off over multi-date OOS

| Field | Detail |
|---|---|
| **Task ID** | GF-08 |
| **Priority** | P2/P3 — required before any "gate-first is default" claim |
| **Order** | LAST — after GF-01, GF-02, GF-03, GF-04 |
| **Depends on** | GF-01, GF-02, GF-03, GF-04 `[BLOCKER prerequisites]` |

**Problem.** A single cross-section proves plumbing (routing, gate classification, veto firing), not alpha — same lesson as the Config E single-cross-section.

**Fix.** Run the §7 bake-off over **≥20 dates or walk-forward**, arms: `base-only / recent-only / gate-first / RRF / fixed-weight / ±veto`. Include an extra arm where `recent` carries more than confirm/veto weight (3-day horizon: fastest signals may be under-used). Validate the vetoes' realized excess explicitly (±veto arm). The 3-day horizon makes independent periods cheap to accumulate.

**Validation.** Per-arm metric row over the date set; explicit realized-excess of the veto pool; gate-first is only crowned default if it dominates after GF-01..04 are in.

---

## Keep (confirmed sound — do not change)

- Coverage-gate as data-shape, not alpha judge — clean separation (`coverage_gate.py` docstring).
- Double defense for base admission: `STABLE_FAMILIES` **and** `overall_missing ≤ 0.30` (`coverage_gate.py:205–211`).
- Events routed to veto/downgrade rather than raw alpha.
- Asymmetric fusion (base sets direction, recent confirms) — honest, auditable, natural home for veto.

---

## Summary Table

| Task ID | Name | Priority | Order | Blocker for bake-off |
|---|---|---|---|---|
| GF-01 | Exhaustive tier tiling (fix UNCLASSIFIED holes) | P1 | 1 | ✅ |
| GF-02 | RISK_VETO strictly last | P1 | 2 | ✅ |
| GF-03 | Separate veto/reject/observe output pools | P2 | 3 | ✅ |
| GF-04 | Fail-closed default + typed PIT registry | P2 | 4 | ✅ |
| GF-05 | Regularize events → veto use; structured flags | P3 | parallel | — |
| GF-06 | Threshold/coverage-metric semantics | P3 | interleave | — |
| GF-07 | Tier-dependent within-tier sort | P3 | interleave | — |
| GF-08 | §7 bake-off, ≥20-date OOS | P2/P3 | last | needs GF-01..04 |

---

## Implementation Status (updated after first dev pass)

All changes landed in `quant_platform/selection/gate_fusion.py`,
`quant_platform/evaluation/coverage_gate.py`, `scripts/run_d3_gate_once.py`,
and `tests/test_gate_first.py` (10 gate tests pass; full suite green except
pre-existing `optuna`/MLflow-Windows env failures unrelated to these files).

| Task | Status | Notes |
|---|---|---|
| GF-01 | ✅ Done | `UNCLASSIFIED` moved to middle priority (3); tier map no longer dumps strong/mid-base names last. Test: `test_tier_map_has_no_dead_last_hole_and_veto_is_last`. |
| GF-02 | ✅ Done | `RISK_VETO` priority = 7 (strictly last). Same test asserts veto has max `final_rank`. |
| GF-03 | ✅ Done | `_ranked.csv` now actionable-only (A/B); observe/veto/reject routed to `{prefix}_observe.csv`. 2-tuple return preserved. Test: `test_output_pools_separated`. |
| GF-04 | ✅ Done | Fail-closed default + **typed PIT registry** landed in `features/registry.py` (`FeatureMetadata`: family/source/pit_safe/known_at/history_start; `feature_metadata_lookup()`). `coverage_gate` reads it: unregistered / `pit_safe=False` / missing `known_at` ⇒ locked out of base. Name-token detection retained as a *secondary* guard (tokens now also include `label`/`target`). CLI gate path wired. |
| GF-04b | ✅ Done | **Extends typed-PIT gating to the recent alpha model** (reviewer ruling on OQ#5). recent admission now requires registered + `pit_safe=True` + non-empty `known_at`; leak tokens block recent too; **event families never drive `recent_pct`** (`EVENT_FAMILIES` routed to the risk channel only). **Risk/veto/downgrade channel hardened**: an event flag drives veto/downgrade only when it is a registered code carrying an explicit `@known_at` timing (`name[:detail]@time`); untimed/unregistered events are inadmissible. Producers (`run_d3_gate_once.py`, `run_d3_gate_event3_once.py`) updated to emit timed flags. Tests: `test_typed_metadata_gates_recent_alpha`, `test_name_tokens_block_base_and_recent_even_if_pit_safe`, `test_event_family_feature_not_recent_alpha`, `test_event_flag_requires_structured_timing_for_risk_channel`, `test_event_flag_drives_veto_not_recent_pct`. |
| GF-05 | ✅ Done (code portion) | Substring matching replaced with exact structured flag codes (`name:detail`, match on `name`). `coverage_ok` no longer trips `coverage`. Producer `run_d3_gate_once.py` updated to `unlock:{d}d`. Test: `test_flag_codes_exact_match_not_substring`. **Deferred:** event-feature min-support/regularization (modeling change, needs data). |
| GF-06 | ✅ Done | **Universe-ratio thresholds** landed: `recent_symbol_ratio`/`recent_20d_symbol_ratio` default `0.83`; ratio takes priority, absolute `*_threshold` retained as fallback. Plus trailing-trading-day window, recent-window trading-day count, 20d-avg coverage gate. Tests: `test_universe_ratio_threshold_scales_with_universe`, `test_single_day_coverage_glitch_does_not_flip_gate`. |
| GF-07 | ✅ Done | Within-tier sort is tier-dependent; `B_SHORT_BOOST`/`D_OBSERVE` lead with `recent_pct`. Test: `test_within_tier_sort_is_recent_led_for_d_observe`. |
| GF-08 | ⏳ Pending | Requires multi-date panel + base/recent models; run after reviewer sign-off on GF-04/GF-06 deferred items. |

**Behavior-change flags for review:**
- GF-04 fail-closed: features the pipeline forgot to tag are now **excluded from base** (previously silently admitted via `raw_aux`). Confirmed on the real 300-symbol panel (`features/80fd2338`): base `26 → 25`, locked_out `8 → 9` — `close` (a raw price level, never a legitimate cross-sectional feature) is dropped from base with reason `unregistered`. No legitimately-tagged feature was dropped; all real tagged features are `pit_safe`.
- GF-06 gate now decides on 20d-average coverage vs a universe-fraction threshold (`0.83 × universe`) rather than an absolute as-of-day count. At the 300-symbol universe this matches the old ~250 floor (0.83×300 = 249); it rescales automatically for other universe sizes.

**Coverage-gate report confirmation (legacy default → typed+ratio, real panel):**
```
ORIGINAL (raw_aux default)          : base=26 recent=26 locked_out=8
CURRENT  (typed fail-closed + ratio): base=25 recent=25 locked_out=9
dropped from base : ['close']   reason: unregistered
```

---

## Summary Table (original scope)

---

## Open Questions for Reviewers

1. GF-04: implement the typed feature registry as a standalone module, or extend the existing `features/registry.py`? Scope and landing files?
2. GF-08: bake-off date set — fixed ≥20-date panel vs walk-forward? Which metric row is authoritative?
3. GF-05: this iteration — events veto-only, or retain as features with regularization?
4. GF-06: land the P3 measurement fixes with GF-01 (one batch), or as a separate second batch after ranking is corrected?
5. **GF-04 scope — RESOLVED (reviewer ruling):** typed PIT gating now applies to the **recent alpha model as well as base** — `pit_safe=False` / missing `known_at` / unregistered / leak-token features are locked out of both; event families are routed to the risk/veto channel only and never drive `recent_pct`. Implemented as **GF-04b**.
