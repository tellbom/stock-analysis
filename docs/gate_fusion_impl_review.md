# Gate-First Fusion — Implementation Review (GF-04 / GF-04b / GF-05 / GF-06)

*Read-only code review. No code was modified; no commit made. Scope: the 8 points requested. Anchored to `file:line` in the current tree.*

**Verdict up front:** the gate logic is correct, internally consistent, and unit-tested. One wiring gap (the three D3 run-scripts call the gate in *legacy* mode, so the typed `pit_safe`/`known_at` layer is dormant there) is the only material risk. It does **not** block a **smoke** bake-off, but should be closed before any *authoritative* bake-off. See §8.

---

## 1. `FeatureMetadata` coverage — ✅ PASS

- `FeatureMetadata` (`registry.py:196-217`) carries `name/family/source/pit_safe/known_at/history_start` as required.
- `build_feature_metadata()` (`registry.py:275-291`) covers: all `FULL_SPECS`, the four event families via `_event_family_specs()`, `_RAW_AUX_COLUMNS`, `_FUNDAMENTAL_COLUMNS`.
- **Verified exhaustive vs the existing family map:** `feature_metadata_lookup()` and `cli._feature_family_lookup()` produce the **same 83 columns, zero difference** in either direction. No known feature column is left unregistered by accident.
- Every family string actually emitted by the spec modules (`technical, cross_sectional, valuation, industry, flow, sector_flow, concept_flow, proxy_flow, margin, event, announcement, announcement_events, dragon_tiger, block_trade, fundamental, raw_aux`) is present in `_FAMILY_PROVENANCE` (`registry.py:221-238`), so none falls through to the fail-closed fallback.

**Minor / by-design notes (not blockers):**
- `_metadata_for_family` fallback for an *unknown* family is `("feature_panel","",False)` → `pit_safe=False, known_at=""` → fail-closed (`registry.py:265-272`). Correct direction, but if a future feature module introduces a **new family string** and forgets to add it to `_FAMILY_PROVENANCE`, its features are silently locked out. Recommend a `build`-time assertion that every `spec.family` ∈ `_FAMILY_PROVENANCE`.
- `history_start` is always `None` (informational only; not gated). Acceptable — documented as such.
- Provenance is **family-level**, not per-column (e.g. every `margin` feature gets `known_at="post_close_T_lag1"`). Fine for now; per-column overrides would need a richer table later.

## 2. base vs recent-alpha PIT gating consistency — ✅ PASS

`base_allowed` (`coverage_gate.py:298-306`) and `recent_allowed` (`coverage_gate.py:328-337`) apply an **identical PIT bar**:
`... and pit_safe and known_at_present and not has_future_field and not has_pit_risk`.

The only differences are the intended eligibility scopes:
- base: `stable_family` + `overall_missing ≤ base_missing_threshold`.
- recent: `recent_alpha_eligible = stable_family or (short_family and not event_family)` + recent-window coverage/recency.

Reason strings mirror on both sides (`not-pit-safe`, `known-at-missing`, `unregistered`) guarded by `has_meta` so legacy mode never emits spurious reasons. **Consistent.**

*Minor:* base coverage is gated by `base_cov_threshold` (derived from `recent_symbol_ratio`) while recent uses `recent_cov_threshold` (from `recent_20d_symbol_ratio`); both default `0.83`. The `recent_symbol_*` naming applied to the *base* threshold is slightly confusing but functionally correct.

## 3. Event families → risk channel only, never `recent_pct` — ✅ PASS

- `EVENT_FAMILIES = {event, announcement, announcement_events, dragon_tiger, block_trade}` (`coverage_gate.py:` family-set block).
- `recent_alpha_eligible = stable_family or (short_family and not event_family)` (`coverage_gate.py:327`) — event families are `short_family=True, event_family=True` ⇒ **excluded from recent alpha**, reason `event-family-risk-channel-only` (`coverage_gate.py:343-344`).
- This exclusion is **family-based, not metadata-dependent**, so it holds even in legacy mode (see §8) — confirmed by the synthetic slice: `dt_x` (dragon_tiger) → `base=False recent=False`.
- Event *flags* reach tiering only through `risk_flags`/`event_flags` → veto/downgrade in `gate_fusion.py`; they never write `recent_pct` (which is a pure model input). **Two independent guarantees, both hold.**

## 4. Structured flag schema `name[:detail]@known_at` — ✅ PASS

- `_parse_flag_codes` (`gate_fusion.py:62-84`) parses each `;`-separated part as `partition("@")` → `code`, `time_part`; `name = code.split(":",1)[0]`; returns `(name, bool(time_part))`.
- Veto/downgrade require **registered code AND timing**: `any(name in veto_set and timed ...)` / `... downgrade_set and timed ...` (`gate_fusion.py:131-132`).
- **Confirmed:** an untimed flag (`high_unlock` with no `@`) does **not** trigger veto/downgrade (unit test `test_event_flag_requires_structured_timing_for_risk_channel`); a timed one does. `coverage_ok` never trips `coverage` (exact-name match). Empty time (`code@`) → `timed=False` → inadmissible (correct).

## 5. Producer flags all carry `@as_of` — ✅ PASS

All six flag-emitting sites are timed:
- `run_d3_gate_once.py:256` `high_unlock:{d}d_ratio_{r}@{asof}`, `:258` `unlock:{d}d@{asof}`.
- `run_d3_gate_event3_once.py:350` `risk_warning@{asof}`, `:352` `major_announcement@{asof}`, `:354` `dragon_tiger@{asof}`, `:356` `large_discount_block_trade@{asof}`.
- `asof = actual_as_of.isoformat()` in both. No untimed producer remains.

*Note (not a defect):* of the event3 flags, only `risk_warning` is a registered veto token; `major_announcement`/`dragon_tiger`/`large_discount_block_trade` are **not** in the veto/downgrade token sets, so they are informational and never change a tier — this matches prior behaviour (they didn't contain the old substrings either).

## 6. `_ranked.csv` / `_observe.csv` backward compatibility — ✅ PASS

- `write_gate_fusion_outputs` still returns the **2-tuple** `(csv_path, md_path)` (`gate_fusion.py`), writing `_observe.csv` as a side output.
- Only unpacking call site: `cli.py:1431` `csv_path, md_path = write_gate_fusion_outputs(...)` — compatible. Script call sites (`run_d3_gate_once.py:446`, `run_d3_gate_event3_once.py:526`) call as a bare statement — compatible.
- **No downstream reader of the gate-fusion `_ranked.csv` exists.** The `_ranked.csv` readers found (`d3_wednesday_review.py:65`, `next_week_d3_prediction.py:428`) operate on `D3_Prediction_*_ranked.csv` — a *different artifact* (has a `selected` column; produced by the prediction pipeline, not `write_gate_fusion_outputs`). The gate-fusion ranked CSV is a terminal artifact.
- **Behavioural change (intended, GF-03):** the gate-fusion `_ranked.csv` now contains **A/B tiers only**; observe/veto/reject moved to `_observe.csv`. No code breaks, but any human/tool eyeballing the old file for veto rows must now read `_observe.csv`.

## 7. Residual dependence on substring flags / old 2-tuple — ✅ PASS (one cosmetic note)

- `_text_has_any` (old substring matcher) is **removed**; only doc references remain.
- No code depends on the old full-frame `_ranked.csv` or a >2 return arity.
- **Cosmetic, unrelated:** `cli.py:1182-1198` groups feature columns by substring (`"flow" in c`, `"unlock" in c`, …) for the **regime-analysis IC breakdown** report only. This is a display grouping, not PIT/flag logic, and predates this work — flagged for awareness, not action.

## 8. ⚠️→✅ Typed metadata wiring into the D3 run scripts — RESOLVED

*Original finding:* `feature_metadata=` was passed only at `cli.py:209`; the three D3 run-scripts called the gate in legacy mode, leaving the typed `pit_safe`/`known_at` layer dormant there.

*Resolution (this pass):* `feature_metadata=feature_metadata_lookup()` is now passed at every gate call site:
- `run_d3_gate_once.py:355, 366` (+ import `:38`, `feature_metadata = feature_metadata_lookup()` at `:351`)
- `run_d3_gate_event3_once.py:428, 439` (+ import `:45`, computed once at `:424`)
- `next_week_d3_prediction.py:245` (+ import `:54`)

All three `py_compile` clean; gate/coverage tests green (91 passed). The typed PIT layer is now active end-to-end through the script path — no call site remains in legacy mode.

---

## Passed items

| # | Area | Status |
|---|---|---|
| 1 | `FeatureMetadata` coverage complete (83/83, all families provenanced) | ✅ |
| 2 | base/recent PIT gating symmetric | ✅ |
| 3 | event families excluded from `recent_pct` (family-based, legacy-safe) | ✅ |
| 4 | untimed/unregistered flags cannot veto/downgrade | ✅ |
| 5 | all producer flags carry `@as_of` | ✅ |
| 6 | `_ranked.csv`/`_observe.csv` backward compatible; no broken readers | ✅ |
| 7 | no residual substring-flag / old-2-tuple dependence | ✅ |

## Potential risks

| Sev | Risk | Location | Status / action |
|---|---|---|---|
| ~~Med~~ ✅ | Typed `pit_safe`/`known_at` gating dormant in the 3 D3 scripts (legacy mode) | `run_d3_gate_once.py:355,366`; `run_d3_gate_event3_once.py:428,439`; `next_week_d3_prediction.py:245` | **RESOLVED** — `feature_metadata=feature_metadata_lookup()` wired into all call sites; py_compile clean, tests green |
| ~~Low~~ ✅ | New feature family not added to `_FAMILY_PROVENANCE` → silently fail-closed | `registry.py` | **RESOLVED** — `build_feature_metadata()` now raises `ValueError` listing any spec/event family missing provenance (helper `_families_without_provenance`); test `test_all_feature_families_have_provenance` |
| Low | `_ranked.csv` now A/B-only; human/tooling expecting all tiers must read `_observe.csv` | `gate_fusion.write_gate_fusion_outputs` | doc note in the run scripts' README/output header |
| Info | `recent_symbol_ratio` naming applied to the *base* coverage threshold | `coverage_gate.py:301` | optional rename for clarity |
| Info | Regime-report substring grouping (unrelated to PIT) | `cli.py:1182-1198` | none |

## GF-08 readiness

- **Smoke bake-off (plumbing / routing / gate classification / veto firing): ✅ GREEN.** The gate logic is correct and unit-tested (18/18 gate tests; full suite 319 passed, only pre-existing `optuna`/MLflow-Windows env failures). Event routing, fail-closed, ratio thresholds, and structured flags are all active through the script path.
- **Authoritative bake-off: ✅ UNBLOCKED.** The typed PIT layer is now wired into all three D3 scripts (Med risk resolved), so the base/recent sets produced during the bake-off will reflect the full GF-04b ruling.

**Recommendation:** GF-08 (smoke and authoritative) is unblocked from the gate-logic side. The remaining items are Low/Info only. Proceed to GF-08 when instructed.
