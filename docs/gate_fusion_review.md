# Gate-First Fusion — Code-Grounded Review

*Review of `gate_fusion_evaluation_review.md` against the actual implementation. Files read: `quant_platform__selection__gate_fusion.py`, `quant_platform__evaluation__coverage_gate.py`, `tests__test_gate_first.py`. Claims are anchored to file:line; where the review draft's excerpts drift from the code, I note it.*

---

## Verdict

**The architecture is sound; the fusion *ranking* has two real bugs; the isolation guarantee is fail-open; and the single-day samples are smoke tests, not alpha evidence.** The direction — a stable long-history `base` plus a short-window `recent` model, combined by an asymmetric, veto-aware gate rather than a fixed-weight blend — is a defensible and interpretable way to admit short-history flow/event factors without polluting the base. It should **not** be declared the default over fixed-weight/RRF until the §7 bake-off runs, and three concrete defects below should be fixed first. Your own §10 (PIT proof-chain, single-day insufficiency, veto output semantics) already names the right risks; the code review sharpens two of them into bugs and adds one more.

---

## What is genuinely sound (keep)

- **Coverage-gate-as-data-shape, not alpha judge** (`coverage_gate.py` docstring) — clean separation of concerns.
- **Double defense for base admission**: a feature must be in `STABLE_FAMILIES` *and* have `overall_missing ≤ 0.30` (`coverage_gate.py:191–197`). Genuinely short-history features fail both, so they're blocked twice.
- **Events routed to veto/downgrade rather than raw alpha** — the Event3 sample supports this: the event recent model barely touches Top20 (`recent_gate_top20: 1`, `all_three_top20: 0`) and its value is the 10 vetoes. Using sparse events as risk flags, not alpha, is the right instinct.
- **Asymmetric fusion** (base sets direction, recent confirms) is more honest and auditable than a fixed weight, and it gives veto a natural home.

---

## Verified code-level findings (prioritized)

### P1 — `UNCLASSIFIED` swallows strong-base names and ranks them dead last (ranking inversion)

The tier conditions (`gate_fusion.py:112–129`) do **not** tile the `(base_pct, recent_pct)` space. Concretely, these regions fall through to `UNCLASSIFIED`:

- `base_pct ≥ 0.80` with `recent_pct ∈ [0.35, 0.60)` → not A (recent<0.60), not C (recent≥0.35) → **UNCLASSIFIED**. A **top-10% base** name with median recent confirmation.
- `base_pct ∈ [0.60, 0.80)` with `recent_pct ∈ [0.35, 0.85)` → **UNCLASSIFIED**. A top-25% base name, neutral recent.

`UNCLASSIFIED` has `_TIER_PRIORITY = 7` (`gate_fusion.py:52`) — **last**, below `E_REJECT` (5) and even `RISK_VETO` (6). So a top-10% base / median-recent stock ranks **below a double-weak reject and below a risk-vetoed name.** For a system whose thesis is "base sets direction," this is backwards. The samples confirm the scale: `UNCLASSIFIED = 85` (28% of universe) in the fund-flow run and `91` (30%) in Event3 — a huge bucket, much of it presumably decent-base names, dumped to the bottom. The test suite hides this: in `test_gate_first.py:77–100`, stock F (`base=0.70, recent=0.80`) would itself be `UNCLASSIFIED`, but it's given a `high_unlock` flag so it becomes `RISK_VETO` instead — the hole is never exercised.

**Fix:** make the tier map exhaustive so every `(base_pct, recent_pct)` maps to a defined, sensibly-ranked tier; at minimum, rank `UNCLASSIFIED` by `base_pct` in a middle position rather than last. This is the highest-impact correctness fix.

### P1 — `RISK_VETO` ranks above `UNCLASSIFIED`

`_TIER_PRIORITY`: `RISK_VETO = 6`, `UNCLASSIFIED = 7` (`gate_fusion.py:51–52`). A risk-vetoed stock outranks an unclassified one. Semantically a veto should be **strictly last**. (Answers your Q5.) Untested — no assertion orders `RISK_VETO` last.

### P2 — Veto / reject / observe are not separated from the recommendation pool

`write_gate_fusion_outputs` writes the **entire** ranked frame to one `{prefix}_ranked.csv` (`gate_fusion.py:155`); `RISK_VETO`, `E_REJECT`, `D_OBSERVE` are distinguished only by the `gate_tier` column and a larger `final_rank`. A consumer taking "top-K by rank" ingests lower tiers whenever the A/B pool is smaller than K. This is your §10 risk #3, confirmed. **Fix:** emit only actionable tiers (A/B) to the main CSV and route observe/veto/reject to a separate file (or a clearly delimited section), so "rank" can never be misread as "buyable."

### P2 — Isolation is **fail-open**: untagged features default to a base-eligible family

`family = family_by_col.get(col, "raw_aux")` (`coverage_gate.py:151`), and `raw_aux ∈ STABLE_FAMILIES` (`coverage_gate.py:19–27`). An **untagged** feature is therefore base-eligible by default. For a gate whose entire purpose is keeping short-history / PIT-risky factors out of the base, the default must be **fail-closed** (unknown → prediction-only / locked out), not fail-open. The `overall_missing ≤ 0.30` check mitigates this for *genuinely* short-history features (they have high missingness), but a **full-coverage, PIT-risky feature with an innocuous name** would pass into base undetected. (Core of your Q8.)

### P2 — PIT-risk / future-field detection is name-substring only

`has_future_field` / `has_pit_risk` come from `_has_token(col, …)` against `FUTURE_FIELD_TOKENS` / `PIT_RISK_TOKENS` (`coverage_gate.py:41–54, 177–178`). This catches only **self-declaring** leaks — the test uses `future_return_hint` (`test_gate_first.py:56`), the obvious case. A leaky feature named `momentum_signal` is undetectable *by design*. Name tokens are a convention, not a guarantee.

### P3 — `available_days` is lifetime, but gated as if recent

`available_days = non_null["_gate_date"].nunique()` over the **full** panel (`coverage_gate.py:156`), then checked as `available_days ≥ min_recent_trading_days` in `recent_allowed` (`coverage_gate.py:210`). So a long-history feature with sparse *recent* coverage can still pass the "recent trading days" floor on its lifetime count. It happens to work for short-history features (lifetime ≈ recent) but doesn't measure what its name implies.

### P3 — `recent_symbol_coverage` is a single-day snapshot

`recent_symbol_coverage` counts non-nulls on the **single** `end_date` (`coverage_gate.py:160–162`), then is used as a hard gate (`≥ 250`). One glitchy as-of day distorts a binary admit/reject decision. Prefer the 20-day average (already computed as `recent_20d_avg`) or a median for the gate threshold.

### P3 — Risk tokens are substring matches over concatenated free text

`veto/downgrade` use `_text_has_any` substring matching over `f"{risk_flags};{event_flags}"` (`gate_fusion.py:100–102, 56–58`). An `event_flag` containing `"coverage_ok"` would trip the `"coverage"` downgrade token; a flag containing `"unlock"` anywhere downgrades. Order (veto checked before downgrade) saves `high_unlock` from mis-classification, but the approach is brittle. Prefer structured flag codes matched exactly.

### P3 — Within-tier sort is `base_pct`-primary for *all* tiers

The final sort is `["_tier_priority", "base_pct", "recent_pct", "short_boost"]` for every tier (`gate_fusion.py:135–137`). For `D_OBSERVE` (recent strong, base weak) this ranks members by `base_pct` — which is weak by definition, so near-random; for `B_SHORT_BOOST` it partly defeats "recent boosts." (Your §4 question.) Sort keys should be **tier-dependent** — B and D should lead with `recent_pct`. More broadly, the percentile→tier discretization plus `base_pct` tie-break means the recent model's fine-grained information is mostly discarded *within* a tier; recent only moves names between coarse buckets.

---

## Direct answers to your Section 9 questions

1. **Gate-first vs fixed-weight / RRF?** Reasonable and more interpretable/risk-expressive, but currently **unproven** and carrying the P1 holes above. RRF and weighted blends are *smooth* (no cliffs, no holes); gate-first trades smoothness for interpretability + veto routing. Don't crown it the default — run it as **one arm** of the §7 bake-off. A hybrid likely dominates: a hard veto/observe filter (gate) wrapping a *smooth* combined score within the actionable pool.

2. **Does the coverage gate stop short-history factors entering base?** For the **currently-tagged** flow/event families, yes (double defense). But it is **fail-open on untagged features** (P2) and **blind to innocuously-named full-coverage leaks** (P2). Adequate for known families as tagged today; not robust to mis-tagging or silent leaks.

3. **Thresholds (120d / 80d / 250/300)?** (a) `recent_window_days=120` is **natural** days (`coverage_gate.py:141`) — switch to trading days; it's inconsistent with `min_recent_trading_days` (trading days) and drifts with holidays. (b) `250/300` is an **absolute** count — make it a fraction of the actual universe so it survives a universe change. (c) `min_recent_trading_days=80` at a 3-day horizon ≈ ~26 non-overlapping periods — thin; fine as a floor but demands regularization (Q4). Also note P3 (single-day coverage, lifetime `available_days`) means these thresholds don't measure quite what they claim.

4. **Event sparsity → recent overfitting?** **Yes, real.** Sparse binary/count event features over a 6-month window let the recent model split on rare coincidences. The Event3 sample already shows the event model contributes ~nothing to Top20 and everything to the 10 vetoes. Recommendation: treat events primarily as **rule-based veto/downgrade flags**, not alpha model inputs; if used as features, add minimum-support/regularization and **validate the vetoes' realized excess** (your §7 with/without-veto arm). Don't let sparse events drive `recent_pct`.

5. **Stricter veto ordering?** Yes — fix the P1 inversion so `RISK_VETO` is strictly last, and better, **exclude veto/reject from the main output** (P2). `RISK_DOWNGRADE` above `E_REJECT` is defensible, but as a clearly separate observation pool, not interleaved with actionable ranks.

6. **A/B/C/D thresholds fit D3 semantics?** Intent is coherent, but (a) the **holes** (P1) break it — strong-base/mid-recent names vanish into `UNCLASSIFIED`; (b) within-tier sort keys aren't tier-appropriate (P3); (c) hard cliffs make near-boundary assignment noise-sensitive. Fix the tiling and the B/D sort before trusting the semantics.

7. **Single-day = smoke test; need multi-date OOS?** **Yes, unequivocally.** One cross-section proves plumbing (routing, gate classification, veto firing), not alpha — the same lesson as the Config E single-cross-section. The §7 bake-off over ≥20 dates or walk-forward is **required** before any "gate-first is better" claim. The 3-day horizon makes many independent periods cheap to accumulate.

8. **Stronger PIT/family/source schema vs name tokens?** **Yes — highest-value hardening.** Replace name-token + convention with a typed per-feature registry record: `family`, `source`, `pit_safe: bool`, `known_at` rule (post-close-T / announce_date / …), `history_start`, `coverage`. The gate reads metadata; untagged ⇒ fail-closed (prediction-only). Add a test asserting a mis-tagged/untagged feature **cannot** reach base.

---

## Recommended sequence

1. **Fix P1 first** (tier tiling + `UNCLASSIFIED`/`RISK_VETO` ordering) — the current ranking demotes good base names below rejects and vetoes; every downstream metric is distorted until this is corrected. Add tests that exercise the holes and assert veto-last.
2. **Separate the output pools (P2)** so "rank" is never mistaken for "buyable."
3. **Harden isolation (P2):** fail-closed default family + typed PIT metadata + a leak test.
4. **Then, and only then, run the §7 bake-off** (base-only / recent-only / gate-first / RRF / fixed-weight / ±veto) over ≥20 dates with your proposed metric row. Until 1–3 are fixed, the bake-off would be measuring a distorted ranking.
5. **Regularize / gate the event features** toward veto-use; validate veto realized-excess explicitly.

One conceptual note for the bake-off: at a **3-day** horizon, privileging a 2023–2026 long-history base as the direction-setter is in tension with how fast short-horizon (flow/event) alpha decays. Worth including an arm where `recent` carries more than confirm/veto weight — the current design may be under-using the fastest signals precisely where they matter most.
