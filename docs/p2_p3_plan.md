# P2 — Modeling, Evaluation & Alpha Validation · P3 — Research Iteration, Lifecycle & Reproducible Comparison: Plan and Tasks

*Execution plan for the modeling half of the platform. Companion to the P0/P1 plan. Assumes P0/P1 are complete: leakage-checked feature store, label store, point-in-time universe, and a model-ready `(symbol, date, features…, labels…)` panel all exist and are trustworthy. No implementation here — this is an implementation-ready plan for review. Same guiding principle carries over: **the dataset is the product; the model is a consumer** — and now the consumer must be made honest.*

---

## Orientation: what the platform must solve after P0/P1

P0/P1 answered *"is the data clean and leakage-free?"* They did **not** answer the two questions that actually decide whether this project produced anything real:

1. **Does a model trained on this panel contain genuine, cost-survivable alpha — and can I prove it rather than hope it?**
2. **Can I keep improving models over hundreds of experiments without the act of searching silently manufacturing fake alpha?**

These are different problems with different failure modes, and that difference is what sets the P2/P3 boundary.

### Why these boundaries (a deliberate departure from the earlier roadmap)

The original sketch split modeling into "first model + evaluation" (P2) and "iteration loop: HPO/SHAP/backtest" (P3). On reflection that boundary is drawn in the wrong place, because it separates activities by *tool* rather than by *the threat to validity each one manages*. Redrawn:

- **P2 = build the honest measuring instrument and render exactly one verdict.** Threat model: leakage inside a single fit, and an IC that looks great but evaporates after transaction costs and T+1 execution. Therefore the **single-model, cost-aware signal backtest lives in P2** — it is the final arbiter of whether one model's alpha is economically real, inseparable from "evaluation." P2 is depth-first on a single validated baseline.
- **P3 = run that instrument many times without corrupting it.** Threat model: *selection bias from repeated experimentation* — the multiple-comparisons / backtest-overfitting problem, which does not exist until you start iterating. It needs machinery P2 never required: a research ledger, a lockbox spent sparingly, deflated metrics, pre-registered promotion criteria. **HPO, explainability, model comparison/selection, registry/governance, and decay monitoring all belong here**, because each is a *many-models-over-time* concern.

The unifying test for "which phase does X belong to": *does X help render one honest verdict (P2), or does it help iterate honestly across many verdicts (P3)?* That single question cleanly places every task below.

P4+ (closing the loop into the prediction/report surface, and depth work like sequence models and meta-labeling) remain downstream and out of scope here; they're noted at the end so the boundary stays honest.

---

## Phase P2 — Modeling, Evaluation & Alpha Validation

### Phase Goal

From the model-ready panel, produce a **single, rigorously validated, fully reproducible baseline model** and an **honest go/no-go verdict on whether it contains genuine, cost-survivable alpha** — and in doing so, build the evaluation machinery and the validation standard that the entire rest of the platform depends on. The deliverable is less "a model" than "a trustworthy answer to *is there a signal here?*, plus the instrument that produced it."

### Definition of Done

1. A LightGBM baseline trains under a **time-aware, purged + embargoed** cross-validation with zero leakage — the P1 leakage harness (T1.7) extended to cover the modeling step.
2. A standardized **evaluation report** emits IC, Rank IC, ICIR, quantile-spread (with monotonicity), precision@k, and calibration from a single call.
3. The model is measured against a documented **baseline gauntlet** (trivial baselines + the Qlib CSI 300 yardstick from T0.9), and the comparison is recorded.
4. A **cost-and-delay-aware signal backtest** (T+1 execution, transaction costs, turnover) confirms or denies that the IC translates into a realizable long-short spread.
5. **Robustness and null tests** (subperiod/regime stability, feature ablation, label-shuffle null, leaked-feature canary) have been run and recorded.
6. Every run carries a **reproducibility manifest** (data-snapshot id + feature-set hash + label-set hash + code commit + environment + seed) sufficient to regenerate it bit-for-bit.
7. A single **alpha-verdict document** states whether genuine alpha is present, with the evidence and the caveats.

### Key Design Decisions

- **The lockbox split is sacred.** Carve a final test slice (the most recent ~12–18 months) and touch it **exactly once**, at the very end of P2, for the verdict. All development happens on train+validation via purged CV. This isn't bureaucracy — it's the asset P3's entire iteration loop must protect, and it can only be created cleanly now, before any tuning pressure exists.
- **Purged k-fold + embargo is the only CV that exists.** Never random k-fold. Drop training samples whose label windows overlap the validation fold; embargo a gap after each fold so autocorrelation can't leak across the boundary (de Prado). `sklearn`'s `TimeSeriesSplit` is a starting scaffold but is *insufficient alone* — it neither purges nor embargoes.
- **The IC family is the primary metric; accuracy is a diagnostic.** Headline on **Rank IC and ICIR** (mean per-day cross-sectional Spearman of prediction vs. forward return; ICIR = mean/std of that daily series). Directional accuracy and AUC are secondary, because a 52%-accuracy model with stable positive Rank IC beats a 70%-accuracy model whose hits cluster in one week.
- **The backtest is the arbiter, never the objective.** Run one deliberately simple, honest long-short decile backtest with T+1 execution and realistic costs to *confirm* the IC is economically real. Do not optimize against it — the moment the backtest becomes a target, overfitting begins. It sits strictly downstream of the metric.
- **Baselines are mandatory, not optional.** Yesterday's return, universe/sector mean reversion, a single momentum factor, and a random predictor. A model that doesn't clear these has no alpha regardless of how its IC reads in isolation. Beating *nothing* is not signal.
- **Tracking from run #1.** MLflow logs params, metrics, artifacts, and the reproducibility manifest from the very first experiment. "I'll add tracking later" is how the first interesting result becomes unreproducible.
- **Null tests separate signal from pipeline artifact.** A **label-shuffle** test (retrain on permuted labels → IC must collapse to ~0) proves the score isn't an artifact of the harness; a **leaked-feature canary** (inject an obviously future-peeking feature → IC must inflate) proves the leakage controls actually bite. If the null doesn't collapse, you've found a bug, not alpha.
- **Calibration measured now, even though probabilities are consumed later.** Raw GBM scores aren't probabilities; the eventual report layer needs calibrated outputs, so reliability curves are part of the standard report from the start.

### Task Breakdown

| ID | Task | Reuses / builds on | Reference | Done when |
|---|---|---|---|---|
| **T2.1** | **Training harness** — time-aware splitter + purged/embargoed CV; deterministic seeds; extends the leakage harness to the fit step | panel loader, leakage harness (T1.7) | de Prado purged k-fold + embargo; `TimeSeriesSplit` as scaffold | A model trains across folds with no leakage flagged |
| **T2.2** | **LightGBM baseline** — regression/ranking objective on cross-sectional features; `sklearn` Pipeline so preprocessing can't detach from the model | feature/label stores | Qlib `LGBModel` config | Out-of-fold predictions exist for the whole universe |
| **T2.3** | **Evaluation module** — IC / Rank IC / ICIR / quantile-spread + monotonicity / precision@k / calibration, from one call | — | eval §2/§7; `sklearn.calibration` | `evaluate(pred, fwd_return)` emits the full report |
| **T2.4** | **Baseline gauntlet** — trivial baselines + Qlib CSI 300 yardstick | Qlib bridge (T0.9) | Qlib LightGBM example | A comparison table positions the model against all baselines |
| **T2.5** | **Cost-aware signal backtest** — long-short decile portfolio, T+1 execution, transaction costs, turnover; net-of-cost spread; IC-to-return reconciliation | calendar (T0.3) | vectorbt / backtesting.py; Qlib backtester | Net spread reported and reconciles with IC sign/magnitude |
| **T2.6** | **Robustness & null tests** — subperiod/regime stability, feature ablation, label-shuffle null, leaked-feature canary | leakage harness (T1.7) | — | All run and recorded; null collapses IC to ~0 |
| **T2.7** | **Experiment tracking + reproducibility manifest** — MLflow; log data-snapshot id, feature/label-set hashes, commit, env, seed | feature-set version (T1.5) | MLflow tracking | Any logged run regenerates identical metrics from its manifest |
| **T2.8** | **Alpha-verdict report** — synthesize all evidence into a go/no-go on genuine alpha, with caveats | T2.3–T2.6 outputs | — | A reviewer can read the verdict and trace its basis |

### Deliverables

The training harness, the LightGBM baseline, the evaluation module, the baseline-comparison table, the cost-aware backtest, the robustness/null suite, MLflow tracking + reproducibility manifests, and the alpha-verdict document.

### Dependencies

- **P1:** feature store, label store, feature-set versioning (T1.5), and the leakage harness (T1.7) — the harness is extended, not rebuilt.
- **P0:** point-in-time universe (T0.2) and calendar (T0.3) for correct cross-sectional evaluation and T+1 shifting; the Qlib bridge (T0.9) for the external yardstick (T2.4).

### Risks & Pitfalls

- **Optimizing the backtest** instead of treating it as confirmation — the fastest route into overfitting. Keep it simple and strictly downstream of the metric.
- **Spending the lockbox more than once** — its verdict is only valid on first contact. Enforce one-shot use in tooling, not willpower.
- **CV leakage via overlapping labels** — without purge+embargo, a multi-day forward label bleeds across the fold boundary; the harness must extend to the modeling step, not stop at the panel.
- **An IC with no economic translation** — a positive Rank IC that dies after costs and turnover is not alpha. The backtest exists precisely to catch this; if it disagrees with the IC, believe the net spread.
- **Skipping calibration** — discovering at the report stage that scores aren't probabilities, with no calibration baseline to compare against.
- **Single-regime flattery** — a strong aggregate IC driven entirely by one market regime. Subperiod stability is the check that exposes it.

---

## Phase P3 — Research Iteration, Lifecycle & Reproducible Comparison

### Phase Goal

Turn the one-off validated baseline into a **sustainable, honest research loop** — where many candidate models can be generated, compared apples-to-apples, explained, selected, versioned, and monitored for decay — while **actively defending against the multiple-comparisons problem** that otherwise manufactures illusory alpha through sheer volume of experimentation. The phase succeeds when iterating *faster* does not mean fooling yourself *more*.

### Definition of Done

1. **HPO (Optuna)** runs inside the purged CV with persisted, resumable studies, optimizing an out-of-fold metric — and never touches the lockbox.
2. A **model zoo** (≥3 model types) sits behind one train/predict/evaluate interface; adding a model is configuration, not a rewrite.
3. Every experiment **auto-emits the standardized P2 evaluation report**, and results aggregate onto a comparison **leaderboard** built on identical folds and feature versions.
4. A **model-selection protocol** exists: champion/challenger with pre-registered promotion thresholds, statistical significance testing of IC differences, and a **research ledger** accounting for how many configurations were tried (multiple-testing / deflated metrics).
5. **SHAP explanations** (global + per-name) and **feature-importance stability** are produced for promotion candidates, gated by an **economic-sensibility review**.
6. A **model registry** stores versioned models with full **lineage** (data → features → labels → model → metrics) and a **model card** per promoted model.
7. **Decay/drift monitoring** re-evaluates the champion on new data on a schedule and flags IC decay, feature drift, and retrain/retire triggers.
8. Any past result is **reproducible by one command** (DVC-versioned data + pinned environment + logged seed), and a CI check verifies metric stability on re-run.

### Key Design Decisions

- **The lockbox is spent sparingly, and the spending is metered.** Iteration happens entirely on train+validation under purged CV. The final test slice is consulted **only** to confirm a *promoted* champion, and every consultation is logged and counted — because each peek erodes its statistical validity. This discipline, enforced in tooling, is the single most important decision of the phase; without it, P2's clean verdict is destroyed within a week of iterating.
- **A research ledger / multiple-testing account is a first-class artifact.** Record every experiment. Report performance **deflated** for the number of trials behind the best result (deflated Sharpe in the spirit of Bailey & López de Prado, or at minimum a visible trials-behind-best count). A leaderboard-topping model must be judged against *how hard you searched* — this is the defining defense against the phase's central failure mode, and it is what separates research from data-dredging.
- **Comparison must be apples-to-apples by construction.** Same panel version, same CV fold definitions, same evaluation code for every candidate. Any score difference must come from the model, not the harness. Feature-set hash and fold seeds are pinned per study and logged with results.
- **Explainability gates promotion; it is not decoration.** A model is promotable only if its SHAP drivers are economically plausible *and* stable across folds and time. A model that is accurate but leans overwhelmingly on a single suspicious feature is a leakage suspect, not a champion — the explanation step is where you catch the leak the metrics missed.
- **Promotion criteria are pre-registered.** Champion/challenger thresholds (e.g., challenger must beat champion on out-of-fold Rank ICIR by a stated margin *and* pass robustness *and* pass the plausibility review) are written down **before** the comparison, to forbid post-hoc rationalization of whatever happened to win.
- **Decay is assumed, monitored, and acted upon.** Alpha decays; a champion with no monitoring becomes a silently-wrong model. Track rolling-window IC, watch for feature drift, and define explicit retrain/retire triggers and a re-validation schedule.
- **Reproducibility is operationalized, not aspirational.** DVC versions datasets; the pipeline is deterministic; reproduction is one command; CI re-runs a reference experiment and asserts the metrics land within tolerance. "Reproducible in principle" is worthless the day the data snapshot moves.

### Task Breakdown

| ID | Task | Reuses / builds on | Reference | Done when |
|---|---|---|---|---|
| **T3.1** | **HPO with Optuna** — search space + pruning + persisted studies; objective = out-of-fold Rank ICIR under purged CV, never the lockbox | training harness (T2.1) | Optuna; MLflow integration | A study resumes and improves OOF ICIR with no leakage |
| **T3.2** | **Model zoo behind a common interface** — XGBoost / CatBoost now, sequence models later; uniform train/predict/evaluate | T2 harness + eval | Qlib model zoo | Adding a model is a config entry, not a rewrite |
| **T3.3** | **Automated report + leaderboard** — every run emits the P2 report; results aggregate into a sortable comparison on identical folds/features | eval module (T2.3), tracking (T2.7) | — | Leaderboard ranks all candidates on identical metrics/folds |
| **T3.4** | **Comparison & selection protocol** — significance tests on per-day IC series; champion/challenger; pre-registered thresholds | T3.3 | paired tests on IC series | Promotion is a rule, not a judgment call |
| **T3.5** | **Research ledger / multiple-testing account** — log trial counts; report deflated metrics / trials-behind-best | tracking (T2.7) | deflated Sharpe; Bailey & López de Prado on backtest overfitting | Every headline result carries its search-effort context |
| **T3.6** | **Explainability suite** — SHAP global + per-name; importance stability across folds/time; economic-sensibility review template | baseline model (T2.2) | SHAP TreeExplainer; ELI5 / permutation as cross-check | Promoted models ship an explanation pack + plausibility sign-off |
| **T3.7** | **Model registry + lineage + model cards** — versioned artifacts, full data→features→label→model→metrics lineage, promotion state | MLflow registry (T2.7) | MLflow Model Registry; model-card pattern | Any champion's full provenance is queryable |
| **T3.8** | **Decay/drift monitoring + scheduled re-validation** — rolling IC, feature drift, retrain/retire triggers; periodic champion re-eval | eval (T2.3), registry (T3.7) | — | A scheduled job flags decay and proposes an action |
| **T3.9** | **Reproducibility hardening** — DVC dataset versioning; pinned env; one-command reproduction; CI metric-stability check | reproducibility manifest (T2.7) | DVC | A year-old result reproduces within tolerance from one command |

### Deliverables

Optuna HPO, the model zoo, the automated report + leaderboard, the selection protocol, the research ledger, the explainability suite, the model registry with lineage and model cards, the decay-monitoring job, and the reproducibility/CI harness.

### Dependencies

All of P2, tightly. Specifically: T2.1 (CV harness) → T3.1 (HPO must live inside it); T2.3 (eval) → T3.3 (leaderboard reuses it verbatim); T2.7 (tracking + manifest) → T3.5 (ledger) and T3.7 (registry) and T3.9 (reproduction). The P2 **lockbox** is a hard dependency for T3.4/T3.5 — the selection protocol and the multiple-testing account are meaningless without a protected test slice to be sparing about.

### Risks & Pitfalls

- **Multiple-comparisons overfitting — the central risk of the phase.** Run 500 configurations, report the best, and you have manufactured alpha that won't survive contact with the future. The research ledger (T3.5), deflated metrics, and sparing lockbox use are the only real defenses; treat them as load-bearing, not paperwork.
- **Lockbox leakage by repeated peeking.** Even well-intentioned per-experiment evaluation on the test set overfits it. Enforce metering in tooling (T3.4), not in personal discipline.
- **HPO leakage.** Tuning on data that improperly touches validation folds inflates everything downstream. The search must live entirely inside the purged CV (T3.1).
- **Apples-to-oranges comparison.** Different folds or feature-set versions across candidates make the leaderboard a fiction. Pin the fold seeds and feature hash per study (T3.3).
- **Explainability theater.** SHAP plots nobody acts on are wasted effort; worse, they create false confidence. The plausibility review must actually gate promotion (T3.6), or it shouldn't exist.
- **A registry without lineage.** A versioned model whose data/feature provenance is unknown can be neither reproduced nor trusted; lineage is the entire point of T3.7.
- **Ignoring decay.** Yesterday's champion is today's liability without monitoring; a model that was real can quietly stop being real (T3.8).

---

## Sequencing & dependencies

```
P2 (render one honest verdict)
  T2.1 (CV harness) ─┬─ T2.2 (LightGBM) ─┬─ T2.3 (eval) ─┬─ T2.4 (baselines + Qlib)
                     │                   │               ├─ T2.5 (cost-aware backtest)
                     │                   │               └─ T2.6 (robustness + null)
                     └─ T2.7 (tracking + manifest) ───────────────────────────┐
                                                                              ▼
                                                          T2.8 (ALPHA VERDICT + lockbox sealed)
                                                                              │
P3 (iterate without corrupting the verdict)                                  ▼
  T2.1 ─▶ T3.1 (Optuna in purged CV) ─┐
  T2.2 ─▶ T3.2 (model zoo) ───────────┼─ T3.3 (leaderboard) ─ T3.4 (selection) ─┐
  T2.3 ─▶ T3.3                        │                                          │
                                      └─ T3.6 (explainability gate) ─────────────┤
  T2.7 ─▶ T3.5 (research ledger) ─ T3.7 (registry + lineage) ─ T3.8 (decay) ─────┤
  T2.7 ─▶ T3.9 (reproducibility + CI) ───────────────────────────────────────────┘
```

**Critical path:** `T2.1 → T2.2 → T2.3 → T2.5 → T2.8 (verdict + lockbox) → T3.1 → T3.4 → T3.7`. The alpha verdict (T2.8) is the gate between the phases: P3 should not begin in earnest until P2 has produced a believed-or-disbelieved verdict and a sealed lockbox, because P3's whole job is to iterate *against that standard* without eroding it.

## Explicitly deferred to P4+ (so this scope stays honest)

Closing the loop into the **prediction/report surface** (the daily inference job that scores the universe and injects calibrated probabilities, risk flags, and SHAP drivers back into the existing HTML report) is **P4** — it depends on a registry-promoted champion (T3.7) and the calibration measured in T2.3, and it is where the old report generator and the new platform finally become one product. **Depth work** — triple-barrier / meta-labeling, sequence and deep models, automated feature mining (tsfresh), and workflow orchestration (Prefect/Dagster) — is **P5**. Pulling either forward is the usual way the modeling core ends up rushed.

---

*One framing note for the review, in the spirit of the P0/P1 document: P2 and P3 are where most people quietly lose the plot, and they lose it in opposite ways. P2 is lost by believing a backtest you optimized; P3 is lost by believing the best of five hundred experiments. The defenses — the sealed lockbox, the cost-aware arbiter, the null tests, the research ledger, the deflated metrics, the pre-registered promotion rules — are unglamorous and they are the entire value of the phase. A platform that can iterate quickly but cannot tell signal from the residue of its own searching is worse than no platform, because it produces confident, traceable, reproducible nonsense. The skill these two phases build is the discipline to keep iteration and honesty growing together.*
