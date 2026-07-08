# SR-07 Strategy Bake-off Results

- decision: **reject**
- baseline_top20_net: 0.027400
- turnover_aware_top20_net: 0.026539
- turnover_aware_turnover_top20: 0.463158
- weak_regime_baseline_top20_net: 0.032916
- weak_regime_turnover_aware_top20_net: 0.029086
- turnover_pass (<= 0.5): True
- net_pass (>= baseline - 0.002): True
- weak_regime_pass: False

## Scoping note

The optional `+confidence` arm from the task doc is not run separately: SR-03 confidence is banner-only metadata and never changes the selected set, so it would be selection-identical to `turnover_aware`.

## Summary Metrics (all dates)

| Metric | Arm | Mean | Std | N | ICIR |
|---|---|---:|---:|---:|---:|
| precision_at_20_excess | baseline | 0.527500 | 0.080255 | 20 | nan |
| precision_at_20_excess | turnover_aware | 0.522500 | 0.083469 | 20 | nan |
| rank_ic_ret_fwd_3d | baseline | 0.137301 | 0.162551 | 20 | 3.777449 |
| rank_ic_ret_fwd_3d | turnover_aware | 0.137301 | 0.162551 | 20 | 3.777449 |
| top20_ret_fwd_3d_net | baseline | 0.027400 | 0.045124 | 20 | nan |
| top20_ret_fwd_3d_net | turnover_aware | 0.026539 | 0.048049 | 20 | nan |
| top50_ret_fwd_3d_net | baseline | 0.019248 | 0.038382 | 15 | nan |
| top50_ret_fwd_3d_net | turnover_aware | 0.019209 | 0.038318 | 15 | nan |
| turnover_top20 | baseline | 0.652632 | nan | 1 | nan |
| turnover_top20 | turnover_aware | 0.463158 | nan | 1 | nan |

## Weak-regime Summary Metrics

| Metric | Arm | Mean | Std | N | ICIR |
|---|---|---:|---:|---:|---:|
| precision_at_20_excess | baseline | 0.520000 | 0.097753 | 10 | nan |
| precision_at_20_excess | turnover_aware | 0.510000 | 0.099443 | 10 | nan |
| rank_ic_ret_fwd_3d | baseline | 0.089696 | 0.176190 | 10 | 1.609881 |
| rank_ic_ret_fwd_3d | turnover_aware | 0.089696 | 0.176190 | 10 | 1.609881 |
| top20_ret_fwd_3d_net | baseline | 0.032916 | 0.041559 | 10 | nan |
| top20_ret_fwd_3d_net | turnover_aware | 0.029086 | 0.044641 | 10 | nan |
| top50_ret_fwd_3d_net | baseline | 0.021699 | 0.033744 | 9 | nan |
| top50_ret_fwd_3d_net | turnover_aware | 0.021634 | 0.033623 | 9 | nan |

## Date Runs

| Date | Rows | Label Rows | Base Features | Recent Features |
|---|---:|---:|---:|---:|
| 2026-04-02 | 300 | 300 | 47 | 47 |
| 2026-04-08 | 300 | 300 | 47 | 47 |
| 2026-04-13 | 300 | 300 | 47 | 47 |
| 2026-04-16 | 300 | 300 | 47 | 47 |
| 2026-04-21 | 299 | 299 | 47 | 52 |
| 2026-04-24 | 299 | 299 | 47 | 52 |
| 2026-04-29 | 299 | 299 | 47 | 52 |
| 2026-05-07 | 300 | 300 | 47 | 52 |
| 2026-05-12 | 300 | 300 | 47 | 52 |
| 2026-05-15 | 300 | 300 | 47 | 52 |
| 2026-05-20 | 300 | 300 | 47 | 52 |
| 2026-05-25 | 300 | 300 | 47 | 52 |
| 2026-05-28 | 300 | 300 | 47 | 52 |
| 2026-06-02 | 300 | 300 | 47 | 52 |
| 2026-06-05 | 300 | 300 | 47 | 52 |
| 2026-06-10 | 300 | 300 | 47 | 52 |
| 2026-06-15 | 300 | 299 | 47 | 52 |
| 2026-06-18 | 300 | 299 | 47 | 52 |
| 2026-06-24 | 285 | 284 | 46 | 42 |
| 2026-06-29 | 298 | 298 | 35 | 40 |