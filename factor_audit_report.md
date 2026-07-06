# Factor Audit Report

审计日期：2026-07-06  
项目根目录：`E:\stock-analysis`  
审计范围：当前代码、文档、feature/label/silver parquet 产物；未修改业务代码，未新增审计脚本。

## 1. 总结结论

当前项目已经具备一套以 CSI300 为主的短线横截面预测框架，因子层主要由技术因子、动态横截面 rank/zscore、估值/规模、行业相对、资金流、融资融券、基本面、解禁事件几组组成。可稳定进入训练或预测的核心因子目前是：

- 技术因子：23 个已落盘，覆盖率高，未来函数风险低。
- 动态横截面因子：8 个在 `build_panel()` 阶段计算，覆盖率高，但 `cs_rank_close` / `cs_zscore_close` 用原始价格，经济含义弱。
- 估值/规模因子：6 个可用，覆盖率高；其中 `cs_pe_ttm_rank` 因负 PE 被置空，缺失率约 7.23%。
- 融资融券因子：2 个可用，覆盖约 92%，实现了 1 日滞后，适合 D+3/D+5 多于 D+1。
- 行业相对因子：代码可用，行业 SCD 覆盖 300 只；但其质量依赖上游字段。当前预测脚本中 `ind_rank_rsi_6`、`sector_momentum_10d` 可用，`ind_rank_turnover`、`ind_rank_main_flow` 为全缺失。

当前不可稳定使用或不应进入主模型的因子：

- 资金流因子：代码存在，但本地 `silver/fund_flow` 只有 6 个股票、720 行，进入全市场训练时缺失率约 99.72%，当前不可用。
- 解禁事件因子：代码存在，但 `silver/lockup` 无 parquet 文件，且主 `model` 命令没有 include-lockup 路径，当前不可用。
- 基本面因子：代码 PIT join 正确，但本地只有 79 个 symbol 文件、168 行，正式指标缺失约 77% 以上，且默认未进入训练，当前不适合短线主模型。
- `reversal_3d`：代码和当前预测脚本使用它，但最新落盘 feature set `d02a4ebf` 没有该列；预测脚本是临时从 OHLCV 重新计算加入。它短线相关性设计合理，但需要重新正式构建 feature set，避免训练/预测特征口径分叉。

对 D+1 / D+3 / D+5 目标的总体判断：

- D+1：最适合 `reversal_3d`、RSI/KDJ/Stoch、量能横截面、资金流 1 日因子。但资金流当前覆盖不足，所以 D+1 信息源偏少。
- D+3：当前最适合的目标。`ret_fwd_3d` 已追加到 300 只标签文件，`next_week_d3_prediction.py` 已改为训练 `ret_fwd_3d`，技术反转 + 行业相对动量可以支撑这个周期。
- D+5：平台默认主标签仍是 `ret_fwd_5d`，样本覆盖好，walk-forward 工具链成熟，适合作为稳健默认。若 D+3 不能在 walk-forward 上同时超过 D+5 的 Rank IC 与 ICIR，应继续保留 D+5 为平台主目标。

## 2. 代码定位

主要因子定义与计算位置：

| 因子族 | 规格定义 | 计算位置 | 进入主流程方式 |
|---|---|---|---|
| 技术因子 | `quant_platform/features/registry.py` 的 `TECHNICAL_SPECS` | `quant_platform/features/technical.py::build_technical_features` | `FeaturePipeline.run()` 落盘 |
| 横截面因子 | `CROSS_SECTIONAL_SPECS` | `quant_platform/features/cross_sectional.py::build_cross_sectional_features` | `FeaturePipeline.build_panel()` 动态生成 |
| 估值/规模 | `VALUATION_SPECS` | `quant_platform/features/valuation.py::build_valuation_features` | `--include-valuation` |
| 行业相对 | `INDUSTRY_SPECS` | `quant_platform/features/industry.py::build_industry_features` | `--include-industry` 或预测脚本手动调用 |
| 资金流 | `FLOW_SPECS` | `quant_platform/features/flow.py::build_flow_features` | `FeaturePipeline` 支持，但 CLI `model` 未暴露 include-flow |
| 融资融券 | `MARGIN_SPECS` | `quant_platform/features/margin.py::build_margin_features` | `--include-margin` |
| 基本面 | 无 registry specs，固定 `fund_*` 列 | `quant_platform/features/fundamental.py::build_fundamental_features` | `--include-fundamentals`，默认关闭 |
| 解禁事件 | `LOCKUP_SPECS` | `quant_platform/features/event.py::build_lockup_features` | `features --include-lockup` 只加到内存 panel，`model` 无对应参数 |

标签与目标：

- 默认标签：`quant_platform/labels/builder.py` 中 `PRIMARY_LABEL_COL = ret_fwd_5d`。
- 默认 horizons：`[1, 5, 10, 20]`。
- `ret_fwd_3d`：通过 `append_horizon_labels()` 追加，不在默认 horizons 中。
- D+3 预测脚本：`scripts/next_week_d3_prediction.py` 当前训练 `ret_fwd_3d`。
- D+3 vs D+5 比较：`scripts/compare_label_horizons.py` 要求 3d 同时在 Rank IC 与 ICIR 上超过 5d 才切换，否则保留 5d。

## 3. 当前数据与覆盖

本地数据快照：

| 数据源 | 文件/符号覆盖 | 日期范围 | 主要缺失情况 | 结论 |
|---|---:|---|---|---|
| OHLCV | 301 文件，301 符号 | 2023-01-03 至 2026-07-03 | 核心 OHLCV 0 缺失；`amount/outstanding_share/turnover` 约 0.53% 缺失 | 可用 |
| 特征集 `1474f2c4` | 300 文件，248,457 行 | 至 2026-06-18 | 23 个技术因子，无 `close/volume/reversal_3d` | 历史技术集 |
| 特征集 `d02a4ebf` | 300 文件，249,797 行 | 至 2026-06-29 | 23 个技术因子 + `close/volume`；无 `reversal_3d` | 当前主要基础集 |
| 估值 | 300 文件，249,033 行 | 至 2026-06-29 | `turnover_pct` 约 0.11% 源缺失；panel 对齐后约 0.53% | 可用 |
| 行业 SCD | 300 符号 | SCD 表 | 静态 fallback 只补当前分类，历史完整性依赖 SCD | 可用但需 PIT 纪律 |
| 资金流 | 6 文件，720 行 | 2025-12-23 至 2026-06-25 | 全 panel 因子缺失约 99.72% | 不可用 |
| 融资融券 | 295 文件，231,625 行 | 至 2026-06-24 | panel 因子缺失约 7.5%-8.1% | 可用 |
| 解禁 | 0 文件 | 无 | 无数据 | 不可用 |
| 基本面 | 79 文件，168 行 | 非日频 | 正式财务字段缺失约 77%+ | 不适合当前主模型 |
| 标签 | 300 符号，251,372 行 | 至 2026-07-03 | `ret_fwd_1d/3d/5d` 缺失分别约 0.24%/0.48%/0.72% | 可用 |

动态 panel 统计：

| 因子 | 缺失率 | 备注 |
|---|---:|---|
| `cs_rank_close`, `cs_zscore_close` | 0.00% | 可计算，但原始价格横截面意义弱 |
| `cs_rank_volume`, `cs_zscore_volume` | 0.00% | 可用 |
| `cs_rank_rsi_6`, `cs_zscore_rsi_6` | 0.72% | 技术 warmup 导致 |
| `cs_rank_roc_10` | 1.20% | 技术 warmup 导致 |
| `cs_rank_ma_5` | 0.60% | 技术 warmup 导致 |
| 估值 rank/size 大部分 | 0.53% | 可用 |
| `cs_pe_ttm_rank` | 7.23% | 负 PE 置 NaN 后排名 |
| 资金流 5 个因子 | 99.72% | 不应入模 |
| 融资融券 2 个因子 | 7.50%-8.09% | 可用 |

当前 D+3 预测产物 `D3_Prediction_2026-07-03_ranked.csv`：

- 298 只股票被打分。
- `reversal_3d`、`ind_rank_rsi_6`、`sector_momentum_10d`：0 缺失。
- `ind_rank_turnover`、`ind_rank_main_flow`：100% 缺失。

## 4. 因子逐族审计

### 4.1 技术因子

现有因子：`ma_5/10/20/60`、`macd_dif/dea/hist`、`kdj_k/d/j`、`rsi_6/12/24`、`boll_upper/boll_mid`、`atr_14`、`adx_14`、`obv`、`cci_14`、`roc_10`、`willr_14`、`stoch_k/d`、`reversal_3d`。

来源与依赖：OHLCV，主要依赖 `close/high/low/volume`。

可用性：

- 已落盘可用：除 `reversal_3d` 外的 23 个技术因子。
- `reversal_3d`：代码已定义并在预测脚本中临时计算，但当前 `d02a4ebf` feature parquet 无此列，属于“代码可用、产物未正式重建”。

未来函数风险：

- 主体使用 rolling / EMA / pct_change / TA 指标，均为 T 日及以前。
- `pandas_ta_classic` 返回值通过原 index reindex，代码注释明确避免 future indicator 对齐到早期日期。
- warmup rows 被置 NaN，避免早期不稳定值。
- 未发现显性未来函数。

缺失与覆盖：

- 技术因子缺失主要来自 warmup，`ma_60` 最高约 7.21%，其余大多低于 4.2%。
- 覆盖 300 只 CSI300，样本足够。

短线适配：

- D+1：`reversal_3d`、`rsi_6`、KDJ/Stoch、`roc_10`、`obv` 更适合。
- D+3：`reversal_3d`、`rsi_6/12`、`roc_10`、`ma_5/10`、`sector_momentum_10d` 类组合适合。
- D+5：`ma_20`、`boll_*`、`macd_*`、`adx_14` 可以作为慢一些的状态变量。
- `ma_60`、`adx_14` 更偏中周期状态，D+1 价值有限。

### 4.2 横截面因子

现有因子：`cs_rank_close`、`cs_rank_volume`、`cs_rank_rsi_6`、`cs_rank_roc_10`、`cs_rank_ma_5`、`cs_zscore_close`、`cs_zscore_volume`、`cs_zscore_rsi_6`。

来源与依赖：当前 feature panel 同日全市场截面；依赖 `close/volume/rsi_6/roc_10/ma_5`。

可用性：

- `build_panel()` 动态生成，不落盘在 per-symbol feature parquet。
- 当前 `d02a4ebf` registry specs 包含 8 个横截面因子。
- `1474f2c4` registry 只有 6 个，未包含 `cs_rank_close/cs_zscore_close`。

未来函数风险：

- 仅使用同日截面，无时间上未来函数。
- 风险点是 universe 是否 PIT。代码注释强调 PIT universe，但本地 `universe/csi300/membership.parquet` 当前是 300 当前成员；若历史成员没有完整有效日期，将存在幸存者偏差，而非单列未来函数。

缺失与覆盖：

- `close/volume` 截面因子 0 缺失。
- 技术派生截面因子缺失随 warmup 约 0.6%-1.2%。

短线适配：

- `cs_rank_volume`、`cs_zscore_volume` 对 D+1/D+3 有短线拥挤度/活跃度信息。
- `cs_rank_rsi_6`、`cs_rank_roc_10` 适合 D+1/D+3。
- `cs_rank_close`、`cs_zscore_close` 不建议作为有效 alpha：原始股价水平不是横截面强信号，且容易引入面值/拆分/市场偏好噪声。

### 4.3 估值/规模因子

现有因子：`cs_log_float_mcap`、`cs_pe_ttm_rank`、`cs_pb_rank`、`cs_turnover_rank`、`cs_log_mcap_rank`、`pe_momentum_5d`。

来源与依赖：`silver/valuation`，字段包括 `pe_ttm/pb/total_mcap_yi/float_mcap_yi/turnover_pct`。

可用性：

- 数据 300 符号完整，panel 对齐后大部分缺失约 0.53%。
- CLI `model` 支持 `--include-valuation`。
- 当前预测脚本未系统纳入估值，主要使用现有 feature parquet + 行业技术因子。

未来函数风险：

- 估值来自 T 日收盘估值，作为收盘后可知的 T 日特征，预测 T+1 之后收益，基本合理。
- `cs_log_float_mcap` 实现先按日期 rank，随后又用全样本 min/max 重写成全局 min-max，这会在严格训练/验证分割下造成跨期归一化泄漏风险。该风险不是未来价格泄漏，但属于 transform 使用全样本信息。建议后续改成同日截面 rank 或在训练 pipeline 内 fit/transform。

缺失与覆盖：

- `cs_pe_ttm_rank` 缺失约 7.23%，主要由负 PE 被置 NaN。
- 其他估值/规模因子缺失约 0.53%。

短线适配：

- D+1：估值/规模信号较慢，直接预测力有限。
- D+3/D+5：规模、换手、PE/PB 可作为风险暴露和风格控制，更适合作为辅助因子。
- `cs_turnover_rank` 对短线更有意义，但需要确认与 OHLCV `turnover` 字段口径一致。

### 4.4 行业相对因子

现有因子：`ind_rank_rsi_6`、`ind_rank_turnover`、`ind_rank_main_flow`、`sector_momentum_10d`。

来源与依赖：`silver/industry_map.parquet` 的 SCD 行业映射；依赖技术、估值换手、资金流等上游列。

可用性：

- 行业映射覆盖 300 符号。
- `ind_rank_rsi_6`、`sector_momentum_10d` 在当前预测日可用。
- `ind_rank_turnover` 只有在 valuation 的 `turnover_pct` 已进入 panel 时才可用；当前预测脚本未引入 valuation，因此为全缺失。
- `ind_rank_main_flow` 依赖资金流，当前资金流覆盖不足，因此全缺失。

未来函数风险：

- `_join_industry()` 使用 `effective_date <= T` 且 `out_date > T`，设计上 PIT 正确。
- 风险来自源数据：如果 industry_map 中部分记录是当前静态 fallback，则历史行业分类会有前视污染。文档已说明存在少量 fallback。

缺失与覆盖：

- 当前 D+3 ranked 文件：`ind_rank_rsi_6`、`sector_momentum_10d` 0 缺失；`ind_rank_turnover`、`ind_rank_main_flow` 100% 缺失。

短线适配：

- D+1/D+3：`ind_rank_rsi_6` 和 `sector_momentum_10d` 适合短线行业轮动/行业内强弱排序。
- D+5：同样适合，并可能比 D+1 稳定。
- 建议训练时显式剔除全缺失行业因子，避免不同脚本靠 NaN 阈值各自处理。

### 4.5 资金流因子

现有因子：`cs_main_flow_rank_1d`、`cs_main_flow_rank_5d`、`cs_small_flow_rank_1d`、`cs_super_flow_rank_1d`、`cs_flow_reversal_5d`。

来源与依赖：`silver/fund_flow`，字段 `main_net/small_net/super_net`，可用估值 `float_mcap_yi` 标准化。

可用性：

- 代码已实现，registry 的 `FULL_SPECS` 包含 flow。
- 本地数据只有 6 个符号、720 行；全 panel 因子缺失约 99.72%。
- CLI `features` 支持 `--include-flow`，但 CLI `model` 参数当前未暴露 `--include-flow`，因此即便 feature 阶段加过，常规模型命令也不会自动重建 flow panel。

未来函数风险：

- 代码注释称 T 日资金流在收盘后发布，特征用 T 日流向预测 T+1 后收益，设计上可接受。
- 若实际数据发布时间晚于收盘后或有修订，需要进一步确认源端发布时间；当前没有强制滞后。

缺失与覆盖：

- 覆盖不足是决定性问题。

短线适配：

- 理论上最适合 D+1/D+3，尤其 `cs_main_flow_rank_1d`。
- 但当前数据状态下不能使用；先补齐覆盖，再做单因子 IC。

### 4.6 融资融券因子

现有因子：`cs_margin_balance_change_5d`、`cs_rzrq_ratio_rank`。

来源与依赖：`silver/margin`，字段 `rzye/rzrqye`，可用 `float_mcap_yi` 标准化。

可用性：

- 295 个符号、231,625 行，覆盖较好。
- CLI `model` 支持 `--include-margin`。
- 缺失约 7.50%-8.09%，可接受。

未来函数风险：

- 实现将 margin 数据做 1 日滞后：T 日特征使用 T-1 可知数据，PIT 纪律较好。
- 注意当前 lag 使用每个符号下一条 margin 日期作为 join date，若某符号停牌或缺交易日，仍是“下一可用日”对齐，整体保守。

短线适配：

- D+1：可能偏慢，日频公布滞后会削弱即时性。
- D+3/D+5：较适合作为杠杆情绪和拥挤度因子。

### 4.7 基本面因子

现有因子：`fund_revenue`、`fund_net_profit`、`fund_eps`、`fund_roe`、`fund_lag_days` 以及 period/announce 元数据。

来源与依赖：`silver/fundamentals`。

可用性：

- 当前只有 79 个文件、168 行。
- 正式指标 `eps/revenue/net_profit` 等缺失约 77%+。
- 默认关闭，且 `_build_feature_cols()` 会排除字符串元数据，仅数值列可能入模。

未来函数风险：

- 使用 `announce_date <= T` as-of join，而不是 `period_end`，设计正确。
- 已过滤 forecast-only 行不覆盖正式财报指标。

短线适配：

- D+1/D+3/D+5 直接信号弱，更多是质量/风险暴露。
- 当前覆盖太低，不建议进入短线主模型。

### 4.8 解禁事件因子

现有因子：`days_to_next_unlock`、`unlock_size_ratio`。

来源与依赖：`silver/lockup`，可选依赖估值 float market cap。

可用性：

- 代码已实现。
- 本地 `silver/lockup` 0 文件。
- `features --include-lockup` 只在内存 panel 加，不会落盘到 feature set；`model` 命令没有 include-lockup 参数。
- `FULL_SPECS` 当前没有包含 `LOCKUP_SPECS`。

未来函数风险：

- 代码只使用 `unlock_date > T` 的严格未来已公告事件，并保守排除 `unlock_date == T`，设计合理。
- 但需确认 collector 是否只保存已公开公告的未来解禁，不应从事后全量日历补入过去不可知事件。

短线适配：

- 若数据补齐，非常适合 D+3/D+5，D+1 只对临近解禁事件有用。
- 当前不可用。

## 5. 标签与目标适配

标签实现采用：

```text
ret_fwd_{h}d(T) = close(T+1+h) / close(T+1) - 1
```

这符合 T+1 执行假设，特征 T 日收盘后形成，最早从 T+1 建仓，避免把 T 日收盘价格作为可交易收益的一部分。

当前标签覆盖：

| 标签 | 缺失率 | 结论 |
|---|---:|---|
| `ret_fwd_1d` | 0.24% | 可用 |
| `ret_fwd_3d` | 0.48% | 已追加，可用 |
| `ret_fwd_5d` | 0.72% | 默认主标签，可用 |
| `excess_vs_csi300_5d` | 1.17% | 可用，但 index 是等权 proxy 还是实盘指数需继续标记 |

注意事项：

- per-symbol label parquet 中 `ret_fwd_*_cs` 和 `ret_fwd_*_bin` 是占位列，当前文件中为全 NaN；只有 `build_label_panel()` 动态计算后才有意义。
- `ret_fwd_3d` 不在默认 horizons 中，后续正式 pipeline 如需 D+3 应把 `[1, 3, 5, 10, 20]` 作为显式参数，避免重建标签时丢失 3d。

## 6. 未来函数与偏差风险清单

高优先级风险：

1. `cs_log_float_mcap` 使用全样本 min/max，严格 OOS 下存在跨期归一化信息泄漏风险。
2. 当前 universe 可能是当前 CSI300 成分为主；如果历史 membership 没有真实 effective/out 日期，会有幸存者偏差。
3. `reversal_3d` 在预测脚本中临时计算，但未正式落盘到 `d02a4ebf`，存在训练/预测口径不一致风险。
4. `features --include-lockup` 生成的 lockup 列不会进入常规 `model`，容易让人误以为解禁因子已在模型中。
5. `flow` 在数据覆盖极低时若被纳入，会被 imputer 或模型缺失处理吞掉，产生不可解释噪声。

中优先级风险：

- `cs_rank_close/cs_zscore_close` 虽然无未来函数，但信号经济含义弱。
- 行业 SCD 的静态 fallback 对历史行业有前视污染可能。
- 估值数据使用 T 日收盘值预测 T+1 后收益是合理的，但若真实获取时间晚于收盘批处理时间，需要加入可用时间约束。
- `vol_fwd_1d` 全 NaN 是公式自然结果，不能作为风险标签使用；1 日 horizon 下 realized vol 没有足够日收益点。

## 7. 短线目标建议

### D+1

建议可用因子：

- `reversal_3d`
- `rsi_6`、KDJ、Stoch、`roc_10`
- `cs_rank_volume` / `cs_zscore_volume`
- `ind_rank_rsi_6`
- 资金流 1 日因子仅在补齐覆盖后使用

不建议：

- 基本面、PE/PB 慢变量作为主信号。
- 融资融券 5 日变化作为 D+1 主信号。

### D+3

当前最值得推进：

- `reversal_3d`
- `ma_5/ma_10`、`roc_10`、`rsi_6/12`
- `cs_rank_rsi_6`、`cs_rank_roc_10`、`cs_rank_volume`
- `ind_rank_rsi_6`、`sector_momentum_10d`
- 融资融券因子作为辅助

前提：

- 正式重建包含 `reversal_3d` 的 feature set。
- 用 `scripts/compare_label_horizons.py` 做 D+3 vs D+5 同模型 walk-forward 比较。

### D+5

保留为平台默认主目标是合理的：

- 样本覆盖好。
- 现有 walk-forward、alpha verdict、Ridge baseline 都围绕 5d 更成熟。
- 技术、估值、行业、融资融券的信号半衰期更容易覆盖 5 日。

若 D+3 未明确胜出，建议继续用 D+5 作为主训练标签，同时输出 D+3 作为预测/监控分支。

## 8. 建议优先级

P0，先修口径：

1. 正式重建包含 `reversal_3d` 的 feature set，不再在预测脚本中临时补列。
2. 把当前训练/预测的最终 feature list 固化输出，并标记每列来源、缺失率、是否动态生成。
3. 对 `cs_rank_close/cs_zscore_close` 做一次单因子 IC，若弱则从默认训练列剔除。

P1，补数据：

1. 补齐 `fund_flow` 至 300 符号，否则 flow 因子保持禁用。
2. 若要使用解禁因子，先补 `silver/lockup`，再把 `LOCKUP_SPECS` 纳入 registry 和 `model` include 参数。
3. 若要行业换手因子可用，预测脚本 build panel 时需要同时引入 valuation，或改为使用 OHLCV 的 `turnover` 口径。

P2，防泄漏：

1. 修正 `cs_log_float_mcap` 的全样本 min/max，改为同日截面 rank 或训练内 scaler。
2. 审计 universe membership 是否真正 PIT；若不是，在报告和模型评估中明确标注幸存者偏差。
3. 对行业 fallback 记录在历史日期上的占比做统计，必要时限制行业因子使用区间。

P3，短线验证：

1. 固定同一模型、同一特征，用 `ret_fwd_3d` vs `ret_fwd_5d` 做 walk-forward 比较。
2. 对每个因子族跑单因子 IC：D+1、D+3、D+5 三个 horizon 分开看。
3. 对 D+3 预测继续执行真实滚动验证，不把单周结果当作结论。

## 9. 审计判定表

| 因子族 | 当前可用 | 未来函数风险 | 缺失率风险 | 样本覆盖风险 | D+1 | D+3 | D+5 |
|---|---|---|---|---|---|---|---|
| 技术因子 | 是 | 低 | 低 | 低 | 高 | 高 | 中高 |
| `reversal_3d` | 半可用 | 低 | 低 | 低 | 高 | 高 | 中 |
| 横截面技术/量能 | 是 | 低 | 低 | 中：需 PIT universe | 高 | 高 | 中 |
| 原始价格横截面 | 可计算但不推荐 | 低 | 低 | 中 | 低 | 低 | 低 |
| 估值/规模 | 是 | 中：全样本 minmax | 低-中 | 低 | 低 | 中 | 中 |
| 行业 RSI/动量 | 是 | 低-中：fallback 风险 | 低 | 中 | 中 | 高 | 高 |
| 行业换手/资金流 | 当前不可用 | 低 | 高 | 高 | 低 | 低 | 低 |
| 资金流 | 否 | 中：需确认发布时间 | 高 | 高 | 理论高 | 理论高 | 中 |
| 融资融券 | 是 | 低 | 中 | 中 | 中 | 中高 | 高 |
| 基本面 | 否 | 低 | 高 | 高 | 低 | 低 | 低 |
| 解禁 | 否 | 低 | 高 | 高 | 中 | 高 | 高 |

## 10. 最终结论

当前最可信的短线因子底座是“技术因子 + 横截面技术/量能 + 行业内 RSI/板块动量 + 融资融券辅助”。D+3 是当前最值得继续验证的短线目标，但平台默认 D+5 仍更稳健，不能只因单个 D+3 预测脚本存在就切换主目标。

在进入下一轮模型迭代前，最重要的不是增加更多模型，而是把 `reversal_3d` 正式落盘、剔除全缺失/伪可用因子、修正全样本归一化风险，并把 D+1/D+3/D+5 的单因子 IC 表固化为每次训练前的审计门槛。
