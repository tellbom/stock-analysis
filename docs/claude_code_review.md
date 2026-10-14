# Claude代码修复与裁决任务

## 当前任务：10月14日历史复权缺陷修复

请首先阅读 **`docs/reviews/2026-10-14-weekly/top_picks_audit.md`**，重点为“交付Claude：代码修复问题与验收”。审阅基准`70119e8`包含Claude的`bae124e`和本机必要修正。

10月14日确实重新训练6个模型，普通/混合基线和输出复算一致；但本机随后确认历史复权拼接错误，**该批次推荐已撤回，尚未完成数据修复或重训**。优先修数据入口、修订传播与推荐前质量校验，再评估真实下跌股为何因反弹信号入选。

### 云端与本机分工

- Claude在云端负责代码修复、fixture测试、方案裁决、commit和push。
- Codex在用户本机拉取修复代码，执行全池数据核验/修复、特征与成熟标签重建、同口径重训和新Excel交付。
- 云端没有`E:`盘、完整数据湖和6个模型二进制；不要声称云端已完成全池修复、完整训练或收益改善。

### 可复现证据

指南针旧9月18日价格82.00，完整新前复权56.55，近期尾部又相同；新913行、旧908行，共有日期中896行差异、最后12行一致，旧文件还缺6月22至26日五个价格日。固定旧Ridge、仅改输入并重算横截面，D3排名1→174、D5排名1→156。该排名是问题回放结果，不是新模型需硬达到的目标。

云端证据已提交`docs/reviews/2026-10-14-weekly/qfq_audit/`。仓库根目录运行：

```bash
python scripts/replay_qfq_audit_fixture.py
python scripts/replay_qfq_audit_fixture.py --rebuild-technical
```

第一条仅需NumPy/Pandas，第二条需项目技术指标依赖。无需联网或本机数据湖。详细问题的文件/行号、验收场景和本机待执行要求在主报告。

### 交付要求

列出已验证缺陷的触发条件、影响、修复方法和回归测试。处理除权在尾部窗口外、完整重取失败、缓存失效、标签成熟重算和数据质量阻断；保留原推荐证据。保持普通/混合基线、日期、池子和约束一致；风险过滤/去重复因子等策略改造需另给公平实验方案，不因当前跌股临时调参。提交并推送修复代码、测试结果和本机执行命令，由本机接续数据修复及重训。

## 历史任务记录（以下非当前数据修复完成声明）

以下保存前次不同批次的交接背景；其中“已完成训练”“未运行资金流训练实验”等描述只适用于对应历史阶段，不能覆盖上面的最新撤回状态。其他批次入口为`docs/original_hybrid_comparison.md`、`docs/batch_time_contract.md`、`docs/reviews/2026-10-14/d3_backtest_status.md`。本次请求不包含全历史回测。

10月14日已完成本机数据更新和新批次训练，请先阅读 `docs/weekly_training_2026-10-14.md`。本轮追加修正了两个入口近期模型仍不一致的问题；当前默认周入口执行完整的Ridge32+近期32/37对照流程。有流主组与周策略导出已逐项核对一致，模型和完整数据湖保留本机。新信号14日，15日收盘入场，D3/D5到期分别20/22日；原9月30日与10月9日报告属于旧批次。

最新增加的同基线受控实验请先看 `docs/original_hybrid_comparison.md`。该实验固定9月30日原版Ridge32预测，比较近期无资金流、近期有资金流和分数Gate；收益只核对10月8日至13日。证据位于 `docs/reviews/2026-09-30-controlled/`。这是事后重建实验，不是9月30日实际交付过的混合推荐，也不是10月9日35/40列LightGBM混合的原样复现。

用户随后明确其实际使用的是9月30日原表、计划10月8日入场。请先阅读 `docs/batch_time_contract.md`。下文10月9日普通/混合描述属于另一批次，不能用于评价原表持仓。原表D3于10月13日到期；后续批次D3于10月15日到期。

请先阅读 `docs/reviews/2026-10-14/d3_backtest_status.md`。目标是判断普通D+3与长期/近期资金流RRF混合是否值得继续采用，并找出可验证的代码缺陷。

## 已知事实

- 普通是2026-10-09已经交付的Ridge50只，混合是同日Gate+RRF43只。
- 目标必须为 `ret_fwd_3d`，T+1收盘至之后第3个交易日收盘；入场10/12，到期10/15。
- 报告生成日10/14，检查截至10/13。本地训练价格缓存到10/9；日线已另外取到10/13用于收益核对。不要从名单差异或重要性推断收益优劣，也不能把新取观察价格回填到原推荐训练。
- 普通源JSON原在被忽略的temp，本仓库固定放在 `docs/reviews/2026-10-14/inputs/ordinary_20261009.json`；两份原Excel均已逐行核对。
- `temp/latest_dual_horizon.py`保留临时Ridge导出逻辑，其中默认日期仍为9/30。10/9输出当时通过导入后覆盖AS_OF/OUT运行；审阅应以冻结JSON及已核实的原Excel（Excel在本地交付包）为名单依据，不能把该文件默认日期当作10/9的运行日志。
- 长期/近期/无分数Gate对照为事后诊断，不是额外交付过的策略。
- 本次只处理最近一周；任何更长历史研究需另行说明，不能改写为本次已完成回测。

## 优先审阅

1. `scripts/run_dual_horizon_rrf.py` 的训练时间和标签成熟约束，确认没有信号之后的价格进入训练。
2. `scripts/run_d3_gate_once.py` 的label_col传递，以及其他入口是否仍从已经带有未来收益的存储标签训练。
3. `quant_platform/selection/gate_fusion.py` 和 `fusion_score.py`：近期模型既影响可买资格又影响RRF，默认门槛与k=60是否有相应验证；区分风险Gate和分数Gate。
4. `quant_platform/features/flow.py` 的资金流单位、方向、归一化、可观察时间和短历史覆盖。树分裂次数不是增量收益证据。
5. `quant_platform/features/pipeline.py` 的严格失败、原子写入，以及同一feature_set_id对不同include_*组合是否可能复用不同列的缓存。
6. `scripts/reconcile_d3_recommendations.py` 的等本金、缺价处理、指数匹配、交易成本、时间边界和事后对照标识。检查股票缺交易日时按行shift的标签是否符合交易所D+3日期。
7. 固定股票池/行业分类的历史可观察性、存活者偏差，以及当前标准化使用价格/成交量绝对水平是否影响跨股票泛化。

## 要求输出

先给可复现的缺陷，每项注明文件与行号、触发条件、影响和修复建议。把“已验证问题”“待验证假设”“当前数据不足”分开。

在相同近期窗口和模型参数下，有/无资金流才是资金流增量对照；本仓库没有运行该训练实验。D5的参数复用也不代表D5独立验证完成。

提出改造前先给公平比较方案：固定信号日、入场/退出、股票数、行业约束和成本；完整D3收益到期后再评价。不要改变标签去适配已有价格，不要用缺价股票剔除后的均值冒充原组合收益。

## Git上的审阅入口

核心入口为 `scripts/run_dual_horizon_rrf.py`、`scripts/reconcile_d3_recommendations.py`。普通Ridge历史导出和因子审计脚本分别为 `temp/latest_dual_horizon.py`、`temp/audit_dual_horizon_factor_lists.py`，已显式纳入Git，供比较原版逻辑。其他temp脚本主要为临时探测和制品处理。

复算当前缺价状态：

```powershell
python scripts/reconcile_d3_recommendations.py --ordinary-json docs/reviews/2026-10-14/inputs/ordinary_20261009.json --fusion-dir docs/reviews/2026-10-14/inputs/fusion_20261009 --as-of 2026-10-13 --prices-csv docs/reviews/2026-10-14/prices.csv --out-dir temp/claude_d3_reconciliation
```

使用项目Python依赖环境和已有calendar/industry_map/membership辅助表。当前快照只有10/8和10/9价格，因此输出收益不可计算；不代表零收益。
