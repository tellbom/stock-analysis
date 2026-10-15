# fa77d31风险审计修复：本机验收通过

日期：2026-10-15上午。核查代码：`fa77d31f611b9d2075e14f4f7ae53c60b6a0221a`，包含Claude对R1/R2/R3的修复。

**结论：本次约定的P0风险可观测性修复验收通过。** 完整pytest通过；R1风险合并三组复现、R2两个周期三个出口的8字段一致性、R3最小日期有效性标记均通过。实际冻结股票名单和排序分数保持一致。本次验收没有增加止跌过滤或重新训练模型，不能当作选股准确性改善的证据。

## 一、完整pytest

```bash
python -m pytest tests/ -q --junitxml=temp/risk_audit_fa77d31_pytest.xml
```

| 项目 | 本机实测 |
| :-- | :-- |
| 测试通过 | **468** |
| 失败 / 错误 / 跳过 | **0 / 0 / 0** |
| 警告 | 97条 |
| 耗时 | 288.04秒 |
| Python | 3.13.0，使用现有sklearn/LightGBM等本机依赖 |
| `test_risk_audit.py`实际收集 | 22项，全部通过 |

上一版风险审计10项，本版实际新增12项；上游回复的“新增13个用例、27项逻辑检查”是不同口径，本机以pytest/JUnit实际收集为准。未删减或跳过用例以获得通过。

## 二、R1三组最小复现

每组均调用真实`combine -> gate_first_fusion -> actionable_pool`，以两个模型均为高分的最小样本核查风险合并的后果。

| 场景 | 合并后risk_flags | Gate结果 | 诊断可选数量 |
| :-- | :-- | :-- | --: |
| base=UNKNOWN，recent=有效否决码 | `extreme_outflow@2026-10-14` | RISK_VETO | 0 |
| 两路分别为coverage及否决码 | `coverage@2026-10-14;extreme_outflow@2026-10-14` | RISK_VETO | 0 |
| 两路有重复否决码和不同有效代码 | 并集合并、重复否决码仅保留一次 | RISK_VETO | 0 |

额外检查了错位索引：按primary索引对齐，匹配位置保留secondary有效代码，缺失位置保持UNKNOWN。结论：此前UNKNOWN遮挡和双有效代码丢失的最小问题已修正。

## 三、R2三个出口逐字段一致

用已提交的299只冻结分数和属性重新运行当前选股/审计逻辑，生成比较结果；随后调用**真实`_run_shared_weekly`导出包装层**。仅模型训练提供方替换为冻结结果，没有执行全池重训。

比较每股`symbol`及8字段：`disagreement_flag, risk_as_of, risk_data_status, risk_reason_codes, eligible, pre_filter_rank, post_filter_rank, rule_version`。

| 周期 | 比较CSV股数 | 顶层CSV股数 | JSON selected股数 | 8字段逐项一致 |
| :-- | --: | --: | --: | :-- |
| D3有流混合 | 42 | 42 | 42 | **通过** |
| D5有流混合 | 38 | 38 | 38 | **通过** |

比较文件为`comparison/D{h}_hybrid_flow_picks.csv`，顶层为`D{h}_picks.csv`，JSON为`run_summary.json.horizons[h].selected`。检查的是落盘JSON，不仅是函数返回值。

- JSON中`disagreement_flag/eligible`均为真正的布尔值，未变成字符串。
- 周摘要`risk_audit`与比较组的规则版本、分歧阈值、标记数量、风险状态一致。
- 三个出口的审计列抽取后分别保存CSV；每个周期的三份提取文件SHA256完全一致，见附件清单。
- 代码已补实验组的分歧阈值摘要字段。当前P0风险状态作用域是被传入名单的集合聚合，不能用作逐股或全299只风险数据完整性证明。

这属于冻结评分驱动的实际导出集成验收，**没有宣称本轮又跑了完整训练CLI**。

## 四、R3最小日期校验

以信号日2026-10-14核查：

| 风险代码 | risk_data_status |
| :-- | :-- |
| `extreme_outflow@2026-10-16` | INVALID |
| `extreme_outflow@tomorrow` | INVALID |
| `extreme_outflow@` | INVALID |
| `extreme_outflow@2026-10-14` | OK |

此前未来日期被标OK的问题已修复。当前验收范围为**审计状态的日级检查**；不等于完整的known_at/observed_at/event_time时序合同，也没有增加硬过滤。

现有Gate对非空`@`文本的判断仍独立于该审计状态，fixture中的未来或错误非空时间仍可能被Gate标为RISK_VETO；无时间戳代码在P0也不作完整时点裁决。后续接真实风险生产者前，需另行定义时分秒/时区、信息获知时间与未来事件日期的区别，以及INVALID如何参与交易资格。本次不将这些P1事项误称已完成。

## 五、冻结基线保持一致

以现有299只评分快照回放D3/D5各5组，股票完整顺序一致，所用排序分数最大绝对差均为0。

| 周期 | 普通 | 无流混合 | 有流混合 | Gate诊断 | 时间权重实验 | 原三组equal_n |
| :-- | --: | --: | --: | --: | --: | --: |
| D3 | 42 | 44 | 42 | 35 | 47 | 42 |
| D5 | 43 | 37 | 38 | 38 | 49 | 37 |

首周合同保持：10/14信号，10/15收盘入场，D3于10/20到期，D5于10/22到期。使用独立的`ret_fwd_3d/ret_fwd_5d`，未改变成本或行业约束，未改写冻结首周名单。

688521在时间权重实验中仍排D3第2、D5第6，分歧为False；301308仍排24/31，分歧为True。两股独立风险数据状态均为UNKNOWN，eligible仍为True。**这是P0只观察合同，不是风险安全认证，也不是已启用止跌规则。**

## 六、后续判断

我接受本次R1/R2/R3在约定P0范围内的修复。下一步可以继续固定规则的四周前瞻模拟，同时保留原策略与实验独立对照。

当前还没有本批次到期收益，不能提升实验为生产基线。是否增加止跌/持续净流出/双模型分歧过滤，仍应遵循前份报告的独立实验和公平比较要求，不能只因688521仍入选而临时调整门槛。

## 七、Git中的证据与原文

附件目录：`docs/reviews/2026-10-15-risk-audit-fa77d31/`。

- `test_summary.json`、`pytest.xml`：完整468项及22项风险审计用例的原始结果。
- `acceptance.json`：R1/R3复现、10组冻结评分/名单检查、R2双周期跨出口结果和两股字段。
- `D3/D5_comparison_audit.csv`、`weekly_audit.csv`、`json_selected_audit.csv`：每周期三个出口的8字段及股票代码。
- `manifest.json`：附件字节哈希和核查源码Git blob哈希。
- [Claude回复原文](reviews/2026-10-15-risk-audit-fa77d31/claude_original.md)：用户转来的原始文本，保留用于复制核对，不能替代本机验收数据。

上一份`docs/risk_audit_local_review_2026-10-15.md`记录的是`738412e`阶段的缺口，本报告是`fa77d31`对其约定修复的接续验收。
