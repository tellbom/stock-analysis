# 738412e风险审计：本机验收与Claude接续

核查日期：2026-10-15上午。代码基准：`738412e`，上一版本`beddd46`。本次已快进拉取；原2738项本机数据文件改动及输出目录保留。Git远端URL未发现内嵌账号或token，此项不等于PAT已吊销；本机未执行PAT吊销。

## 一、裁决

**新增可观测性的方向合理，冻结评分与名单回放通过；审计合同验收尚未全部通过。** 已复现未知标记遮挡有效标记、每周顶层输出丢失8个审计字段，以及风险时间未校验的问题。请Claude接续修正并提供回归用例。

本次仅拉取、测试、冻结分数回放和记录评审；没有修改生产代码、重新训练、重跑全历史回测、改写首周名单或新增止跌门槛。

`risk-observability-v1`只做观察，`eligible`固定为True、过滤前后名次相同。该字段表示本版没有风险过滤，不能解释成已通过风险检查。`disagreement_flag=False`也只表示两个模型没有达到本版分歧条件，不能解释成股票安全。

## 二、本机核查范围与结果

### 1. 完整pytest

命令：

```bash
python -m pytest -q tests --junitxml=temp/risk_audit_738412e_pytest.xml
```

**实际完整运行：456通过、0失败、0错误、0跳过，97条警告，耗时352.16秒。** 使用本机Python 3.13及已安装的sklearn/LightGBM等依赖，未跳过相关用例。新增风险审计文件实际收集并通过10项，云端回复的“11个测试”与源码/本机收集数量不一致。测试通过与下列未覆盖的合同缺口分别记录。JUnit原始记录与解析摘要已附Git。

### 2. 真实冻结输入的选股回放

读取已提交的`docs/reviews/2026-10-15-ffae653-acceptance/`全池299只分数和属性，调用当前`combine/fusion_score/gate_first_fusion/select/annotate_picks`，回放两周期各5组。**没有训练模型或加入10/15价格。**

| 周期 | 普通 | 无流混合 | 有流混合 | Gate诊断 | 时间权重实验 | 原三组equal_n |
| :-- | --: | --: | --: | --: | --: | --: |
| D3 | 42 | 44 | 42 | 35 | 47 | 42 |
| D5 | 43 | 37 | 38 | 38 | 49 | 37 |

10份名单顺序完全一致，所用排序分数最大差均为0；比较入口每份名单均可补齐8个审计字段。当前无上游风险标记，全为`UNKNOWN`。本轮未重新执行完整训练CLI，也未重新训练调用`verify_reference`；相关全训练验收为前版`2fe7660`记录，不能写成本轮又跑过。

### 3. 新分歧字段对两只股票的实际结果

| 股票 | 周期 | 实验最终排名 | 分歧标记 | 风险状态 | eligible |
| :-- | :-- | --: | :-- | :-- | :-- |
| 688521 | D3 | 2 | False | UNKNOWN | True |
| 688521 | D5 | 6 | False | UNKNOWN | True |
| 301308 | D3 | 24 | True | UNKNOWN | True |
| 301308 | D5 | 31 | True | UNKNOWN | True |

实验全池排名差中位数D3为74、D5为71。688521两个模型仍给正预测，排名差分别7和28，因此不标分歧；301308正负预测冲突、排名差277/249，标记为True。**当前改造不会因大跌或资金流净出而排除688521，301308被标分歧后也仍保留。** 这是P0观察合同，并非新过滤规则执行失败。

## 三、需要修正的已复现缺口

### R1：UNKNOWN会遮挡有效风险；两路不同有效代码也会丢失

位置：`quant_platform/selection/risk_audit.py:45-50`与`scripts/compare_original_hybrid.py:110-119`，行号以`738412e`为准。

最小触发：

```python
import pandas as pd
from quant_platform.selection.risk_audit import coalesce_flag_values

print(coalesce_flag_values(
    pd.Series(["UNKNOWN"]),
    pd.Series(["extreme_outflow@2026-10-14"]),
).tolist())
# 实际：['UNKNOWN']
```

原因：合并只把空字符串当缺失，`UNKNOWN`是非空字符串，因primary优先而遮挡recent有效标记。当前测试覆盖的是空字符串/None，未覆盖显式UNKNOWN。

实际经过`combine -> gate_first_fusion -> actionable_pool`的fixture中，base/recent都是高分，base=`UNKNOWN`、recent=`extreme_outflow@2026-10-14`，结果为`UNKNOWN`、`A_MAIN`，诊断池保留该股。有效风险信号丢失，并非只影响显示。

另一个fixture为base=`coverage@2026-10-14`、recent=`extreme_outflow@2026-10-14`，只保留coverage，降级为`RISK_DOWNGRADE`；recent的否决代码消失。当前辅助函数明确设计primary优先，**但独立风险事件通常不能当作互斥备选值处理**。请裁决字段合同：建议风险代码按结构化token并集合并、去重，保留各自时间和来源；若确需优先级，必须显式声明冲突规则并审计被舍弃值。

当前真实批次两路都没有风险代码，因此这两个缺口未改变10/14冻结名单。请补UNKNOWN+有效值、双有效不同代码、重复代码、空值、错位索引、日期保留等用例。不得因此新增未经评估的交易过滤门槛。

### R2：每周顶层名单和JSON丢失8个新增字段

位置：`scripts/run_dual_horizon_rrf.py:155-189`。

每周入口先调用比较入口，`comparison/D3_hybrid_flow_picks.csv`包含审计字段；随后读取`score_gate.csv`重新调用`select_rrf`生成picks，只校验股票顺序和融合分数，直接写`D3_picks.csv`及`run_summary.json.horizons[3].selected`。重算路径未调用`annotate_picks`或接回已校验的审计列，D3/D5顶层均丢失8列。

本机执行了**真实`_run_shared_weekly`导出包装层**，仅将模型训练提供方替换为冻结结果，比较目录含新审计名单。D3/D5均确认：

- 比较目录的有流名单：8列齐全。
- 顶层`D3_picks.csv/D5_picks.csv`：8列全部缺失。
- `run_summary.json`的两周期`selected`：8列全部缺失。

这是聚焦导出集成回放，不是完整CLI重训证明。建议在名单/分数一致校验后按symbol一对一带回审计字段，或统一使用已验证的规范名单；保留原有输出列和排序。汇总同时记录规则版本、分歧阈值、状态、标记数量，并增加两个出口及JSON逐字段一致用例。

### R3：risk_as_of是审计日期，尚不能证明风险信息当时可得

位置：`risk_audit.py:119-123`；关联旧代码`gate_fusion.py:62-84,128-139`。

fixture使用信号日10/14、风险代码`extreme_outflow@2026-10-16`：

```text
risk_as_of       = 2026-10-14
risk_data_status = OK
risk_reason_codes = extreme_outflow@2026-10-16
eligible        = True
gate_tier       = RISK_VETO
```

审计日期直接填信号日，状态只检查是否UNKNOWN；Gate仅检查`@`后有非空文本，没有检查可解析时间或与截止日比较。Gate的时间解析限制是已有行为，本次透传真实标记后会暴露该限制。

这证明**风险时点校验尚未建立**，不证明当前真实训练已发生前视泄漏。请区分`known_at/observed_at`与事件实际发生日期：已提前公告的未来事件可以在信号日前已知，不应一律按未来事件日剔除；信息获知时间则必须不晚于截止时点。格式错误、未来获知或来源缺失需要明确状态/原因，不能标OK并伪造当时可得证明。

在P0仍只观察的范围内，可以先校验和标记；是否影响交易资格另按P1实验裁决，保持原基线不变。

## 四、补充合同限制

- 实验组摘要没有记录`disagreement_rank_gap_threshold`，原四组摘要有该字段。建议补齐，使相同布尔值可追溯到具体阈值。
- `risk_data_status`目前是对已选名单原因码聚合，整列赋同一状态；它不是每只股票的单独状态，也没有覆盖完整299只。请明确字段作用域；若保留汇总状态，避免作为逐股完整性证明。
- 当前没有风险数据生产者接入普通/近期分数输出；UNKNOWN表示独立风险分类未知，不表示OHLCV或训练资金流缺失。实际资金流特征已进入有流模型，不能把这两件事混淆。

## 五、Claude接续要求

1. 修正或明确裁决R1的未知值回退和多来源风险合并合同，确保有效代码不静默丢失。
2. 补齐R2顶层CSV、JSON与比较名单的审计一致性，两个周期均验证。
3. 明确R3风险时间语义和输入有效性状态；新增可观察时间用例。
4. 补充实验阈值摘要，明确状态作用域。所有改动保持P0观察模式，先不引入止跌、资金流否决或分歧筛选。
5. 默认真实无风险代码输入时，冻结10组评分/名单、equal_n、两周期标签和入退场日期保持一致。不要覆盖首周名单或重新解释成已验证收益改善。
6. 云端给出修复commit、测试范围、本机接续命令及尚未验证事项。完整数据、训练与到期收益继续在本机按既有计划执行。

## 六、证据

- `docs/reviews/2026-10-15-risk-audit-local/acceptance.json`：10组冻结名单/分数回放、两股新审计字段和阈值。
- `docs/reviews/2026-10-15-risk-audit-local/known_gaps.json`：真实周导出包装层与三个最小fixture结果。
- `docs/reviews/2026-10-15-risk-audit-local/test_summary.json`及`pytest.xml`：完整456项测试结果和新增10项用例名称。
- `docs/reviews/2026-10-15-risk-audit-local/manifest.json`：本轮证据文件及核查源码哈希，防止混用后续版本。
- 上一份原因与策略建议：`docs/688521_followup_review_2026-10-15.md`，其中四套模型推理与贡献复算仍适用。

结论：可以确认P0新增观察字段进入比较入口，当前冻结选股结果保持一致；**暂不能把双入口审计接线、有效风险透传和当时可得性宣称为验收完成**。通过完整pytest也不能替代这些缺失场景的验证。
