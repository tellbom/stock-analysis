# A-Stock Data Event3 Integration Report

- actual_as_of_date: 2026-07-03
- gate_mode: event3_enhanced_gate

## 1. cninfo 公告覆盖情况

- coverage: {'path': 'silver/announcement_events', 'files': 300, 'symbol_file_coverage': '300/300', 'as_of_files': 6, 'latest_min': '2026-04-28', 'latest_max': '2026-07-03'}
- entered_recent_model: yes

## 2. 龙虎榜覆盖情况

- coverage: {'path': 'silver/dragon_tiger', 'files': 300, 'symbol_file_coverage': '300/300', 'as_of_files': 8, 'latest_min': '2026-01-06', 'latest_max': '2026-07-03'}
- entered_recent_model: yes

## 3. 大宗交易覆盖情况

- coverage: {'path': 'silver/block_trade', 'files': 300, 'symbol_file_coverage': '300/300', 'as_of_files': 15, 'latest_min': '2026-01-08', 'latest_max': '2026-07-03'}
- entered_recent_model: yes

## 4. 三类事件因子是否进入 recent model

- announcement_events: yes
- dragon_tiger: yes
- block_trade: yes

## 5. 融资融券补充是否仅用于交叉验证

- yes; margin features in model: recent=none, base=none

## 6. 是否重跑 D3 Gate

- yes; requested_as_of_date=2026-07-06; actual_as_of_date=2026-07-03

## 7. Base Top20

1. 301165
2. 002422
3. 000975
4. 688012
5. 300476
6. 000807
7. 002532
8. 601600
9. 002371
10. 002625
11. 688082
12. 688256
13. 600188
14. 002460
15. 688521
16. 002594
17. 600489
18. 600547
19. 688506
20. 300450

## 8. Recent Top20

1. 002916
2. 002028
3. 000100
4. 300502
5. 002384
6. 300274
7. 600176
8. 000776
9. 600999
10. 300033
11. 603259
12. 600026
13. 601211
14. 300442
15. 300433
16. 688183
17. 300866
18. 000338
19. 600276
20. 002460

## 9. Gate Fused Top20

1. 002422
2. 000975
3. 688012
4. 300476
5. 000807
6. 002532
7. 601600
8. 688082
9. 688256
10. 600188
11. 688521
12. 688506
13. 300450
14. 605499
15. 300661
16. 002353
17. 300502
18. 601066
19. 688303
20. 688223

## 10. 生成文件路径

- base_ranked: `E:\stock-analysis\models\data\reports\D3_base_ranked_2026-07-03.csv`
- recent_ranked: `E:\stock-analysis\models\data\reports\D3_recent_ranked_2026-07-03.csv`
- gate_ranked: `E:\stock-analysis\models\data\reports\D3_gate_fused_ranked_2026-07-03.csv`
- report: `E:\stock-analysis\models\data\reports\D3_gate_event3_run_report_2026-07-03.md`
- json: `E:\stock-analysis\models\data\reports\D3_gate_event3_run_summary_2026-07-03.json`
- integration_report: `E:\stock-analysis\a_stock_data_event3_integration_report.md`
- base_X_train_columns: `E:\stock-analysis\models\data\reports\D3_base_X_train_columns_2026-07-03.csv`
- recent_X_train_columns: `E:\stock-analysis\models\data\reports\D3_recent_event3_X_train_columns_2026-07-03.csv`
- event3_coverage_gate: `E:\stock-analysis\models\data\reports\D3_event3_coverage_gate_2026-07-03.csv`
- event3_missing_nonzero: `E:\stock-analysis\models\data\reports\D3_event3_feature_missing_nonzero_2026-07-03.csv`
- event3_feature_importance: `E:\stock-analysis\models\data\reports\D3_event3_feature_importance_2026-07-03.csv`

## 11. 是否建议 Claude 评审

- yes; 建议重点评审 PIT 日期、事件稀疏 coverage gate、以及 event3 特征进入 recent-only 的边界。