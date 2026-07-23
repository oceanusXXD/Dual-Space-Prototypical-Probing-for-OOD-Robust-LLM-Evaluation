# FLASK Direct Judge / 分类头 / LoRA 对比实验表

## 结论先写

本轮按你的要求把一个 Domain × 一个 Skill 当成一个数据集，选择单一 Domain 视图里样本量最大的 2×2 transfer grid：Language / Culture × Comprehension / Commonsense Understanding。这样得到 4 个数据集，避免多 Domain membership 带来的跨 cell 重复。文档计数合计 5405 条 B-space 评分行；真实可用行数是 5383，每个 cell 内按 question_id 分组切成 train 60% / validation 10% / test 30%。

Direct Judge 没有训练集，训练数据量记为 0，只在四个 target cell 的 test split 上跑。分类头和 LoRA 各训练 4 个模型：每个 source cell 训练 1 个分类头和 1 个 LoRA，然后分别测试四个 target cell，所以分类头是 4×4 次测试，LoRA 也是 4×4 次测试。两种训练方法使用完全相同的 row ids。

## 选中的大样本 cells

| Cell | Doc rows | Train 60% | Val 10% | Test 30% |
| --- | --- | --- | --- | --- |
| Language::Comprehension | 1140 | 684 | 114 | 342 |
| Language::Commonsense Understanding | 1015 | 609 | 102 | 304 |
| Culture::Comprehension | 1978 | 1187 | 198 | 593 |
| Culture::Commonsense Understanding | 1272 | 763 | 127 | 382 |

## 三方法数据量

| Method | Train rows | Validation rows | Test rows | Notes |
| --- | --- | --- | --- | --- |
| Direct Judge | 0 | not used | 1625 | No supervised training; evaluate once on each target test cell. |
| Classification head | 3205 total; see source table | 553 total; see source table | 1625 rows × 4 heads = 6500 eval rows | Train 4 separate heads; each source head is evaluated on all 4 target test cells. |
| LoRA | 3205 total; see source table | 553 total; see source table | 1625 rows × 4 adapters = 6500 eval rows | Train 4 separate LoRA adapters; each adapter is evaluated on all 4 target test cells. |

## 4 个 source 训练集

| Source cell | Head/LoRA train rows | Head/LoRA validation rows | Own-cell test rows |
| --- | --- | --- | --- |
| Language::Comprehension | 672 | 119 | 342 |
| Language::Commonsense Understanding | 597 | 104 | 312 |
| Culture::Comprehension | 1176 | 195 | 596 |
| Culture::Commonsense Understanding | 760 | 135 | 375 |


## 已生成的实际 split

- 总行数：5383
- Train / Val / Test：{'train': 3205, 'validation': 553, 'test': 1625}
- 共享集合文件：'artifacts/flask_direct_head_lora_comparison/comparison_rows.jsonl'
- Split manifest：'artifacts/flask_direct_head_lora_comparison/split_manifest.json'
- Cell audit CSV：'artifacts/flask_direct_head_lora_comparison/split_audit_by_cell.csv'


## 运行顺序

~~~bash
# 推荐：在 GPU 环境一条命令跑完整 pipeline
python scripts/llm_judge_ood/72_run_flask_direct_head_lora_pipeline.py

# 手动分步：
# 1) 下载/准备 FLASK domain-task B-space（数据下载完成后）
python scripts/llm_judge_ood/48_prepare_flask_domain_task_splits.py

# 2) 固定共同 row ids：question_id 分组，60/10/30
python scripts/llm_judge_ood/67_prepare_flask_direct_head_lora_comparison.py

# 3) 0.8B Direct Judge + strict final-prelogit features（同一批 rows）
python scripts/llm_judge_ood/68_run_flask_comparison_direct_and_features.py \
  --rows artifacts/flask_direct_head_lora_comparison/comparison_rows.jsonl \
  --split-manifest artifacts/flask_direct_head_lora_comparison/split_manifest.json

# 4) 训练分类头（同一 train/validation/test）
python scripts/llm_judge_ood/69_train_flask_comparison_head.py \
  --rows artifacts/flask_direct_head_lora_comparison/direct_and_features/b_space_with_direct_judge.jsonl \
  --split-manifest artifacts/flask_direct_head_lora_comparison/split_manifest.json \
  --features artifacts/flask_direct_head_lora_comparison/direct_and_features/strict_final_prelogit_features.npz

# 5) 训练 LoRA（同一 train/validation/test）
python scripts/llm_judge_ood/70_train_flask_comparison_lora.py \
  --rows artifacts/flask_direct_head_lora_comparison/comparison_rows.jsonl \
  --split-manifest artifacts/flask_direct_head_lora_comparison/split_manifest.json

# 6) 汇总三种方法的 performance 表
python scripts/llm_judge_ood/71_summarize_flask_comparison_results.py
~~~

## 结果表模板

实际训练/测试完成后，用 71_summarize_flask_comparison_results.py 填充下面字段；空表 CSV 已写到 'artifacts/flask_direct_head_lora_comparison/result_template.csv'。Direct Judge 行没有 source_cell_id；分类头和 LoRA 行用 source_cell_id → target_cell_id 表示 4×4 迁移测试。

| method | source_cell_id | target_cell_id | split | rows | parse_rate | mae | exact_accuracy | plus_minus_1_accuracy | quadratic_weighted_kappa | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| direct_judge |  | Language::Comprehension | test |  |  |  |  |  |  |  |
| direct_judge |  | Language::Commonsense Understanding | test |  |  |  |  |  |  |  |
| direct_judge |  | Culture::Comprehension | test |  |  |  |  |  |  |  |
| direct_judge |  | Culture::Commonsense Understanding | test |  |  |  |  |  |  |  |
| classification_head | Language::Comprehension | Language::Comprehension | test |  |  |  |  |  |  |  |
| classification_head | Language::Comprehension | Language::Commonsense Understanding | test |  |  |  |  |  |  |  |
| classification_head | Language::Comprehension | Culture::Comprehension | test |  |  |  |  |  |  |  |
| classification_head | Language::Comprehension | Culture::Commonsense Understanding | test |  |  |  |  |  |  |  |
| classification_head | Language::Commonsense Understanding | Language::Comprehension | test |  |  |  |  |  |  |  |
| classification_head | Language::Commonsense Understanding | Language::Commonsense Understanding | test |  |  |  |  |  |  |  |
| classification_head | Language::Commonsense Understanding | Culture::Comprehension | test |  |  |  |  |  |  |  |
| classification_head | Language::Commonsense Understanding | Culture::Commonsense Understanding | test |  |  |  |  |  |  |  |
| classification_head | Culture::Comprehension | Language::Comprehension | test |  |  |  |  |  |  |  |
| classification_head | Culture::Comprehension | Language::Commonsense Understanding | test |  |  |  |  |  |  |  |
| classification_head | Culture::Comprehension | Culture::Comprehension | test |  |  |  |  |  |  |  |
| classification_head | Culture::Comprehension | Culture::Commonsense Understanding | test |  |  |  |  |  |  |  |
| classification_head | Culture::Commonsense Understanding | Language::Comprehension | test |  |  |  |  |  |  |  |
| classification_head | Culture::Commonsense Understanding | Language::Commonsense Understanding | test |  |  |  |  |  |  |  |
| classification_head | Culture::Commonsense Understanding | Culture::Comprehension | test |  |  |  |  |  |  |  |
| classification_head | Culture::Commonsense Understanding | Culture::Commonsense Understanding | test |  |  |  |  |  |  |  |
| lora | Language::Comprehension | Language::Comprehension | test |  |  |  |  |  |  |  |
| lora | Language::Comprehension | Language::Commonsense Understanding | test |  |  |  |  |  |  |  |
| lora | Language::Comprehension | Culture::Comprehension | test |  |  |  |  |  |  |  |
| lora | Language::Comprehension | Culture::Commonsense Understanding | test |  |  |  |  |  |  |  |
| lora | Language::Commonsense Understanding | Language::Comprehension | test |  |  |  |  |  |  |  |
| lora | Language::Commonsense Understanding | Language::Commonsense Understanding | test |  |  |  |  |  |  |  |
| lora | Language::Commonsense Understanding | Culture::Comprehension | test |  |  |  |  |  |  |  |
| lora | Language::Commonsense Understanding | Culture::Commonsense Understanding | test |  |  |  |  |  |  |  |
| lora | Culture::Comprehension | Language::Comprehension | test |  |  |  |  |  |  |  |
| lora | Culture::Comprehension | Language::Commonsense Understanding | test |  |  |  |  |  |  |  |
| lora | Culture::Comprehension | Culture::Comprehension | test |  |  |  |  |  |  |  |
| lora | Culture::Comprehension | Culture::Commonsense Understanding | test |  |  |  |  |  |  |  |
| lora | Culture::Commonsense Understanding | Language::Comprehension | test |  |  |  |  |  |  |  |
| lora | Culture::Commonsense Understanding | Language::Commonsense Understanding | test |  |  |  |  |  |  |  |
| lora | Culture::Commonsense Understanding | Culture::Comprehension | test |  |  |  |  |  |  |  |
| lora | Culture::Commonsense Understanding | Culture::Commonsense Understanding | test |  |  |  |  |  |  |  |
