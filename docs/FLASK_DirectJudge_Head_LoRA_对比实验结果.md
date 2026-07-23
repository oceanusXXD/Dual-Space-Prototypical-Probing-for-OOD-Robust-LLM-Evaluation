# FLASK Direct Judge / 分类头 / LoRA Performance Summary

## Scope

- 数据集：FLASK single-domain 2×2 transfer grid。
- 模型：Direct Judge 与 LoRA 使用 Qwen3.5-0.8B；分类头使用同一批 strict final-prelogit hidden features。
- Split：所有方法共享 question_id 分组后的 60% train / 10% validation / 30% test；Direct Judge 只在 test 上评估。
- Output CSV：artifacts/flask_direct_head_lora_comparison/performance_summary.csv

## Runtime

| Item | Value |
| --- | --- |
| status | blocked |
| local_deps | /workspace/.deps/python_min |
| torch_cuda_available | False |
| errors | CUDA is required for the full Direct Judge + LoRA run, but torch cannot see a GPU. |

## Results

| method | source_cell_id | target_cell_id | rows | mae | exact_accuracy | plus_minus_1_accuracy | quadratic_weighted_kappa | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |


## Pending

| method | source_cell_id | target_cell_id | status |
| --- | --- | --- | --- |
| direct_judge |  | Culture::Commonsense Understanding | missing: artifacts/flask_direct_head_lora_comparison/direct_and_features/direct_judge_metrics.csv |
| direct_judge |  | Culture::Comprehension | missing: artifacts/flask_direct_head_lora_comparison/direct_and_features/direct_judge_metrics.csv |
| direct_judge |  | Language::Commonsense Understanding | missing: artifacts/flask_direct_head_lora_comparison/direct_and_features/direct_judge_metrics.csv |
| direct_judge |  | Language::Comprehension | missing: artifacts/flask_direct_head_lora_comparison/direct_and_features/direct_judge_metrics.csv |
| classification_head | Culture::Commonsense Understanding | Culture::Commonsense Understanding | missing: artifacts/flask_direct_head_lora_comparison/classification_head/classification_head_4x4_metrics.csv |
| classification_head | Culture::Commonsense Understanding | Culture::Comprehension | missing: artifacts/flask_direct_head_lora_comparison/classification_head/classification_head_4x4_metrics.csv |
| classification_head | Culture::Commonsense Understanding | Language::Commonsense Understanding | missing: artifacts/flask_direct_head_lora_comparison/classification_head/classification_head_4x4_metrics.csv |
| classification_head | Culture::Commonsense Understanding | Language::Comprehension | missing: artifacts/flask_direct_head_lora_comparison/classification_head/classification_head_4x4_metrics.csv |
| classification_head | Culture::Comprehension | Culture::Commonsense Understanding | missing: artifacts/flask_direct_head_lora_comparison/classification_head/classification_head_4x4_metrics.csv |
| classification_head | Culture::Comprehension | Culture::Comprehension | missing: artifacts/flask_direct_head_lora_comparison/classification_head/classification_head_4x4_metrics.csv |
| classification_head | Culture::Comprehension | Language::Commonsense Understanding | missing: artifacts/flask_direct_head_lora_comparison/classification_head/classification_head_4x4_metrics.csv |
| classification_head | Culture::Comprehension | Language::Comprehension | missing: artifacts/flask_direct_head_lora_comparison/classification_head/classification_head_4x4_metrics.csv |
| classification_head | Language::Commonsense Understanding | Culture::Commonsense Understanding | missing: artifacts/flask_direct_head_lora_comparison/classification_head/classification_head_4x4_metrics.csv |
| classification_head | Language::Commonsense Understanding | Culture::Comprehension | missing: artifacts/flask_direct_head_lora_comparison/classification_head/classification_head_4x4_metrics.csv |
| classification_head | Language::Commonsense Understanding | Language::Commonsense Understanding | missing: artifacts/flask_direct_head_lora_comparison/classification_head/classification_head_4x4_metrics.csv |
| classification_head | Language::Commonsense Understanding | Language::Comprehension | missing: artifacts/flask_direct_head_lora_comparison/classification_head/classification_head_4x4_metrics.csv |
| classification_head | Language::Comprehension | Culture::Commonsense Understanding | missing: artifacts/flask_direct_head_lora_comparison/classification_head/classification_head_4x4_metrics.csv |
| classification_head | Language::Comprehension | Culture::Comprehension | missing: artifacts/flask_direct_head_lora_comparison/classification_head/classification_head_4x4_metrics.csv |
| classification_head | Language::Comprehension | Language::Commonsense Understanding | missing: artifacts/flask_direct_head_lora_comparison/classification_head/classification_head_4x4_metrics.csv |
| classification_head | Language::Comprehension | Language::Comprehension | missing: artifacts/flask_direct_head_lora_comparison/classification_head/classification_head_4x4_metrics.csv |
| lora | Culture::Commonsense Understanding | Culture::Commonsense Understanding | missing: artifacts/flask_direct_head_lora_comparison/lora/lora_4x4_metrics.csv |
| lora | Culture::Commonsense Understanding | Culture::Comprehension | missing: artifacts/flask_direct_head_lora_comparison/lora/lora_4x4_metrics.csv |
| lora | Culture::Commonsense Understanding | Language::Commonsense Understanding | missing: artifacts/flask_direct_head_lora_comparison/lora/lora_4x4_metrics.csv |
| lora | Culture::Commonsense Understanding | Language::Comprehension | missing: artifacts/flask_direct_head_lora_comparison/lora/lora_4x4_metrics.csv |
| lora | Culture::Comprehension | Culture::Commonsense Understanding | missing: artifacts/flask_direct_head_lora_comparison/lora/lora_4x4_metrics.csv |
| lora | Culture::Comprehension | Culture::Comprehension | missing: artifacts/flask_direct_head_lora_comparison/lora/lora_4x4_metrics.csv |
| lora | Culture::Comprehension | Language::Commonsense Understanding | missing: artifacts/flask_direct_head_lora_comparison/lora/lora_4x4_metrics.csv |
| lora | Culture::Comprehension | Language::Comprehension | missing: artifacts/flask_direct_head_lora_comparison/lora/lora_4x4_metrics.csv |
| lora | Language::Commonsense Understanding | Culture::Commonsense Understanding | missing: artifacts/flask_direct_head_lora_comparison/lora/lora_4x4_metrics.csv |
| lora | Language::Commonsense Understanding | Culture::Comprehension | missing: artifacts/flask_direct_head_lora_comparison/lora/lora_4x4_metrics.csv |
| lora | Language::Commonsense Understanding | Language::Commonsense Understanding | missing: artifacts/flask_direct_head_lora_comparison/lora/lora_4x4_metrics.csv |
| lora | Language::Commonsense Understanding | Language::Comprehension | missing: artifacts/flask_direct_head_lora_comparison/lora/lora_4x4_metrics.csv |
| lora | Language::Comprehension | Culture::Commonsense Understanding | missing: artifacts/flask_direct_head_lora_comparison/lora/lora_4x4_metrics.csv |
| lora | Language::Comprehension | Culture::Comprehension | missing: artifacts/flask_direct_head_lora_comparison/lora/lora_4x4_metrics.csv |
| lora | Language::Comprehension | Language::Commonsense Understanding | missing: artifacts/flask_direct_head_lora_comparison/lora/lora_4x4_metrics.csv |
| lora | Language::Comprehension | Language::Comprehension | missing: artifacts/flask_direct_head_lora_comparison/lora/lora_4x4_metrics.csv |
