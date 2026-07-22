#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

mkdir -p logs artifacts/llm_judge_ood_asap

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"

gpu_memory_mib="$(.venv/bin/python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available")
print(torch.cuda.get_device_properties(0).total_memory // (1024 * 1024))
PY
)"
if (( gpu_memory_mib < 80000 )); then
  echo "This preset requires an 80+ GB GPU; detected ${gpu_memory_mib} MiB." >&2
  exit 2
fi

.venv/bin/python scripts/llm_judge_ood/20_prepare_llm_judge_ood_hidden.py \
  --input artifacts/llm_judge_ood_asap/asap_prepared_contract_v1.jsonl \
  --model-path /home/ubuntu/models/qwen3.5-4b \
  --model-id Qwen/Qwen3.5-4B \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --local-files-only \
  --output artifacts/llm_judge_ood_asap/qwen3_5_4b_input_document_masked_mean_v1.npz \
  --parts-dir artifacts/llm_judge_ood_asap/qwen3_5_4b_input_document_masked_mean_v1.fast.parts \
  --layers -10 -1 \
  --batch-size 128 \
  --max-batch-tokens 65536 \
  --tokenization-batch-size 512 \
  --max-length 2048 \
  --device cuda \
  --torch-dtype bfloat16 \
  --hidden-dtype float16 \
  --attn-implementation sdpa \
  --feature-scope input_document \
  --pooling masked_mean \
  --seed 42 \
  --fast \
  2>&1 | tee logs/20260719_rtx_pro_6000_full_feature_extraction.log

extract_auxiliary_cache() {
  local input_path="$1"
  local output_path="$2"
  local feature_scope="$3"
  local log_path="$4"
  .venv/bin/python scripts/llm_judge_ood/20_prepare_llm_judge_ood_hidden.py \
    --input "$input_path" \
    --model-path /home/ubuntu/models/qwen3.5-4b \
    --model-id Qwen/Qwen3.5-4B \
    --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
    --local-files-only \
    --output "$output_path" \
    --layers -10 -1 \
    --batch-size 128 \
    --max-batch-tokens 65536 \
    --tokenization-batch-size 512 \
    --max-length 2048 \
    --device cuda \
    --torch-dtype bfloat16 \
    --hidden-dtype float16 \
    --attn-implementation sdpa \
    --feature-scope "$feature_scope" \
    --pooling masked_mean \
    --seed 42 \
    --fast \
    2>&1 | tee "$log_path"
}

extract_auxiliary_cache \
  artifacts/llm_judge_ood_asap/asap_prepared_contract_v1.jsonl \
  artifacts/llm_judge_ood_asap/qwen3_5_4b_judge_input_asap_rubric_v1.npz \
  judge_input \
  logs/qwen3_5_4b_judge_input_asap_rubric_v1.log
extract_auxiliary_cache \
  artifacts/llm_judge_ood_asap/asap_prepared_contract_v1_within_prompt_covariate_v1.jsonl \
  artifacts/llm_judge_ood_asap/asap_within_prompt_input_document_v1.npz \
  input_document \
  logs/asap_within_prompt_input_document_v1.log
extract_auxiliary_cache \
  artifacts/llm_judge_ood_asap/asap_prepared_contract_v1_within_prompt_covariate_v1.jsonl \
  artifacts/llm_judge_ood_asap/asap_within_prompt_judge_input_v1.npz \
  judge_input \
  logs/asap_within_prompt_judge_input_v1.log
extract_auxiliary_cache \
  artifacts/llm_judge_ood_asap/asap_prepared_contract_v1_semantic_task_shift_v1.jsonl \
  artifacts/llm_judge_ood_asap/asap_semantic_task_input_document_v1.npz \
  input_document \
  logs/asap_semantic_task_input_document_v1.log
extract_auxiliary_cache \
  artifacts/llm_judge_ood_asap/asap_prepared_contract_v1_semantic_task_shift_v1.jsonl \
  artifacts/llm_judge_ood_asap/asap_semantic_task_judge_input_v1.npz \
  judge_input \
  logs/asap_semantic_task_judge_input_v1.log
