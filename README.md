# LLM Judge Document-Level Covariate-Shift Monitoring and Controlled Adaptation

This repository implements document-level covariate-shift monitoring for an LLM
Judge and a label-efficient, gated adaptation path. The method separates three
claims that must be evaluated independently: distribution drift, persistent
drift, and harmful Judge degradation. Detecting OOD is not by itself evidence
that the Judge is wrong or that an update is beneficial.

Generated outputs belong under `artifacts/` or `outputs/` and are not source artifacts.

The repository specifications are maintained in:

- [`docs/LLM_Judge_OOD/LLMJudgeOOD_完整方案_ResidualViM_MMD定稿版.md`](docs/LLM_Judge_OOD/LLMJudgeOOD_完整方案_ResidualViM_MMD定稿版.md): authoritative method, benchmark, implementation, and formal acceptance report;
- [`docs/LLM_Judge_OOD/LLM_Judge_OOD_算法流程与HiddenState采集验收清单_代码核对版.md`](docs/LLM_Judge_OOD/LLM_Judge_OOD_算法流程与HiddenState采集验收清单_代码核对版.md): hidden-state cache contract and code-level acceptance checklist;
- [`docs/LLM_Judge_OOD/benchmark_ground_truth_full_with_samples.md`](docs/LLM_Judge_OOD/benchmark_ground_truth_full_with_samples.md): cross-domain/rubric benchmark ground-truth plan and real sample fields.

## Repository Layout

```text
configs/llm_judge_ood/  JSON configs for LLM Judge local smoke and SummEval runs
datasets/               Local source datasets, when available
docs/LLM_Judge_OOD/     LLM Judge OOD design, acceptance, and historical diagnostics
scripts/llm_judge_ood/  LLM Judge OOD entrypoints
src/common/             IO, metrics, seed, and device helpers
src/models/             Frozen hidden-state loading utilities
src/llm_judge_ood/      Standalone LLM Judge OOD implementation
```

## Setup

```bash
python3 -m pip install -r requirements.lock
python3 scripts/llm_judge_ood/00_validate_environment.py \
  --output artifacts/llm_judge_ood_asap/environment_cpu_contract_v1.json
```

`requirements.txt` remains the broad dependency declaration; `requirements.lock` is the
2026-07-20 formal-environment lock. The command above performs imports and records the
CPU environment only. GPU readiness is a separate, later prerequisite check and does not
run extraction or experiments.

The exact local frozen-model path used by the hidden-state scripts in this workspace is:

```text
/home/ubuntu/models/qwen3.5-4b
```

The protocol pins this snapshot to Hugging Face commit `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. The available `/home/ubuntu/models/qwen3-4b-instruct-2507` directory is intentionally not used: `Qwen3-4B-Instruct-2507` is an older, different checkpoint.

All production and smoke pipelines require a frozen `Qwen/Qwen3.5-4B` hidden-state cache. Missing or unpinned caches fail closed; there is no TF-IDF, SVD, or other handcrafted text-feature fallback.

## LLM Judge Document OOD

### Production Boundary

The production OOD question is:

> Does this deployment input document deviate from the training input-document distribution?

It is not a generator, candidate-system, candidate-response, query, or label detector. The document OOD contract has exactly these document-side fields:

| Field | Meaning |
|---|---|
| `input_document_id` | Stable identity for one input document. |
| `input_document_text` | Exact document text consumed by the OOD detector. |
| `document_distribution_role` | `training`, `development`, `benchmark`, or `deployment`. |

`query_text`, candidate-response text, labels, and `candidate_system_id` never enter the document-OOD extractor. They may enter the Judge extractor only when the task itself requires response-aware, query-conditioned judging.

The frozen cache contract separates document OOD from Judge behavior:

| Scope | Consumers | Allowed text |
|---|---|---|
| `input_document` | A-space and document OOD monitor | Raw `input_document_text` only. |
| `judge_input` | Query-conditioned Judge and B-space | The prepared Judge input, including the query/dimension, source document, and candidate response. |

ASAP uses `judge_feature_scope=judge_input`: its Judge/B cache encodes the frozen scoring template, task, rubric, and essay, while A-space encodes only the essay text. SummEval also uses `judge_feature_scope=judge_input` and a per-query linear head: its Judge/B cache is separate, while A-space remains document-only. Every cache and resumable-parts manifest must declare the pinned requested/resolved revision and architecture, `model_id=Qwen/Qwen3.5-4B`, its scope-specific `feature_scope`, prompt-template version, representation kind, selected token policy or pooling formula, `labels_in_prompt=false`, `max_length=2048`, `model_eval=true`, and `requires_grad=false`. Other cache contracts are rejected.

### Hidden-State Representation Policy

Do not treat one hidden-state rule as universal. A-space and B-space answer different questions, and even within B-space there are at least three separable signals:

1. **document / response distribution**: what kind of text is being judged;
2. **rubric / task conditioning**: which criterion the Judge is applying;
3. **decision state**: the hidden state immediately before the Judge would emit a score, label, or class option.

The final protocol should therefore store a named representation bundle instead of pretending that one vector is always correct:

| View name | What it captures | How to compute | Typical consumer |
|---|---|---|---|
| `input_document_masked_mean` | Raw input or candidate-answer distribution | Attention-mask mean over `input_document_text` only | A-space OOD, clustering, retrieval/localization |
| `candidate_span_mean` | Candidate response semantics inside the Judge prompt | Mean over the candidate-response token span after template rendering | Response-quality diagnostics, long-input robustness |
| `rubric_task_span_mean` | The current task/rubric/check definition | Mean over rubric, task, criteria, or check tokens | Task/rubric OOD, check clustering |
| `judge_prompt_masked_mean` | Coarse mixture of rubric, instruction, reference, and candidate text | Attention-mask mean over the full label-free `judge_input_text` | Auxiliary B-space diagnostics and legacy-cache reuse |
| `pre_answer_token` | The Judge state after reading the full prompt and answer prefix | Append a fixed answer prefix, then take the final prompt token hidden state | Parent view for score/label/classification decisions |
| `pre_score_token` | Numeric score decision state | `pre_answer_token` with a score prefix such as `Score:` or `{"score":` | Ordinal/regression score heads, Direct Judge scoring |
| `pre_label_token` | Binary or categorical label decision state | `pre_answer_token` with `Label:`, `Answer:`, or class-option prefix | Pass/fail checks, intent/topic classification |
| `answer_logits` | The actual next-token preference over allowed score/label tokens | Logits at `pre_answer_token`, restricted to the frozen allowed vocabulary | Direct Judge baselines, confidence/energy diagnostics |

No view is a universal replacement for the others. `pre_score_token` is often the best single vector for a scoring head, but it can hide useful covariate-shift structure because it compresses the whole prompt into a decision boundary state. Conversely, `judge_prompt_masked_mean` is good at noticing rubric/content distribution changes, but it is usually weaker for predicting the final score because it averages many tokens that are not near the decision point.

Dataset-specific policy:

| Dataset family | Label / decision structure | Recommended A-space | Recommended B-space bundle | Rationale |
|---|---|---|---|---|
| ASAP AES / ELLIPSE | Ordinal essay-quality scores under a rubric | `input_document_masked_mean` over the essay | `candidate_span_mean` + `rubric_task_span_mean` + `pre_score_token`; optionally `answer_logits` over score tokens | Essay distribution shift and rubric-conditioned scoring are different. The score head should see the decision token, while OOD/localization still benefits from essay-span and rubric-span views. |
| SummEval | Query-specific summary quality scores for candidate summaries against a source article | For production document OOD: article `input_document_masked_mean`; for candidate-response diagnostics: optional summary `candidate_span_mean` | One B row per query with `candidate_span_mean`, query/rubric span mean, `pre_score_token`, and score logits | Full-prompt mean can be dominated by the source article. Dimension-level scoring should be anchored near the answer token, while summary/content shift should be measured separately. |
| CLINC150 / ROSTD / AG News | Intent, OOS, or topic classification, not scalar judging | `input_document_masked_mean` over utterance/news text | `pre_label_token` plus allowed-class logits; `judge_prompt_masked_mean` is optional | The decision is a class label, so “score token” is the wrong abstraction. OOD may be stronger in input-text mean, while classifier behavior should use label-option state/logits. |
| LongJudgeBench | Long report quality; native dimensions or weighted multi-dimension totals | `input_document_masked_mean` over the candidate report | For final dense matrix: per-dimension `rubric_task_span_mean` + `candidate_span_mean` + `pre_score_token`; for weighted-total legacy runs, keep a separate representation identity | Per-dimension labels and weighted totals are different tasks. Do not mix a weighted-total cache with future per-dimension score-token caches. |
| RuVerBench | Atomic binary rubric/check coverage over reports or agent traces | `input_document_masked_mean` over report/trace | `rubric_task_span_mean` for each check + `candidate_span_mean` + `pre_label_token` / yes-no logits; optionally aggregate check-level B views to task-level summaries | These are binary check decisions, not numeric scores. The check text itself is a major part of the task distribution, so keeping the check/rubric span is important. |
| BiGGen-Bench | Human score or frozen rubric score for candidate responses; some cross-task labels may be missing | `input_document_masked_mean` over candidate response | `rubric_task_span_mean` + `candidate_span_mean` + `pre_score_token`; separate metadata for human-GT vs teacher-filled labels | Instance-specific natural unit tests/rubrics make the rubric view important. Teacher-completed cross-task rows must not be mixed with original human-score rows without track metadata. |
| FLASK | Skill-specific 1–5 GPT-4 teacher scores; multi-domain memberships share the same response-skill label | `input_document_masked_mean` over `candidate_response` | Final score-head cache should use `rubric_task_span_mean` + `candidate_span_mean` + `pre_score_token` / score logits. The existing `artifacts/flask_full_b_space/` cache is `judge_prompt_masked_mean` and should be treated as auxiliary or an explicit ablation. | FLASK task/rubric is central, but the candidate answer also carries domain/style shift. The uploaded mean-pooled B-space is useful, but it is not the final decision-state representation if the protocol moves to score-token features. |
| Prometheus | Custom-rubric teacher scores over original responses | `input_document_masked_mean` over `orig_response` | `rubric_task_span_mean` + `candidate_span_mean` + `pre_score_token`; optionally score logits | Free-form criteria can dominate behavior. Domain/task labels and teacher scores remain metadata and must never enter the prompt as labels. |

Acceptance rule: cache metadata must include `representation_kind` or `representation_bundle`, `feature_scope`, `prompt_template_version`, selected layers, truncation policy, span-boundary policy, answer-prefix text, and the exact token-selection or pooling formula. A cache containing only `judge_prompt_masked_mean` must not be silently consumed where a config asks for `pre_score_token`, `pre_label_token`, segmented span views, or logits.

### Judge Behavior OOD

The B-space detector matrix operates on the deployed Judge representation and, where required, its K-class raw logits. ASAP reuses the whitened input-document representation; SummEval uses a separate query/source/candidate representation and retains query-level logits and metrics for every Judge row. The deployed `vim` score is explicitly the residual-only variant and does not consume logits. The executable OpenOOD-style matrix covers MSP, ODIN, Energy, MaxLogit, Mahalanobis, RMD, kNN, ViM residual, ReAct, DICE, ASH-B, GEN, KL-Matching, and GradNorm. Head-transforming postprocessors consume the exact deployed affine parameters; no least-squares head reconstruction is allowed. These are repository mechanism ports, not a claim of official OpenOOD numerical equivalence; DICE, ASH, GEN, and KL-Matching retain the protocol document's pending-primary-source-verification status.

`judge_ood_selection` fits on `training_train`, calibrates empirical score thresholds on `training_calibration`, and evaluates all configured postprocessors against held-out `development` records. The deployed B-space detector is fixed to ViM residual-only; development metrics select only among configured residual rank variants by AUROC, then AUPR and FPR95. RMD, kNN, and the other OpenOOD-style methods remain reported baselines and cannot replace it. No paired-bootstrap detector-selection test is performed. The pipeline writes `judge_behavior_ood_scorer.npz`, records `score_variant=residual_only`, `uses_logits=false`, and the Judge fingerprint, and adds `judge_behavior_ood_*` fields to `sample_ood_scores.jsonl`.

Before detector selection, `representation_separability.json` reports document-level, out-of-fold logistic AUROC for every frozen layer on source-validation versus known near/far development shifts. It selects the A-space monitoring layer without reading deployment records; an undetectable shift is routed to the label safety net.

`window_drift` runs block-permutation RBF-MMD, block-aware logistic C2ST, and a scalar residual-norm KS auxiliary test on non-overlapping deployment windows. A-space retains the input-document representation for interpretation and clustering. B-main is the residual vector from the exact source-fitted residual-only ViM subspace; raw logits, probabilities, confidence, and scalar OOD scores are excluded. Formal decisions use the conservative MMD permutation p-value only; randomized tie-breaking, C2ST accuracy/Binomial p, and KS are diagnostics. ASAP uses a finite eight-window Pocock design with three consecutive rejections and checks that every planned alpha is resolvable by 1,000 permutations. Independent calibration replays complete sequential episodes and must pass the episode-level FWER Wilson-upper-bound audit; otherwise the run fails closed. At the first persistent rejection, monitoring freezes an episode snapshot and stops.

For live streams, provide the same integer `stream_order` (or `arrival_index`) on every Judge record from an input document. Without it, the pipeline uses a deterministic document-ID replay and marks that choice in `summary.json.monitoring_stream.deployment_order_source`.

After B-space persistence, the standard localizers cluster A-space document embeddings in the first rejection segment whose ViM residual status is `soft_ood` or `hard_ood`. The `hybrid` localizer instead finds HDBSCAN cores in document-aggregated B-residual direction space, then attaches only nearby HDBSCAN-noise documents within each core's frozen radius multiplier. Its prototypes and Gate/Future routing reuse that same B-direction geometry. ATC, DoC, Agreement-on-the-Line, confidence, and margin form an advisory harm-risk warning; ViM and Energy are shift evidence used for reporting/ranking and cannot trigger that warning alone. Warning is never required for a random document-level Probe. Probe remains the sole cluster-harmfulness decision. The safety net labels at least four random documents by the first 200 arrivals and at most 10 by 1,000, with budgets counted in unique documents and query ratings reported separately. Single-episode safety routing is explicitly recorded as offline replay.

Adaptation copies the deployed logistic head, freezes Qwen, reuses every confirmed harmful Probe label, replays source labels at no more than a 1:1 ratio, and optimizes the documented weighted source/deployment loss plus the sum of squared parameter displacement from the copied head, with optimizer weight decay set to zero. Gate rows are independently labeled and never used for training. Promotion requires target improvement >=0.1, a positive paired-bootstrap lower bound, source NFR <=0.05, and source QWK drop <=0.02. Accepted updates save a candidate detector/reference version, but next-episode monitoring remains disabled until a new independent, exchangeable post-update reference/calibration split passes formal calibration.

### Data, Scoring, And Monitoring

```text
input documents
  -> assign training/development/deployment role once per input_document_id
  -> expand Judge rows only after that assignment
  -> build input_document features for A-space
  -> build Judge features from input_document (ASAP) or judge_input (SummEval)
  -> fit one global document reference bank from unique training documents
  -> calibrate global document thresholds
  -> select on development documents
  -> score each deployment document once
  -> broadcast that score to related Judge rows for joined artifacts
  -> use one document event per monitoring stream position
```

Training, development, and deployment document IDs must be pairwise disjoint. Audit reports may group final results by document origin or corpus through `audit_document_group_id`; it is never available to monitoring, routing, Probe, Adapt, Gate, or future-test decisions. Per-query Judge heads and per-query Judge metrics remain valid Judge behavior; there is no per-query OOD bank or OOD threshold.

### SummEval Controlled Document OOD

SummEval is a summarization-quality dataset, not a native cross-corpus OOD benchmark. It provides a controlled held-out-document simulation:

- An article is the `input_document_text`; its stable article identity is `input_document_id`.
- Articles receive `training`, `development`, or `deployment` roles before their candidate summaries and four Judge queries are expanded.
- All Judge rows derived from one article share its document role and one document OOD score.
- Candidate summaries and query dimensions enter `judge_input_text`, not the document-OOD input. Human scores and generator indexes remain labels or audit provenance.
- Development documents select the OOD detector. Deployment documents never select preprocessing, representation, detector, threshold, or lifecycle settings.

Candidate-system comparisons and same-article generator-shift pairs can remain in experiment artifacts, but they are non-production diagnostics. They cannot select a document OOD configuration or appear as a production OOD headline result.

### Configuration Names

Document OOD configurations set `ood_definition` to `document_distribution` and use `training_document_*`, `development_document_*`, and `deployment_document_*` keys. The explicit training stages are `training_train`, `training_calibration`, `training_validation`, `training_guard`, and `training_test`; the deployment stages are `deployment_stream`, `deployment_ood_evaluation`, `deployment_gate`, and `deployment_future_test`. A retained `deployment_adapt` split is evaluation-only: Adapt consumes labels already obtained by the online Probe and requests no additional target labels. `deployment_ood_evaluation` is an offline reporting pool and is never the Probe source; Probe samples only observed documents localized to a persistent stream cluster.

Prepare the SummEval data:

```bash
python scripts/llm_judge_ood/19_prepare_llm_judge_ood_summeval.py \
  --output artifacts/llm_judge_ood_summeval/summeval_prepared_document_ood_v1.jsonl \
  --split-config configs/llm_judge_ood/summeval_document_split_profile.json
```

Prepare the Newsroom human-evaluation data:

```bash
python scripts/llm_judge_ood/23_prepare_llm_judge_ood_newsroom.py
```

The official Newsroom thin archive is retained under `datasets/raw/newsroom/`. Its
extractive density and coverage fields are not human quality labels. The prepared
LLM Judge artifact uses the public three-rater Newsroom human-evaluation mirror,
aggregates each article/system/dimension to a rounded 1-5 human-quality label,
and assigns all rows from one `input_document_id` to the same document-level split.
The aggregate file and one JSONL file per split are written under
`artifacts/llm_judge_ood_newsroom/`.

Prepare the ASAP-AES essay-quality data (the raw Kaggle download is not
redistributed by this repository):

```bash
python scripts/llm_judge_ood/24_prepare_llm_judge_ood_asap.py \
  --input datasets/raw/asap_aes/training_set_rel3.tsv \
  --output artifacts/llm_judge_ood_asap/asap_prepared_contract_v1.jsonl \
  --write-flows
```

The ASAP adapter keeps prompts 1/2 as ID, 3/4 as near shift, and 7/8 as far
shift. It writes fixed prompt-specific official-rubric range mappings, document-disjoint
split IDs, overlap audits, paired rater scores, and deterministic abrupt,
gradual, and harmless streams for five seeds. Since ASAP has no observed
arrival batch, its adapter sets `arrival_batch_id = input_document_id`: each
essay (or each controlled-flow event) is an independent permutation block;
prompt/shift identity remains audit-only. Formal result directories also write
`table0_block_calibration_audit.csv`, `table4a_probe_acceptance.csv`,
`table6_acceptance.csv`, and `table7_label_cost.csv`; see
`docs/LLM_Judge_OOD/2026-07-19_数据区块与验收表规范.md`. The dataset terms and
download license remain the responsibility of the experiment owner. Raw ASAP
data and generated experiment artifacts are not tracked by Git.

The formal ASAP source allocation is 33% `training_train`, 35%
`training_calibration`, 8% `training_validation`, 6% `training_guard`, 6%
`training_test`, and 12% ID `deployment_stream`. Deployment also has disjoint
`deployment_ood_evaluation`, `deployment_adapt`, `deployment_gate`, and
`deployment_future_test` pools. `deployment_ood_evaluation` is offline
evaluation only and is not sampled by Probe. `deployment_adapt` is retained
for evaluation compatibility, but adaptation requests no new labels: it reuses
confirmed harmful Probe labels.

Formal lifecycle candidates require at least 20 valid calibration windows in
both A and B, five usable C2ST folds, no insufficient target blocks, and no
deployment data in selection. With the current ASAP source size, a 50-document
window is supported; 100/200/500-document candidates are normally rejected by
the calibration preflight unless more independent source/calibration documents
are provided. Lowering the formal minimum or reusing deployment documents is
not allowed.

Run the frozen ASAP base split with its separate, aligned A and B caches:

```bash
.venv/bin/python scripts/llm_judge_ood/13_run_llm_judge_ood_end_to_end.py \
  --config configs/llm_judge_ood/llm_judge_ood_asap.json \
  --output-dir artifacts/llm_judge_ood_asap/document_ood_contract_v1
```

Do not pass one input-document cache as both `--document-hidden-feature-path`
and `--judge-hidden-feature-path`. The retained controlled-flow scripts and
their shared-cache results predate the rubric-aware B contract; the current
scope and missing flow B-cache step are recorded in the HiddenState runbook.

### Feature Caches

The legacy five-dataset A/B extraction registry is
`configs/llm_judge_ood/llm_judge_ood_hiddenstate_datasets.json`. The
cross-domain/rubric benchmark registry for LongJudgeBench, RuVerBench,
BiGGen-Bench, FLASK, and Prometheus is
`configs/llm_judge_ood/llm_judge_ood_benchmark_ground_truth_hiddenstate.json`.

Prepare the benchmark-ground-truth adapters, then extract aligned A and B
hidden-state caches:

```bash
.venv/bin/python scripts/llm_judge_ood/34_prepare_llm_judge_ood_hidden_datasets.py \
  --config configs/llm_judge_ood/llm_judge_ood_benchmark_ground_truth_hiddenstate.json

.venv/bin/python scripts/llm_judge_ood/35_extract_all_hiddenstate_ab.py \
  --config configs/llm_judge_ood/llm_judge_ood_benchmark_ground_truth_hiddenstate.json \
  --model-path /home/ubuntu/models/qwen3.5-4b \
  --local-files-only
```

The adapters have been checked against official files for LongJudgeBench,
RuVerBench, BiGGen-Bench, FLASK, and Prometheus. Before launching a GPU
extraction, choose the representation bundle from the policy table above. The
retained script-20/script-35 path currently writes single-view `masked_mean`
caches; use it for A-space and for auxiliary B-space diagnostics only. Final
scoring/classification heads for the benchmark-ground-truth datasets should use
a bundle extractor that records the relevant span means, `pre_score_token` or
`pre_label_token`, and optional allowed-answer logits unless the experiment
explicitly declares a `judge_prompt_masked_mean` ablation.

This registry uses `truncation_strategy=head_tail`. For long Judge inputs the
frozen v2 templates put the rubric before the instruction, then retain the
first and last halves of the 2048-token window. Cache metadata records and
script 35 validates this policy, so a right-truncated cache cannot be silently
reused as a head-tail cache.

Input-document caches contain one row per unique document and have shape:

```text
[num_unique_input_documents, num_selected_layers, 2560]
```

The loader validates the document fingerprint and broadcasts each document row to related Judge records. Judge-input caches instead contain one row per Judge record, validate the record fingerprint, and align by `sample_id`, `query_id`, and `input_document_id`. Cache metadata records the exact model identity, freeze/eval state, representation kind, token-selection or pooling rule, truncation length and strategy, source fingerprint, and the corresponding alignment identifiers.

Build the document/OOD cache required by every configuration:

```bash
.venv/bin/python scripts/llm_judge_ood/20_prepare_llm_judge_ood_hidden.py \
  --input artifacts/llm_judge_ood_summeval/summeval_prepared_document_ood_v1.jsonl \
  --model-path /home/ubuntu/models/qwen3.5-4b \
  --model-id Qwen/Qwen3.5-4B \
  --local-files-only \
  --output artifacts/llm_judge_ood_summeval/gpu_qwen_hidden/qwen3_5_4b_input_document_masked_mean_v1.npz \
  --feature-scope input_document \
  --pooling masked_mean \
  --max-length 2048 \
  --torch-dtype bfloat16
```

SummEval configurations also require a second, response-aware Judge cache. Existing SummEval `input_document` Judge caches do not satisfy this contract and must be re-extracted. The command below produces the legacy/auxiliary `judge_prompt_masked_mean` cache; for final score-head runs, replace it with a bundle extractor that uses the same prompt template and records `candidate_span_mean`, query/rubric span metadata, `pre_score_token`, and optional score logits:

```bash
.venv/bin/python scripts/llm_judge_ood/20_prepare_llm_judge_ood_hidden.py \
  --input artifacts/llm_judge_ood_summeval/summeval_prepared_document_ood_v1.jsonl \
  --model-path /home/ubuntu/models/qwen3.5-4b \
  --model-id Qwen/Qwen3.5-4B \
  --local-files-only \
  --output artifacts/llm_judge_ood_summeval/gpu_qwen_hidden/qwen3_5_4b_judge_input_masked_mean_v1.npz \
  --feature-scope judge_input \
  --pooling masked_mean \
  --max-length 2048 \
  --torch-dtype bfloat16
```

On a 96 GB RTX PRO 6000, the optimized resumable ASAP extraction is a single
command. It preserves the same cache contract while using selected-layer hooks
and length-bucketed dynamic batches:

```bash
bash scripts/llm_judge_ood/27_prepare_asap_hidden_rtx_pro_6000.sh
```

Run a SummEval GPU configuration only after both its document and Judge-input caches exist:

```bash
python scripts/llm_judge_ood/13_run_llm_judge_ood_end_to_end.py \
  --config configs/llm_judge_ood/llm_judge_ood_summeval_gpu.json \
  --output-dir artifacts/llm_judge_ood_summeval/document_ood_qwen3_5_4b
```

For the SummEval kNN diagnostic sweep, run `configs/llm_judge_ood/llm_judge_ood_summeval_document_knn_grid.json`. It fixes the scorer family to kNN and searches `k=[1, 3, 5, 7, 10, 15]`; the production configurations use the narrower `k=[3, 5, 7, 10]` grid. Every k must be no greater than the unique training-document bank. Invalid or duplicated values fail instead of being silently clipped.

### Evaluation Baselines

The paper comparison baselines are implemented as a separate evaluation
runner. Given an NPZ containing `source_features`, `target_features`, and
optionally `source_logits`/`target_logits`, it reports detection rate versus
target sample size for `NoRed`, `PCA`, `SRP`, `TAE`, `UAE`, `BBSDs`, `BBSDh`,
and `Classif`. All representations are fitted on source rows only; target
labels are never consumed by the detector. The same runner can add operational
strategy and harmfulness label-cost curves from JSON inputs.

```bash
python scripts/llm_judge_ood/25_run_llm_judge_ood_baselines.py \
  --features artifacts/llm_judge_ood_baselines/features.npz \
  --output artifacts/llm_judge_ood_baselines/results.json \
  --sample-sizes 10 20 50 100 200 500 1000 10000
```

`results.json` contains per-seed p-values, detection rates, false-alarm and
detection-delay summaries for `no-monitor`, `always-retrain`,
`confidence-only`, and `final-accuracy-only`, plus the label-cost/error curve
when those optional inputs are supplied. These are evaluation baselines, not
additional production decision paths.

### Outputs And Interpretation

The end-to-end output directory contains `summary.json`, `sample_ood_scores.jsonl`, preprocessing artifacts, selected detector metadata, Judge diagnostics, and tables. A score row must expose `input_document_id` and document-level OOD status. Repeated Judge rows for the same document carry the same A-space score only because the document scorer broadcasts one decision back to those rows; their B-space Judge-behavior scores remain record-level.

`window_drift.jsonl` records the conservative MMD p-value used for decisions, the randomized MMD p-value and C2ST/KS diagnostics, A/B status, alpha-spending allocation, and B-space persistence decision. `summary.json.dual_space_drift.first_persistent_episode` records the confirmation window, visible prefix, rejection segment, and alpha state. `document_cluster_lifecycle.jsonl` contains only localized clusters created from that first episode.

`summary.json.ood_selection.candidate_results` records each candidate's effective embedding dimension, training-bank size, calibration-document count, and development-document count. Threshold metadata also records `quantile_resolution=1/calibration_count`: quantiles are empirical rank cutoffs, not a claim of an exact false-positive rate when the calibration set is small.

The lifecycle artifact is `document_cluster_lifecycle.jsonl`. Every event denotes a discovered document cluster and counts each input document once.

### Historical Candidate-System Diagnostics

Some retained artifacts and older notes compare candidate systems, generator indexes, candidate pooling, or same-article generator changes. They may help audit a Judge experiment, but they do not measure deployment input-document OOD. Treat them as non-production diagnostics only. The documents in `docs/LLM_Judge_OOD/` label these historical sections explicitly.

## Lightweight Verification

This research repository does not retain a regression test suite. Before a
full experiment, compile the source, validate the checked-in JSON configs, and
run one relevant full-data workflow against its real feature cache:

```bash
python -m compileall -q src scripts
python -c 'import json, pathlib; [json.loads(p.read_text()) for p in pathlib.Path("configs").rglob("*.json")]'
python scripts/llm_judge_ood/13_run_llm_judge_ood_end_to_end.py \
  --config configs/llm_judge_ood/llm_judge_ood_asap.json \
  --output-dir artifacts/llm_judge_ood_asap/document_ood_contract_v1
```

Full frozen-Qwen and licensed ASAP runs remain explicit experiment steps because
their inputs are not redistributed with the repository.

## Formal Result Acceptance

Static tables establish only component-level feasibility. A formal static run
must report source-test Judge QWK/MAE, near/far representation AUROC,
near/far document-OOD AUROC, and ViM residual alongside RMD, kNN, and MSP
controls. Its rank and every other detector or lifecycle setting must be selected without
deployment records.

A harmful predicted cluster is accepted only when all of the following hold:

```text
harm_delta > 0.15
AND 95% lower confidence bound > 0.15
AND BH-FDR rejects the cluster null
AND status = harmful
```

Probe uses at most 20 independently routed documents per predicted cluster.
Adapt reuses those labels and requests zero additional target labels. Gate uses
an independent pool and accepts an update only when:

```text
target excess-error improvement >= 0.10
AND paired-bootstrap 95% lower bound > 0
AND source NFR <= 0.05
AND source guard QWK drop <= 0.02
```

Passing Gate is not sufficient to claim a beneficial update: independent
`deployment_future_test` MAE must decrease and QWK must increase. All requested
and reused labels must be reported separately, with the safety-net cost counted
outside the normal single-cluster Probe/Gate budget.

The minimum formal evidence that the end-to-end path executes is one harmful
abrupt-far run completing Detect -> Persistent -> Cluster -> Probe -> Adapt ->
Gate -> Future, plus one harmless run that does not update. The minimum matrix
for an "initially supported" result is nine formal runs: abrupt-near,
abrupt-far, and harmless for seeds 42/43/44. Near and far must each be detected
in at least two of three seeds and show Future improvement when harmful;
harmless must avoid updates in at least two of three seeds. Every run must have
valid formal calibration and no nominal fallback.

## Current Evidence Boundary

The current workspace has a complete 9,371-document ASAP cache. Formal evidence
is the three-scenario by three-seed matrix under
`artifacts/llm_judge_ood_asap/formal_acceptance_v5/`. Residual-only ViM reaches
overall AUROC/AUPR/FPR95 0.9961/0.9921/0.0120 at rank 64, effectively tied with
Mahalanobis and well ahead of Full ViM, RMD, kNN, Energy, and MSP. B-residual
MMD has H0 FPR 0.030 at alpha 0.05 and 0.005 at alpha 0.01 over 200 trials; its
Near/Far power is 1.0 for every tested window size from 50 through 500.

The finite H=8 Pocock monitor has 0/200 H0 episode alerts (Wilson 95% upper
0.01885). Near and Far are persistent in 6/6 runs and at least one harmful
cluster is confirmed in all six. Harmless is a real distribution shift and is
therefore persistent in 3/3; Probe returns `uncertain`, not `benign`, but no
Harmless run updates. The four-condition Gate accepts only Near seed 43, whose
routed Future MAE improves from 1.812 to 1.050. Near seeds 42/44 are rolled back
for NFR 0.0605/0.0512, and all Far candidates are rolled back after negative
target Gate improvement.

Accordingly, detector, MMD, sequential calibration, and fail-closed safety are
supported; the complete algorithm is not yet accepted because HDBSCAN routing
recall, Harmless benign confirmation, and Far/multi-seed adaptation power remain
below the formal targets. The authoritative report includes all six component
tables and the exact evidence boundary. Regenerate the cached report with:

```bash
.venv/bin/python scripts/llm_judge_ood/31_run_formal_acceptance.py \
  --trials 200 --permutations 1000 --workers 4
```

ASAP also reuses the document representation as the Judge penultimate feature,
so it cannot by itself establish general dual-space complementarity. SummEval
uses a separate query/document/candidate Judge representation and is the
available repository setting for A-only, B-only, and A+B evidence. SummEval
remains a held-out-document simulation, not proof of unrestricted cross-corpus
generalization.

The implementation covers the ASAP preparation and audit contracts,
source-only PCA, fixed-family ViM residual selection, OpenOOD-style controls,
block-aware MMD/C2ST/KS drift auditing, small-sample harmfulness, gated
adaptation, B-reference versioning, result-table generation, and the baseline
runner. Semantic-drift diagnosis, persistent Gate-failure state across
episodes, and automatic multi-episode retraining are explicitly unsupported.
Power analysis remains an offline stage and uses the source-only,
block-preserving `power_reference_max_samples=512` cap.
