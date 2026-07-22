# Benchmark Ground-Truth HiddenState Notebooks

These notebooks run the repository's benchmark-ground-truth preparation and A/B
hidden-state extraction one dataset at a time on Google Colab, Kaggle, or
ModelScope Notebook.

- [LongJudgeBench](https://colab.research.google.com/github/oceanusXXD/Dual-Space-Prototypical-Probing-for-OOD-Robust-LLM-Evaluation/blob/main/notebooks/llm_judge_ood_benchmark_hiddenstate_longjudgebench.ipynb)
- [RuVerBench](https://colab.research.google.com/github/oceanusXXD/Dual-Space-Prototypical-Probing-for-OOD-Robust-LLM-Evaluation/blob/main/notebooks/llm_judge_ood_benchmark_hiddenstate_ruverbench.ipynb)
- [BiGGen-Bench](https://colab.research.google.com/github/oceanusXXD/Dual-Space-Prototypical-Probing-for-OOD-Robust-LLM-Evaluation/blob/main/notebooks/llm_judge_ood_benchmark_hiddenstate_biggen_bench.ipynb)
- [FLASK](https://colab.research.google.com/github/oceanusXXD/Dual-Space-Prototypical-Probing-for-OOD-Robust-LLM-Evaluation/blob/main/notebooks/llm_judge_ood_benchmark_hiddenstate_flask.ipynb)
- [Prometheus](https://colab.research.google.com/github/oceanusXXD/Dual-Space-Prototypical-Probing-for-OOD-Robust-LLM-Evaluation/blob/main/notebooks/llm_judge_ood_benchmark_hiddenstate_prometheus.ipynb)

Each notebook expects the raw files to be staged under the paths declared in
`configs/llm_judge_ood/llm_judge_ood_benchmark_ground_truth_hiddenstate.json`. If the raw files are mounted elsewhere, set `RAW_DATA_SOURCE` in the
first code cell. An empty value auto-detects
`MyDrive/llm_judge_ood_data/datasets/raw` on Colab,
`/kaggle/input/llm-judge-ood-raw/datasets/raw` on Kaggle, or
`/mnt/workspace/llm_judge_ood_data/datasets/raw` on ModelScope.

LongJudgeBench, RuVerBench, BiGGen-Bench, FLASK, and Prometheus automatically
download pinned official-source revisions when local raw files are absent.

The notebooks default to `MODEL_PATH = ""` and `MODEL_DOWNLOAD_BACKEND =
"huggingface"`, which downloads the pinned `Qwen/Qwen3.5-4B` revision into the
notebook cache before extraction. On ModelScope this cache defaults to
`/mnt/workspace/.cache/huggingface`. To use a mounted local snapshot instead,
set `MODEL_PATH` to that directory and set `LOCAL_FILES_ONLY = True`. The
`MODEL_DOWNLOAD_BACKEND = "modelscope"` option is available for mirror testing,
but the Hugging Face backend remains the default because the hidden-state cache
contract records the exact Hugging Face Qwen revision.

Outputs are archived to `MyDrive/llm_judge_ood_outputs` on Colab,
`/kaggle/working/llm_judge_ood_outputs` on Kaggle, and
`/mnt/workspace/llm_judge_ood_outputs` on ModelScope. Set `DOWNLOAD_AFTER_RUN =
True` to start a browser download after a Colab run finishes.
