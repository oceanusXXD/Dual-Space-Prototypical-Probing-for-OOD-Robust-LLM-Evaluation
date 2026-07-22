#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_DIR = ROOT / "notebooks"
REPO_URL = "https://github.com/oceanusXXD/Dual-Space-Prototypical-Probing-for-OOD-Robust-LLM-Evaluation.git"
GITHUB_REPO = "oceanusXXD/Dual-Space-Prototypical-Probing-for-OOD-Robust-LLM-Evaluation"
CONFIG = "configs/llm_judge_ood/llm_judge_ood_benchmark_ground_truth_hiddenstate.json"

DATASETS = {
    "longjudgebench": "LongJudgeBench",
    "ruverbench": "RuVerBench",
    "biggen_bench": "BiGGen-Bench",
    "flask": "FLASK",
    "prometheus": "Prometheus",
}

OFFICIAL_SOURCES = {
    "longjudgebench": {
        "revision": "f35d00d8949dbc34cae17f100168c12454400ad3",
        "files": [
            {
                "url": "https://raw.githubusercontent.com/cjj826/LongJudgeBench/"
                "f35d00d8949dbc34cae17f100168c12454400ad3/"
                "data_standardized/" + filename,
                "path": f"datasets/raw/longjudgebench/data_standardized/{filename}",
            }
            for filename in (
                "deepresearch_bench.jsonl",
                "realdr.jsonl",
            )
        ],
    },
    "ruverbench": {
        "revision": "4e2992e3fa85448b4ba7a85741b65e09e4bec016",
        "files": [
            {
                "url": "https://raw.githubusercontent.com/THU-KEG/RuVerBench/"
                "4e2992e3fa85448b4ba7a85741b65e09e4bec016/data/benchmark/"
                + filename,
                "path": f"datasets/raw/ruverbench/data/benchmark/{filename}",
            }
            for filename in (
                "deepresearch_dataset.json",
                "deepresearch_responses.json",
                "deepresearch_labels.json",
            )
        ],
    },
    "biggen_bench": {
        "revision": "3a9589efbad801052bb2e153b44ce027498c27e4",
        "files": [
            {
                "url": "https://huggingface.co/datasets/prometheus-eval/BiGGen-Bench/resolve/"
                "3a9589efbad801052bb2e153b44ce027498c27e4/data/test-00000-of-00001.parquet",
                "path": "datasets/raw/biggen_bench/human_eval.parquet",
            }
        ],
    },
    "flask": {
        "revision": "3b4e22bc34aa9dc15ea0c51be3cbf5f3c8b1b5e5",
        "files": [
            {
                "url": "https://raw.githubusercontent.com/kaistAI/FLASK/"
                "3b4e22bc34aa9dc15ea0c51be3cbf5f3c8b1b5e5/gpt_review/outputs/"
                "gpt4_review.jsonl",
                "path": "datasets/raw/flask/gpt_review/outputs/gpt4_review.jsonl",
            }
        ],
    },
    "prometheus": {
        "revision": "22a339f961c40b07261bea44ec10849c7440b75f",
        "files": [
            {
                "url": "https://huggingface.co/datasets/prometheus-eval/Feedback-Collection/resolve/"
                "22a339f961c40b07261bea44ec10849c7440b75f/new_feedback_collection.json",
                "path": "datasets/raw/prometheus/new_feedback_collection.json",
            }
        ],
    },
}


def main() -> None:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    for dataset, title in DATASETS.items():
        write_notebook(dataset, title)
    write_readme()
    print(json.dumps({"written": sorted(path.name for path in NOTEBOOK_DIR.glob("*.ipynb"))}, indent=2))


def write_notebook(dataset: str, title: str) -> None:
    cells = [
        markdown(
            f"# {title} A/B Hidden-State Extraction\n\n"
            "This notebook prepares one benchmark-ground-truth dataset and extracts its aligned "
            "A-space input-document cache plus B-space judge-input cache. It is intended for "
            "Google Colab, Kaggle, or ModelScope GPU notebooks.\n\n"
            "Expected input files must be staged under the repository's `datasets/raw/...` "
            "paths described in the config. Set `RAW_DATA_SOURCE` to a mounted Drive/Kaggle/"
            "ModelScope directory if the raw files live outside the cloned repo."
        ),
        code(parameters_cell(dataset)),
        code(setup_cell()),
        code(preflight_cell()),
        code(run_prepare_cell()),
        code(run_extract_cell()),
        code(verify_and_archive_cell()),
    ]
    for index, cell in enumerate(cells):
        cell["id"] = f"{dataset}-{index}"
    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {
                "gpuType": "T4",
                "provenance": [],
            },
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "pygments_lexer": "ipython3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path = NOTEBOOK_DIR / f"llm_judge_ood_benchmark_hiddenstate_{dataset}.ipynb"
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def markdown(source: str) -> dict[str, object]:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip("\n").splitlines(keepends=True),
    }


def parameters_cell(dataset: str) -> str:
    official_source = json.dumps(OFFICIAL_SOURCES.get(dataset), indent=2)
    return f"""
# Parameters
DATASET = "{dataset}"
REPO_URL = "{REPO_URL}"
BRANCH_OR_COMMIT = "main"
CONFIG = "{CONFIG}"

# Leave MODEL_PATH empty to download the pinned Qwen snapshot into the notebook cache.
# Set it to a mounted local snapshot path when reusing an existing model directory.
MODEL_PATH = ""
MODEL_DOWNLOAD_BACKEND = "huggingface"  # "huggingface" or "modelscope"
HF_MODEL_ID = "Qwen/Qwen3.5-4B"
HF_MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
HUGGINGFACE_CACHE_DIR = ""  # ModelScope default: /mnt/workspace/.cache/huggingface
MOUNT_GOOGLE_DRIVE = False  # Set True only when RAW_DATA_SOURCE or OUTPUT_ARCHIVE_DIR uses Drive.
DISABLE_CUDNN_CONV1D = True  # T4-safe fallback for Qwen3.5 linear-attention Conv1d.

# Optional ModelScope hub mirror. Keep the Hugging Face backend unless you intentionally
# want to test the ModelScope mirror against the repository's strict Qwen revision contract.
MODEL_SCOPE_MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_SCOPE_MODEL_REVISION = ""
MODEL_SCOPE_CACHE_DIR = ""  # ModelScope default: /mnt/workspace/.cache/modelscope
LOCAL_FILES_ONLY = False

# Optional: a mounted directory containing either datasets/raw/... or the raw dataset folders directly.
# Colab example: "/content/drive/MyDrive/llm_judge_ood_data/datasets/raw"
# Kaggle example: "/kaggle/input/llm-judge-ood-raw/datasets/raw"
# ModelScope example: "/mnt/workspace/llm_judge_ood_data/datasets/raw"
RAW_DATA_SOURCE = ""
AUTO_DOWNLOAD_OFFICIAL_DATA = True
OFFICIAL_SOURCE = {official_source}

# Optional: where the final .tar.gz is written. Defaults to the cloned repository
# on Colab, /kaggle/working on Kaggle, and /mnt/workspace on ModelScope.
OUTPUT_ARCHIVE_DIR = ""
DOWNLOAD_AFTER_RUN = False

RUN_PREPARE = True
RUN_EXTRACT = True
OVERWRITE = False

# Conservative T4 defaults. Increase only after a successful runtime-specific smoke.
BATCH_SIZE = 4
MAX_BATCH_TOKENS = 2048
PAD_TO_MULTIPLE_OF = 1
"""


def setup_cell() -> str:
    return r"""
import glob
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path


def run(cmd, cwd=None, env=None, attempts=1):
    printable = " ".join(str(part) for part in cmd)
    for attempt in range(1, attempts + 1):
        print(f"+ {printable} (attempt {attempt}/{attempts})", flush=True)
        try:
            subprocess.run([str(part) for part in cmd], cwd=cwd, env=env, check=True)
            return
        except subprocess.CalledProcessError:
            if attempt == attempts:
                raise
            time.sleep(min(30, 5 * attempt))


def in_colab():
    return not Path("/kaggle/working").exists() and Path("/var/colab/hostname").exists()


def in_kaggle():
    return Path("/kaggle/working").exists()


def in_modelscope():
    return (not in_colab()) and (not in_kaggle()) and Path("/mnt/workspace").exists()


BASE_DIR = Path(
    "/kaggle/working"
    if in_kaggle()
    else "/content"
    if in_colab()
    else "/mnt/workspace"
    if in_modelscope()
    else "."
).resolve()
REPO_NAME = "Dual-Space-Prototypical-Probing-for-OOD-Robust-LLM-Evaluation"
CURRENT_DIR = Path.cwd().resolve()
REPO_DIR = CURRENT_DIR if (CURRENT_DIR / "scripts/llm_judge_ood").exists() else BASE_DIR / REPO_NAME

if in_modelscope():
    os.environ.setdefault("HF_HOME", str(Path(HUGGINGFACE_CACHE_DIR or "/mnt/workspace/.cache/huggingface").expanduser()))
    os.environ.setdefault("MODELSCOPE_CACHE", str(Path(MODEL_SCOPE_CACHE_DIR or "/mnt/workspace/.cache/modelscope").expanduser()))

if DISABLE_CUDNN_CONV1D:
    os.environ["LLM_JUDGE_OOD_DISABLE_CUDNN_CONV1D"] = "1"

if in_colab() and MOUNT_GOOGLE_DRIVE:
    from google.colab import drive  # type: ignore
    drive.mount("/content/drive", force_remount=False)

if not REPO_DIR.exists():
    run(["git", "clone", "--depth", "1", REPO_URL, str(REPO_DIR)], attempts=3)
else:
    run(["git", "fetch", "--depth", "1", "origin"], cwd=REPO_DIR, attempts=3)

run(["git", "fetch", "--depth", "1", "origin", BRANCH_OR_COMMIT], cwd=REPO_DIR, attempts=3)
run(["git", "checkout", "FETCH_HEAD"], cwd=REPO_DIR)

os.chdir(REPO_DIR)
print("Repository:", REPO_DIR)
run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], attempts=3)
if str(MODEL_DOWNLOAD_BACKEND).lower() == "modelscope":
    run([sys.executable, "-m", "pip", "install", "-q", "modelscope"], attempts=3)

try:
    run(["nvidia-smi"])
except Exception as error:
    print("nvidia-smi failed; make sure the notebook runtime has a GPU:", error)
"""


def preflight_cell() -> str:
    return r"""
cfg = json.loads(Path(CONFIG).read_text(encoding="utf-8"))
spec = cfg["datasets"][DATASET]

raw_data_source = RAW_DATA_SOURCE
if not raw_data_source:
    candidates = [
        Path("/content/drive/MyDrive/llm_judge_ood_data/datasets/raw"),
        Path("/kaggle/input/llm-judge-ood-raw/datasets/raw"),
        Path("/mnt/workspace/llm_judge_ood_data/datasets/raw"),
        Path("/mnt/workspace/datasets/raw"),
    ]
    raw_data_source = next((str(path) for path in candidates if path.exists()), "")

if raw_data_source:
    src = Path(raw_data_source).expanduser()
    if (src / "datasets" / "raw").exists():
        src = src / "datasets" / "raw"
    dst = REPO_DIR / "datasets" / "raw"
    print(f"Copying raw data from {src} to {dst}")
    shutil.copytree(src, dst, dirs_exist_ok=True)

def missing_input_patterns():
    return [
        pattern
        for pattern in spec.get("input_paths", [spec.get("input_path")])
        if pattern and not glob.glob(str(REPO_DIR / pattern))
    ]


missing = missing_input_patterns()
if missing and AUTO_DOWNLOAD_OFFICIAL_DATA and OFFICIAL_SOURCE:
    print("Downloading official data revision:", OFFICIAL_SOURCE["revision"])
    for item in OFFICIAL_SOURCE["files"]:
        destination = REPO_DIR / item["path"]
        if destination.exists():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".download")
        request = urllib.request.Request(item["url"], headers={"User-Agent": "llm-judge-ood-notebook"})
        for attempt in range(1, 4):
            try:
                with urllib.request.urlopen(request) as response, temporary.open("wb") as output:
                    shutil.copyfileobj(response, output)
                break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(min(30, 5 * attempt))
        os.replace(temporary, destination)
        print("Downloaded", destination.relative_to(REPO_DIR))
    missing = missing_input_patterns()

for pattern in spec.get("input_paths", [spec.get("input_path")]):
    if not pattern:
        continue
    matches = sorted(glob.glob(str(REPO_DIR / pattern)))
    if matches:
        print("Matched", pattern, "->", len(matches), "file(s)")

if missing:
    raise FileNotFoundError(
        "Missing raw dataset path(s): "
        + ", ".join(missing)
        + "\nStage the files under datasets/raw/... or set RAW_DATA_SOURCE. "
        + "This dataset has no verified public source configured when OFFICIAL_SOURCE is None."
    )

print("Dataset:", DATASET)
print("Prepared path:", spec["prepared_path"])
print("A cache:", spec["a_cache_path"])
print("B cache:", spec["b_cache_path"])

def _default_hf_cache_dir():
    if in_modelscope():
        return "/mnt/workspace/.cache/huggingface"
    if in_kaggle():
        return "/kaggle/working/.cache/huggingface"
    if in_colab():
        return "/content/.cache/huggingface"
    return "/tmp/llm_judge_ood_huggingface_cache"


def _default_modelscope_cache_dir():
    if in_modelscope():
        return "/mnt/workspace/.cache/modelscope"
    return "/tmp/llm_judge_ood_modelscope_cache"


EFFECTIVE_LOCAL_FILES_ONLY = bool(LOCAL_FILES_ONLY)
if MODEL_PATH:
    MODEL_PATH = str(Path(MODEL_PATH).expanduser())
else:
    backend = str(MODEL_DOWNLOAD_BACKEND).lower()
    if backend == "huggingface":
        from huggingface_hub import snapshot_download

        cache_dir = str(Path(HUGGINGFACE_CACHE_DIR or _default_hf_cache_dir()).expanduser())
        os.environ.setdefault("HF_HOME", cache_dir)
        model_dir = Path(cache_dir) / "models" / "qwen3.5-4b"
        MODEL_PATH = snapshot_download(
            repo_id=HF_MODEL_ID,
            revision=HF_MODEL_REVISION,
            local_dir=str(model_dir),
            local_files_only=bool(LOCAL_FILES_ONLY),
        )
        EFFECTIVE_LOCAL_FILES_ONLY = True
    elif backend == "modelscope":
        from modelscope import snapshot_download

        cache_dir = str(Path(MODEL_SCOPE_CACHE_DIR or _default_modelscope_cache_dir()).expanduser())
        os.environ.setdefault("MODELSCOPE_CACHE", cache_dir)
        kwargs = {"model_id": MODEL_SCOPE_MODEL_ID, "cache_dir": cache_dir}
        if MODEL_SCOPE_MODEL_REVISION:
            kwargs["revision"] = MODEL_SCOPE_MODEL_REVISION
        MODEL_PATH = snapshot_download(**kwargs)
        EFFECTIVE_LOCAL_FILES_ONLY = True
        print(
            "ModelScope backend selected. If strict Qwen identity validation fails, "
            "switch MODEL_DOWNLOAD_BACKEND back to 'huggingface' or provide a verified local snapshot."
        )
    else:
        raise ValueError("MODEL_DOWNLOAD_BACKEND must be 'huggingface' or 'modelscope'")

print("Model backend:", MODEL_DOWNLOAD_BACKEND)
print("Model path:", MODEL_PATH)
print("Local files only:", EFFECTIVE_LOCAL_FILES_ONLY)
"""


def run_prepare_cell() -> str:
    return r"""
if RUN_PREPARE:
    run([
        sys.executable,
        "scripts/llm_judge_ood/34_prepare_llm_judge_ood_hidden_datasets.py",
        "--config",
        CONFIG,
        "--dataset",
        DATASET,
    ], cwd=REPO_DIR)
else:
    print("Skipping prepare step")
"""


def run_extract_cell() -> str:
    return r"""
if RUN_EXTRACT:
    cmd = [
        sys.executable,
        "scripts/llm_judge_ood/35_extract_all_hiddenstate_ab.py",
        "--config",
        CONFIG,
        "--dataset",
        DATASET,
        "--model-path",
        MODEL_PATH,
        "--batch-size",
        str(BATCH_SIZE),
        "--max-batch-tokens",
        str(MAX_BATCH_TOKENS),
        "--pad-to-multiple-of",
        str(PAD_TO_MULTIPLE_OF),
    ]
    if EFFECTIVE_LOCAL_FILES_ONLY:
        cmd.append("--local-files-only")
    if OVERWRITE:
        cmd.append("--overwrite")
    started = time.perf_counter()
    run(cmd, cwd=REPO_DIR)
    print(f"Elapsed wall time: {(time.perf_counter() - started) / 3600:.2f} hours")
else:
    print("Skipping extraction step")
"""


def verify_and_archive_cell() -> str:
    return r"""
import numpy as np

cfg = json.loads(Path(CONFIG).read_text(encoding="utf-8"))
spec = cfg["datasets"][DATASET]
paths = [
    Path(spec["prepared_path"]),
    Path(spec["a_cache_path"]),
    Path(spec["a_cache_path"]).with_suffix(".metadata.json"),
    Path(spec["b_cache_path"]),
    Path(spec["b_cache_path"]).with_suffix(".metadata.json"),
]

for path in paths:
    if not path.exists():
        raise FileNotFoundError(path)
    print(path, f"{path.stat().st_size / (1024 ** 2):.2f} MiB")

for cache_key in ["a_cache_path", "b_cache_path"]:
    cache_path = Path(spec[cache_key])
    metadata = json.loads(cache_path.with_suffix(".metadata.json").read_text(encoding="utf-8"))
    with np.load(cache_path, allow_pickle=True) as payload:
        feature_key = "features" if "features" in payload.files else payload.files[0]
        print(cache_key, "shape=", payload[feature_key].shape, "records=", metadata["num_records"])
        print("  feature_scope=", metadata["feature_scope"], "truncation=", metadata["truncation_strategy"])

if OUTPUT_ARCHIVE_DIR:
    archive_dir = Path(OUTPUT_ARCHIVE_DIR).expanduser()
elif in_colab() and Path("/content/drive/MyDrive").exists():
    archive_dir = Path("/content/drive/MyDrive/llm_judge_ood_outputs")
elif in_kaggle():
    archive_dir = Path("/kaggle/working/llm_judge_ood_outputs")
elif in_modelscope():
    archive_dir = Path("/mnt/workspace/llm_judge_ood_outputs")
else:
    archive_dir = REPO_DIR / "artifacts" / "download_archives"

archive_dir.mkdir(parents=True, exist_ok=True)
archive_path = archive_dir / f"{DATASET}_benchmark_gt_hiddenstate_ab.tar.gz"
dataset_artifact_dir = Path(spec["a_cache_path"]).parent

with tarfile.open(archive_path, "w:gz") as tar:
    tar.add(Path(spec["prepared_path"]), arcname=Path(spec["prepared_path"]).name)
    tar.add(dataset_artifact_dir, arcname=dataset_artifact_dir.name)

print("Archive written:", archive_path)
print("On Kaggle, download it from the Output panel. On Colab or ModelScope, use the path above.")

if DOWNLOAD_AFTER_RUN and in_colab():
    from google.colab import files  # type: ignore
    files.download(str(archive_path))
"""


def write_readme() -> None:
    links = "\n".join(
        f"- [{title}](https://colab.research.google.com/github/{GITHUB_REPO}/blob/main/"
        f"notebooks/llm_judge_ood_benchmark_hiddenstate_{dataset}.ipynb)"
        for dataset, title in DATASETS.items()
    )
    readme = f"""# Benchmark Ground-Truth HiddenState Notebooks

These notebooks run the repository's benchmark-ground-truth preparation and A/B
hidden-state extraction one dataset at a time on Google Colab, Kaggle, or
ModelScope Notebook.

{links}

Each notebook expects the raw files to be staged under the paths declared in
`{CONFIG}`. If the raw files are mounted elsewhere, set `RAW_DATA_SOURCE` in the
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
"""
    (NOTEBOOK_DIR / "README.md").write_text(readme, encoding="utf-8")


if __name__ == "__main__":
    main()
