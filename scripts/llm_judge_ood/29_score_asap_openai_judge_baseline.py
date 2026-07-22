#!/usr/bin/env python3
"""Score the label-free ASAP manifest with an OpenAI-compatible API."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.io import read_jsonl


SYSTEM_PROMPT = """You are an automated essay-scoring judge. Evaluate only the supplied essay for its assignment. Return a JSON object with exactly one field named score, whose value is an integer from 1 to 5. Use 1 for very poor, 2 for weak, 3 for adequate, 4 for strong, and 5 for excellent. Do not include explanations."""


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a GPT single-answer ASAP Judge baseline.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True, help="Pinned API model id, not a floating family name.")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--base-url", default="https://api.openai.com/v1")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=6)
    args = parser.parse_args()

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"{args.api_key_env} is not set; refusing to create placeholder GPT scores")
    manifest = read_jsonl(args.manifest)
    output_path = Path(args.output)
    existing = read_jsonl(output_path) if output_path.exists() else []
    completed = {
        (str(row.get("model")), str(row.get("sample_id")))
        for row in existing
        if "score" in row
    }
    pending = [
        row
        for row in manifest
        if (str(args.model), str(row["sample_id"])) not in completed
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
            futures = {
                executor.submit(
                    _score_one,
                    row,
                    model=str(args.model),
                    api_key=api_key,
                    base_url=str(args.base_url),
                    timeout=float(args.timeout_seconds),
                    max_retries=int(args.max_retries),
                ): row
                for row in pending
            }
            for index, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                handle.write(json.dumps(result, ensure_ascii=False))
                handle.write("\n")
                handle.flush()
                if index % 25 == 0 or index == len(pending):
                    print(f"completed {index}/{len(pending)} new rows", flush=True)
    print(json.dumps({"output": str(output_path), "model": args.model, "rows": len(manifest)}))


def _score_one(
    row: dict[str, Any],
    *,
    model: str,
    api_key: str,
    base_url: str,
    timeout: float,
    max_retries: int,
) -> dict[str, Any]:
    user_prompt = f"Assignment:\n{row['task']}\n\nEssay:\n{row['essay']}"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    delay = 1.0
    for attempt in range(max_retries + 1):
        request = urllib.request.Request(
            f"{base_url.rstrip('/')}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
            content = body["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            score = int(parsed["score"])
            if score not in {1, 2, 3, 4, 5}:
                raise ValueError(f"API returned out-of-range score {score}")
            usage = body.get("usage") or {}
            return {
                "sample_id": str(row["sample_id"]),
                "split": str(row["split"]),
                "asap_prompt_id": int(row["asap_prompt_id"]),
                "model": model,
                "model_returned": body.get("model"),
                "score": score,
                "temperature": 0,
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "request_id": body.get("id"),
            }
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, KeyError, ValueError, json.JSONDecodeError) as error:
            if attempt >= max_retries:
                raise RuntimeError(f"Failed to score {row['sample_id']} with {model}: {error}") from error
            time.sleep(delay)
            delay = min(delay * 2.0, 30.0)
    raise AssertionError("retry loop exhausted")


if __name__ == "__main__":
    main()
