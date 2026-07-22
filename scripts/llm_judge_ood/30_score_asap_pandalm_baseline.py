#!/usr/bin/env python3
"""Run PandaLM's native pairwise protocol on the ASAP pair manifest."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.io import read_jsonl


_SPECIAL = re.compile(
    r"<unk>|<pad>|<s>|</s>|\[PAD\]|<\|endoftext\|>|\[UNK\]|\[CLS\]|\[MASK\]|"
    r"<\|startofpiece\|>|<\|endofpiece\|>|\[gMASK\]|\[sMASK\]"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the native PandaLM pairwise Judge baseline.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="WeOpenML/PandaLM-7B-v1")
    parser.add_argument("--revision", default="f7e28bda625e6e72b4638165ef9964a35a12a4fc")
    parser.add_argument("--quantization", choices=("none", "8bit", "4bit"), default="none")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--max-response-tokens", type=int, default=700)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--repetition-penalty", type=float, default=1.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as error:
        raise RuntimeError("PandaLM requires torch, transformers, and sentencepiece") from error
    if args.quantization != "none":
        try:
            import accelerate  # noqa: F401
            import bitsandbytes  # noqa: F401
        except ImportError as error:
            raise RuntimeError("Quantized PandaLM requires accelerate and bitsandbytes") from error

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.revision,
        use_fast=False,
        cache_dir=args.cache_dir,
    )
    quantization_config = None
    if args.quantization == "8bit":
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    elif args.quantization == "4bit":
        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        torch_dtype=torch.float16,
        device_map="auto",
        quantization_config=quantization_config,
        cache_dir=args.cache_dir,
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    manifest = read_jsonl(args.manifest)
    output_path = Path(args.output)
    existing = read_jsonl(output_path) if output_path.exists() else []
    completed = {(str(row.get("model")), str(row.get("pair_id"))) for row in existing}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        for index, row in enumerate(manifest, start=1):
            if (str(args.model), str(row["pair_id"])) in completed:
                continue
            response1 = _truncate(row["response1"], tokenizer, args.max_response_tokens)
            response2 = _truncate(row["response2"], tokenizer, args.max_response_tokens)
            prompt = _prompt(row["instruction"], row["input"], response1, response2)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            input_ids = inputs["input_ids"].to(model.device)
            attention_mask = inputs["attention_mask"].to(model.device)
            with torch.inference_mode():
                generated = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    do_sample=False,
                    num_beams=int(args.num_beams),
                    max_new_tokens=int(args.max_new_tokens),
                    repetition_penalty=float(args.repetition_penalty),
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            text = tokenizer.decode(generated[0, input_ids.shape[1] :], skip_special_tokens=True).strip()
            preference, parse_status = _parse_preference(text)
            result = {
                "pair_id": str(row["pair_id"]),
                "asap_prompt_id": int(row["asap_prompt_id"]),
                "model": str(args.model),
                "revision": str(args.revision),
                "quantization": str(args.quantization),
                "seed": int(args.seed),
                "num_beams": int(args.num_beams),
                "max_new_tokens": int(args.max_new_tokens),
                "repetition_penalty": float(args.repetition_penalty),
                "preference": preference,
                "parse_status": parse_status,
                "raw_response": text,
            }
            handle.write(json.dumps(result, ensure_ascii=False))
            handle.write("\n")
            handle.flush()
            if index % 10 == 0 or index == len(manifest):
                print(f"processed {index}/{len(manifest)}", flush=True)


def _prompt(instruction: str, input_text: str, response1: str, response2: str) -> str:
    clean1 = _SPECIAL.sub("", response1).strip()
    clean2 = _SPECIAL.sub("", response2).strip()
    return (
        "Below are two responses for a given task. The task is defined by the Instruction with an Input "
        "that provides further context. Evaluate the responses and generate a reference answer for the task.\n\n"
        f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n"
        f"### Response 1:\n{clean1}\n\n### Response 2:\n{clean2}\n\n### Evaluation:\n"
    )


def _truncate(text: str, tokenizer: Any, maximum_tokens: int) -> str:
    token_ids = tokenizer(str(text), add_special_tokens=False)["input_ids"][: int(maximum_tokens)]
    return tokenizer.decode(token_ids, skip_special_tokens=True)


def _parse_preference(text: str) -> tuple[int, str]:
    first = text.strip().splitlines()[0].strip().lower() if text.strip() else ""
    if first == "1":
        return 1, "parsed"
    if first == "2":
        return 2, "parsed"
    if first == "tie":
        return 0, "parsed"
    return 0, "invalid_treated_as_tie"


if __name__ == "__main__":
    main()
