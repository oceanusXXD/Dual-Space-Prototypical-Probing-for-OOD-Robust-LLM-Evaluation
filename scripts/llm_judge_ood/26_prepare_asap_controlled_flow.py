#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.io import read_jsonl, write_json, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine a deterministic ASAP reference corpus with a controlled flow."
    )
    parser.add_argument("--base", required=True)
    parser.add_argument("--flow", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-base-documents", type=int, default=469)
    parser.add_argument("--events-per-window", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.max_base_documents < 1 or args.events_per_window < 1:
        raise ValueError("Document and per-window limits must be positive")

    base_rows = read_jsonl(args.base)
    flow_rows = read_jsonl(args.flow)
    selected_ids = _select_base_ids(
        base_rows,
        limit=int(args.max_base_documents),
        seed=int(args.seed),
    )
    selected_base = [row for row in base_rows if str(row["input_document_id"]) in selected_ids]
    retained_base = [row for row in selected_base if str(row["split"]) != "deployment_stream"]
    selected_flow = _select_flow_events(
        flow_rows,
        per_window=int(args.events_per_window),
        seed=int(args.seed),
    )
    for stream_order, row in enumerate(selected_flow):
        row["stream_order"] = int(stream_order)

    combined = retained_base + selected_flow
    document_ids = [str(row["input_document_id"]) for row in combined]
    if len(document_ids) != len(set(document_ids)):
        raise ValueError("Controlled-flow input document IDs must be unique")

    output = Path(args.output)
    write_jsonl(output, combined)
    all_base_document_ids = {str(row["input_document_id"]) for row in base_rows}
    metadata = {
        "artifact_type": "llm_judge_ood_asap_controlled_flow_input",
        "base_path": str(args.base),
        "flow_path": str(args.flow),
        "seed": int(args.seed),
        "selected_base_documents": len(selected_base),
        "all_base_documents": len(all_base_document_ids),
        "is_full_base_corpus": len(selected_ids) == len(all_base_document_ids),
        "selected_base_fraction": len(selected_ids) / max(len(all_base_document_ids), 1),
        "removed_base_stream_documents": len(selected_base) - len(retained_base),
        "retained_base_documents": len(retained_base),
        "events_per_window": int(args.events_per_window),
        "selected_flow_events": len(selected_flow),
        "combined_documents": len(combined),
        "roles": dict(Counter(str(row["document_distribution_role"]) for row in combined)),
        "splits": dict(Counter(str(row["split"]) for row in combined)),
        "flow_windows": dict(
            Counter(str(row.get("flow_window_index")) for row in selected_flow)
        ),
        "permutation_block_key": "arrival_batch_id",
        "permutation_assumption": "each controlled-flow simulation event is one arrival block",
    }
    write_json(output.with_suffix(".metadata.json"), metadata)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


def _select_base_ids(
    rows: list[dict[str, object]],
    *,
    limit: int,
    seed: int,
) -> set[str]:
    document_ids = list(dict.fromkeys(str(row["input_document_id"]) for row in rows))
    ranked = sorted(
        document_ids,
        key=lambda document_id: hashlib.sha256(
            f"{seed}::{document_id}".encode("utf-8")
        ).hexdigest(),
    )
    return set(ranked[:limit])


def _select_flow_events(
    rows: list[dict[str, object]],
    *,
    per_window: int,
    seed: int,
) -> list[dict[str, object]]:
    by_window: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_window[int(row["flow_window_index"])].append(row)
    selected: list[dict[str, object]] = []
    for window_index, window_rows in sorted(by_window.items()):
        ranked = sorted(
            window_rows,
            key=lambda row: hashlib.sha256(
                f"{seed}::{window_index}::{row['input_document_id']}".encode("utf-8")
            ).hexdigest(),
        )
        chosen = sorted(
            ranked[:per_window],
            key=lambda row: int(row["stream_order"]),
        )
        selected.extend(dict(row) for row in chosen)
    return selected


if __name__ == "__main__":
    main()
