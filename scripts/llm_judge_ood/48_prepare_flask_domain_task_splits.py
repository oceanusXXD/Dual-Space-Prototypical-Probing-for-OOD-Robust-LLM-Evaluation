from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable
from urllib.request import urlopen


FLASK_REVISION = "3b4e22bc34aa9dc15ea0c51be3cbf5f3c8b1b5e5"
RAW_OUTPUTS_SUBDIR = Path("gpt_review/outputs")
SKILLSET_DESCRIPTION_PATH = Path(
    "metadata_annotation/skillset/src/skillset_description.json"
)

REVIEW_FILES = (
    "alpaca_13b.jsonl",
    "bard_review.jsonl",
    "chatgpt_review.jsonl",
    "claude_v1_review.jsonl",
    "davinci_003_review.jsonl",
    "gpt4_review.jsonl",
    "llama2_chat_13b.jsonl",
    "llama2_chat_70b.jsonl",
    "tulu_13b_review.jsonl",
    "tulu_30b_review.jsonl",
    "tulu_65b_review.jsonl",
    "tulu_7b_review.jsonl",
    "vicuna_13b.jsonl",
    "vicuna_33b.jsonl",
    "wizardlm_13b.jsonl",
)

DOMAINS = (
    "Humanities",
    "Language",
    "Social Science",
    "History",
    "Culture",
    "Technology",
    "Coding",
    "Math",
    "Natural Science",
    "Health",
)

SKILLS = (
    "Comprehension",
    "Factuality",
    "Logical Correctness",
    "Commonsense Understanding",
    "Completeness",
    "Insightfulness",
    "Metacognition",
    "Readability",
    "Conciseness",
    "Harmlessness",
    "Logical Robustness",
    "Logical Efficiency",
)

EXPECTED_FULL_MATRIX = {
    "Humanities": [4139, 1158, 825, 2700, 1800, 1139, 740, 1439, 825, 1004, 330, 75],
    "Language": [2204, 925, 705, 1986, 810, 1064, 234, 1110, 1035, 195, 90, 60],
    "Social Science": [3388, 1752, 644, 1313, 1065, 643, 564, 1049, 838, 599, 90, 45],
    "History": [990, 799, 225, 503, 449, 283, 90, 360, 210, 30, 15, 15],
    "Culture": [4258, 2301, 780, 2553, 1380, 1644, 442, 1874, 1317, 420, 254, 195],
    "Technology": [3163, 1453, 479, 1535, 1423, 1018, 246, 1364, 733, 253, 210, 210],
    "Coding": [1513, 750, 1168, 761, 645, 314, 73, 660, 420, 45, 1229, 1590],
    "Math": [1215, 469, 2655, 1587, 495, 164, 162, 510, 540, 15, 1379, 627],
    "Natural Science": [1799, 1382, 808, 973, 900, 285, 134, 480, 300, 90, 135, 75],
    "Health": [1469, 689, 210, 704, 645, 330, 164, 389, 389, 210, 90, 120],
}

EXPECTED_SINGLE_DOMAIN_MATRIX = {
    "Humanities": [1484, 365, 210, 937, 735, 360, 176, 585, 285, 419, 75, 0],
    "Language": [1140, 570, 465, 1015, 375, 540, 117, 630, 630, 90, 30, 30],
    "Social Science": [1228, 809, 329, 373, 315, 104, 194, 419, 344, 134, 30, 15],
    "History": [255, 296, 135, 89, 120, 30, 15, 90, 60, 15, 0, 15],
    "Culture": [1978, 1341, 570, 1272, 600, 476, 204, 795, 582, 105, 104, 90],
    "Technology": [974, 521, 150, 495, 374, 209, 72, 464, 224, 58, 90, 60],
    "Coding": [960, 482, 959, 435, 375, 90, 30, 465, 270, 15, 1140, 1457],
    "Math": [780, 277, 2070, 1122, 315, 60, 117, 360, 390, 15, 960, 434],
    "Natural Science": [974, 914, 538, 463, 510, 45, 60, 225, 180, 30, 90, 30],
    "Health": [509, 343, 75, 240, 285, 90, 30, 164, 164, 60, 15, 45],
}

EXPECTED_CLEANING = {
    "prompts": 1700,
    "candidate_models": 15,
    "a_space_responses": 25500,
    "raw_score_slots": 76499,
    "valid_b_space_rows": 75977,
    "score_na": 476,
    "score_empty": 28,
    "score_out_of_range": 18,
    "full_memberships": 103812,
    "single_domain_prompts": 1077,
    "single_domain_a_space_responses": 16155,
    "single_domain_b_space_rows": 48142,
    "single_domain_memberships": 48142,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download official FLASK GPT-4 review outputs and build the 10x12 "
            "Domain-Task views described in docs/FLASK_完整Domain-Task划分表.md."
        )
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("datasets/raw/flask"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("datasets/processed/flask_domain_task_v1"),
    )
    parser.add_argument("--revision", default=FLASK_REVISION)
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Require files to already exist under --raw-dir.",
    )
    parser.add_argument(
        "--no-cell-files",
        action="store_true",
        help="Skip per Domain-Task JSONL shards and write only compact index files.",
    )
    parser.add_argument(
        "--no-strict",
        action="store_true",
        help="Do not fail if counts differ from docs/FLASK_完整Domain-Task划分表.md.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir: Path = args.raw_dir
    output_dir: Path = args.output_dir
    if not args.no_download:
        download_sources(raw_dir, args.revision)

    skill_descriptions = load_skill_descriptions(raw_dir / SKILLSET_DESCRIPTION_PATH)
    b_rows, memberships, stats = build_rows(raw_dir, skill_descriptions)
    single_b_rows = [row for row in b_rows if len(row["domain_ids"]) == 1]
    single_memberships = [
        row for row in memberships if bool(row["is_single_domain_prompt"])
    ]

    output_dir.mkdir(parents=True, exist_ok=True)

    write_jsonl(output_dir / "b_space_full.jsonl", b_rows)
    write_jsonl(output_dir / "domain_task_memberships_full.jsonl", memberships)
    write_jsonl(output_dir / "b_space_single_domain.jsonl", single_b_rows)
    write_jsonl(
        output_dir / "domain_task_memberships_single_domain.jsonl",
        single_memberships,
    )
    full_matrix = write_matrix(
        output_dir / "cell_counts_full.csv",
        cell_counts(memberships),
    )
    single_matrix = write_matrix(
        output_dir / "cell_counts_single_domain.csv",
        cell_counts(single_memberships),
    )
    if not args.no_cell_files:
        write_cell_files(output_dir / "cells_full", memberships, b_rows)
        write_cell_files(output_dir / "cells_single_domain", single_memberships, b_rows)

    metadata = build_metadata(
        raw_dir=raw_dir,
        output_dir=output_dir,
        revision=str(args.revision),
        b_rows=b_rows,
        memberships=memberships,
        single_b_rows=single_b_rows,
        single_memberships=single_memberships,
        stats=stats,
        full_matrix=full_matrix,
        single_matrix=single_matrix,
    )
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    verify(metadata, strict=not args.no_strict)
    print(json.dumps(metadata["summary"], ensure_ascii=False, indent=2))


def download_sources(raw_dir: Path, revision: str) -> None:
    outputs_dir = raw_dir / RAW_OUTPUTS_SUBDIR
    outputs_dir.mkdir(parents=True, exist_ok=True)
    base = f"https://raw.githubusercontent.com/kaistAI/FLASK/{revision}/"
    for name in REVIEW_FILES:
        download_if_missing(base + str(RAW_OUTPUTS_SUBDIR / name), outputs_dir / name)
    download_if_missing(base + str(SKILLSET_DESCRIPTION_PATH), raw_dir / SKILLSET_DESCRIPTION_PATH)


def download_if_missing(url: str, path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url, timeout=120) as response:
        path.write_bytes(response.read())


def load_skill_descriptions(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing FLASK skill description file: {path}. Run without --no-download."
        )
    rows = json.loads(path.read_text(encoding="utf-8"))
    descriptions: dict[str, dict[str, Any]] = {}
    for row in rows:
        skill = canonical_skill(str(row["Skill"]))
        scoring = row.get("Scoring", {})
        score_guide = "\n".join(
            f"{score}: {text}" for score, text in sorted(scoring.items(), key=lambda x: int(x[0]))
        )
        descriptions[skill] = {
            "rubric": str(row.get("Criteria", "")).strip(),
            "score_guide": score_guide,
        }
    missing = [skill for skill in SKILLS if skill not in descriptions]
    if missing:
        raise ValueError(f"Skill description file is missing official skills: {missing}")
    return descriptions


def build_rows(
    raw_dir: Path, skill_descriptions: dict[str, dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    paths = [raw_dir / RAW_OUTPUTS_SUBDIR / name for name in REVIEW_FILES]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing FLASK review files: {missing}")

    b_rows: list[dict[str, Any]] = []
    memberships: list[dict[str, Any]] = []
    seen_b_ids: set[str] = set()
    questions: set[int] = set()
    single_domain_questions: set[int] = set()
    score_status = Counter()
    domain_length_by_response = Counter()
    raw_score_slots = 0
    row_count_by_file: dict[str, int] = {}

    for path in paths:
        generator_id = generator_from_filename(path.name)
        row_count = 0
        with path.open(encoding="utf-8") as handle:
            for source_row_index, line in enumerate(handle, start=1):
                row_count += 1
                raw = json.loads(line)
                question_id = int(raw["question_id"])
                questions.add(question_id)
                domains = [canonical_domain(domain) for domain in raw.get("domain_labeled", [])]
                if not domains:
                    raise ValueError(f"{path}:{source_row_index} has no domain_labeled")
                unknown_domains = [domain for domain in domains if domain not in DOMAINS]
                if unknown_domains:
                    raise ValueError(
                        f"{path}:{source_row_index} has unknown domains {unknown_domains}"
                    )
                if len(domains) == 1:
                    single_domain_questions.add(question_id)
                domain_length_by_response[str(len(domains))] += 1
                response_id = f"flask::{generator_id}::q{question_id:04d}"
                metrics = raw.get("metrics") or []
                scores = raw.get("score") or {}
                for metric in metrics:
                    raw_score_slots += 1
                    task_id = canonical_skill(str(metric))
                    if task_id not in SKILLS:
                        raise ValueError(f"{path}:{source_row_index} has unknown skill {metric!r}")
                    # The FLASK contract is score[metric.lower()].  Do not normalize
                    # raw score keys: a trailing-space key is a missing score rather
                    # than a valid annotation under the documented cleaning rule.
                    raw_score = scores.get(task_id.lower())
                    status, ground_truth = clean_score(raw_score)
                    score_status[status] += 1
                    if status != "valid":
                        continue
                    task_slug = slug(task_id)
                    b_id = f"{response_id}::{task_slug}"
                    if b_id in seen_b_ids:
                        raise ValueError(f"Duplicate B-space id: {b_id}")
                    seen_b_ids.add(b_id)
                    description = skill_descriptions[task_id]
                    b_row = {
                        "b_id": b_id,
                        "response_id": response_id,
                        "base_id": question_id,
                        "split_group_key": question_id,
                        "generator_id": generator_id,
                        "instruction": raw.get("text") or "",
                        "reference_answer": raw.get("answer") or "",
                        "candidate_response": raw.get("target_txt") or "",
                        "domain_ids": domains,
                        "task_id": task_id,
                        "task_slug": task_slug,
                        "rubric": description["rubric"],
                        "score_guide": description["score_guide"],
                        "metric_explanation": raw.get("metric_explanation") or "",
                        "ground_truth": ground_truth,
                        "raw_score": raw_score,
                        "source_task": raw.get("task") or "",
                        "has_answer": bool(raw.get("has_answer", False)),
                        "source_path": str(path),
                        "source_file": path.name,
                        "source_row_index": source_row_index,
                    }
                    b_rows.append(b_row)
                    for domain_id in domains:
                        memberships.append(
                            {
                                "membership_id": f"{b_id}::{slug(domain_id)}",
                                "b_id": b_id,
                                "response_id": response_id,
                                "base_id": question_id,
                                "split_group_key": question_id,
                                "generator_id": generator_id,
                                "domain_id": domain_id,
                                "domain_slug": slug(domain_id),
                                "task_id": task_id,
                                "task_slug": task_slug,
                                "cell_id": f"{slug(domain_id)}__{task_slug}",
                                "ground_truth": ground_truth,
                                "is_single_domain_prompt": len(domains) == 1,
                            }
                        )
        row_count_by_file[path.name] = row_count

    stats = {
        "row_count_by_file": row_count_by_file,
        "prompts": len(questions),
        "candidate_models": len(paths),
        "a_space_responses": sum(row_count_by_file.values()),
        "single_domain_prompts": len(single_domain_questions),
        "single_domain_a_space_responses": int(domain_length_by_response["1"]),
        "domain_length_by_response": dict(sorted(domain_length_by_response.items())),
        "raw_score_slots": raw_score_slots,
        "score_status": dict(sorted(score_status.items())),
    }
    return b_rows, memberships, stats


def clean_score(value: Any) -> tuple[str, int | float | None]:
    if value is None or value == "":
        return "empty", None
    if isinstance(value, str) and value.strip().upper() == "N/A":
        return "na", None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return "invalid", None
    if not 1 <= score <= 5:
        return "out_of_range", None
    if score.is_integer():
        return "valid", int(score)
    return "valid", score


def generator_from_filename(filename: str) -> str:
    stem = Path(filename).stem
    return stem.removesuffix("_review")


def canonical_domain(domain: str) -> str:
    normalized = normalize_key(domain)
    for candidate in DOMAINS:
        if normalize_key(candidate) == normalized:
            return candidate
    return domain


def canonical_skill(skill: str) -> str:
    normalized = normalize_key(skill)
    for candidate in SKILLS:
        if normalize_key(candidate) == normalized:
            return candidate
    return skill


def normalize_key(value: str) -> str:
    return " ".join(str(value).strip().lower().replace("_", " ").split())


def slug(value: str) -> str:
    return normalize_key(value).replace(" ", "_").replace("/", "_")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def cell_counts(memberships: Iterable[dict[str, Any]]) -> dict[str, list[int]]:
    counts = {(domain, skill): 0 for domain in DOMAINS for skill in SKILLS}
    for row in memberships:
        counts[(str(row["domain_id"]), str(row["task_id"]))] += 1
    return {domain: [counts[(domain, skill)] for skill in SKILLS] for domain in DOMAINS}


def write_matrix(path: Path, matrix: dict[str, list[int]]) -> dict[str, list[int]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Domain", *SKILLS, "Total"])
        for domain in DOMAINS:
            row = matrix[domain]
            writer.writerow([domain, *row, sum(row)])
        skill_totals = [sum(matrix[domain][i] for domain in DOMAINS) for i in range(len(SKILLS))]
        writer.writerow(["Skill Total", *skill_totals, sum(skill_totals)])
    return matrix


def write_cell_files(
    root: Path,
    memberships: Iterable[dict[str, Any]],
    b_rows: list[dict[str, Any]],
) -> None:
    b_by_id = {row["b_id"]: row for row in b_rows}
    root.mkdir(parents=True, exist_ok=True)
    handles: dict[tuple[str, str], Any] = {}
    try:
        for domain in DOMAINS:
            for skill in SKILLS:
                path = root / slug(domain) / f"{slug(skill)}.jsonl"
                path.parent.mkdir(parents=True, exist_ok=True)
                handles[(domain, skill)] = path.open("w", encoding="utf-8")
        for membership in memberships:
            domain = str(membership["domain_id"])
            skill = str(membership["task_id"])
            cell_row = dict(b_by_id[str(membership["b_id"])])
            cell_row.update(
                {
                    "membership_id": membership["membership_id"],
                    "domain_id": domain,
                    "domain_slug": membership["domain_slug"],
                    "cell_id": membership["cell_id"],
                    "is_single_domain_prompt": membership["is_single_domain_prompt"],
                }
            )
            handles[(domain, skill)].write(
                json.dumps(cell_row, ensure_ascii=False, sort_keys=True) + "\n"
            )
    finally:
        for handle in handles.values():
            handle.close()


def build_metadata(
    *,
    raw_dir: Path,
    output_dir: Path,
    revision: str,
    b_rows: list[dict[str, Any]],
    memberships: list[dict[str, Any]],
    single_b_rows: list[dict[str, Any]],
    single_memberships: list[dict[str, Any]],
    stats: dict[str, Any],
    full_matrix: dict[str, list[int]],
    single_matrix: dict[str, list[int]],
) -> dict[str, Any]:
    status = stats["score_status"]
    summary = {
        "prompts": stats["prompts"],
        "candidate_models": stats["candidate_models"],
        "a_space_responses": stats["a_space_responses"],
        "raw_score_slots": stats["raw_score_slots"],
        "valid_b_space_rows": len(b_rows),
        "score_na": int(status.get("na", 0)),
        "score_empty": int(status.get("empty", 0)),
        "score_out_of_range": int(status.get("out_of_range", 0)),
        "full_memberships": len(memberships),
        "single_domain_prompts": stats["single_domain_prompts"],
        "single_domain_a_space_responses": stats["single_domain_a_space_responses"],
        "single_domain_b_space_rows": len(single_b_rows),
        "single_domain_memberships": len(single_memberships),
    }
    output_paths = {
        "b_space_full": str(output_dir / "b_space_full.jsonl"),
        "domain_task_memberships_full": str(output_dir / "domain_task_memberships_full.jsonl"),
        "b_space_single_domain": str(output_dir / "b_space_single_domain.jsonl"),
        "domain_task_memberships_single_domain": str(
            output_dir / "domain_task_memberships_single_domain.jsonl"
        ),
        "cell_counts_full": str(output_dir / "cell_counts_full.csv"),
        "cell_counts_single_domain": str(output_dir / "cell_counts_single_domain.csv"),
        "cells_full": str(output_dir / "cells_full"),
        "cells_single_domain": str(output_dir / "cells_single_domain"),
    }
    return {
        "artifact_type": "flask_domain_task_split_v1",
        "source": {
            "name": "FLASK",
            "revision": revision,
            "raw_dir": str(raw_dir),
            "review_files": [
                {
                    "path": str(raw_dir / RAW_OUTPUTS_SUBDIR / name),
                    "sha256": sha256_file(raw_dir / RAW_OUTPUTS_SUBDIR / name),
                    "rows": stats["row_count_by_file"][name],
                    "generator_id": generator_from_filename(name),
                }
                for name in REVIEW_FILES
            ],
            "skillset_description": {
                "path": str(raw_dir / SKILLSET_DESCRIPTION_PATH),
                "sha256": sha256_file(raw_dir / SKILLSET_DESCRIPTION_PATH),
            },
        },
        "domains": list(DOMAINS),
        "skills": list(SKILLS),
        "summary": summary,
        "cleaning_rules": {
            "keep": "numeric score in inclusive range [1, 5], including decimals",
            "drop": ["N/A", "empty", "non-numeric", "scores outside [1, 5]"],
            "task_source": "only skills listed in each raw record's metrics field",
            "split_group_key": "question_id",
            "multi_domain_rule": (
                "keep one B-space row per response_id x task_id; expand memberships "
                "to each domain_labeled value without changing ground_truth"
            ),
        },
        "stats": stats,
        "full_matrix": full_matrix,
        "single_domain_matrix": single_matrix,
        "single_domain_empty_cells": [
            {"domain_id": domain, "task_id": skill}
            for domain in DOMAINS
            for skill in SKILLS
            if single_matrix[domain][SKILLS.index(skill)] == 0
        ],
        "output_paths": output_paths,
        "expected_summary_from_doc": EXPECTED_CLEANING,
        "expected_full_matrix_from_doc": EXPECTED_FULL_MATRIX,
        "expected_single_domain_matrix_from_doc": EXPECTED_SINGLE_DOMAIN_MATRIX,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(metadata: dict[str, Any], *, strict: bool) -> None:
    errors: list[str] = []
    for key, expected in EXPECTED_CLEANING.items():
        actual = metadata["summary"].get(key)
        if actual != expected:
            errors.append(f"summary.{key}: expected {expected}, got {actual}")
    if metadata["full_matrix"] != EXPECTED_FULL_MATRIX:
        errors.append("full Domain-Task matrix differs from docs/FLASK_完整Domain-Task划分表.md")
    if metadata["single_domain_matrix"] != EXPECTED_SINGLE_DOMAIN_MATRIX:
        errors.append(
            "single-domain Domain-Task matrix differs from docs/FLASK_完整Domain-Task划分表.md"
        )
    if errors and strict:
        raise SystemExit("FLASK split verification failed:\n" + "\n".join(errors))
    if errors:
        print(json.dumps({"verification_warnings": errors}, ensure_ascii=False, indent=2))
    else:
        print("FLASK split verification passed.")


if __name__ == "__main__":
    main()
