from __future__ import annotations


def benchmark_prompt(*, dataset: str, instruction: str, response: str, rubric: str) -> str:
    return (
        f"Dataset: {dataset}\n"
        f"Instruction:\n{instruction}\n\n"
        f"Candidate response:\n{response}\n\n"
        f"Rubric:\n{rubric}"
    )
