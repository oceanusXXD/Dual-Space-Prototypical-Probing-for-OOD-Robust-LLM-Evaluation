from __future__ import annotations


DIRECT_JUDGE_TEMPLATE_VERSION = "flask_direct_judge_v1"


def direct_judge_prompt(*, instruction: str, response: str, rubric: str) -> str:
    return (
        "Instruction:\n"
        f"{instruction}\n\n"
        "Candidate response:\n"
        f"{response}\n\n"
        "Rubric:\n"
        f"{rubric}\n\n"
        "Return only the numeric score."
    )
