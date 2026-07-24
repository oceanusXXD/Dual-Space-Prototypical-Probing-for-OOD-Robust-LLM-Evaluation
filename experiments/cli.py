from __future__ import annotations

import argparse
import json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m experiments.cli")
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available experiment groups.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    groups = ["flask", "asap", "benchmark", "studies"]
    if args.list:
        print(json.dumps({"experiment_groups": groups}, indent=2))
        return
    parser.print_help()


if __name__ == "__main__":
    main()
