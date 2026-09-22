"""Compare two saved evaluation runs."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping


def load(path_value: str | Path) -> dict[str, Any]:
    """Load one evaluation summary from JSON."""

    try:
        contents = json.loads(Path(path_value).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Evaluation run could not be loaded as JSON.") from error
    if not isinstance(contents, dict) or "results" not in contents:
        raise ValueError("Evaluation run does not have the required summary shape.")
    return contents


def classify_changes(
    first: Mapping[str, Any], second: Mapping[str, Any]
) -> tuple[list[str], list[str], list[tuple[str, str]]]:
    """Return query IDs improved, regressed, and unchanged between two runs."""

    first_by_id = {result["id"]: result for result in first["results"]}
    second_by_id = {result["id"]: result for result in second["results"]}
    if first_by_id.keys() != second_by_id.keys():
        raise ValueError("Runs use different test sets and cannot be compared.")
    improved: list[str] = []
    regressed: list[str] = []
    unchanged: list[tuple[str, str]] = []
    for query_id in sorted(first_by_id):
        before = first_by_id[query_id]["status"]
        after = second_by_id[query_id]["status"]
        if before == "FAIL" and after == "PASS":
            improved.append(query_id)
        elif before == "PASS" and after == "FAIL":
            regressed.append(query_id)
        else:
            unchanged.append((query_id, before))
    return improved, regressed, unchanged


def compare(first: Mapping[str, Any], second: Mapping[str, Any]) -> None:
    """Print a query-by-query comparison of two compatible evaluation runs."""

    improved, regressed, unchanged = classify_changes(first, second)
    first_by_id = {result["id"]: result for result in first["results"]}
    print("\nComparing:")
    print(f"  A: {first['commit']} ({first['passed']}/{first['total']} = {first['pass_rate'] * 100:.1f}%)")
    print(f"  B: {second['commit']} ({second['passed']}/{second['total']} = {second['pass_rate'] * 100:.1f}%)")
    for label, query_ids in (("IMPROVED", improved), ("REGRESSED", regressed)):
        print(f"\n{label} ({len(query_ids)}):")
        for query_id in query_ids:
            print(f"  {query_id}  {first_by_id[query_id]['query']}")
    print(f"\nUNCHANGED ({len(unchanged)}):")
    for query_id, status in unchanged:
        print(f"  {query_id}  [{status}]  {first_by_id[query_id]['query']}")
    delta = second["passed"] - first["passed"]
    print(f"\nNET CHANGE: {delta:+d} ({first['passed']} → {second['passed']})\n")


def main() -> None:
    """Parse two paths and print their evaluation comparison."""

    if len(sys.argv) != 3:
        raise SystemExit("Usage: python eval/compare.py <run_a.json> <run_b.json>")
    compare(load(sys.argv[1]), load(sys.argv[2]))


if __name__ == "__main__":
    main()
