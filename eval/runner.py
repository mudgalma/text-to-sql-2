"""Run a fixed correctness suite against a single-query analytics pipeline."""
from __future__ import annotations

import json
import logging
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


TOLERANCE = 0.05
PERCENTAGE_TOLERANCE = 0.1
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEST_SET = Path(__file__).with_name("test_set.json")
DEFAULT_RUNS_DIR = Path(__file__).with_name("runs")
LOGGER = logging.getLogger(__name__)


def _is_number(value: Any) -> bool:
    """Return whether a value is numeric but not a boolean."""

    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _as_rows(result: Any) -> list[Mapping[str, Any]] | None:
    """Validate the JSON-row result shape emitted by the pipeline."""

    if isinstance(result, Mapping):
        return [result]
    if isinstance(result, Sequence) and not isinstance(result, (str, bytes)):
        rows = list(result)
        if all(isinstance(row, Mapping) for row in rows):
            return rows
    return None


def _compare_scalar(expected: float, result: Any) -> bool:
    """Compare one expected numeric aggregate to a result row."""

    rows = _as_rows(result)
    if not rows:
        return False
    row = rows[0]
    for key in ("value", "total", "revenue", "profit", "count"):
        value = row.get(key)
        if _is_number(value):
            return abs(float(value) - expected) < TOLERANCE
    numeric_values = [float(value) for value in row.values() if _is_number(value)]
    return len(numeric_values) == 1 and abs(numeric_values[0] - expected) < TOLERANCE


def _compare_rows(expected: Mapping[str, Any], result: Any) -> bool:
    """Compare unordered grouped results, including text-valued rankings."""

    rows = _as_rows(result)
    if not rows or len(rows) != len(expected):
        return False
    found: dict[str, Any] = {}
    numeric_values_expected = all(_is_number(value) for value in expected.values())
    for row in rows:
        text_values = [str(value) for value in row.values() if not _is_number(value)]
        numeric_values = [float(value) for value in row.values() if _is_number(value)]
        if numeric_values_expected:
            if not text_values or len(numeric_values) != 1:
                return False
            key, value = "|".join(text_values), numeric_values[0]
        else:
            if len(text_values) != 2:
                return False
            key, value = text_values
        if key in found:
            return False
        found[key] = value

    for key, expected_value in expected.items():
        actual_value = found.get(key)
        if actual_value is None:
            return False
        if _is_number(expected_value):
            if not _is_number(actual_value) or abs(float(actual_value) - expected_value) > TOLERANCE:
                return False
        elif actual_value != expected_value:
            return False
    return True


def _compare_rows_ordered(expected: Sequence[Sequence[Any]], result: Any) -> bool:
    """Compare ordered [label, numeric value] result rows."""

    rows = _as_rows(result)
    if rows is None or len(rows) != len(expected):
        return False
    for (expected_key, expected_value), row in zip(expected, rows):
        text_values = [str(value) for value in row.values() if not _is_number(value)]
        numeric_values = [float(value) for value in row.values() if _is_number(value)]
        if expected_key not in text_values or len(numeric_values) != 1:
            return False
        if abs(numeric_values[0] - float(expected_value)) > TOLERANCE:
            return False
    return True


def _compare_rows_with_pct(expected: Mapping[str, float], result: Any) -> bool:
    """Compare unordered percentage contribution results."""

    rows = _as_rows(result)
    if not rows or len(rows) != len(expected):
        return False
    found: dict[str, float] = {}
    for row in rows:
        labels = [str(value) for value in row.values() if not _is_number(value)]
        percentages = [
            float(value)
            for column, value in row.items()
            if "pct" in column.lower() and _is_number(value)
        ]
        if len(labels) != 1 or len(percentages) != 1 or labels[0] in found:
            return False
        found[labels[0]] = percentages[0]
    return all(
        key in found and abs(found[key] - value) <= PERCENTAGE_TOLERANCE
        for key, value in expected.items()
    )


def _compare_set(expected: Sequence[str], result: Any) -> bool:
    """Compare an unordered set of one-label result rows."""

    rows = _as_rows(result)
    if rows is None:
        return False
    labels: set[str] = set()
    for row in rows:
        text_values = [str(value) for value in row.values() if not _is_number(value)]
        if not text_values:
            return False
        labels.add(text_values[0])
    return labels == set(expected)


def check(expected_spec: Mapping[str, Any], result: Any) -> bool:
    """Check a pipeline result against one versioned expected-result schema."""

    kind = expected_spec.get("kind")
    if kind == "scalar":
        return _compare_scalar(float(expected_spec["value"]), result)
    if kind == "rows":
        return _compare_rows(expected_spec["values"], result)
    if kind == "rows_ordered":
        return _compare_rows_ordered(expected_spec["values"], result)
    if kind == "rows_with_pct":
        return _compare_rows_with_pct(expected_spec["values"], result)
    if kind == "set":
        return _compare_set(expected_spec["values"], result)
    if kind == "empty":
        rows = _as_rows(result)
        return result is None or rows == []
    if kind == "rejected":
        return result is None
    return False


def _git_commit() -> str:
    """Return the checked-out commit when Git metadata is available."""

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "no-git"


def _load_test_set(path: Path) -> list[dict[str, Any]]:
    """Load and minimally validate the versioned evaluation test cases."""

    try:
        cases = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Evaluation test set could not be loaded as JSON.") from error
    if not isinstance(cases, list) or not cases:
        raise ValueError("Evaluation test set must be a non-empty JSON array.")
    if not all(isinstance(case, dict) and {"id", "query", "category", "expected"} <= case.keys() for case in cases):
        raise ValueError("Each evaluation case requires id, query, category, and expected fields.")
    return cases


def run_eval(
    run_query: Callable[[str], Mapping[str, Any]],
    *,
    test_set_path: str | Path = DEFAULT_TEST_SET,
    runs_dir: str | Path = DEFAULT_RUNS_DIR,
) -> dict[str, Any]:
    """Run every case, persist a comparable result file, and return its summary."""

    cases = _load_test_set(Path(test_set_path))
    results: list[dict[str, Any]] = []
    passed = 0
    for item in cases:
        try:
            output = run_query(item["query"])
            if not isinstance(output, Mapping):
                raise TypeError("Pipeline output must be a mapping.")
            ok = check(item["expected"], output.get("result"))
            error = output.get("error")
        except Exception as exc:
            LOGGER.warning(
                "evaluation_query_failed",
                extra={"query_id": item["id"], "error_type": type(exc).__name__},
                exc_info=True,
            )
            output = {}
            ok = False
            error = f"{type(exc).__name__}: {exc}"
        passed += int(ok)
        results.append(
            {
                "id": item["id"],
                "query": item["query"],
                "category": item["category"],
                "status": "PASS" if ok else "FAIL",
                "expected": item["expected"],
                "actual_result": output.get("result"),
                "actual_sql": output.get("generated_logic"),
                "confidence": output.get("confidence_score"),
                "error": error,
            }
        )

    timestamp = datetime.now(UTC)
    summary: dict[str, Any] = {
        "commit": _git_commit(),
        "timestamp": timestamp.isoformat(),
        "total": len(cases),
        "passed": passed,
        "failed": len(cases) - passed,
        "pass_rate": round(passed / len(cases), 3),
        "results": results,
    }
    output_dir = Path(runs_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{timestamp.strftime('%Y%m%d_%H%M%S')}_{summary['commit']}.json"
    output_path.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    summary["run_file"] = str(output_path)
    return summary


def print_summary(summary: Mapping[str, Any]) -> None:
    """Print a concise, human-readable result summary."""

    print(f"\nEVAL RESULTS — commit {summary['commit']}")
    print(f"Passed: {summary['passed']} / {summary['total']} ({summary['pass_rate'] * 100:.1f}%)")
    for result in summary["results"]:
        if result["status"] == "FAIL":
            print(f"[FAIL] {result['id']}  {result['query']}")
            print(f"       expected: {result['expected']}")
            print(f"       actual:   {result['actual_result']}")
            if result["error"]:
                print(f"       error:    {result['error']}")
    print(f"Saved: {summary['run_file']}\n")


def main() -> None:
    """Evaluate the repository's current TextToSQLEngine implementation."""

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from main import TextToSQLEngine

    engine = TextToSQLEngine()
    try:
        print_summary(run_eval(engine.run_query))
    finally:
        engine.close()


if __name__ == "__main__":
    main()
