"""Tests for evaluation comparison and result persistence."""
from __future__ import annotations

import json
from pathlib import Path

from eval.compare import classify_changes
from eval.runner import check, run_eval


def test_check_supports_each_result_shape() -> None:
    """The harness accepts the result shapes used by the fixed suite."""

    assert check({"kind": "scalar", "value": 4.0}, [{"value": 4.02}])
    assert check(
        {"kind": "rows", "values": {"APAC|2024-01": 2.0}},
        [{"region": "APAC", "month": "2024-01", "value": 2.0}],
    )
    assert check(
        {"kind": "rows", "values": {"APAC": "Ergo Chair"}},
        [{"region": "APAC", "product_name": "Ergo Chair"}],
    )
    assert check(
        {"kind": "rows_ordered", "values": [["APAC", 2.0]]},
        [{"region": "APAC", "value": 2.0}],
    )
    assert check(
        {"kind": "rows_with_pct", "values": {"APAC": 20.0}},
        [{"region": "APAC", "pct": 20.08}],
    )
    assert check({"kind": "set", "values": ["APAC"]}, [{"region": "APAC"}])
    assert check({"kind": "empty"}, [])
    assert check({"kind": "rejected"}, None)


def test_check_supports_tolerance_ties_and_zero_filled_group_rows() -> None:
    """Extended cases can declare valid rounding, ties, and zero-filled rows."""

    assert check(
        {"kind": "scalar", "value": 10.0, "tolerance": 0.5},
        [{"value": 10.4}],
    )
    assert check(
        {
            "kind": "rows",
            "values": {"NA|2024-02": 200.0},
            "alternatives": [{"NA|2024-03": 200.0}],
        },
        [{"region": "NA", "month": "2024-03", "profit": 200.0}],
    )
    assert check(
        {
            "kind": "rows",
            "values": {"USA|Technology": 10.0},
            "allow_extra_zero_rows": True,
        },
        [
            {"country": "USA", "product_category": "Technology", "revenue": 10.0},
            {"country": "USA", "product_category": "Furniture", "revenue": 0.0},
        ],
    )
    assert check(
        {"kind": "rows", "values": {"Consumer|MacBook Air": 200.0}},
        [{"product_name": "MacBook Air", "customer_segment": "Consumer", "value": 200.0}],
    )


def test_run_eval_persists_results_and_captures_failures(tmp_path: Path) -> None:
    """A run writes JSON output and isolates a failed pipeline query."""

    cases = [
        {"id": "pass", "query": "good", "category": "aggregate", "expected": {"kind": "scalar", "value": 1}},
        {"id": "fail", "query": "bad", "category": "aggregate", "expected": {"kind": "scalar", "value": 1}},
    ]
    test_set = tmp_path / "test_set.json"
    test_set.write_text(json.dumps(cases), encoding="utf-8")

    def run_query(query: str) -> dict[str, object]:
        if query == "bad":
            raise RuntimeError("test failure")
        return {"result": [{"value": 1}], "generated_logic": "SELECT 1", "confidence_score": 1.0}

    summary = run_eval(run_query, test_set_path=test_set, runs_dir=tmp_path / "runs")
    assert summary["passed"] == 1
    assert summary["failed"] == 1
    saved = Path(summary["run_file"])
    assert saved.is_file()
    assert json.loads(saved.read_text(encoding="utf-8"))["results"][1]["error"] == "RuntimeError: test failure"


def test_classify_changes_identifies_improvements_and_regressions() -> None:
    """Comparison classifies each matching test ID exactly once."""

    first = {"results": [{"id": "a", "status": "FAIL"}, {"id": "b", "status": "PASS"}]}
    second = {"results": [{"id": "a", "status": "PASS"}, {"id": "b", "status": "FAIL"}]}
    assert classify_changes(first, second) == (["a"], ["b"], [])
