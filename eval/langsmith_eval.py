"""Upload the extended evaluation set and run a tracked LangSmith experiment.

The local JSON files remain the source of truth. This module mirrors a
versioned test set to LangSmith and evaluates the existing single-query engine
without changing its production pipeline.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langsmith import Client
from langsmith.evaluation import evaluate
from langsmith.utils import LangSmithNotFoundError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_NAME = "analytics-engine-extended-v2"
DEFAULT_TEST_SET = Path(__file__).with_name("test_set_extended.json")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.runner import _git_commit, _load_test_set, check  # noqa: E402


def _example_payload(case: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Convert one local case to the stable LangSmith example schema."""

    return {
        "inputs": {"query": case["query"]},
        "outputs": {"expected": case["expected"]},
        "metadata": {
            "case_id": case["id"],
            "category": case["category"],
            "difficulty": case.get("difficulty", "unspecified"),
        },
    }


def _matches_case(existing: Any, payload: Mapping[str, Any]) -> bool:
    """Return whether a remote example is identical to its local definition."""

    return (
        getattr(existing, "inputs", None) == payload["inputs"]
        and getattr(existing, "outputs", None) == payload["outputs"]
        and (getattr(existing, "metadata", {}) or {}).get("case_id")
        == payload["metadata"]["case_id"]
    )


def sync_dataset(
    client: Client,
    cases: Sequence[Mapping[str, Any]],
    *,
    dataset_name: str = DEFAULT_DATASET_NAME,
) -> Any:
    """Create an idempotent LangSmith mirror without overwriting remote data.

    Existing examples must be byte-for-byte equivalent to the versioned local
    case. A mismatch is intentionally rejected so an experiment cannot silently
    compare results against changed ground truth.
    """

    try:
        dataset = client.read_dataset(dataset_name=dataset_name)
    except LangSmithNotFoundError:
        dataset = client.create_dataset(
            dataset_name,
            description="Hand-verified extended Text-to-SQL evaluation suite.",
            metadata={"source": "eval/test_set_extended.json", "version": "v1"},
        )

    existing_by_case_id: dict[str, Any] = {}
    for example in client.list_examples(dataset_id=dataset.id):
        metadata = getattr(example, "metadata", {}) or {}
        case_id = metadata.get("case_id")
        if isinstance(case_id, str):
            existing_by_case_id[case_id] = example

    local_case_ids = {str(case["id"]) for case in cases}
    remote_case_ids = set(existing_by_case_id)
    unexpected = remote_case_ids - local_case_ids
    if unexpected:
        names = ", ".join(sorted(unexpected))
        raise ValueError(
            f"Dataset {dataset_name!r} contains cases absent from the local source: {names}. "
            "Create a new versioned dataset instead of mutating this one."
        )

    for case in cases:
        payload = _example_payload(case)
        existing = existing_by_case_id.get(str(case["id"]))
        if existing is None:
            client.create_example(dataset_id=dataset.id, **payload)
        elif not _matches_case(existing, payload):
            raise ValueError(
                f"Dataset {dataset_name!r} case {case['id']!r} differs from the local source. "
                "Create a new versioned dataset instead of overwriting expected results."
            )
    return dataset


def accuracy_evaluator(run: Any, example: Any) -> dict[str, Any]:
    """Score a LangSmith run with the same comparator as the local harness."""

    outputs = getattr(run, "outputs", None) or {}
    expected_outputs = getattr(example, "outputs", None) or {}
    expected = expected_outputs.get("expected")
    confidence = outputs.get("confidence_score")
    matched = isinstance(expected, Mapping) and check(expected, outputs.get("result"), confidence)
    return {"key": "exact_match", "score": float(matched)}


def run_langsmith_eval(
    client: Client,
    *,
    dataset_name: str,
    experiment_prefix: str,
) -> Any:
    """Run one serial, traceable experiment against the configured LLM engine."""

    from main import TextToSQLEngine

    engine = TextToSQLEngine()

    def target(inputs: Mapping[str, Any]) -> dict[str, Any]:
        query = inputs.get("query")
        if not isinstance(query, str):
            raise ValueError("LangSmith evaluation input requires a string query.")
        return engine.run_query(query)

    try:
        return evaluate(
            target,
            data=dataset_name,
            evaluators=[accuracy_evaluator],
            experiment_prefix=experiment_prefix,
            description="Tracked extended Text-to-SQL correctness evaluation.",
            metadata={"git_commit": _git_commit(), "suite": "extended-v1"},
            max_concurrency=1,
            client=client,
            blocking=True,
        )
    finally:
        engine.close()


def main() -> None:
    """Mirror the versioned suite to LangSmith and optionally run an experiment."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--test-set", type=Path, default=DEFAULT_TEST_SET)
    parser.add_argument("--experiment-prefix", default="llm-direct-extended")
    parser.add_argument(
        "--upload-only",
        action="store_true",
        help="Create or verify the dataset without invoking the LLM pipeline.",
    )
    parser.add_argument(
        "--no-sync",
        action="store_true",
        help="Skip dataset sync and evaluate directly against the existing remote dataset."
        " Use this to measure against a frozen baseline dataset whose cases differ from"
        " the local test set (e.g. an older version with a different question wording).",
    )
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    if not os.environ.get("LANGSMITH_API_KEY"):
        raise RuntimeError("LANGSMITH_API_KEY is required to upload or evaluate this suite.")
    if not os.environ.get("LANGSMITH_TRACING"):
        os.environ["LANGSMITH_TRACING"] = "true"

    client = Client()
    if args.no_sync:
        print(f"Skipping sync — evaluating against existing remote dataset: {args.dataset}")
    else:
        cases = _load_test_set(args.test_set)
        sync_dataset(client, cases, dataset_name=args.dataset)
        print(f"LangSmith dataset ready: {args.dataset} ({len(cases)} versioned cases)")
    if args.upload_only:
        return
    run_langsmith_eval(
        client,
        dataset_name=args.dataset,
        experiment_prefix=args.experiment_prefix,
    )
    print(f"LangSmith experiment completed: {args.experiment_prefix}")


if __name__ == "__main__":
    main()
