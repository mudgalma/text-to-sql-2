"""Tests for the LangSmith evaluation adapter without network access."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from langsmith.utils import LangSmithNotFoundError

from eval.langsmith_eval import accuracy_evaluator, sync_dataset


class FakeClient:
    """Small in-memory subset of the LangSmith dataset client."""

    def __init__(self) -> None:
        self.dataset = SimpleNamespace(id="dataset-1")
        self.examples: list[Any] = []
        self.created_dataset = False

    def read_dataset(self, *, dataset_name: str) -> Any:
        """Return the dataset after it has been created once."""

        del dataset_name
        if not self.created_dataset:
            raise LangSmithNotFoundError("dataset missing")
        return self.dataset

    def create_dataset(self, dataset_name: str, **kwargs: Any) -> Any:
        """Record local dataset creation."""

        del dataset_name, kwargs
        self.created_dataset = True
        return self.dataset

    def list_examples(self, *, dataset_id: str) -> list[Any]:
        """Return existing in-memory examples."""

        assert dataset_id == self.dataset.id
        return self.examples

    def create_example(self, *, dataset_id: str, **payload: Any) -> None:
        """Append one in-memory example."""

        assert dataset_id == self.dataset.id
        self.examples.append(SimpleNamespace(**payload))


def test_sync_dataset_is_idempotent_and_preserves_case_metadata() -> None:
    """A repeated upload adds no duplicate examples or hidden mutations."""

    cases = [
        {
            "id": "case-1",
            "query": "Total revenue",
            "category": "aggregate",
            "difficulty": "easy",
            "expected": {"kind": "scalar", "value": 1.0},
        }
    ]
    client = FakeClient()
    sync_dataset(client, cases, dataset_name="test-suite")
    sync_dataset(client, cases, dataset_name="test-suite")

    assert len(client.examples) == 1
    assert client.examples[0].inputs == {"query": "Total revenue"}
    assert client.examples[0].metadata == {
        "case_id": "case-1",
        "category": "aggregate",
        "difficulty": "easy",
    }


def test_accuracy_evaluator_reuses_the_local_comparator() -> None:
    """LangSmith scores exactly the same result contract as local evaluation."""

    run = SimpleNamespace(outputs={"result": [{"value": 3.0}]})
    example = SimpleNamespace(outputs={"expected": {"kind": "scalar", "value": 3.0}})

    assert accuracy_evaluator(run, example) == {"key": "exact_match", "score": 1.0}
