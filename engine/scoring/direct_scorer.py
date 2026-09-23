"""Confidence scoring from direct LLM-to-SQL pipeline evidence."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class DirectConfidence:
    """Inspectable signals used to decide whether a result is ready to return."""

    execution_success: float
    validation_success: float
    intent_completeness: float
    retry_score: float
    final: float


class DirectConfidenceScorer:
    """Score direct pipeline results only from observed, reproducible evidence."""

    _REQUIRED_INTENT_FIELDS = frozenset(
        {
            "operation", "metrics", "dimensions", "partition_by", "filters",
            "time_window", "percentages_requested", "rank_kind", "limit",
            "order_by", "transforms",
        }
    )

    def score(
        self,
        intent: Mapping[str, Any],
        *,
        execution_success: bool,
        validation_success: bool,
        retries_used: int,
    ) -> DirectConfidence:
        """Return a bounded score without inspecting untrusted SQL text heuristically."""

        execution = 1.0 if execution_success else 0.0
        validation = 1.0 if validation_success else 0.0
        completeness = len(self._REQUIRED_INTENT_FIELDS.intersection(intent)) / len(self._REQUIRED_INTENT_FIELDS)
        retry_score = max(0.0, 1.0 - 0.15 * max(0, retries_used))
        final = 0.35 * execution + 0.30 * validation + 0.20 * completeness + 0.15 * retry_score
        if not execution_success or not validation_success:
            final = min(final, 0.25)
        return DirectConfidence(
            execution_success=execution,
            validation_success=validation,
            intent_completeness=round(completeness, 3),
            retry_score=round(retry_score, 3),
            final=round(max(0.0, min(1.0, final)), 3),
        )
