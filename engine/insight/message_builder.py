"""Build consistent user-facing output messages for direct query outcomes."""
from __future__ import annotations

from typing import Any, Mapping


def success_message(intent: Mapping[str, Any], confidence: float) -> dict[str, str]:
    """Return the standard explanation shape for a validated result."""

    return {
        "understood": str(intent.get("reasoning", "LLM-generated analytics intent.")),
        "generated": f"Generated validated read-only SQL. Confidence: {confidence:.2f}.",
    }


def low_confidence_message(
    intent: Mapping[str, Any], confidence: float, warning: str
) -> dict[str, str]:
    """Return a result explanation that clearly asks the user to verify it."""

    return {
        "understood": str(intent.get("reasoning", "LLM-generated analytics intent.")),
        "generated": f"Low confidence ({confidence:.2f}): {warning}",
    }


def unresolved_message(query: str, attempts: list[Mapping[str, Any]]) -> dict[str, str]:
    """Return a concise clarification request after bounded repair is exhausted."""

    recent = "; ".join(str(item.get("issue", "unknown issue")) for item in attempts[-3:])
    return {
        "understood": f"Could not reliably answer: {query}",
        "generated": "Please clarify the metric, time period, or grouping. " + recent,
    }
