"""Post-execution dataframe validation for an LLM-produced analytical result."""
from __future__ import annotations

import math
from typing import Any, Mapping

import pandas as pd


def validate_result(
    dataframe: pd.DataFrame | None, intent: Mapping[str, Any]
) -> tuple[bool, str | None]:
    """Validate result shape against explicit intent without inspecting SQL text."""

    if dataframe is None:
        return False, "Execution returned no dataframe."
    if intent.get("operation") == "rank":
        rank_error = _validate_rank_rows(dataframe, intent)
        if rank_error:
            return False, rank_error
    if intent.get("percentages_requested"):
        percentage_error = _validate_percentages(
            dataframe,
            require_total=intent.get("comparison_mode") != "attainment"
            and intent.get("calculation") != "growth_percent",
        )
        if percentage_error:
            return False, percentage_error
    if len(dataframe) == 0 and intent.get("dimensions") and intent.get("operation") != "compare":
        return False, "Grouped query returned no rows."
    if len(dataframe) == 1 and intent.get("filters") and dataframe.iloc[0].isna().all():
        return False, "Filtered aggregate returned only null values."
    return True, None


def _validate_rank_rows(dataframe: pd.DataFrame, intent: Mapping[str, Any]) -> str | None:
    """Ensure global and partitioned rankings respect their explicit limits."""

    rank_kind = intent.get("rank_kind")
    expected_limit = {"top_n": intent.get("limit"), "single_winner": 1, "default_top": 5}.get(rank_kind)
    if expected_limit is None:
        return None
    if not isinstance(expected_limit, int) or expected_limit < 1:
        return "Ranking has an invalid limit."
    partitions = [str(value) for value in intent.get("partition_by", [])]
    if not partitions:
        return None if len(dataframe) <= expected_limit else f"Ranking returned more than {expected_limit} rows."
    missing = [column for column in partitions if column not in dataframe.columns]
    if missing and not _partitions_are_filtered(intent, missing):
        return f"Ranking result is missing partition columns: {', '.join(missing)}."
    if missing:
        return None if len(dataframe) <= expected_limit else "Filtered ranking returned too many rows."
    counts = dataframe.groupby(partitions, dropna=False).size()
    if (counts > expected_limit).any():
        return f"Partitioned ranking returned more than {expected_limit} rows in a group."
    return None


def _partitions_are_filtered(intent: Mapping[str, Any], partitions: list[str]) -> bool:
    """Return whether omitted partition columns are fixed by intent filters."""

    filters = intent.get("filters", [])
    if not isinstance(filters, list):
        return False
    filtered_columns = {
        item.get("column")
        for item in filters
        if isinstance(item, Mapping) and isinstance(item.get("column"), str)
    }
    return set(partitions).issubset(filtered_columns)


def _validate_percentages(dataframe: pd.DataFrame, *, require_total: bool) -> str | None:
    """Require a percentage column whose values are within the valid range."""

    if "pct" not in dataframe.columns:
        return "Percentage query returned no pct column."
    values = dataframe["pct"].dropna()
    if not values.empty and (values.max() > 100.5 or values.min() < -0.5):
        return "Percentage values must be between 0 and 100."
    if require_total and not values.empty and not math.isclose(float(values.sum()), 100.0, abs_tol=0.2):
        return "Percentage values must sum to approximately 100."
    return None
