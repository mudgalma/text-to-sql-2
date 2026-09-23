"""Pre-execution SQL validation using DuckDB's parser and binder."""
from __future__ import annotations

import re
from typing import Any, Mapping

from engine.llm_sql import clean_read_only_sql
from engine.semantic_layer import SemanticLayerError


_MONTH_LITERAL = re.compile(r"\bmonth\s*=\s*'([^']+)'", re.IGNORECASE)
_AGGREGATE_CALL = re.compile(
    r"\b(count|sum|avg|min|max)\s*\(\s*(?:distinct\s+)?"
    r"(?:[a-z_][a-z0-9_]*\.)?([a-z_][a-z0-9_]*)\s*\)",
    re.IGNORECASE,
)


def validate_sql(
    semantic_layer: Any, sql: str, intent: Mapping[str, Any]
) -> tuple[bool, str | None]:
    """Validate safe SQL with ``EXPLAIN`` and intent-aware structural checks."""

    try:
        safe_sql = clean_read_only_sql(sql)
    except ValueError as error:
        return False, str(error)
    if not re.search(r"\bFROM\s+(?:v_sales|targets|ranked|actual|period_totals|monthly)\b", safe_sql, re.IGNORECASE):
        return False, "SQL must reference a known table or CTE (v_sales or targets)."
    try:
        semantic_layer.execute(f"EXPLAIN {safe_sql}")
    except SemanticLayerError as error:
        return False, _bound_error(str(error))

    upper_sql = safe_sql.upper()
    invalid_months = [
        literal for literal in _MONTH_LITERAL.findall(safe_sql)
        if not re.fullmatch(r"\d{4}-\d{2}", literal)
    ]
    if invalid_months:
        return False, "Month filters must use YYYY-MM format."
    if intent.get("operation") == "compare":
        required = ("TARGETS", "LEFT JOIN", "COALESCE", "TARGET_REVENUE")
        if any(token not in upper_sql for token in required):
            return False, "Target comparisons require targets, LEFT JOIN, COALESCE, and target_revenue."
    if intent.get("percentages_requested") and not re.search(
        r"\bAS\s+pct\b", safe_sql, re.IGNORECASE
    ):
        return False, "A percentage request must return a pct column."
    if not intent.get("percentages_requested") and re.search(
        r"\bAS\s+pct\b", safe_sql, re.IGNORECASE
    ):
        return False, "A plain result must not return contribution percentages."
    metric_error = _validate_metric_formula(semantic_layer, safe_sql, intent)
    if metric_error is not None:
        return False, metric_error
    if intent.get("operation") == "rank" and not intent.get("partition_by"):
        rank_error = _validate_global_rank_limit(safe_sql, intent)
        if rank_error is not None:
            return False, rank_error
    for dimension in intent.get("partition_by", []):
        pattern = rf"\bPARTITION\s+BY\s+(?:[A-Za-z_][A-Za-z0-9_]*\.)?{re.escape(str(dimension))}\b"
        if not re.search(pattern, safe_sql, re.IGNORECASE):
            return False, f"Ranking must partition by {dimension}."
    if intent.get("partition_by") and re.search(r"\bLIMIT\s+\d+\b", safe_sql, re.IGNORECASE):
        return False, "Partitioned ranking must not use a global LIMIT."
    return True, None


def _validate_metric_formula(
    semantic_layer: Any, sql: str, intent: Mapping[str, Any]
) -> str | None:
    """Require aggregate calls implied by a selected derived business metric."""

    metrics = intent.get("metrics")
    if not isinstance(metrics, list):
        return None
    sql_calls = {
        (function.lower(), column.lower())
        for function, column in _AGGREGATE_CALL.findall(sql)
    }
    for metric_name in metrics:
        if not isinstance(metric_name, str):
            continue
        formula = semantic_layer.get_metric_sql(metric_name)
        if not isinstance(formula, str):
            continue
        required_calls = {
            (function.lower(), column.lower())
            for function, column in _AGGREGATE_CALL.findall(formula)
        }
        missing = required_calls.difference(sql_calls)
        if missing:
            function, column = sorted(missing)[0]
            return f"Metric {metric_name} requires {function.upper()}({column})."
    return None


def _validate_global_rank_limit(sql: str, intent: Mapping[str, Any]) -> str | None:
    """Check the explicit rank contract without guessing a limit."""

    rank_kind = intent.get("rank_kind")
    expected_limit = {
        "top_n": intent.get("limit"),
        "single_winner": 1,
        "default_top": 5,
    }.get(rank_kind)
    if expected_limit is None:
        return None
    if not re.search(rf"\bLIMIT\s+{expected_limit}\b", sql, re.IGNORECASE):
        rank_filter = rf"\brnk\s*(?:=|<=)\s*{expected_limit}\b"
        if not re.search(rank_filter, sql, re.IGNORECASE):
            return f"Ranking requires LIMIT {expected_limit} or an equivalent rank filter."
    return None


def _bound_error(error: str, limit: int = 2_000) -> str:
    """Bound database diagnostics before they become repair-model context."""

    return error[:limit] + (" [truncated]" if len(error) > limit else "")
