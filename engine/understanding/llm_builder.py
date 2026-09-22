"""LLM-backed analytical specification builder using tool calling."""
from __future__ import annotations

import json
import logging
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from engine.interfaces import SemanticLayerProtocol
from engine.understanding.prompts import build_system_prompt, build_user_prompt
from engine.types import (
    AnalyticalSpec,
    Filter,
    MetricFilter,
    OrderSpec,
    TaggedQuery,
    TimeComparison,
    TimeConstraint,
)

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output schema — defines exactly what JSON the LLM must return.
# ---------------------------------------------------------------------------

class FilterOutput(BaseModel):
    """A WHERE-clause filter on a single schema column."""
    column: str = Field(description="Exact schema column name to filter on")
    operator: str = Field(description="SQL operator: =, !=, >, <, >=, <=, IN, LIKE")
    value: Any = Field(description="The filter value")


class MetricFilterOutput(BaseModel):
    """A HAVING-clause filter on an aggregated metric."""
    metric: str = Field(description="Metric name (e.g. revenue, profit)")
    operator: str = Field(description="SQL operator")
    target_value: float | None = None
    target_metric: str | None = None


class TimeConstraintOutput(BaseModel):
    """A time period constraint."""
    raw: str = Field(description="Raw period string from the user query, e.g. '2', 'February', '2024-02'")
    column: str = Field(
        default="order_date",
        description="Schema column to apply this filter to. Use 'month' for month-level, 'order_date' for date-level."
    )
    resolution: Literal[
        "EXACT_PERIOD", "LATEST_PERIOD", "PERIOD_OVER_PERIOD", "ALL_PERIODS"
    ] = Field(default="EXACT_PERIOD")


class TimeComparisonOutput(BaseModel):
    """A period-over-period comparison."""
    baseline: TimeConstraintOutput
    target: TimeConstraintOutput | None = None
    delta_type: Literal["CUSTOM", "MOM", "YOY"] = "CUSTOM"


class AnalyticalSpecOutput(BaseModel):
    """The analytical specification to build from the user query."""

    operation: Literal["aggregate", "rank", "compare", "trend"] = Field(
        default="aggregate",
        description="Query type: aggregate (sum/avg), rank (top-N), compare (vs target), trend (over time)"
    )
    metrics: list[str] = Field(description="List of metric names e.g. ['revenue', 'profit']")
    group_by: list[str] = Field(
        default_factory=list,
        description="Dimension columns to GROUP BY"
    )
    partition_by: list[str] = Field(
        default_factory=list,
        description="Dimensions to partition within for window functions (e.g. rank within group)"
    )
    limit: int | None = Field(default=None, description="TOP-N limit for rank queries")
    transforms: list[
        Literal["contribution_pct", "yoy", "mom", "rolling_avg"]
    ] = Field(default_factory=list, description="Post-aggregation transforms to apply")
    filters: list[FilterOutput] = Field(default_factory=list, description="WHERE clause filters")
    metric_filters: list[MetricFilterOutput] = Field(default_factory=list, description="HAVING clause filters")
    order_by: dict[str, str] | None = Field(
        default=None,
        description="e.g. {'metric': 'revenue', 'direction': 'DESC'}"
    )
    time_window: TimeConstraintOutput | None = Field(default=None, description="Time period filter")
    time_comparison: TimeComparisonOutput | None = Field(default=None, description="Period-over-period comparison")
    defaults_applied: list[str] = Field(
        default_factory=list,
        description="List 'default_metric' here if no metric was mentioned in the query"
    )


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

class LLMSpecBuilder:
    """Build an AnalyticalSpec via a single tool-call to the LLM."""

    MODEL = "anthropic/claude-3-haiku"

    def __init__(self, client: Any, semantic_layer: SemanticLayerProtocol) -> None:
        self._client = client
        self._sl = semantic_layer

    def build(self, tagged: TaggedQuery) -> AnalyticalSpec | None:
        """Ask the LLM to fill the AnalyticalSpec tool schema and convert the result."""
        try:
            response = self._client.chat.completions.create(
                model=self.MODEL,
                messages=[
                    {"role": "system", "content": self._system_prompt()},
                    {"role": "user", "content": build_user_prompt(tagged)},
                ],
                tools=[{
                    "type": "function",
                    "function": {
                        "name": "build_analytical_spec",
                        "description": "Build a structured analytical specification from the tagged query.",
                        "parameters": AnalyticalSpecOutput.model_json_schema(),
                    }
                }],
                tool_choice={"type": "function", "function": {"name": "build_analytical_spec"}},
                temperature=0.0,
            )
            tool_calls = response.choices[0].message.tool_calls
            if not tool_calls:
                LOGGER.warning("llm_build_failed: no tool call returned")
                return None
            raw = json.loads(tool_calls[0].function.arguments)
            return self._to_spec(raw)
        except Exception as exc:
            LOGGER.warning("llm_build_failed: %s", exc)
            return None

    def _system_prompt(self) -> str:
        """Inject schema context so the LLM can use correct column names."""
        base = build_system_prompt(self._sl)
        schema_cols = list(self._sl.get_view_schema().keys())
        return (
            f"{base}\n\n"
            f"Available schema columns: {schema_cols}\n"
            f"For time_window: use column='month' for month-number filters (e.g. month 2 = February), "
            f"column='order_date' for full date filters. "
            f"Set 'raw' to the exact period from the user query."
        )

    @staticmethod
    def _to_spec(data: dict[str, Any]) -> AnalyticalSpec | None:
        """Validate LLM output and convert to domain types."""
        try:
            parsed = AnalyticalSpecOutput.model_validate(data)
            order_by = OrderSpec(**parsed.order_by) if parsed.order_by else None
            time_window = (
                TimeConstraint(**parsed.time_window.model_dump())
                if parsed.time_window else None
            )
            comparison = (
                TimeComparison(
                    baseline=TimeConstraint(**parsed.time_comparison.baseline.model_dump()),
                    target=(
                        TimeConstraint(**parsed.time_comparison.target.model_dump())
                        if parsed.time_comparison.target else None
                    ),
                    delta_type=parsed.time_comparison.delta_type,
                )
                if parsed.time_comparison else None
            )
            return AnalyticalSpec(
                operation=parsed.operation,
                metrics=parsed.metrics,
                group_by=parsed.group_by,
                partition_by=parsed.partition_by,
                order_by=order_by,
                filters=[Filter(**f.model_dump()) for f in parsed.filters],
                metric_filters=[MetricFilter(**mf.model_dump()) for mf in parsed.metric_filters],
                limit=parsed.limit,
                transforms=parsed.transforms,
                time_window=time_window,
                time_comparison=comparison,
                defaults_applied=parsed.defaults_applied,
            )
        except (KeyError, TypeError, ValidationError, ValueError) as exc:
            print("llm_spec_conversion_failed:", exc)
            LOGGER.warning("llm_spec_conversion_failed: %s", exc)
            return None
