"""Run bounded, schema-grounded SQL repair after an execution failure."""
from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any, Protocol
from langsmith import traceable

from engine.interfaces import SemanticLayerProtocol, ValueIndexProtocol
from engine.generation.generator import SQLGenerator
from engine.correction.prompts import SQL_REPAIR_SYSTEM, build_sql_repair_prompt
from engine.execution.executor import ExecutionResult, Executor, is_empty_result
from engine.types import AnalyticalSpec


LOGGER = logging.getLogger(__name__)


class SQLRepairError(ValueError):
    """Raised internally when an optional SQL-repair request cannot be used."""


class SQLRepairClient(Protocol):
    """Minimal contract for an injected SQL-repair provider."""

    # TODO(provider): enforce timeout and transient-network retry policy in its adapter.
    def generate(self, *, system: str, user: str) -> str:
        """Return one proposed SQL repair."""


@dataclass
class CorrectionResult:
    """The final execution result and auditable history of bounded repair attempts."""

    final: ExecutionResult
    attempts: list[ExecutionResult] = field(default_factory=list)
    retries_used: int = 0
    repair_errors: list[str] = field(default_factory=list)


class SelfCorrector:
    """Retry failed read-only SQL at most twice using an injected repair client."""

    _MAX_RETRIES = 2

    def __init__(
        self,
        executor: Executor,
        semantic_layer: SemanticLayerProtocol,
        value_index: ValueIndexProtocol,
        llm_client: SQLRepairClient | None = None,
        max_retries: int = _MAX_RETRIES,
    ) -> None:
        if isinstance(max_retries, bool) or not isinstance(max_retries, int):
            raise ValueError("max_retries must be an integer.")
        if not 0 <= max_retries <= self._MAX_RETRIES:
            raise ValueError(f"max_retries must be between 0 and {self._MAX_RETRIES}.")
        self._executor = executor
        self._semantic_layer = semantic_layer
        self._value_index = value_index
        self._llm = llm_client
        self._max_retries = max_retries
        self._view_name = getattr(semantic_layer, "VIEW_NAME", "v_sales")
        self._target_columns = {
            metric: target
            for metric in semantic_layer.get_metric_names()
            if (target := semantic_layer.get_target_column(metric)) is not None
        }

    @traceable
    def execute(self, sql: str, spec: AnalyticalSpec) -> CorrectionResult:
        """Execute SQL, requesting at most two safe repairs after failures."""

        attempts = [self._executor.run(sql)]
        repair_errors: list[str] = []
        retries_used = 0

        while self._llm is not None:
            latest = attempts[-1]
            needs_repair = not latest.success
            
            # Treat empty results as failures to heal formatting (unless it's a compare query)
            is_empty = is_empty_result(latest.df)
            if latest.success and is_empty and spec.operation != "compare":
                needs_repair = True
                
            if not needs_repair:
                break
                
            if retries_used >= self._max_retries:
                break
            retries_used += 1
            
            try:
                error_msg = latest.error or ""
                if latest.success and is_empty:
                    context = self._gather_context(spec)
                    error_msg = f"Query executed successfully but returned exactly 0 rows. Please check if your WHERE clause filters (string casing, exact formatting) might have caused this, and intelligently reformulate the time or string values. Context: {context}"
                repaired_sql = self._repair(latest.sql, error_msg, spec)
            except SQLRepairError as error:
                repair_errors.append(str(error))
                continue
            attempts.append(self._executor.run(repaired_sql))

        return CorrectionResult(
            final=attempts[-1],
            attempts=attempts,
            retries_used=retries_used,
            repair_errors=repair_errors,
        )

    def _repair(self, failed_sql: str, error_message: str, spec: AnalyticalSpec) -> str:
        """Request and validate one read-only SQL repair from the optional client."""

        try:
            raw = self._llm.generate(
                system=SQL_REPAIR_SYSTEM,
                user=build_sql_repair_prompt(
                    spec,
                    failed_sql,
                    error_message,
                    self._view_name,
                    self._semantic_layer.get_view_schema(),
                    self._target_columns,
                ),
            )
            return SQLGenerator.clean_read_only_sql(raw)
        except Exception as error:
            LOGGER.warning(
                "sql_repair_failed",
                extra={"error_type": type(error).__name__},
                exc_info=error,
            )
            raise SQLRepairError(f"SQL repair was unavailable: {type(error).__name__}: {error}") from error

    def _gather_context(self, spec: AnalyticalSpec) -> str:
        """Gather database context for failing filters to help the LLM repair."""
        context_parts = []
        
        # 1. Gather context for normal dimension filters using ValueIndex
        for f in spec.filters:
            matches = self._value_index.match(str(f.value))
            if matches:
                top_matches = [m.value for m in matches[:3]]
                context_parts.append(f"For '{f.column}', the closest database matches to '{f.value}' are {top_matches}.")
                
        # 2. Gather context for time columns using a quick sample query
        if spec.time_window and spec.time_window.column:
            col = spec.time_window.column
            try:
                res = self._executor.run(f'SELECT DISTINCT "{col}" FROM {self._view_name} ORDER BY "{col}" DESC LIMIT 5')
                if res.success and res.df is not None and not res.df.empty:
                    samples = res.df.iloc[:, 0].dropna().tolist()
                    context_parts.append(f"For '{col}', sample formats in the database are {samples}.")
            except Exception:
                pass
                
        if not context_parts:
            return "No additional context found."
        return " ".join(context_parts)
