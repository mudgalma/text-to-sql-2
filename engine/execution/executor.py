"""Execute validated, read-only SQL against the semantic layer."""
from __future__ import annotations

from dataclasses import dataclass
import logging
from langsmith import traceable

import pandas as pd

from engine.interfaces import SemanticLayerProtocol
from engine.llm_sql import clean_read_only_sql


LOGGER = logging.getLogger(__name__)


def is_empty_result(df: pd.DataFrame | None) -> bool:
    """Return True if the dataframe is completely empty or contains only a single row of nulls."""
    if df is None:
        return True
    if len(df) == 0:
        return True
    if len(df) == 1 and df.iloc[0].isnull().all():
        return True
    return False


@dataclass
class ExecutionResult:
    """One SQL execution outcome, including either data or an error description."""

    success: bool
    sql: str
    df: pd.DataFrame | None = None
    error: str | None = None


class Executor:
    """Run one safe SQL query and return a structured result instead of raising."""

    def __init__(self, semantic_layer: SemanticLayerProtocol) -> None:
        self._semantic_layer = semantic_layer

    @traceable
    def run(self, sql: str) -> ExecutionResult:
        """Execute one read-only query and capture any expected execution failure."""

        try:
            safe_sql = clean_read_only_sql(sql)
            dataframe = self._semantic_layer.execute(safe_sql)
            if not isinstance(dataframe, pd.DataFrame):
                raise TypeError("Semantic-layer execution must return a pandas DataFrame.")
            return ExecutionResult(success=True, sql=safe_sql, df=dataframe)
        except Exception as error:
            LOGGER.warning(
                "sql_execution_failed",
                extra={"error_type": type(error).__name__},
                exc_info=error,
            )
            return ExecutionResult(
                success=False,
                sql=sql,
                error=f"{type(error).__name__}: {error}",
            )
