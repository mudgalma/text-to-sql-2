"""Offline tests for SQL validation, repair diagnostics, and bounded correction."""
from __future__ import annotations

import pandas as pd
import pytest

from engine.correction.robust_corrector import RobustCorrector
from engine.execution.executor import Executor
from engine.scoring.direct_scorer import DirectConfidenceScorer
from engine.semantic_layer import DuckDBSemanticLayer
from engine.verification.result_validator import validate_result
from engine.verification.sql_validator import validate_sql


@pytest.fixture
def layer() -> DuckDBSemanticLayer:
    """Create the real semantic schema used by offline validator tests."""

    instance = DuckDBSemanticLayer(
        "dataset/sales_data.csv", "dataset/targets.csv", "dataset/data_dictionary.json"
    )
    yield instance
    instance.close()


def test_explain_accepts_valid_ctes_aliases_and_target_columns(layer: DuckDBSemanticLayer) -> None:
    """DuckDB binding accepts valid SQL that a regex allowlist would reject."""

    sql = (
        "WITH actual AS (SELECT region, SUM(revenue) AS actual FROM v_sales "
        "WHERE month = '2024-02' GROUP BY region) "
        "SELECT t.region, COALESCE(a.actual, 0) AS actual, "
        "CAST(t.target_revenue AS DOUBLE) AS target FROM targets t "
        "LEFT JOIN actual a ON t.region = a.region WHERE t.month = '2024-02' "
        "AND COALESCE(a.actual, 0) < CAST(t.target_revenue AS DOUBLE)"
    )
    intent = {"operation": "compare", "time_window": {"value": "2024-02"}}
    assert validate_sql(layer, sql, intent) == (True, None)


def test_explain_rejects_invalid_grouping(layer: DuckDBSemanticLayer) -> None:
    """Binding errors are detected before query execution returns data."""

    sql = "SELECT region, SUM(revenue) AS value FROM v_sales"
    valid, issue = validate_sql(layer, sql, {"operation": "aggregate"})
    assert valid is False
    assert issue is not None


def test_metric_validation_rejects_quantity_for_order_count(layer: DuckDBSemanticLayer) -> None:
    """A selected derived metric must use its semantic aggregate expression."""

    wrong_sql = "SELECT product_category, SUM(quantity) AS value FROM v_sales GROUP BY product_category"
    valid_sql = "SELECT product_category, COUNT(order_id) AS value FROM v_sales GROUP BY product_category"
    intent = {"operation": "aggregate", "metrics": ["orders"]}

    valid, issue = validate_sql(layer, wrong_sql, intent)
    assert valid is False
    assert issue == "Metric orders requires COUNT(order_id)."
    assert validate_sql(layer, valid_sql, intent) == (True, None)


def test_metric_validation_accepts_qualified_order_id(layer: DuckDBSemanticLayer) -> None:
    """Qualified SQL columns implement the same metric definition safely."""

    sql = "SELECT COUNT(v_sales.order_id) AS value FROM v_sales"
    assert validate_sql(layer, sql, {"operation": "aggregate", "metrics": ["orders"]}) == (
        True,
        None,
    )


def test_partitioned_rank_rejects_global_limit(layer: DuckDBSemanticLayer) -> None:
    """Top-per-group queries must not drop groups with a global limit."""

    sql = (
        "WITH ranked AS (SELECT region, product_name, SUM(revenue) AS value, "
        "ROW_NUMBER() OVER (PARTITION BY region ORDER BY SUM(revenue) DESC) AS rnk "
        "FROM v_sales GROUP BY region, product_name) "
        "SELECT region, product_name, value FROM ranked WHERE rnk = 1 LIMIT 1"
    )
    valid, issue = validate_sql(
        layer,
        sql,
        {"operation": "rank", "partition_by": ["region"], "rank_kind": "top_n", "limit": 1},
    )
    assert valid is False
    assert issue == "Partitioned ranking must not use a global LIMIT."


def test_global_rank_accepts_outer_rank_filter(layer: DuckDBSemanticLayer) -> None:
    """A CTE rank filter is equivalent to a one-row global LIMIT."""

    sql = (
        "WITH ranked AS (SELECT customer_id, SUM(revenue) AS value, "
        "ROW_NUMBER() OVER (ORDER BY SUM(revenue) DESC) AS rnk "
        "FROM v_sales GROUP BY customer_id) "
        "SELECT customer_id, value FROM ranked WHERE rnk = 1"
    )
    assert validate_sql(
        layer, sql, {"operation": "rank", "rank_kind": "single_winner"}
    ) == (True, None)


def test_result_validation_checks_partition_limits_and_percentages() -> None:
    """Post-validation enforces intent-specific dataframe constraints."""

    ranked = pd.DataFrame(
        {"region": ["APAC", "APAC"], "product_name": ["a", "b"], "value": [2, 1]}
    )
    valid, issue = validate_result(
        ranked,
        {"operation": "rank", "rank_kind": "top_n", "limit": 1, "partition_by": ["region"]},
    )
    assert valid is False
    assert issue is not None
    percentages = pd.DataFrame({"region": ["APAC", "NA"], "pct": [20.0, 80.0]})
    assert validate_result(percentages, {"percentages_requested": True}) == (True, None)


class FakeLLM:
    """Return one deterministic repair without network access."""

    def __init__(self, response: str) -> None:
        self.response = response
        self.last_user = ""

    def generate(self, *, system: str, user: str) -> str:
        """Record repair context and return the configured SQL."""

        self.last_user = user
        return self.response


def test_corrector_repairs_prevalidation_failure_with_bounded_context(layer: DuckDBSemanticLayer) -> None:
    """A bind failure reaches the repair LLM as structured diagnostic data."""

    llm = FakeLLM("SELECT SUM(revenue) AS value FROM v_sales")
    corrector = RobustCorrector(
        Executor(layer), layer, llm, DirectConfidenceScorer(), max_attempts=2
    )
    outcome = corrector.run(
        "Total revenue", {"operation": "aggregate", "metrics": ["revenue"]},
        "SELECT missing_column FROM v_sales",
    )
    assert outcome["status"] == "success"
    assert outcome["result"].iloc[0]["value"] == pytest.approx(6134.4)
    assert '"stage": "pre_validation"' in llm.last_user
    assert "missing_column" in llm.last_user


def test_executor_retains_bounded_duckdb_detail_for_repairs(layer: DuckDBSemanticLayer) -> None:
    """Execution diagnostics retain DuckDB's useful grouping failure details."""

    result = Executor(layer).run("SELECT region, SUM(revenue) AS value FROM v_sales")
    assert result.success is False
    assert result.error is not None
    assert "GROUP BY" in result.error
