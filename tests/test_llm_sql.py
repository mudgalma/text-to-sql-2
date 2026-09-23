"""Tests for the direct LLM-to-SQL safety boundary."""
from __future__ import annotations

from engine.llm_sql import IntentAnalyzer, SQLGenerator, build_dimension_values
from main import TextToSQLEngine


class FakeLLM:
    """Return preconfigured responses without external network calls."""

    def __init__(self, *responses: str) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[str, str]] = []

    def generate(self, *, system: str, user: str) -> str:
        """Return the next deterministic fake response."""

        self.calls.append((system, user))
        return self._responses.pop(0)


def test_direct_pipeline_executes_valid_llm_sql() -> None:
    """A valid intent and SQL response produce JSON-ready query results."""

    llm = FakeLLM(
        '{"operation":"aggregate","metrics":["revenue"]}',
        "SELECT SUM(revenue) AS value FROM v_sales",
    )
    engine = TextToSQLEngine("dataset", llm_client=llm)
    try:
        result = engine.run_query("Total revenue")
    finally:
        engine.close()
    assert result["result"] == [{"value": 6134.4}]
    assert result["generated_logic"] == "SELECT SUM(revenue) AS value FROM v_sales"


def test_unsafe_llm_sql_is_rejected_before_execution() -> None:
    """Mutating SQL from a model cannot pass the direct generation boundary."""

    llm = FakeLLM("DROP TABLE v_sales")
    generator = SQLGenerator(llm)
    assert generator.generate("Delete all orders", {"operation": "aggregate"}) is None


def test_sql_generator_injects_validated_metric_definitions() -> None:
    """SQL generation receives the business meaning of metrics from the semantic layer."""

    llm = FakeLLM("SELECT COUNT(order_id) AS value FROM v_sales")
    engine = TextToSQLEngine("dataset", llm_client=llm)
    try:
        assert engine._generator is not None
        engine._generator.generate("How many orders?", {"operation": "aggregate"})
    finally:
        engine.close()

    system, _ = llm.calls[-1]
    assert "<metric_definitions>" in system
    assert "- orders = count(order_id)" in system


def test_intent_analyzer_injects_validated_metric_definitions() -> None:
    """Intent generation receives the same canonical metric catalog as SQL generation."""

    llm = FakeLLM('{"operation":"aggregate","metrics":["orders"]}')
    engine = TextToSQLEngine("dataset", llm_client=llm)
    try:
        intent = engine._intent.analyze("Which category has the most orders?")
    finally:
        engine.close()

    assert intent is not None
    assert intent["metrics"] == ["orders"]
    system, _ = llm.calls[-1]
    assert "<metric_definitions>" in system
    assert "- orders = count(order_id)" in system


def test_dimension_values_are_grounded_for_relevant_filter_terms() -> None:
    """A real query value exposes its configured dimension values to the LLM."""

    engine = TextToSQLEngine("dataset", llm_client=FakeLLM())
    try:
        values = build_dimension_values(
            engine.layer,
            {"dimensions": ["product_category"], "partition_by": [], "filters": []},
            "Which category generates the most profit in NA?",
        )
    finally:
        engine.close()

    assert '- region: ["APAC", "EMEA", "NA"]' in values
    assert "- product_category:" in values


def test_sql_prompt_documents_generic_calculation_shapes() -> None:
    """The SQL LLM receives reusable, rather than query-specific, calculation rules."""

    llm = FakeLLM("SELECT 1 AS value")
    generator = SQLGenerator(llm)
    generator.generate(
        "Average monthly profit",
        {"operation": "aggregate", "calculation": "average_per_period"},
    )

    system, _ = llm.calls[-1]
    assert "calculation=average_per_period" in system
    assert "calculation=growth_percent" in system


def test_intent_uses_semantic_default_for_metric_free_rank_query() -> None:
    """A metric-free best-product query cannot silently become a profit query."""

    llm = FakeLLM(
        '{"operation":"rank","metrics":["profit"],'
        '"order_by":{"metric":"profit","direction":"DESC"}}'
    )
    engine = TextToSQLEngine("dataset", llm_client=llm)
    try:
        intent = engine._intent.analyze("Show me the best product per region")
    finally:
        engine.close()

    assert intent is not None
    assert intent["metrics"] == ["revenue"]
    assert intent["order_by"]["metric"] == "revenue"


def test_intent_disables_percentages_for_plain_breakdown() -> None:
    """A breakdown is not a contribution unless percentage terms are present."""

    llm = FakeLLM('{"operation":"aggregate","metrics":["revenue"],"percentages_requested":true}')
    engine = TextToSQLEngine("dataset", llm_client=llm)
    try:
        intent = engine._intent.analyze("Monthly revenue breakdown")
    finally:
        engine.close()

    assert intent is not None
    assert intent["percentages_requested"] is False


def test_invalid_intent_json_uses_complete_fallback() -> None:
    """Malformed intent output includes every field required by SQL generation."""

    intent = IntentAnalyzer(FakeLLM("not JSON")).analyze("Total revenue")
    assert intent is not None
    assert intent["percentages_requested"] is False
    assert intent["rank_kind"] is None
