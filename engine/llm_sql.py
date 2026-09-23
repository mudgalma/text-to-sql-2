"""Translate a raw analytics question into validated read-only SQL with an LLM."""
from __future__ import annotations

import json
import re
from typing import Any, Mapping, Protocol

from engine.interfaces import SemanticLayerProtocol


_METRIC_DEFINITIONS_TOKEN = "__METRIC_DEFINITIONS__"
_DIMENSION_VALUES_TOKEN = "__DIMENSION_VALUES__"
_VALUE_TOKEN = re.compile(r"[a-z0-9]+")
_PERCENTAGE_TERMS = re.compile(
    r"\b(?:contribution|distribution|percentage|share|split)\b|%", re.IGNORECASE
)


INTENT_SYSTEM = """You analyze analytics questions and return only a JSON intent.
<schema>
v_sales: order_id, order_date, month, year, region, country, city, customer_id,
customer_segment, product_category, product_subcategory, product_name, quantity,
unit_price, discount, shipping_cost, profit, revenue.
targets: region, month, target_revenue.
</schema>
<metric_definitions>
__METRIC_DEFINITIONS__
</metric_definitions>
<dimension_values>
__DIMENSION_VALUES__
</dimension_values>
<rules>
OPERATION:
- "top N", "best N", "bottom N", or "highest N" -> rank, rank_kind="top_n", limit=N.
- "who", "which one", "the single", or "the most" -> rank,
  rank_kind="single_winner", limit=1.
- "rank all", "list by", "sorted by", or "order by" -> rank,
  rank_kind="all_sorted", limit=null.
- "highest", "best", or "top" with no number -> rank,
  rank_kind="default_top", limit=5.
- "contribution", "distribution", "share", "percentage", or "%" ->
  percentages_requested=true, except when calculation is growth_percent or
  comparison_mode is attainment.
- "breakdown", "split", "by X", "per X", or "each X" ->
  percentages_requested=false unless share, distribution, contribution, or % also occurs.
- "target", "missed", "quota", "attainment", "vs", "against", or "beat" -> compare.
CALCULATIONS:
- "average monthly/weekly/quarterly/yearly X" -> calculation="average_per_period"
  and time_grain="month/week/quarter/year". This averages period totals, not raw rows.
- "combined", "together", or "total ... and ..." across periods ->
  calculation="combined_total" and periods=["YYYY-MM", ...].
- "percent change", "growth", or "change from X to Y" ->
  calculation="growth_percent" and periods=[earlier, later].
- Otherwise calculation=null and periods=[].
COMPARISONS:
- "attainment" -> comparison_mode="attainment" and percentages_requested=false.
- "missed", "below", or "under" target -> comparison_mode="below_target".
- "beat", "above", or "exceeded" target -> comparison_mode="above_target".
- Otherwise comparison_mode=null.
MONTH RESOLUTION (mandatory):
- Convert month names to YYYY-MM. The data contains only year 2024:
  January/Jan -> 2024-01, February/Feb -> 2024-02, March/Mar -> 2024-03.
- time_window.value must be YYYY-MM, never a month word.
- "yoy" or "year over year" -> transforms=["yoy"]. Otherwise -> aggregate.
DIMENSIONS:
- "per X", "each X", or "in every X" puts X in dimensions and partition_by.
- "by X", "across X", or "for each X" puts X in dimensions only.
- For a best/top result within every group, set tie_policy="stable_first". This returns
  one deterministic row per group, using a secondary ascending dimension sort for ties.
METRICS:
- Resolve a named metric to the exact canonical name in metric_definitions. A named
  metric must never fall back to revenue. Use the default metric only when none is named.
- sales, income, and turnover mean revenue; earnings means profit; AOV and average
  order value mean avg_order_value.
FILTERS:
- Country, city, and region names are filters on their matching column.
- Month names use time_window {"column":"month","value":"YYYY-MM"}; years use year.
REJECTION:
- If the query is conversational, off-topic, or too vague to be answered definitively with the available metrics and dimensions, set operation="reject".
</rules>
Return only valid JSON:
{"operation":"aggregate|rank|compare|trend|reject","metrics":["revenue"],"dimensions":[],
"partition_by":[],"filters":[],"time_window":null,"percentages_requested":false,
"rank_kind":"top_n|single_winner|all_sorted|default_top|null","limit":null,
"order_by":null,"transforms":[],"calculation":null,"comparison_mode":null,
"periods":[],"time_grain":null,"tie_policy":null,"reasoning":"one sentence"}
<examples>
Q: Top 2 cities by profit
{"operation":"rank","metrics":["profit"],"dimensions":["city"],"partition_by":[],"filters":[],"time_window":null,"percentages_requested":false,"rank_kind":"top_n","limit":2,"order_by":{"metric":"profit","direction":"DESC"},"transforms":[],"reasoning":"Top two cities by profit."}
Q: Sales distribution across all months
{"operation":"aggregate","metrics":["revenue"],"dimensions":["month"],"partition_by":[],"filters":[],"time_window":null,"percentages_requested":true,"rank_kind":null,"limit":null,"order_by":null,"transforms":[],"reasoning":"Monthly revenue as percentages."}
Q: Who spent the most?
{"operation":"rank","metrics":["revenue"],"dimensions":["customer_id"],"partition_by":[],"filters":[],"time_window":null,"percentages_requested":false,"rank_kind":"single_winner","limit":1,"order_by":{"metric":"revenue","direction":"DESC"},"transforms":[],"reasoning":"One customer with the highest revenue."}
Q: Rank all products by profit
{"operation":"rank","metrics":["profit"],"dimensions":["product_name"],"partition_by":[],"filters":[],"time_window":null,"percentages_requested":false,"rank_kind":"all_sorted","limit":null,"order_by":{"metric":"profit","direction":"DESC"},"transforms":[],"reasoning":"All products sorted by profit."}
</examples>
No prose or Markdown."""


SQL_SYSTEM = """You write one read-only DuckDB SQL query from a question and intent.
<schema>
v_sales: order_id, order_date, month, year, region, country, city, customer_id,
customer_segment, product_category, product_subcategory, product_name, quantity,
unit_price, discount, shipping_cost, profit, revenue.
targets: region, month, target_revenue.
</schema>
<metric_definitions>
__METRIC_DEFINITIONS__
</metric_definitions>
<dimension_values>
__DIMENSION_VALUES__
</dimension_values>
<rules>
READ-ONLY: Only SELECT or WITH. Never DROP, DELETE, INSERT, UPDATE, ALTER, or TRUNCATE.
ALIASES: Use only value, actual, target, and pct. Never alias to a database column name.
CONTRIBUTION:
- If intent.percentages_requested is true, add
  ROUND(100.0 * SUM(metric) / SUM(SUM(metric)) OVER (), 2) AS pct.
- If false, use a plain group by with no pct column.
RANK:
- rank_kind=top_n -> LIMIT intent.limit; single_winner -> LIMIT 1;
  all_sorted -> no LIMIT; default_top -> LIMIT 5. rank_kind is authoritative.
RANK WITH PARTITION_BY:
- Use this CTE and never add a global LIMIT after a partitioned rank. For a stable
  single winner per partition, use ROW_NUMBER with the ranked dimension ascending as
  the secondary tie-breaker:
  WITH ranked AS (
    SELECT d1, d2, SUM(metric) AS value,
           ROW_NUMBER() OVER (PARTITION BY d1 ORDER BY SUM(metric) DESC, d2 ASC) AS rnk
    FROM v_sales GROUP BY d1, d2
  ) SELECT d1, d2, value FROM ranked WHERE rnk <= intent.limit or 1.
COMPARE:
- Use this exact shape. Do not deviate:
  WITH actual AS (
    SELECT region, SUM(revenue) AS actual FROM v_sales
    WHERE month = 'YYYY-MM' GROUP BY region
  )
  SELECT t.region, COALESCE(a.actual, 0) AS actual,
         CAST(t.target_revenue AS DOUBLE) AS target
  FROM targets t LEFT JOIN actual a ON t.region = a.region
  WHERE t.month = 'YYYY-MM' AND COALESCE(a.actual, 0) < CAST(t.target_revenue AS DOUBLE).
- Month filters must use YYYY-MM, never a month word. Use LEFT JOIN, aggregate v_sales
  inside the CTE, CAST target_revenue to DOUBLE, and COALESCE actual to 0. Use < for
  missed/below/under and > for beat/exceeded/above.
CALCULATION SHAPES:
- calculation=average_per_period: first aggregate the metric by intent.time_grain in a
  CTE, then SELECT AVG(value) AS value from that CTE. Never average raw order rows.
- calculation=combined_total: SELECT one SUM(metric) AS value with month IN intent.periods.
  Do not GROUP BY month.
- calculation=growth_percent: first aggregate each requested period in a CTE, then return
  100.0 * (later_value - earlier_value) / NULLIF(earlier_value, 0) AS value.
ATTAINMENT:
- comparison_mode=attainment: calculate 100.0 * actual / target, using revenue grouped
  before a LEFT JOIN to targets. For a total attainment by month, aggregate actual and
  target across regions by month before dividing. Return the requested value; do not use
  contribution pct rules because attainment percentages do not sum to 100.
METRIC: Metric definitions are canonical. Aggregate a metric's validated expression;
for example, use COUNT(order_id) for a metric defined as count(order_id), never
SUM(quantity) unless the question explicitly asks for units or quantity.
OUTPUT: Return only the requested dimensions and value, actual/target, or pct. Return
one row for a specific winner. Return SQL only: no prose, Markdown, comments, or multiple statements.
</rules>
<examples>
Q: Top 2 cities by profit
SQL: SELECT city, SUM(profit) AS value FROM v_sales GROUP BY city ORDER BY value DESC LIMIT 2
Q: Who spent the most?
SQL: SELECT customer_id, SUM(revenue) AS value FROM v_sales GROUP BY customer_id ORDER BY value DESC LIMIT 1
Q: Sales distribution across all months
SQL: SELECT month, SUM(revenue) AS value, ROUND(100.0 * SUM(revenue) / SUM(SUM(revenue)) OVER (), 2) AS pct FROM v_sales GROUP BY month ORDER BY value DESC
Q: Which regions are below target for February?
SQL: WITH actual AS (SELECT region, SUM(revenue) AS actual FROM v_sales WHERE month = '2024-02' GROUP BY region) SELECT t.region, COALESCE(a.actual, 0) AS actual, CAST(t.target_revenue AS DOUBLE) AS target FROM targets t LEFT JOIN actual a ON t.region = a.region WHERE t.month = '2024-02' AND COALESCE(a.actual, 0) < CAST(t.target_revenue AS DOUBLE)
Q: What is the average monthly revenue?
SQL: WITH period_totals AS (SELECT month, SUM(revenue) AS value FROM v_sales GROUP BY month) SELECT AVG(value) AS value FROM period_totals
Q: Total revenue for January and February together
SQL: SELECT SUM(revenue) AS value FROM v_sales WHERE month IN ('2024-01', '2024-02')
Q: What is the attainment percentage by region for March?
SQL: WITH actual AS (SELECT region, SUM(revenue) AS actual FROM v_sales WHERE month = '2024-03' GROUP BY region) SELECT t.region, ROUND(100.0 * COALESCE(a.actual, 0) / NULLIF(CAST(t.target_revenue AS DOUBLE), 0), 2) AS value FROM targets t LEFT JOIN actual a ON t.region = a.region WHERE t.month = '2024-03'
</examples>"""

_FENCE = re.compile(r"^```(?:json|sql)?\s*|\s*```$", re.IGNORECASE)
_FORBIDDEN = re.compile(
    r"\b(?:ALTER|ATTACH|COPY|CREATE|DELETE|DROP|EXPORT|IMPORT|INSERT|INSTALL|LOAD|"
    r"PRAGMA|REPLACE|TRUNCATE|UPDATE|VACUUM)\b",
    re.IGNORECASE,
)


class SQLClient(Protocol):
    """Minimal LLM contract used by direct SQL generation."""

    def generate(self, *, system: str, user: str) -> str:
        """Return the model response for one bounded prompt."""


def build_metric_definitions(semantic_layer: SemanticLayerProtocol | None) -> str:
    """Format validated business metric formulas for bounded LLM context."""

    if semantic_layer is None:
        return "- No semantic metric definitions are available."
    definitions: list[str] = []
    for name in semantic_layer.get_metric_names():
        formula = semantic_layer.get_metric_definition(name)
        if formula:
            definitions.append(f"- {name} = {formula}")
    return "\n".join(definitions) or "- No semantic metric definitions are available."


def build_dimension_values(
    semantic_layer: SemanticLayerProtocol | None,
    intent: Mapping[str, Any] | None,
    query: str,
) -> str:
    """Return bounded real dimension values relevant to one analytics question."""

    if semantic_layer is None or not isinstance(query, str):
        return "- No relevant dimension values are available."
    dimensions = semantic_layer.get_dimension_names()
    selected = _selected_dimensions(intent, dimensions)
    query_tokens = set(_VALUE_TOKEN.findall(query.lower()))
    for dimension in dimensions:
        values = semantic_layer.get_dimension_values(dimension)
        if any(_value_tokens(value).issubset(query_tokens) for value in values):
            selected.add(dimension)
    lines = []
    for dimension in sorted(selected):
        values = semantic_layer.get_dimension_values(dimension)
        lines.append(f"- {dimension}: {json.dumps(values)}")
    return "\n".join(lines) or "- No relevant dimension values are available."


def render_metric_system(
    template: str, metric_definitions: str, dimension_values: str
) -> str:
    """Render one prompt template using only validated semantic metadata."""

    return template.replace(_METRIC_DEFINITIONS_TOKEN, metric_definitions).replace(
        _DIMENSION_VALUES_TOKEN, dimension_values
    )


def clean_read_only_sql(raw_sql: str) -> str:
    """Strip fences and reject SQL that is not exactly one read-only statement."""

    if not isinstance(raw_sql, str):
        raise ValueError("Generated SQL must be text.")
    sql = _FENCE.sub("", raw_sql.strip()).strip()
    if not sql.upper().startswith(("SELECT", "WITH")):
        raise ValueError("Generated SQL must start with SELECT or WITH.")
    if _FORBIDDEN.search(sql):
        raise ValueError("Generated SQL contains a non-read-only keyword.")
    if ";" in sql.rstrip(";"):
        raise ValueError("Generated SQL must contain exactly one statement.")
    return sql.rstrip(";").strip()


class IntentAnalyzer:
    """Request a structured intent without applying local query rules."""

    def __init__(
        self, llm_client: SQLClient, semantic_layer: SemanticLayerProtocol | None = None
    ) -> None:
        self._llm = llm_client
        self._semantic_layer = semantic_layer
        self._metric_definitions = build_metric_definitions(semantic_layer)

    def analyze(self, query: str) -> dict[str, Any] | None:
        """Return intent JSON or a complete fallback for malformed JSON output."""

        if not isinstance(query, str) or not query.strip() or len(query) > 2_000:
            return None
        try:
            system = render_metric_system(
                INTENT_SYSTEM,
                self._metric_definitions,
                build_dimension_values(self._semantic_layer, None, query),
            )
            raw = self._llm.generate(
                system=system, user=f"Question:\n{query}\nJSON:"
            )
            intent = json.loads(_clean(raw))
        except json.JSONDecodeError:
            return _fallback_intent()
        except Exception:
            return None
        if not isinstance(intent, dict):
            return _fallback_intent()
        return _normalize_intent(intent, query, self._semantic_layer)


class SQLGenerator:
    """Request SQL for an LLM-produced intent and enforce the read-only boundary."""

    def __init__(
        self, llm_client: SQLClient, semantic_layer: SemanticLayerProtocol | None = None
    ) -> None:
        self._llm = llm_client
        self._semantic_layer = semantic_layer
        self._metric_definitions = build_metric_definitions(semantic_layer)

    @property
    def system_prompt(self) -> str:
        """Return the base SQL prompt for backward-compatible repair configuration."""

        return render_metric_system(
            SQL_SYSTEM, self._metric_definitions, "- No relevant dimension values are available."
        )

    def build_system_prompt(self, query: str, intent: Mapping[str, Any]) -> str:
        """Return SQL instructions enriched with relevant real dimension values."""

        return render_metric_system(
            SQL_SYSTEM,
            self._metric_definitions,
            build_dimension_values(self._semantic_layer, intent, query),
        )

    def generate(self, query: str, intent: dict[str, Any]) -> str | None:
        """Return validated SQL for a query and intent, or ``None`` when unsafe."""

        if not isinstance(query, str) or not isinstance(intent, dict):
            return None
        user = (
            f"Question:\n{query}\n\n<intent_json>\n"
            f"{json.dumps(intent, sort_keys=True)}\n</intent_json>\n\nSQL:"
        )
        try:
            return clean_read_only_sql(
                self._llm.generate(system=self.build_system_prompt(query, intent), user=user)
            )
        except Exception:
            return None


def _clean(raw: str) -> str:
    """Remove one optional Markdown fence from an LLM response."""

    return _FENCE.sub("", raw.strip()).strip()


def _selected_dimensions(
    intent: Mapping[str, Any] | None, dimensions: list[str]
) -> set[str]:
    """Select configured dimensions explicitly named by a valid intent."""

    if not isinstance(intent, Mapping):
        return set()
    available = set(dimensions)
    selected: set[str] = set()
    for field in ("dimensions", "partition_by"):
        values = intent.get(field, [])
        if isinstance(values, list):
            selected.update(
                value for value in values if isinstance(value, str) and value in available
            )
    filters = intent.get("filters", [])
    if isinstance(filters, list):
        selected.update(
            item["column"]
            for item in filters
            if isinstance(item, Mapping)
            and isinstance(item.get("column"), str)
            and item["column"] in available
        )
    return selected


def _value_tokens(value: str) -> set[str]:
    """Normalize one trusted dimension value for relevance matching."""

    return set(_VALUE_TOKEN.findall(value.lower()))


def _normalize_intent(
    intent: dict[str, Any], query: str, semantic_layer: SemanticLayerProtocol | None
) -> dict[str, Any]:
    """Apply narrow semantic defaults when the question names no business metric."""

    normalized = dict(intent)
    if semantic_layer is None:
        return normalized
    query_tokens = set(_VALUE_TOKEN.findall(query.lower()))
    named_metric = _named_metric(query, query_tokens, semantic_layer)
    if named_metric is None:
        default_metric = semantic_layer.get_default_metric()
        normalized["metrics"] = [default_metric]
        normalized["reasoning"] = f"Uses the default metric {default_metric}."
        order_by = normalized.get("order_by")
        if isinstance(order_by, Mapping):
            normalized["order_by"] = {**order_by, "metric": default_metric}
    if not _PERCENTAGE_TERMS.search(query):
        normalized["percentages_requested"] = False
    return normalized


def _named_metric(
    query: str, query_tokens: set[str], semantic_layer: SemanticLayerProtocol
) -> str | None:
    """Return a canonical metric explicitly named by the query or its synonyms."""

    metric_names = semantic_layer.get_metric_names()
    metric_name_set = set(metric_names)
    candidates = [(name, name) for name in metric_names]
    candidates.extend(
        (phrase, canonical)
        for phrase, canonical in semantic_layer.get_synonyms().items()
        if canonical in metric_name_set
    )
    for phrase, canonical in sorted(candidates, key=lambda item: len(item[0]), reverse=True):
        phrase_tokens = _value_tokens(phrase)
        if phrase_tokens and phrase_tokens.issubset(query_tokens):
            return canonical
    return None


def _fallback_intent() -> dict[str, Any]:
    """Return the complete intent schema when model output is malformed JSON."""

    return {
        "operation": "aggregate",
        "metrics": ["revenue"],
        "dimensions": [],
        "partition_by": [],
        "filters": [],
        "time_window": None,
        "percentages_requested": False,
        "rank_kind": None,
        "limit": None,
        "order_by": None,
        "transforms": [],
        "calculation": None,
        "comparison_mode": None,
        "periods": [],
        "time_grain": None,
        "tie_policy": None,
        "reasoning": "Fallback intent.",
    }
