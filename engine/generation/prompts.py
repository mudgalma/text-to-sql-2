"""Prompt construction for the optional Phase 5 SQL fallback."""
from __future__ import annotations

import json
from dataclasses import asdict
from typing import Mapping

from engine.types import AnalyticalSpec


SQL_GEN_SYSTEM = """You generate one read-only DuckDB SQL query.

<schema>
View v_sales:
  order_id, order_date (DATE), month ('YYYY-MM'), year (INT),
  region, country, city, customer_id, customer_segment,
  product_category, product_subcategory, product_name,
  quantity, unit_price, discount, shipping_cost, profit,
  revenue  -- pre-computed: quantity * unit_price * (1 - discount)

Table targets:
  region, month ('YYYY-MM'), target_revenue
</schema>

<business_rules>
- Use v_sales.revenue directly. Never recompute it.
- avg_order_value = SUM(revenue) / COUNT(order_id)
- "contribution"/"distribution"/"share"/"%" -> add pct column:
    ROUND(100.0 * SUM(m) / SUM(SUM(m)) OVER (), 2) AS pct
- "top N per group" / "each" -> DENSE_RANK() OVER (PARTITION BY ...)
- Target comparison -> JOIN targets on region AND month.
  Use LEFT JOIN from targets so regions with no sales still appear.
- YoY -> LAG(value) OVER (ORDER BY year)
- Aliases ONLY: value, actual, target, pct, month, year, region, city
- Never use a column name from another table as an alias.

Return SQL only. No prose. No markdown fences. No comments.
</business_rules>

<examples>
Q: Top 2 cities by profit
SQL: SELECT city, SUM(profit) AS value FROM v_sales GROUP BY city ORDER BY value DESC LIMIT 2

Q: Sales distribution across all months
SQL: SELECT month, SUM(revenue) AS value,
            ROUND(100.0 * SUM(revenue) / SUM(SUM(revenue)) OVER (), 2) AS pct
     FROM v_sales GROUP BY month ORDER BY value DESC

Q: Revenue by region and month
SQL: SELECT region, month, SUM(revenue) AS value
     FROM v_sales GROUP BY region, month ORDER BY region, month

Q: Which region missed its target in Feb?
SQL: WITH actual AS (
       SELECT region, SUM(revenue) AS actual FROM v_sales
       WHERE month = '2024-02' GROUP BY region)
     SELECT t.region, COALESCE(a.actual, 0) AS actual, t.target_revenue AS target
     FROM targets t LEFT JOIN actual a ON t.region = a.region
     WHERE t.month = '2024-02' AND COALESCE(a.actual, 0) < t.target_revenue

Q: Top product in each region
SQL: WITH ranked AS (
       SELECT region, product_name, SUM(revenue) AS value,
              DENSE_RANK() OVER (PARTITION BY region ORDER BY SUM(revenue) DESC) AS rnk
       FROM v_sales GROUP BY region, product_name)
     SELECT region, product_name, value FROM ranked WHERE rnk = 1
</examples>
"""



def build_sql_schema_context(
    view_name: str,
    view_schema: Mapping[str, str],
    target_columns: Mapping[str, str],
) -> str:
    """Return the trusted schema context shared by SQL-generation prompts."""

    schema_lines = "\n".join(
        f"- {name} ({data_type})" for name, data_type in sorted(view_schema.items())
    )
    targets = json.dumps(dict(sorted(target_columns.items())), sort_keys=True)
    return f"""<schema>
View {view_name}:
{schema_lines}
Target table: targets
Metric-to-target-column mapping: {targets}
</schema>"""


def build_sql_user_prompt(
    spec: AnalyticalSpec,
    view_name: str,
    view_schema: Mapping[str, str],
    target_columns: Mapping[str, str],
) -> str:
    """Build a schema-grounded, data-delimited prompt for a fallback client."""

    serialized_spec = json.dumps(asdict(spec), default=str, sort_keys=True)
    return f"""{build_sql_schema_context(view_name, view_schema, target_columns)}

<analytical_spec>
{serialized_spec}
</analytical_spec>

Generate the SQL now."""
