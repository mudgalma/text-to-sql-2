import re

with open('README.md', 'r') as f:
    content = f.read()

# 1. Overview tagline
old_overview = "Unlike end-to-end LLM approaches, this engine uses a **multi-phase deterministic pipeline** with optional LLM augmentation at controlled seams, providing **auditability**, **reproducibility**, and **safety** without sacrificing flexibility."
new_overview = "The system uses two LLM calls per query — intent analysis and SQL generation — with deterministic validation, execution, and confidence scoring between them. LLM where judgment is needed; code where correctness is provable."
content = content.replace(old_overview, new_overview)

# 2. Architecture
arch_start = content.find("## 🏗 Architecture")
arch_end = content.find("## 📁 Project Structure")
new_arch = """## 🏗 Architecture

### High-Level Pipeline

```mermaid
flowchart LR
    Q[Query] -->|LLM| I[Intent]
    I -->|LLM| S[SQL]
    S -->|Code| V[Validate]
    V -->|Code| E[Execute]
    E -->|Code| C[Score]
    C -->|LLM| X[Explain]
    X --> J[JSON]
```

### Phase Responsibilities

| Phase | Engine | Responsibility |
|:------|:-------|:---------------|
| Intent | LLM | Parse raw query into structured JSON intent |
| Generation | LLM | Translate intent to DuckDB SQL |
| Validation | Code | Reject non-SELECT/WITH statements via EXPLAIN |
| Execution | Code | Run query and return dataframe |
| Scoring | Code | Compute algorithmic confidence from pipeline signals |
| Explanation | LLM | Summarize execution state into a short sentence |

### The "No Hardcoding" Guarantee

| Aspect | How it's handled |
|:-------|:-----------------|
| Dimensions & Metrics | Injected dynamically into prompts |
| Value Matching | Injected dynamically (up to cardinality limits) |
| SQL Templates | Zero templates; LLM handles grammar |
| Intent Routing | Handled purely by LLM reasoning |

### Three Output States

1. **Success**: Valid JSON intent → Valid SQL → Results Returned
2. **Rejection**: LLM explicitly returns `operation="reject"` for off-topic/vague queries
3. **Fallback**: LLM fails to generate valid JSON/SQL → Empty result + 0.0 confidence

---

"""
content = content[:arch_start] + new_arch + content[arch_end:]

# 3. Project Structure
ps_start = content.find("## 📁 Project Structure")
ps_end = content.find("## ⚖️ Trade-offs")
new_ps = """## 📁 Project Structure

```
text-tosql/
├── main.py
├── dataset/
│   ├── sales_data.csv
│   ├── targets.csv
│   ├── data_dictionary.json
│   └── nl_queries.json
├── engine/
│   ├── interfaces.py
│   ├── types.py
│   ├── semantic_layer.py
│   ├── llm_sql.py              # ← the whole pipeline
│   ├── execution/
│   │   └── executor.py
│   ├── correction/
│   │   └── repairer.py
│   ├── scoring/
│   │   └── scorer.py
│   ├── insight/
│   │   └── explainer.py
│   └── memory/
│       └── feedback_store.py
├── eval/
│   ├── runner.py
│   ├── test_set_unseen.json
│   ├── test_set_complex.json
│   └── test_set_edge.json
├── tests/
├── scripts/
└── pyproject.toml
```

---

"""
content = content[:ps_start] + new_ps + content[ps_end:]

# 4. Trade-offs
to_start = content.find("### Deterministic Rules vs. LLM Flexibility")
to_end = content.find("### Design Decisions")
new_to = """### LLM-Direct Design Tradeoffs

| Decision | Chosen | Alternative | Rationale |
|:---------|:-------|:------------|:----------|
| Pipeline style | LLM-direct (2 calls) | Rules + templates | LLM generalizes; no code change for new synonyms |
| Intent representation | Structured JSON | Free-form LLM output | Auditable — the intent is visible proof of understanding |
| Validation | DuckDB `EXPLAIN` | Regex allowlist | DuckDB's parser is authoritative |
| Confidence | Algorithmic | LLM self-reported | Auditable — no fabricated numbers |
| Self-correction | Bounded (max 5) | Unbounded or none | Balances coverage vs. cost |
| Examples | In prompt | None | Shape guidance is standard practice |

"""
content = content[:to_start] + new_to + content[to_end:]

# 5. Sample Output string
old_output = "I understood this as a ranking query for profit, grouped by city, limited to 2. I generated the SQL with the rule-based specification and a validated SQL template, executed it with no repairs, and returned 2 row(s). Confidence is 0.93."
new_output = "I understood this as a ranking query for profit, grouped by city, limited to 2. I interpreted your question into a structured intent, generated the SQL from it, executed against the database, and validated the result. Confidence is 0.93."
content = content.replace(old_output, new_output)

with open('README.md', 'w') as f:
    f.write(content)

print("DONE")
