# Evaluation Harness

This is a fixed, hand-checked correctness suite for the Text-to-SQL pipeline.
It evaluates the `TextToSQLEngine.run_query` single-query interface and saves a
JSON record for each run.

## Run a baseline

From the repository root, with dependencies installed:

```bash
uv run python eval/runner.py
```

If you use an activated virtual environment rather than `uv`, replace
`uv run python` with `python`.

The command runs all 18 cases in `test_set.json`, prints the pass rate and
failures, and saves a result to `eval/runs/<UTC timestamp>_<git commit>.json`.
Generated run files are intentionally ignored by Git.

## Compare two runs

First capture a baseline, make one scoped change, then run the harness again.
Compare the two saved files:

```bash
uv run python eval/compare.py eval/runs/RUN_A.json eval/runs/RUN_B.json
```

The comparison identifies improved and regressed query IDs, plus the net
change in passed tests. Runs must have the same test-case IDs.

## Test set result schema

`test_set.json` contains queries and expected values calculated from
`dataset/sales_data.csv`. Supported expectation kinds are:

- `scalar`: one numeric aggregate
- `rows`: unordered grouped values (numeric or text-valued)
- `rows_ordered`: ordered label/value rankings
- `rows_with_pct`: unordered contribution percentages
- `set`: unordered labels
- `empty`: no returned rows
- `rejected`: a query rejected before data is returned

Keep expected values hand-verified and commit changes to the test set with a
clear explanation. Do not rewrite expectations merely to make a changed
pipeline pass.

## LangSmith extended evaluation

`test_set_extended.json` is a separate, 30-case suite for broader capability
testing. It remains the version-controlled source of truth; the LangSmith
dataset is an idempotent mirror, named `analytics-engine-extended-v2` by
default. It records category, difficulty, and case ID as example metadata.

Upload or verify the dataset only:

```bash
uv run python eval/langsmith_eval.py --upload-only
```

Run a tracked experiment (this calls the configured LLM for every case):

```bash
LANGSMITH_TRACING=true uv run python eval/langsmith_eval.py \
  --experiment-prefix robust-corrector-extended
```

Open **Datasets & Experiments** in LangSmith, select
`analytics-engine-extended-v2`, and compare experiments by the `exact_match`
feedback score. The upload never overwrites a mismatched remote case: create a
new versioned dataset name when expected results change.

### Recorded baselines

| Suite | Result | Run |
|---|---:|---|
| Normal local regression suite (`test_set.json`) | **18 / 18 (100%)** | 2026-09-22 22:08 UTC, commit metadata `9192c21` |
| Complex LangSmith suite v1 (`test_set_extended.json`) | **21 / 30 (70.0%)** | `robust-corrector-extended-591032af`, 2026-09-22 |

The complex baseline failures are `p06`, `p03`, `n02`, `m02`, `t04`, `p04`,
`t02`, `n06`, and `m04`. Keep those cases unchanged; they are evidence of the
starting point, not errors to delete from the evaluation.

### Improve a score with LangSmith

1. Open the baseline experiment and filter the failed `exact_match` rows by
   `category` or `difficulty`. Inspect the query, generated SQL, validator or
   repair message, and returned data to identify one root cause.
2. Make one scoped change—for example, teach the intent prompt that “orders”
   means `COUNT(order_id)`, or add an attainment SQL shape. Do not place an
   expected answer or evaluation SQL in the production prompt.
3. Run unit tests and the normal regression suite first:

   ```bash
   uv run pytest -q
   uv run python eval/runner.py
   ```

4. Run a new LangSmith experiment name for that one change:

   ```bash
   LANGSMITH_TRACING=true uv run python eval/langsmith_eval.py \
     --experiment-prefix attainment-fix-v1
   ```

5. In **Datasets & Experiments**, select `analytics-engine-extended-v2` and
   compare `attainment-fix-v1` with `robust-corrector-extended-591032af`.
   Keep a change only when `exact_match` improves without regression in the
   normal 18-case suite.

The 30 complex cases are now a development evaluation set because they are
visible while improving the system. Before making a generalisation claim, add
a separate, locked holdout suite with new queries and never expose its expected
answers to prompts, feedback retrieval, or tuning decisions.

## Use a different pipeline version

Check out that version, run `uv run python eval/runner.py`, and retain its saved run
file before switching back. The run metadata records the Git commit, so the
files can be compared later without needing both versions checked out at once.
