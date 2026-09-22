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

The command runs all 25 cases in `test_set.json`, prints the pass rate and
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

## Use a different pipeline version

Check out that version, run `uv run python eval/runner.py`, and retain its saved run
file before switching back. The run metadata records the Git commit, so the
files can be compared later without needing both versions checked out at once.
