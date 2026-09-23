"""Run the LLM-direct analytics query engine over a JSON query batch."""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from engine.correction.robust_corrector import RobustCorrector
from engine.execution.executor import Executor
from engine.insight.message_builder import (
    low_confidence_message,
    success_message,
    unresolved_message,
)
from engine.llm_sql import IntentAnalyzer, SQLClient, SQLGenerator
from engine.memory.feedback_store import FeedbackStore
from engine.scoring.direct_scorer import DirectConfidenceScorer
from engine.semantic_layer import DuckDBSemanticLayer


LOGGER = logging.getLogger(__name__)


class OpenRouterAdapter:
    """Adapt the OpenAI-compatible OpenRouter client to the local LLM protocol."""

    def __init__(self, client: Any, model: str = "anthropic/claude-3-haiku") -> None:
        self._client = client
        self._model = model

    def generate(self, *, system: str, user: str) -> str:
        """Return one model response or raise a provider error to the safe boundary."""

        response = self._client.chat.completions.create(
            model=self._model,
            temperature=0,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        content = response.choices[0].message.content
        if not isinstance(content, str):
            raise RuntimeError("LLM returned no text content.")
        return content


def load_queries(path_value: str | Path) -> list[str]:
    """Load a bounded list of query strings from the assignment JSON format."""

    path = Path(path_value).expanduser().resolve()
    try:
        raw_entries = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Query file could not be loaded as JSON.") from error
    if not isinstance(raw_entries, list):
        raise ValueError("Query JSON must be an array of objects.")
    queries = [entry.get("query") for entry in raw_entries if isinstance(entry, dict)]
    if len(queries) != len(raw_entries) or not all(isinstance(query, str) for query in queries):
        raise ValueError("Every query entry must contain a string query field.")
    return queries


class TextToSQLEngine:
    """Own the direct LLM-to-SQL components and semantic database connection."""

    def __init__(
        self,
        dataset_dir: str | Path = "dataset",
        feedback_path: str | Path | None = None,
        explanation_client: SQLClient | None = None,
        llm_client: SQLClient | None = None,
    ) -> None:
        load_dotenv()
        self.data_dir = Path(dataset_dir).expanduser().resolve()
        self.layer = DuckDBSemanticLayer(
            self.data_dir / "sales_data.csv",
            self.data_dir / "targets.csv",
            self.data_dir / "data_dictionary.json",
        )
        self.feedback_path = Path(feedback_path).expanduser().resolve() if feedback_path else None
        self._feedback_store = FeedbackStore(self.feedback_path) if self.feedback_path else None
        self._llm = llm_client or explanation_client or self._load_llm_client()
        self._intent = IntentAnalyzer(self._llm, self.layer) if self._llm is not None else None
        self._generator = SQLGenerator(self._llm, self.layer) if self._llm is not None else None
        self._corrector = (
            RobustCorrector(
                Executor(self.layer),
                self.layer,
                self._llm,
                DirectConfidenceScorer(),
                sql_system_prompt=self._generator.build_system_prompt,
                feedback_store=self._feedback_store,
            )
            if self._llm is not None
            else None
        )

    @staticmethod
    def _load_llm_client() -> SQLClient | None:
        """Create the configured OpenRouter client only when its API key is available."""

        api_key = os.environ.get("OPEN_ROUTER_KEY") or os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            return None
        from openai import OpenAI

        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            timeout=30.0,
            max_retries=2,
        )
        return OpenRouterAdapter(client)

    def add_feedback(self, query: str, corrected_sql: str) -> None:
        """Add a manual SQL correction for a query to the feedback CSV."""
        
        if not self._feedback_store or not self.feedback_path:
            raise ValueError("No feedback_path was provided during engine initialization.")
            
        self._feedback_store.save_correction(self.feedback_path, query, corrected_sql)

    def run_query(self, query: str) -> dict[str, Any]:
        """Translate, validate, execute, and return one analytics query safely."""

        # 1. Check Feedback Cache (Bypass LLM completely on exact match)
        if self._feedback_store:
            cached_sql = self._feedback_store.get_exact_correction(query)
            if cached_sql:
                # Execute cached SQL directly
                executor = Executor(self.layer)
                try:
                    exec_result = executor.run(cached_sql)
                    if not exec_result.success:
                        return _failure_output(query, f"Cached SQL failed: {exec_result.error}")
                        
                    return {
                        "query": query,
                        "generated_logic": cached_sql,
                        "result": exec_result.df.to_dict(orient="records") if exec_result.df is not None else [],
                        "confidence_score": 1.0,
                        "explanation": {
                            "understood": "Matched exactly with verified feedback cache.",
                            "generated": "Result served from verified feedback cache."
                        },
                    }
                except Exception as e:
                    return _failure_output(query, f"Cached SQL crashed: {e}")

        # 2. Normal LLM Pipeline
        if self._intent is None or self._generator is None or self._corrector is None:
            return _failure_output(query, "No LLM provider is configured.")
        intent = self._intent.analyze(query)
        if intent is None:
            return _failure_output(query, "Could not interpret this query.")
        if intent.get("operation") == "reject":
            return _failure_output(query, str(intent.get("reasoning", "Query rejected by intent analyzer.")))
        sql = self._generator.generate(query, intent)
        if sql is None:
            return _failure_output(query, "Could not generate safe SQL.")
        correction = self._corrector.run(query, intent, sql)
        if correction["status"] == "success":
            return _success_output(query, intent, correction)
        if correction["status"] == "low_confidence":
            return _low_confidence_output(query, intent, correction)
        return {
            "query": query,
            "generated_logic": correction.get("sql"),
            "result": None,
            "confidence_score": 0.0,
            "explanation": unresolved_message(query, correction.get("attempts", [])),
        }

    def close(self) -> None:
        """Close the underlying database connection."""

        self.layer.close()


def run_pipeline(
    dataset_dir: str | Path = "dataset",
    feedback_path: str | Path | None = None,
    explanation_client: SQLClient | None = None,
) -> list[dict[str, Any]]:
    """Run the assignment batch while preserving the existing public entry point."""

    engine = TextToSQLEngine(dataset_dir, feedback_path, explanation_client)
    try:
        return [engine.run_query(query) for query in load_queries(engine.data_dir / "nl_queries.json")]
    finally:
        engine.close()


def _failure_output(query: str, reason: str, sql: str | None = None) -> dict[str, Any]:
    """Return the standard output shape without exposing internal provider errors."""

    return {
        "query": query,
        "generated_logic": sql,
        "result": None,
        "confidence_score": 0.0,
        "explanation": {"understood": reason, "generated": "No result was generated."},
    }


def _success_output(
    query: str, intent: dict[str, Any], correction: dict[str, Any]
) -> dict[str, Any]:
    """Return one validated high-confidence correction result."""

    result = correction["result"]
    return {
        "query": query,
        "generated_logic": correction["sql"],
        "result": result.to_dict(orient="records"),
        "confidence_score": correction["confidence"],
        "explanation": success_message(intent, correction["confidence"]),
    }


def _low_confidence_output(
    query: str, intent: dict[str, Any], correction: dict[str, Any]
) -> dict[str, Any]:
    """Return the best bounded repair with a clear verification warning."""

    result = correction["result"]
    return {
        "query": query,
        "generated_logic": correction["sql"],
        "result": result.to_dict(orient="records"),
        "confidence_score": correction["confidence"],
        "explanation": low_confidence_message(
            intent,
            correction["confidence"],
            correction["warning"],
        ),
    }


def main() -> None:
    """Run the dataset query batch and persist its JSON results."""

    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--output", default="outputs.json")
    parser.add_argument("--feedback-path", default=None, help="Optional CSV with verified query corrections")
    args = parser.parse_args()
    outputs = run_pipeline(args.dataset_dir, feedback_path=args.feedback_path)
    serialized = json.dumps(outputs, indent=2, default=str)
    print(serialized)
    Path(args.output).write_text(serialized + "\n", encoding="utf-8")
    LOGGER.info("pipeline_completed", extra={"queries": len(outputs), "output": args.output})


if __name__ == "__main__":
    main()
