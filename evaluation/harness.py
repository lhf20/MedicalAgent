"""Dataset-driven baseline evaluator for the production Agent."""

import json
import logging
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

from app.agents.graph import MedicalImagingAgent
from app.rag.loader import build_chunks
from app.rag.reranker import SiliconFlowReranker
from app.rag.retriever import LocalVectorStore
from app.rag.siliconflow import SiliconFlowClient
from app.rag.workflow import MedicalRAGWorkflow
from app.tools.petct_result import query_petct_result
from evaluation.metrics import aggregate_results, score_case
from evaluation.tracing import EvaluationTrace, TracingReranker, TracingVectorStore


logger = logging.getLogger(__name__)


class EvaluationHarness:
    """Run the existing Agent and score outputs with deterministic rules."""

    def __init__(self, agent: MedicalImagingAgent, trace: EvaluationTrace) -> None:
        self.agent = agent
        self.trace = trace

    def _traced_tool(self, study_id: str, location: str, lesion_id: str | None = None) -> dict[str, Any]:
        result = query_petct_result(study_id, location, lesion_id)
        self.trace.tool_calls.append(
            {
                "name": "query_petct_result",
                "arguments": {
                    "study_id": study_id,
                    "location": location,
                    "lesion_id": lesion_id,
                },
                "result": result,
            }
        )
        return result

    def run_case(self, case: dict[str, Any]) -> dict[str, Any]:
        turns = case["turns"]
        if not turns:
            raise ValueError(f"Evaluation case {case.get('id')} has no turns")
        self.agent.reset_conversation()

        with patch("app.agents.graph.query_petct_result", side_effect=self._traced_tool):
            for setup_query in turns[:-1]:
                self.agent.invoke(setup_query)
            self.trace.reset()
            started = time.perf_counter()
            state = self.agent.invoke(turns[-1])
            latency_ms = (time.perf_counter() - started) * 1000

        final_top3 = self.trace.final_top_k(3)
        actual = {
            "intent": state["intent"],
            "resolved_query": state["resolved_query"],
            "final_answer": state["final_answer"],
            "initial_recall": list(self.trace.initial_recall),
            "reranked": list(self.trace.reranked),
            "rag_top3": final_top3,
            "top1_relevance_score": final_top3[0]["score"] if final_top3 else None,
            "tool_calls": list(self.trace.tool_calls),
        }
        return {
            "id": case["id"],
            "category": case["category"],
            "query": turns[-1],
            "latency_ms": latency_ms,
            "expected": case["expected"],
            "actual": actual,
            "scores": score_case(case, actual),
        }

    def run(self, dataset: dict[str, Any]) -> dict[str, Any]:
        cases = dataset.get("cases")
        if not isinstance(cases, list):
            raise ValueError("Dataset must contain a cases list")
        results = []
        for index, case in enumerate(cases, start=1):
            logger.info("Evaluation case %d/%d: %s", index, len(cases), case["id"])
            try:
                results.append(self.run_case(case))
            except Exception as error:
                logger.exception("Evaluation case failed: %s", case.get("id"))
                results.append(
                    {
                        "id": case.get("id"),
                        "category": case.get("category"),
                        "query": case.get("turns", [""])[-1] if case.get("turns") else "",
                        "latency_ms": 0.0,
                        "expected": case.get("expected", {}),
                        "actual": {"error": str(error)},
                        "scores": {},
                    }
                )
        return {
            "dataset": dataset.get("name", "unknown"),
            "case_count": len(cases),
            "summary": aggregate_results(results),
            "cases": results,
        }


def build_production_harness(project_dir: Path) -> EvaluationHarness:
    """Compose tracing wrappers around existing production components."""
    client = SiliconFlowClient(str(project_dir))
    knowledge_dir = project_dir / "data" / "knowledge"
    chunks = build_chunks(knowledge_dir)
    if not chunks:
        raise RuntimeError(f"No Markdown knowledge documents found in: {knowledge_dir}")
    base_store = LocalVectorStore(chunks, client.embed([chunk.content for chunk in chunks]))
    trace = EvaluationTrace()
    vector_store = TracingVectorStore(base_store, trace)
    reranker = TracingReranker(SiliconFlowReranker(client), trace)
    workflow = MedicalRAGWorkflow(client, vector_store, reranker)
    return EvaluationHarness(MedicalImagingAgent(workflow, client), trace)


def load_dataset(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as dataset_file:
        return json.load(dataset_file)
