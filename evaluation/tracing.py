"""Non-invasive wrappers that capture retrieval and Tool behavior for evaluation."""

from dataclasses import dataclass, field
from typing import Any

from app.rag.models import RerankResult, SearchResult


@dataclass
class EvaluationTrace:
    initial_recall: list[dict[str, Any]] = field(default_factory=list)
    reranked: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)

    def reset(self) -> None:
        self.initial_recall.clear()
        self.reranked.clear()
        self.tool_calls.clear()

    def final_top_k(self, top_k: int = 3) -> list[dict[str, Any]]:
        return (self.reranked or self.initial_recall)[:top_k]


class TracingVectorStore:
    """Delegate to the production vector store and record returned candidates."""

    def __init__(self, delegate: Any, trace: EvaluationTrace) -> None:
        self.delegate = delegate
        self.trace = trace

    def search(self, query_vector: list[float], top_k: int = 3) -> list[SearchResult]:
        results = self.delegate.search(query_vector, top_k=top_k)
        self.trace.initial_recall = [
            {
                "chunk_id": item.chunk.chunk_id,
                "source": item.chunk.source,
                "score": item.score,
            }
            for item in results
        ]
        return results


class TracingReranker:
    """Delegate to the production reranker and record its ranking and scores."""

    def __init__(self, delegate: Any, trace: EvaluationTrace) -> None:
        self.delegate = delegate
        self.trace = trace

    def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        top_k: int,
    ) -> list[RerankResult]:
        results = self.delegate.rerank(query, candidates, top_k=top_k)
        self.trace.reranked = [
            {
                "chunk_id": item.result.chunk.chunk_id,
                "source": item.result.chunk.source,
                "score": item.score,
            }
            for item in results
        ]
        return results
