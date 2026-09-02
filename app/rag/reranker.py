"""Second-stage reranking for vector-retrieval candidates."""

from app.rag.models import RerankResult, SearchResult
from app.rag.siliconflow import SiliconFlowClient


class SiliconFlowReranker:
    """Rerank vector-search candidates with SiliconFlow's BGE reranker."""

    def __init__(self, client: SiliconFlowClient) -> None:
        self.client = client

    def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        top_k: int,
    ) -> list[RerankResult]:
        """Return the highest scoring candidates in cross-encoder score order."""
        if not candidates or top_k <= 0:
            return []

        response = self.client.rerank(
            query=query,
            documents=[candidate.chunk.content for candidate in candidates],
            top_n=min(top_k, len(candidates)),
        )
        reranked: list[RerankResult] = []
        for item in response.get("results", []):
            index = item.get("index")
            score = item.get("relevance_score")
            if not isinstance(index, int) or not 0 <= index < len(candidates):
                continue
            if not isinstance(score, (int, float)):
                continue
            reranked.append(RerankResult(result=candidates[index], score=float(score)))
        return sorted(reranked, key=lambda result: result.score, reverse=True)
