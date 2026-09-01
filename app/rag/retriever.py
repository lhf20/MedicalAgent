"""In-memory cosine-similarity retriever for the first RAG demo."""

import math

from app.rag.models import SearchResult, TextChunk


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Compute cosine similarity without NumPy or a vector database."""
    if len(left) != len(right):
        raise ValueError("Embedding vectors must have the same dimensions")
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0


class LocalVectorStore:
    """Stores chunks and vectors in memory for a single command-line session."""

    def __init__(self, chunks: list[TextChunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("Each chunk must have exactly one embedding vector")
        self.chunks = chunks
        self.vectors = vectors

    def search(self, query_vector: list[float], top_k: int = 3) -> list[SearchResult]:
        scored = [
            SearchResult(chunk=chunk, score=cosine_similarity(query_vector, vector))
            for chunk, vector in zip(self.chunks, self.vectors)
        ]
        return sorted(scored, key=lambda result: result.score, reverse=True)[:top_k]
