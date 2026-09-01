"""Data models used by the minimal local RAG pipeline."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TextChunk:
    """A searchable piece of a Markdown knowledge document."""

    chunk_id: str
    source: str
    content: str


@dataclass(frozen=True)
class SearchResult:
    """A chunk and its cosine-similarity score."""

    chunk: TextChunk
    score: float
