"""Markdown loading and dependency-free character-based text splitting."""

from pathlib import Path

from app.rag.models import TextChunk


def load_markdown_documents(knowledge_dir: Path) -> list[tuple[str, str]]:
    """Return (filename, cleaned text) for every Markdown file in a directory."""
    documents: list[tuple[str, str]] = []
    for path in sorted(knowledge_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8").strip()
        if text:
            documents.append((path.name, text))
    return documents


def split_text(text: str, chunk_size: int = 600, chunk_overlap: int = 100) -> list[str]:
    """Split on paragraph boundaries first, then use a sliding character window.

    The overlap preserves context at chunk boundaries while keeping this first demo
    simple and independent of third-party text-splitting libraries.
    """
    if chunk_size <= chunk_overlap:
        raise ValueError("chunk_size must be greater than chunk_overlap")

    paragraphs = [part.strip() for part in text.split("\n\n") if part.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        if current:
            chunks.append(current)
        while len(paragraph) > chunk_size:
            chunks.append(paragraph[:chunk_size])
            paragraph = paragraph[chunk_size - chunk_overlap :]
        current = paragraph
    if current:
        chunks.append(current)
    return chunks


def build_chunks(knowledge_dir: Path) -> list[TextChunk]:
    """Load all knowledge documents and create stable, source-labelled chunks."""
    chunks: list[TextChunk] = []
    for source, text in load_markdown_documents(knowledge_dir):
        for index, content in enumerate(split_text(text)):
            chunks.append(TextChunk(chunk_id=f"{source}:{index}", source=source, content=content))
    return chunks
