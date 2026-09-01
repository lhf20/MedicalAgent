"""Command-line entry point for the minimal medical-imaging RAG demo."""

from pathlib import Path

from app.rag.loader import build_chunks
from app.rag.retriever import LocalVectorStore
from app.rag.siliconflow import SiliconFlowClient


TOP_K = 3
PROJECT_DIR = Path(__file__).resolve().parent
KNOWLEDGE_DIR = PROJECT_DIR / "data" / "knowledge"


def main() -> None:
    chunks = build_chunks(KNOWLEDGE_DIR)
    if not chunks:
        raise RuntimeError(f"No Markdown knowledge documents found in: {KNOWLEDGE_DIR}")

    client = SiliconFlowClient(str(PROJECT_DIR))
    print(f"已加载 {len(chunks)} 个知识片段，正在创建本地向量索引...")
    vector_store = LocalVectorStore(chunks, client.embed([chunk.content for chunk in chunks]))
    print("RAG Demo 已就绪。输入 exit 或 quit 可结束。")

    while True:
        question = input("\n医学影像问题> ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue

        results = vector_store.search(client.embed([question])[0], top_k=TOP_K)
        context = "\n\n".join(f"[来源：{item.chunk.source}]\n{item.chunk.content}" for item in results)
        print("\n检索到的知识片段：")
        for number, item in enumerate(results, start=1):
            print(f"\n{number}. {item.chunk.source}（相似度：{item.score:.4f}）\n{item.chunk.content}")

        print("\n回答：")
        print(client.answer(question, context))


if __name__ == "__main__":
    main()
