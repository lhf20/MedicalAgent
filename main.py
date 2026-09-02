"""Command-line entry point for the minimal medical-imaging RAG demo."""

import logging
from pathlib import Path

from app.rag.loader import build_chunks
from app.rag.reranker import SiliconFlowReranker
from app.rag.retriever import LocalVectorStore
from app.rag.siliconflow import SiliconFlowClient


RECALL_TOP_K = 10
FINAL_TOP_K = 3
PROJECT_DIR = Path(__file__).resolve().parent
KNOWLEDGE_DIR = PROJECT_DIR / "data" / "knowledge"
logger = logging.getLogger(__name__)


def main() -> None:
    chunks = build_chunks(KNOWLEDGE_DIR)
    if not chunks:
        raise RuntimeError(f"No Markdown knowledge documents found in: {KNOWLEDGE_DIR}")

    client = SiliconFlowClient(str(PROJECT_DIR))
    reranker = SiliconFlowReranker(client)
    print(f"已加载 {len(chunks)} 个知识片段，正在创建本地向量索引...")
    vector_store = LocalVectorStore(chunks, client.embed([chunk.content for chunk in chunks]))
    print("RAG Demo 已就绪。输入 exit 或 quit 可结束。")

    while True:
        question = input("\n医学影像问题> ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue

        recalled_results = vector_store.search(client.embed([question])[0], top_k=RECALL_TOP_K)
        logger.info("初始向量召回：%d 个 chunk（请求 Top-%d）", len(recalled_results), RECALL_TOP_K)
        if not recalled_results:
            print("\n未检索到相关知识片段，无法生成基于知识库的回答。")
            continue

        try:
            reranked_results = reranker.rerank(question, recalled_results, top_k=RECALL_TOP_K)
        except RuntimeError as error:
            logger.warning("Rerank 调用失败，回退到向量召回结果：%s", error)
            reranked_results = []

        if reranked_results:
            final_reranked_results = reranked_results[:FINAL_TOP_K]
            final_results = [item.result for item in final_reranked_results]
            rerank_scores = {item.result.chunk.chunk_id: item.score for item in final_reranked_results}
            logger.info("Rerank 排序结果：%s", [
                f"{item.result.chunk.chunk_id}={item.score:.4f}" for item in reranked_results
            ])
        else:
            final_results = recalled_results[:FINAL_TOP_K]
            rerank_scores: dict[str, float] = {}
            logger.warning("Rerank 未返回有效结果，使用向量召回前 %d 个 chunk。", len(final_results))

        logger.info("最终送入 LLM：%d 个 chunk", len(final_results))
        context = "\n\n".join(
            f"[来源：{item.chunk.source}]\n{item.chunk.content}"
            for item in final_results
        )
        print("\nRerank 后的知识片段：")
        for number, item in enumerate(final_results, start=1):
            rerank_score = rerank_scores.get(item.chunk.chunk_id)
            score_label = (
                f"Rerank：{rerank_score:.4f}"
                if rerank_score is not None
                else f"向量相似度：{item.score:.4f}"
            )
            print(f"\n{number}. {item.chunk.source}（{score_label}）\n{item.chunk.content}")

        print("\n回答：")
        print(client.answer(question, context))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
