"""Reusable orchestration of the already-verified RAG and rerank components."""

import logging
from dataclasses import dataclass
from pathlib import Path

from app.rag.loader import build_chunks
from app.rag.reranker import SiliconFlowReranker
from app.rag.retriever import LocalVectorStore
from app.rag.siliconflow import SiliconFlowClient


RECALL_TOP_K = 10
FINAL_TOP_K = 3
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RAGResponse:
    """The final RAG answer together with the context used to create it."""

    context: str
    answer: str


class MedicalRAGWorkflow:
    """Uses the existing embedding, vector search, rerank, and Qwen components."""

    def __init__(
        self,
        client: SiliconFlowClient,
        vector_store: LocalVectorStore,
        reranker: SiliconFlowReranker,
    ) -> None:
        self.client = client
        self.vector_store = vector_store
        self.reranker = reranker

    @classmethod
    def from_knowledge_dir(cls, client: SiliconFlowClient, knowledge_dir: Path) -> "MedicalRAGWorkflow":
        """Build the existing in-memory vector index once for the agent session."""
        chunks = build_chunks(knowledge_dir)
        if not chunks:
            raise RuntimeError(f"No Markdown knowledge documents found in: {knowledge_dir}")
        logger.info("已加载 %d 个知识片段，正在创建本地向量索引...", len(chunks))
        vector_store = LocalVectorStore(chunks, client.embed([chunk.content for chunk in chunks]))
        return cls(client=client, vector_store=vector_store, reranker=SiliconFlowReranker(client))

    def answer(self, query: str) -> RAGResponse:
        """Run the verified Top-10 vector recall -> rerank -> Top-3 RAG flow."""
        recalled_results = self.vector_store.search(self.client.embed([query])[0], top_k=RECALL_TOP_K)
        logger.info("初始向量召回：%d 个 chunk（请求 Top-%d）", len(recalled_results), RECALL_TOP_K)
        if not recalled_results:
            return RAGResponse(context="", answer="未检索到相关医学知识片段，无法生成基于知识库的回答。")

        try:
            reranked_results = self.reranker.rerank(query, recalled_results, top_k=RECALL_TOP_K)
        except RuntimeError as error:
            logger.warning("Rerank 调用失败，回退到向量召回结果：%s", error)
            reranked_results = []

        if reranked_results:
            final_results = [item.result for item in reranked_results[:FINAL_TOP_K]]
            logger.info("Rerank 排序结果：%s", [
                f"{item.result.chunk.chunk_id}={item.score:.4f}" for item in reranked_results
            ])
        else:
            final_results = recalled_results[:FINAL_TOP_K]
            logger.warning("Rerank 未返回有效结果，使用向量召回前 %d 个 chunk。", len(final_results))

        logger.info("最终送入 LLM：%d 个 chunk", len(final_results))
        context = "\n\n".join(
            f"[来源：{item.chunk.source}]\n{item.chunk.content}" for item in final_results
        )
        return RAGResponse(context=context, answer=self.client.answer(query, context))
