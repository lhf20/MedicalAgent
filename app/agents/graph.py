"""Minimal LangGraph agent: intent routing to RAG, chat, or safe fallback."""

import logging
import json
from typing import Any, Literal, Protocol

from langgraph.graph import END, START, StateGraph

from app.agents.intent_router import classify_intent
from app.agents.state import AgentState, Intent
from app.rag.workflow import RAGResponse, RAGRetrieval
from app.rag.siliconflow import SiliconFlowClient
from app.tools.petct_result import query_petct_result


logger = logging.getLogger(__name__)


class RAGWorkflow(Protocol):
    """The small interface needed by the medical question node."""

    def answer(self, query: str) -> RAGResponse: ...

    def retrieve(self, query: str) -> RAGRetrieval: ...


class MedicalImagingAgent:
    """Routes a user query through a minimal LangGraph StateGraph."""

    def __init__(self, rag_workflow: RAGWorkflow, llm_client: SiliconFlowClient) -> None:
        self.rag_workflow = rag_workflow
        self.llm_client = llm_client
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("intent_router", self._intent_router_node)
        graph.add_node("medical_qa", self._medical_qa_node)
        graph.add_node("petct_result_query", self._petct_result_query_node)
        graph.add_node("petct_tool_answer", self._petct_tool_answer_node)
        graph.add_node("petct_tool_rag", self._petct_tool_rag_node)
        graph.add_node("general_chat", self._general_chat_node)
        graph.add_node("unsupported", self._unsupported_node)
        graph.add_edge(START, "intent_router")
        graph.add_conditional_edges(
            "intent_router",
            self._route_by_intent,
            {
                "medical_qa": "medical_qa",
                "petct_result_query": "petct_result_query",
                "general_chat": "general_chat",
                "unsupported": "unsupported",
            },
        )
        graph.add_edge("medical_qa", END)
        graph.add_conditional_edges(
            "petct_result_query",
            self._route_petct_follow_up,
            {"tool_answer": "petct_tool_answer", "tool_rag": "petct_tool_rag"},
        )
        graph.add_edge("petct_tool_answer", END)
        graph.add_edge("petct_tool_rag", END)
        graph.add_edge("general_chat", END)
        graph.add_edge("unsupported", END)
        return graph.compile()

    def _intent_router_node(self, state: AgentState) -> dict[str, Intent]:
        query = state["user_query"]
        intent = classify_intent(query)
        logger.info("当前 query：%s", query)
        logger.info("识别 intent：%s", intent)
        return {"intent": intent}

    @staticmethod
    def _route_by_intent(state: AgentState) -> Intent:
        return state["intent"]

    def _medical_qa_node(self, state: AgentState) -> dict[str, str]:
        logger.info("进入节点：medical_qa")
        response = self.rag_workflow.answer(state["user_query"])
        return {"retrieved_context": response.context, "final_answer": response.answer}

    def _petct_result_query_node(self, state: AgentState) -> dict[str, Any]:
        """Extract natural-language parameters and call the existing PET-CT Tool."""
        logger.info("进入节点：petct_result_query")
        query = state["user_query"]
        try:
            parameters = self.llm_client.extract_petct_parameters(query)
        except RuntimeError as error:
            logger.warning("PET-CT Tool 参数提取失败，未调用 Tool：%s", error)
            return {"needs_rag": False, "final_answer": "无法解析 PET-CT 查询参数，请提供 study_id 和 location。"}

        study_id = parameters.get("study_id")
        location = parameters.get("location")
        lesion_id = parameters.get("lesion_id")
        if not isinstance(study_id, str) or not study_id.strip() or not isinstance(location, str) or not location.strip():
            logger.info("PET-CT Tool 未调用：缺少必要参数")
            return {"needs_rag": False, "final_answer": "请提供 PET-CT 的 study_id 和 location 后再查询。"}

        logger.info(
            "PET-CT Tool 调用：study_id=%s, location=%s, lesion_id=%s",
            study_id,
            location,
            lesion_id,
        )
        try:
            tool_result = query_petct_result(study_id, location, lesion_id)
        except Exception as error:  # Tool exceptions must not terminate the graph.
            logger.exception("PET-CT Tool 调用异常：%s", error)
            return {"needs_rag": False, "final_answer": "PET-CT 结果查询暂时不可用，请稍后重试。"}

        found = tool_result.get("found") is True
        logger.info("PET-CT Tool 返回状态：found=%s", found)
        needs_rag = self._needs_medical_explanation(query)
        logger.info("PET-CT 联合推理：是否调用 RAG=%s", needs_rag)
        return {"tool_result": tool_result, "needs_rag": needs_rag}

    @staticmethod
    def _needs_medical_explanation(query: str) -> bool:
        normalized = query.lower()
        return any(keyword in normalized for keyword in (
            "怎么理解", "如何理解", "是什么意思", "是什么", "一般", "意义", "反映", "解释", "说明",
        ))

    @staticmethod
    def _route_petct_follow_up(state: AgentState) -> Literal["tool_answer", "tool_rag"]:
        return "tool_rag" if state["needs_rag"] else "tool_answer"

    def _petct_tool_answer_node(self, state: AgentState) -> dict[str, str]:
        """Return a direct Tool-based response when the query asks only for facts."""
        logger.info("进入节点：petct_tool_answer")
        if state["final_answer"]:
            return {}
        tool_result = state["tool_result"]
        if not tool_result or tool_result.get("found") is not True:
            reason = tool_result.get("error", "unknown_error") if tool_result else "tool_not_called"
            return {"final_answer": f"PET-CT 结果查询失败：{reason}。未返回任何医学结果。"}
        query = state["user_query"]
        tool_context = json.dumps(tool_result, ensure_ascii=False)
        try:
            answer = self.llm_client.answer_petct_result(query, tool_result)
        except RuntimeError as error:
            logger.warning("PET-CT Tool 结果生成失败：%s", error)
            return {
                "retrieved_context": tool_context,
                "final_answer": "PET-CT Tool 查询成功，但暂时无法生成说明。请参考返回的结构化结果。",
            }
        return {"retrieved_context": tool_context, "final_answer": answer}

    def _petct_tool_rag_node(self, state: AgentState) -> dict[str, str]:
        """Combine structured Tool facts with general RAG knowledge for explanation queries."""
        logger.info("进入节点：petct_tool_rag")
        tool_result = state["tool_result"]
        if not tool_result:
            return {"final_answer": state["final_answer"] or "PET-CT Tool 未返回可用结果。"}
        logger.info("PET-CT 联合推理：调用 RAG")
        try:
            retrieval = self.rag_workflow.retrieve(state["user_query"])
            rag_context = retrieval.context
            rag_count = retrieval.chunk_count
        except RuntimeError as error:
            logger.warning("PET-CT 联合推理 RAG 调用失败：%s", error)
            rag_context = ""
            rag_count = 0
        logger.info("PET-CT 联合推理：最终使用 %d 条 RAG context", rag_count)
        combined_context = (
            f"Tool Result（当前检查/病灶事实）：\n{json.dumps(tool_result, ensure_ascii=False)}\n\n"
            f"RAG Context（一般医学知识）：\n{rag_context or '无足够相关知识片段'}"
        )
        try:
            answer = self.llm_client.answer_petct_with_rag(state["user_query"], tool_result, rag_context)
        except RuntimeError as error:
            logger.warning("PET-CT 联合推理回答生成失败：%s", error)
            if tool_result.get("found") is True:
                fallback = "PET-CT Tool 查询成功，但暂时无法生成综合说明。请参考结构化 Tool 结果。"
            else:
                fallback = "未查到具体 PET-CT 数据；知识解释暂时无法生成。"
            return {"retrieved_context": combined_context, "final_answer": fallback}
        return {"retrieved_context": combined_context, "final_answer": answer}

    def _general_chat_node(self, state: AgentState) -> dict[str, str]:
        logger.info("进入节点：general_chat")
        return {"final_answer": self.llm_client.chat(state["user_query"])}

    @staticmethod
    def _unsupported_node(state: AgentState) -> dict[str, str]:
        logger.info("进入节点：unsupported")
        return {
            "final_answer": "抱歉，我目前仅支持医学影像知识问答和简单的普通聊天，无法处理该问题。"
        }

    def invoke(self, query: str) -> AgentState:
        """Run one graph invocation with a clean initial state."""
        return self.graph.invoke(
            {
                "user_query": query,
                "intent": "unsupported",
                "tool_result": None,
                "needs_rag": False,
                "retrieved_context": "",
                "final_answer": "",
            }
        )
