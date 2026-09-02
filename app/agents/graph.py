"""Minimal LangGraph agent: intent routing to RAG, chat, or safe fallback."""

import logging
from typing import Protocol

from langgraph.graph import END, START, StateGraph

from app.agents.intent_router import classify_intent
from app.agents.state import AgentState, Intent
from app.rag.workflow import RAGResponse
from app.rag.siliconflow import SiliconFlowClient


logger = logging.getLogger(__name__)


class RAGWorkflow(Protocol):
    """The small interface needed by the medical question node."""

    def answer(self, query: str) -> RAGResponse: ...


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
        graph.add_node("general_chat", self._general_chat_node)
        graph.add_node("unsupported", self._unsupported_node)
        graph.add_edge(START, "intent_router")
        graph.add_conditional_edges(
            "intent_router",
            self._route_by_intent,
            {
                "medical_qa": "medical_qa",
                "general_chat": "general_chat",
                "unsupported": "unsupported",
            },
        )
        graph.add_edge("medical_qa", END)
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
                "retrieved_context": "",
                "final_answer": "",
            }
        )
