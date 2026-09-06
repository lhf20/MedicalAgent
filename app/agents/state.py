"""State definition for the minimal LangGraph agent."""

from typing import Any, Literal, TypedDict


Intent = Literal["medical_qa", "petct_result_query", "general_chat", "unsupported"]


class AgentState(TypedDict):
    """Data passed between intent routing and answer nodes."""

    user_query: str
    intent: Intent
    tool_result: dict[str, Any] | None
    needs_rag: bool
    retrieved_context: str
    final_answer: str
