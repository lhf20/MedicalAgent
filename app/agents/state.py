"""State definition for the minimal LangGraph agent."""

from typing import Literal, TypedDict


Intent = Literal["medical_qa", "general_chat", "unsupported"]


class AgentState(TypedDict):
    """Data passed between intent routing and answer nodes."""

    user_query: str
    intent: Intent
    retrieved_context: str
    final_answer: str
