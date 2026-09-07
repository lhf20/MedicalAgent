"""State definition for the minimal LangGraph agent."""

from typing import Any, Literal, TypedDict


Intent = Literal["medical_qa", "petct_result_query", "general_chat", "unsupported", "clarification"]


class ConversationTurn(TypedDict):
    """One bounded user/assistant exchange."""

    user: str
    assistant: str


class AgentState(TypedDict):
    """Data passed between intent routing and answer nodes."""

    user_query: str
    resolved_query: str
    intent: Intent
    tool_result: dict[str, Any] | None
    needs_rag: bool
    retrieved_context: str
    final_answer: str
    conversation_history: list[ConversationTurn]
    last_petct_result: dict[str, Any] | None
    last_study_id: str | None
    last_lesion_id: str | None
    last_location: str | None
    last_medical_metric: str | None
    clarification_needed: bool
    clarification_message: str
