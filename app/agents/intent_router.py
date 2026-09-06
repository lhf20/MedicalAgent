"""Deterministic, dependency-free intent routing for the first agent version."""

import re

from app.agents.state import Intent


MEDICAL_KEYWORDS = (
    "医学影像", "影像", "放射", "pet", "pet-ct", "ct", "mri", "磁共振", "超声",
    "x线", "x 光", "suv", "示踪剂", "病灶", "肿瘤", "诊断", "检查",
)
GENERAL_CHAT_KEYWORDS = (
    "你好", "您好", "你是谁", "介绍一下你自己", "谢谢", "再见", "hello", "hi",
)


def _is_petct_result_request(query: str) -> bool:
    """Identify requests for a specific PET-CT study, not general medical knowledge."""
    has_demo_study_id = bool(re.search(r"petct-demo-\d+", query))
    has_petct_result_request = (
        ("pet-ct" in query or "petct" in query)
        and any(keyword in query for keyword in ("查询", "查一下", "查看", "获取", "显示"))
    )
    return has_demo_study_id or has_petct_result_request


def classify_intent(query: str) -> Intent:
    """Classify the supported MVP intents without another LLM/API call."""
    normalized = query.strip().lower()
    if not normalized:
        return "unsupported"
    if _is_petct_result_request(normalized):
        return "petct_result_query"
    if any(keyword in normalized for keyword in MEDICAL_KEYWORDS):
        return "medical_qa"
    if any(keyword in normalized for keyword in GENERAL_CHAT_KEYWORDS):
        return "general_chat"
    return "unsupported"
