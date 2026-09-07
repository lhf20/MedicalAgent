"""Deterministic, dependency-free intent routing for the first agent version."""

import re

from app.agents.state import Intent
from app.utils.input_normalizer import normalize_for_matching


MEDICAL_KEYWORDS = (
    "医学影像", "影像", "放射", "磁共振", "超声", "x线", "x 光",
    "示踪剂", "病灶", "肿瘤", "诊断", "检查",
)
MEDICAL_TERMS_PATTERN = re.compile(r"(?<![a-z0-9])(?:pet-ct|petct|ct|mri|suvmax|suv)(?![a-z0-9])")
GENERAL_CHAT_KEYWORDS = (
    "你好", "您好", "你是谁", "介绍一下你自己", "谢谢", "再见",
)
GENERAL_CHAT_EXACT = frozenset({"hello", "hi", "hey"})


def _is_petct_result_request(query: str) -> bool:
    """Identify requests for a specific PET-CT study, not general medical knowledge."""
    has_demo_study_id = bool(re.search(r"petct-demo-\d+", query))
    has_petct_result_request = (
        ("pet-ct" in query or "petct" in query)
        and any(keyword in query for keyword in ("查询", "查一下", "查看", "获取", "显示"))
    )
    return has_demo_study_id or has_petct_result_request


def _is_medical_question(query: str) -> bool:
    return any(keyword in query for keyword in MEDICAL_KEYWORDS) or bool(
        MEDICAL_TERMS_PATTERN.search(query)
    )


def _is_general_chat(query: str) -> bool:
    return query in GENERAL_CHAT_EXACT or any(keyword in query for keyword in GENERAL_CHAT_KEYWORDS)


def classify_intent(query: str) -> Intent:
    """Classify the supported MVP intents without another LLM/API call."""
    normalized = normalize_for_matching(query)
    if not normalized:
        return "unsupported"
    if _is_petct_result_request(normalized):
        return "petct_result_query"
    if _is_medical_question(normalized):
        return "medical_qa"
    if _is_general_chat(normalized):
        return "general_chat"
    return "unsupported"
