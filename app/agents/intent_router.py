"""Hybrid intent routing: deterministic rules first, contextual LLM fallback."""

import json
import logging
import re
from collections.abc import Callable

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
UNSUPPORTED_KEYWORDS = (
    "预测明天股票", "股票走势", "投资建议", "写代码", "编程", "法律意见", "天气预报",
)
LLM_ROUTABLE_INTENTS = frozenset({
    "medical_qa", "petct_query", "general_chat", "unsupported", "clarify",
})
logger = logging.getLogger(__name__)


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


def _rule_router(query: str) -> Intent | None:
    """Return an intent only when a fast rule can decide reliably."""
    if not query:
        return "unsupported"
    if any(keyword in query for keyword in UNSUPPORTED_KEYWORDS):
        return "unsupported"
    if _is_petct_result_request(query):
        return "petct_query"
    if _is_medical_question(query):
        return "medical_qa"
    if _is_general_chat(query):
        return "general_chat"
    return None


def _llm_router_prompt(query: str, conversation_history: list[dict[str, str]]) -> str:
    history = json.dumps(conversation_history, ensure_ascii=False)
    return (
        "你是医学影像 Agent 的意图分类器。能力边界只有："
        "医学影像知识问答、PET-CT 结构化结果查询与解释、基础聊天；其他任务均不支持。\n"
        "请结合已完成指代补全的当前问题和最近对话历史，分类为以下一个 intent：\n"
        "medical_qa：一般医学影像知识；petct_query：查询具体 PET-CT study/病灶事实；"
        "general_chat：基础问候聊天；unsupported：能力范围外；"
        "clarify：仍缺少关键指代或信息，必须向用户澄清。\n"
        "只返回严格 JSON，不要 Markdown 或解释：{\"intent\": \"medical_qa\"}\n"
        f"最近对话历史：{history}\n当前问题：{query}"
    )


def classify_intent(
    query: str,
    *,
    conversation_history: list[dict[str, str]] | None = None,
    llm_classifier: Callable[[str], str] | None = None,
) -> Intent:
    """Classify by rules, calling the contextual LLM only for ambiguous input."""
    normalized = normalize_for_matching(query)
    rule_intent = _rule_router(normalized)
    if rule_intent is not None:
        logger.info("rule_router: intent=%s", rule_intent)
        return rule_intent

    if llm_classifier is None:
        logger.info("rule_router: ambiguous; llm_router unavailable, fallback=unsupported")
        return "unsupported"

    logger.info("rule_router: ambiguous; invoking llm_router")
    try:
        raw_result = llm_classifier(
            _llm_router_prompt(query, conversation_history or [])
        )
        parsed = json.loads(raw_result)
        intent = parsed.get("intent") if isinstance(parsed, dict) else None
        if intent not in LLM_ROUTABLE_INTENTS:
            raise ValueError("invalid intent value")
    except (RuntimeError, TypeError, ValueError, json.JSONDecodeError) as error:
        logger.warning("llm_router: invalid response or call failure; fallback=unsupported (%s)", error)
        return "unsupported"

    logger.info("llm_router: intent=%s", intent)
    return intent
