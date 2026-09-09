"""Deterministic metric calculations; no LLM-as-judge is used."""

import math
import re
import statistics
from typing import Any


ABSTENTION_MARKERS = (
    "无法", "未找到", "未检索到", "没有足够", "未提供足够", "不足", "不支持", "请提供", "请说明", "查询失败",
)
UNSUPPORTED_NUMERIC_FACT_PATTERN = re.compile(
    r"(?:SUV(?:max)?\s*(?:为|=|是|约)?\s*\d+(?:\.\d+)?|"
    r"(?:volume_cm3|体积)\s*(?:为|=|是|约|:|：)?\s*\d+(?:\.\d+)?|"
    r"\d+(?:\.\d+)?\s*(?:cm[³3]|立方厘米))",
    re.IGNORECASE,
)


def _nearest_rank_percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def score_case(case: dict[str, Any], actual: dict[str, Any]) -> dict[str, Any]:
    """Score one case only on metrics for which it provides expectations."""
    expected = case["expected"]
    answer = actual.get("final_answer", "")
    scores: dict[str, Any] = {}

    if "intent" in expected:
        scores["intent_correct"] = actual.get("intent") == expected["intent"]

    if "rag_sources" in expected:
        expected_sources = set(expected["rag_sources"])
        actual_sources = {item["source"] for item in actual.get("rag_top3", [])}
        scores["rag_hit_at_3"] = bool(expected_sources & actual_sources)

    if "tool_called" in expected:
        calls = actual.get("tool_calls", [])
        expected_called = expected["tool_called"]
        call_presence_correct = bool(calls) == expected_called
        expected_name = expected.get("tool_name")
        if expected_called and expected_name:
            call_name_correct = bool(calls) and all(
                call.get("name") == expected_name for call in calls
            )
            scores["tool_call_correct"] = call_presence_correct and call_name_correct
        else:
            scores["tool_call_correct"] = call_presence_correct

    if "tool_args" in expected:
        actual_args = actual.get("tool_calls", [{}])[-1].get("arguments", {}) if actual.get("tool_calls") else {}
        fields = expected["tool_args"]
        scores["argument_field_results"] = {
            field: actual_args.get(field) == value for field, value in fields.items()
        }
        scores["argument_exact_match"] = all(scores["argument_field_results"].values())

    if "tool_found" in expected:
        calls = actual.get("tool_calls", [])
        actual_found = calls[-1].get("result", {}).get("found") if calls else None
        scores["tool_result_correct"] = actual_found is expected["tool_found"]

    if "answer_contains" in expected:
        required_values = [str(value) for value in expected["answer_contains"]]
        query = case.get("turns", [case.get("query", "")])[-1]
        # A generated answer need not repeat a Tool-returned lesion ID unless the
        # user explicitly asked for that ID (or asked for a lesion list). Other
        # expected facts remain mandatory exact substrings.
        if actual.get("tool_calls") and "哪些病灶" not in query:
            required_values = [
                value
                for value in required_values
                if not re.fullmatch(r"LESION-\d+", value, re.IGNORECASE)
                or value.lower() in query.lower()
            ]
        scores["answer_correct"] = all(value in answer for value in required_values)

    forbidden = [str(value) for value in expected.get("forbidden_facts", [])]
    if expected.get("abstain") or forbidden:
        contains_forbidden = any(value in answer for value in forbidden)
        # Repeating a study/lesion ID from the question is not a fabricated fact.
        # For abstention cases, only concrete clinical values are generically
        # detected here. Invalid entity IDs/locations belong in forbidden_facts.
        unsupported_fact = (
            bool(UNSUPPORTED_NUMERIC_FACT_PATTERN.search(answer))
            if expected.get("abstain")
            else False
        )
        hallucinated = contains_forbidden or unsupported_fact
        scores["hallucinated"] = hallucinated
        if expected.get("abstain"):
            scores["abstention_correct"] = (
                (actual.get("intent") == "clarify" or any(marker in answer for marker in ABSTENTION_MARKERS))
                and not hallucinated
            )
    return scores


def aggregate_results(case_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate applicable per-case scores and latency percentiles."""
    score_rows = [row["scores"] for row in case_results]

    def ratio(field: str) -> tuple[float | None, int, int]:
        values = [row[field] for row in score_rows if field in row]
        correct = sum(bool(value) for value in values)
        return ((correct / len(values)) if values else None, correct, len(values))

    intent_accuracy, intent_correct, intent_n = ratio("intent_correct")
    rag_hit, rag_correct, rag_n = ratio("rag_hit_at_3")
    tool_accuracy, tool_correct, tool_n = ratio("tool_call_correct")
    tool_result_accuracy, tool_result_correct, tool_result_n = ratio("tool_result_correct")
    answer_accuracy, answer_correct, answer_n = ratio("answer_correct")
    abstention_accuracy, abstention_correct, abstention_n = ratio("abstention_correct")
    hallucination_rows = [row["hallucinated"] for row in score_rows if "hallucinated" in row]
    argument_values = [
        value
        for row in score_rows
        for value in row.get("argument_field_results", {}).values()
    ]
    argument_exact_values = [
        row["argument_exact_match"] for row in score_rows if "argument_exact_match" in row
    ]
    argument_field_correct = sum(bool(value) for value in argument_values)
    argument_exact_correct = sum(bool(value) for value in argument_exact_values)
    latencies = [row["latency_ms"] for row in case_results]

    return {
        "intent_accuracy": intent_accuracy,
        "intent_correct": intent_correct,
        "intent_cases": intent_n,
        "rag_hit_at_3": rag_hit,
        "rag_correct": rag_correct,
        "rag_cases": rag_n,
        "tool_call_accuracy": tool_accuracy,
        "tool_call_correct": tool_correct,
        "tool_call_cases": tool_n,
        "argument_field_accuracy": (
            sum(argument_values) / len(argument_values) if argument_values else None
        ),
        "argument_field_correct": argument_field_correct,
        "argument_fields": len(argument_values),
        "argument_exact_match": (
            sum(argument_exact_values) / len(argument_exact_values)
            if argument_exact_values else None
        ),
        "argument_exact_correct": argument_exact_correct,
        "argument_cases": len(argument_exact_values),
        "tool_result_accuracy": tool_result_accuracy,
        "tool_result_correct": tool_result_correct,
        "tool_result_cases": tool_result_n,
        "answer_accuracy": answer_accuracy,
        "answer_correct": answer_correct,
        "answer_cases": answer_n,
        "abstention_accuracy": abstention_accuracy,
        "abstention_correct": abstention_correct,
        "abstention_cases": abstention_n,
        "hallucination_count": sum(hallucination_rows),
        "hallucination_rate": (
            sum(hallucination_rows) / len(hallucination_rows) if hallucination_rows else None
        ),
        "hallucination_cases": len(hallucination_rows),
        "average_latency_ms": statistics.fmean(latencies) if latencies else 0.0,
        "p50_latency_ms": _nearest_rank_percentile(latencies, 0.50),
        "p95_latency_ms": _nearest_rank_percentile(latencies, 0.95),
    }
