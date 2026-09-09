"""Deterministic failure extraction for evaluation reports."""

from typing import Any


FAILURE_TYPES = (
    "intent_failure",
    "rag_failure",
    "tool_call_failure",
    "argument_failure",
    "answer_failure",
    "abstention_failure",
    "hallucination_failure",
)


def _failed_metrics(scores: dict[str, Any], actual: dict[str, Any]) -> tuple[list[str], list[str]]:
    failed: list[str] = []
    types: list[str] = []
    checks = (
        ("intent_correct", "intent_failure"),
        ("rag_hit_at_3", "rag_failure"),
        ("tool_call_correct", "tool_call_failure"),
        ("tool_result_correct", "tool_call_failure"),
        ("argument_exact_match", "argument_failure"),
        ("answer_correct", "answer_failure"),
        ("abstention_correct", "abstention_failure"),
    )
    for metric, failure_type in checks:
        if metric in scores and scores[metric] is False:
            failed.append(metric)
            types.append(failure_type)

    for field, correct in scores.get("argument_field_results", {}).items():
        if not correct:
            failed.append(f"argument_field:{field}")
            types.append("argument_failure")

    if scores.get("hallucinated") is True:
        failed.append("hallucinated")
        types.append("hallucination_failure")

    if actual.get("error"):
        failed.append("evaluation_error")
        types.append("answer_failure")

    return list(dict.fromkeys(failed)), list(dict.fromkeys(types))


def build_failure_analysis(report: dict[str, Any]) -> dict[str, Any]:
    """Return detailed failed cases and a deterministic index by failure type."""
    failures: list[dict[str, Any]] = []
    categories: dict[str, list[str]] = {failure_type: [] for failure_type in FAILURE_TYPES}

    for row in report.get("cases", []):
        actual = row.get("actual", {})
        failed_metrics, failure_types = _failed_metrics(row.get("scores", {}), actual)
        if not failed_metrics:
            continue
        detail = {
            "case_id": row.get("id"),
            "query": row.get("query", ""),
            "expected": row.get("expected", {}),
            "actual": actual,
            "failed_metrics": failed_metrics,
            "failure_types": failure_types,
            "final_answer": actual.get("final_answer", ""),
            "intent": actual.get("intent"),
            "rag_top3": actual.get("rag_top3", []),
            "tool_calls": actual.get("tool_calls", []),
            "latency_ms": row.get("latency_ms", 0.0),
        }
        failures.append(detail)
        for failure_type in failure_types:
            categories[failure_type].append(row.get("id"))

    return {
        "dataset": report.get("dataset", "unknown"),
        "generated_at": report.get("generated_at"),
        "failure_count": len(failures),
        "failures": failures,
        "categories": categories,
    }
