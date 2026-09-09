"""Analyze saved reranker scores without selecting or applying a threshold."""

import statistics
from typing import Any


def _distribution(items: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [item["score"] for item in items]
    return {
        "count": len(scores),
        "min": min(scores) if scores else None,
        "max": max(scores) if scores else None,
        "mean": statistics.fmean(scores) if scores else None,
        "median": statistics.median(scores) if scores else None,
        "items": sorted(items, key=lambda item: item["score"], reverse=True),
    }


def build_relevance_analysis(case_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Split top-1 relevance scores into explicit positive/negative examples."""
    positive: list[dict[str, Any]] = []
    negative: list[dict[str, Any]] = []
    for row in case_results:
        expected = row.get("expected", {})
        actual = row.get("actual", {})
        score = actual.get("top1_relevance_score")
        if not isinstance(score, (int, float)):
            continue
        item = {
            "case_id": row.get("id"),
            "query": row.get("query", ""),
            "score": float(score),
            "rag_top3": actual.get("rag_top3", []),
        }
        if expected.get("rag_sources"):
            item["expected_sources"] = expected["rag_sources"]
            positive.append(item)
        elif expected.get("abstain") and not actual.get("tool_calls"):
            negative.append(item)

    all_items = [dict(item, label="positive") for item in positive]
    all_items.extend(dict(item, label="negative") for item in negative)
    return {
        "definition": {
            "positive": "case has expected.rag_sources",
            "negative": "case expects abstention, has a RAG score, and made no Tool call",
        },
        "positive": _distribution(positive),
        "negative": _distribution(negative),
        "all_sorted": sorted(all_items, key=lambda item: item["score"], reverse=True),
        "threshold_selected": False,
    }
