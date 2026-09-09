"""Terminal formatting shared by live evaluation and offline rescore."""

from typing import Any


def _format_metric(value: Any) -> str:
    return "N/A" if value is None else f"{value:.2%}"


def _metric_line(label: str, accuracy: Any, correct: int, total: int) -> str:
    return f"{label:<23} {_format_metric(accuracy):>8}  ({correct}/{total})"


def print_summary(summary: dict[str, Any]) -> None:
    print("\nBaseline Evaluation Summary")
    print(_metric_line("Intent Accuracy", summary["intent_accuracy"], summary["intent_correct"], summary["intent_cases"]))
    print(_metric_line("RAG Hit@3", summary["rag_hit_at_3"], summary["rag_correct"], summary["rag_cases"]))
    print(_metric_line("Tool Call Accuracy", summary["tool_call_accuracy"], summary["tool_call_correct"], summary["tool_call_cases"]))
    print(_metric_line("Argument Field Acc.", summary["argument_field_accuracy"], summary["argument_field_correct"], summary["argument_fields"]))
    print(_metric_line("Argument Exact Match", summary["argument_exact_match"], summary["argument_exact_correct"], summary["argument_cases"]))
    print(_metric_line("Tool Result Accuracy", summary["tool_result_accuracy"], summary["tool_result_correct"], summary["tool_result_cases"]))
    print(_metric_line("Answer Accuracy", summary["answer_accuracy"], summary["answer_correct"], summary["answer_cases"]))
    print(_metric_line("Abstention Accuracy", summary["abstention_accuracy"], summary["abstention_correct"], summary["abstention_cases"]))
    print(
        f"Hallucination:        {summary['hallucination_count']} "
        f"/ {summary['hallucination_cases']} ({_format_metric(summary['hallucination_rate'])})"
    )
    print(f"Average Latency:      {summary['average_latency_ms']:.1f} ms")
    print(f"P50 / P95 Latency:    {summary['p50_latency_ms']:.1f} / {summary['p95_latency_ms']:.1f} ms")
