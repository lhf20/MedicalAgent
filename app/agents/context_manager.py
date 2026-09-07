"""Bounded conversation context and deterministic medical-reference resolution."""

from dataclasses import dataclass
from typing import Any


MAX_HISTORY_TURNS = 6


@dataclass(frozen=True)
class ContextResolution:
    """A context-completed query or a clarification request."""

    query: str
    clarification: str = ""


def trim_history(history: list[dict[str, str]]) -> list[dict[str, str]]:
    """Keep only the most recent bounded number of user/assistant turns."""
    return history[-MAX_HISTORY_TURNS:]


def resolve_query(query: str, state: dict[str, Any]) -> ContextResolution:
    """Resolve supported Chinese references from structured conversation slots."""
    resolved = query.strip()
    last_study_id = state.get("last_study_id")
    last_lesion_id = state.get("last_lesion_id")
    last_location = state.get("last_location")
    last_metric = state.get("last_medical_metric")

    metric_references = ("这个指标", "该指标", "刚才的指标", "上述指标")
    if any(reference in resolved for reference in metric_references):
        if not last_metric:
            return ContextResolution(
                query=resolved,
                clarification="请说明你指的是哪个医学指标，例如 SUVmax 或病灶体积。",
            )
        for reference in metric_references:
            resolved = resolved.replace(reference, f"{last_metric} 指标")

    lesion_references = ("这个病灶", "该病灶", "刚才的病灶", "上述病灶")
    if any(reference in resolved for reference in lesion_references):
        if not last_study_id or not last_location:
            return ContextResolution(
                query=resolved,
                clarification="请先提供该病灶所属的 study_id 和 location。",
            )
        lesion_context = " ".join(
            part for part in (last_study_id, last_location, last_lesion_id, "病灶") if part
        )
        for reference in lesion_references:
            resolved = resolved.replace(reference, lesion_context)

    study_references = ("刚才那个检查", "上一个检查", "这个检查", "该检查")
    if any(reference in resolved for reference in study_references):
        if not last_study_id:
            return ContextResolution(query=resolved, clarification="请提供要查询的 PET-CT study_id。")
        study_context = " ".join(part for part in (last_study_id, last_location) if part)
        for reference in study_references:
            resolved = resolved.replace(reference, study_context)

    # A terse factual follow-up such as “SUVmax 是多少？” inherits the last lesion.
    asks_for_previous_fact = (
        any(metric in resolved.lower() for metric in ("suvmax", "suv", "volume"))
        or "体积" in resolved
    ) and any(word in resolved for word in ("多少", "数值", "结果"))
    has_explicit_study = "petct-demo-" in resolved.lower()
    if asks_for_previous_fact and not has_explicit_study:
        if not last_study_id or not last_location:
            return ContextResolution(
                query=resolved,
                clarification="请提供要查询病灶的 study_id 和 location。",
            )
        prefix = " ".join(part for part in (last_study_id, last_location, last_lesion_id) if part)
        resolved = f"{prefix} {resolved}"

    return ContextResolution(query=resolved)
