"""Structured query tool for fictional, de-identified PET-CT demo results."""

import json
import logging
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)
DEFAULT_DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "petct_demo_results.json"

TOOL_DEFINITION = {
    "name": "query_petct_result",
    "description": "查询虚构、脱敏的 PET-CT 结构化病灶结果。仅用于演示，不返回自然语言诊断。",
    "parameters": {
        "type": "object",
        "properties": {
            "study_id": {"type": "string", "description": "PET-CT 检查 ID，例如 PETCT-DEMO-001。"},
            "location": {"type": "string", "description": "病灶位置，例如 右肺上叶。"},
            "lesion_id": {"type": "string", "description": "可选的病灶 ID，例如 LESION-001。"},
        },
        "required": ["study_id", "location"],
        "additionalProperties": False,
    },
}


class PETCTResultRepository:
    """Read-only repository for the local fictional PET-CT demonstration data."""

    def __init__(self, data_path: Path = DEFAULT_DATA_PATH) -> None:
        self.data_path = data_path

    def studies(self) -> list[dict[str, Any]]:
        """Load studies from the local JSON file and validate its top-level shape."""
        with self.data_path.open(encoding="utf-8") as data_file:
            data = json.load(data_file)
        studies = data.get("studies")
        if not isinstance(studies, list):
            raise ValueError("PET-CT demo data must contain a studies list")
        return studies


def _failure(reason: str) -> dict[str, Any]:
    """Create the common failure result without fabricating medical findings."""
    return {"found": False, "error": reason, "results": []}


def _validate_text(value: object, field_name: str, required: bool = True) -> str | None:
    if value is None and not required:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _serialize_lesion(lesion: dict[str, Any]) -> dict[str, Any]:
    """Return only the structured fields exposed by this demo tool."""
    return {
        "lesion_id": lesion["lesion_id"],
        "location": lesion["location"],
        "suv_max": lesion["suv_max"],
        "volume_cm3": lesion["volume_cm3"],
    }


def query_petct_result(
    study_id: str,
    location: str,
    lesion_id: str | None = None,
    *,
    repository: PETCTResultRepository | None = None,
) -> dict[str, Any]:
    """Query fictional PET-CT results by study, location, and optional lesion ID.

    This function is intentionally side-effect free and JSON-serializable so it can
    later be registered with LangGraph or an LLM tool-calling layer unchanged.
    """
    try:
        normalized_study_id = _validate_text(study_id, "study_id")
        normalized_location = _validate_text(location, "location")
        normalized_lesion_id = _validate_text(lesion_id, "lesion_id", required=False)
    except ValueError as error:
        logger.warning("PET-CT Tool 参数校验失败：%s", error)
        return _failure(f"invalid_parameters: {error}")

    data_repository = repository or PETCTResultRepository()
    try:
        studies = data_repository.studies()
    except (OSError, json.JSONDecodeError, ValueError) as error:
        logger.error("PET-CT Tool 演示数据不可用：%s", error)
        return _failure("demo_data_unavailable")

    study = next((item for item in studies if item.get("study_id") == normalized_study_id), None)
    if study is None:
        logger.info("PET-CT Tool 查询未命中：study_not_found")
        return _failure("study_not_found")

    lesions = study.get("lesions")
    if not isinstance(lesions, list):
        logger.error("PET-CT Tool 演示数据格式错误：lesions")
        return _failure("demo_data_unavailable")

    location_matches = [lesion for lesion in lesions if lesion.get("location") == normalized_location]
    if not location_matches:
        logger.info("PET-CT Tool 查询未命中：location_not_found")
        return _failure("location_not_found")

    if normalized_lesion_id is not None:
        location_matches = [
            lesion for lesion in location_matches if lesion.get("lesion_id") == normalized_lesion_id
        ]
        if not location_matches:
            logger.info("PET-CT Tool 查询未命中：lesion_not_found")
            return _failure("lesion_not_found")

    try:
        results = [_serialize_lesion(lesion) for lesion in location_matches]
    except KeyError:
        logger.error("PET-CT Tool 演示数据格式错误：lesion fields")
        return _failure("demo_data_unavailable")

    logger.info("PET-CT Tool 查询成功：返回 %d 个病灶", len(results))
    primary_result = results[0] if len(results) == 1 else None
    return {
        "found": True,
        "study_id": normalized_study_id,
        "lesion_id": primary_result["lesion_id"] if primary_result else None,
        "location": normalized_location,
        "suv_max": primary_result["suv_max"] if primary_result else None,
        "volume_cm3": primary_result["volume_cm3"] if primary_result else None,
        "query": {"location": normalized_location, "lesion_id": normalized_lesion_id},
        "result_count": len(results),
        "results": results,
    }
