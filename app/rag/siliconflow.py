"""Small OpenAI-compatible SiliconFlow client built with the Python standard library."""

import json
import os
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


def _read_dotenv(project_dir: str) -> None:
    """Load missing environment values from a local .env without external packages."""
    dotenv_path = os.path.join(project_dir, ".env")
    if not os.path.isfile(dotenv_path):
        return
    with open(dotenv_path, encoding="utf-8") as dotenv_file:
        for line in dotenv_file:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


class SiliconFlowClient:
    """Calls SiliconFlow's embeddings and chat-completions endpoints."""

    def __init__(self, project_dir: str) -> None:
        _read_dotenv(project_dir)
        self.api_key = os.getenv("LLM_API_KEY", "")
        self.base_url = os.getenv("LLM_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
        self.llm_model = os.getenv("LLM_MODEL", "")
        self.embedding_model = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        self.reranker_model = DEFAULT_RERANKER_MODEL
        if not self.api_key:
            raise RuntimeError("LLM_API_KEY is missing. Configure it in your existing .env file.")
        if not self.llm_model:
            raise RuntimeError("LLM_MODEL is missing. Configure it in your existing .env file.")

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=90) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"SiliconFlow API request failed ({error.code}): {detail}") from error
        except URLError as error:
            raise RuntimeError(f"Unable to reach SiliconFlow API: {error.reason}") from error

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self._post("/embeddings", {"model": self.embedding_model, "input": texts})
        return [item["embedding"] for item in sorted(response["data"], key=lambda item: item["index"])]

    def rerank(self, query: str, documents: list[str], top_n: int) -> dict[str, Any]:
        """Rerank text candidates using the existing SiliconFlow credentials."""
        return self._post(
            "/rerank",
            {
                "model": self.reranker_model,
                "query": query,
                "documents": documents,
                "top_n": top_n,
                "return_documents": False,
            },
        )

    def answer(self, question: str, context: str) -> str:
        system_prompt = (
            "你是医学影像知识问答助手。仅依据给定知识库上下文回答；"
            "若上下文不足，请明确说明知识库未提供足够信息。"
            "回答仅供学习参考，不能替代专业医疗诊疗建议。"
        )
        user_prompt = f"知识库上下文：\n{context}\n\n用户问题：{question}"
        response = self._post(
            "/chat/completions",
            {
                "model": self.llm_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.2,
            },
        )
        return response["choices"][0]["message"]["content"]

    def chat(self, message: str) -> str:
        """Use the existing Qwen client for a non-RAG conversational response."""
        response = self._post(
            "/chat/completions",
            {
                "model": self.llm_model,
                "messages": [
                    {
                        "role": "system",
                        "content": "你是医学影像智能问答助手，可以进行简短、友好的普通聊天。",
                    },
                    {"role": "user", "content": message},
                ],
                "temperature": 0.2,
            },
        )
        return response["choices"][0]["message"]["content"]

    def extract_petct_parameters(self, query: str) -> dict[str, str | None]:
        """Use Qwen JSON output to extract only explicitly stated PET-CT Tool parameters."""
        response = self._post(
            "/chat/completions",
            {
                "model": self.llm_model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "从用户的 PET-CT 查询中提取参数。只返回 JSON 对象，键必须为 "
                            "study_id、location、lesion_id。未在问题中明确提供的值必须为 null，绝不猜测。"
                        ),
                    },
                    {"role": "user", "content": query},
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0,
            },
        )
        content = response["choices"][0]["message"]["content"]
        if not isinstance(content, str):
            raise RuntimeError("PET-CT parameter extraction returned non-text content")
        try:
            parameters = json.loads(content)
        except json.JSONDecodeError as error:
            raise RuntimeError("PET-CT parameter extraction did not return valid JSON") from error
        if not isinstance(parameters, dict):
            raise RuntimeError("PET-CT parameter extraction did not return an object")
        return {
            field: parameters.get(field)
            if isinstance(parameters.get(field), str) or parameters.get(field) is None
            else None
            for field in ("study_id", "location", "lesion_id")
        }

    def answer_petct_result(self, query: str, tool_result: dict[str, Any]) -> str:
        """Answer exclusively from a successful structured PET-CT Tool result."""
        response = self._post(
            "/chat/completions",
            {
                "model": self.llm_model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "你是医学影像助手。仅依据给定的 PET-CT Tool JSON 结果回答。"
                            "不得补充、推测或改写任何未出现的 SUV、体积、位置或病灶数据；"
                            "请说明结果仅供演示和学习参考。"
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"用户问题：{query}\n\nPET-CT Tool JSON：{json.dumps(tool_result, ensure_ascii=False)}",
                    },
                ],
                "temperature": 0,
            },
        )
        return response["choices"][0]["message"]["content"]

    def answer_petct_with_rag(
        self,
        query: str,
        tool_result: dict[str, Any],
        rag_context: str,
    ) -> str:
        """Answer using distinct PET-CT facts and general RAG medical knowledge."""
        response = self._post(
            "/chat/completions",
            {
                "model": self.llm_model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "你是医学影像助手。必须严格区分两类输入：Tool Result 是当前检查的具体事实；"
                            "RAG Context 是一般医学知识。具体 SUV、体积、位置、病灶数量只能来自 Tool Result，"
                            "不得猜测或补充。若 Tool Result 的 found 为 false，必须明确没有查到具体检查数据；"
                            "若 RAG Context 为空，必须明确知识库没有足够解释依据。回答仅供演示和学习参考。"
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"用户问题：{query}\n\n"
                            f"Tool Result（当前检查/病灶事实）：{json.dumps(tool_result, ensure_ascii=False)}\n\n"
                            f"RAG Context（一般医学知识）：{rag_context or '无足够相关知识片段'}"
                        ),
                    },
                ],
                "temperature": 0,
            },
        )
        return response["choices"][0]["message"]["content"]
