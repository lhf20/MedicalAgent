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
