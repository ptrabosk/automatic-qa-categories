from __future__ import annotations

import json
import re
from typing import Any

import requests

from src.adapters.base_llm import BaseLLMAdapter


class OllamaAdapter(BaseLLMAdapter):
    backend_name = "ollama"

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.1:8b-instruct-q8_0",
        timeout_seconds: int = 120,
        options: dict[str, Any] | None = None,
        use_json_schema: bool = True,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model
        self.timeout_seconds = timeout_seconds
        self.options = options or {"temperature": 0}
        self.use_json_schema = use_json_schema

    def generate_json(self, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": self.options,
            "format": schema if self.use_json_schema else "json",
        }
        response = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        raw = response.json().get("response", "")
        return _parse_json_object(raw)


def _parse_json_object(text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("LLM response must be a JSON object")
    return parsed
