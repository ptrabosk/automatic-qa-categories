from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseLLMAdapter(ABC):
    """Interface for local JSON-producing model backends."""

    backend_name: str
    model_name: str | None

    @abstractmethod
    def generate_json(self, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        """Return one JSON object matching the requested schema."""
