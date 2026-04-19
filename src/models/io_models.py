from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.utils.enums import Role


class NormalizedMessage(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    message_id: str | None = None
    role: Role
    content: str = ""
    index: int
    timestamp: str | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


class ConversationRecord(BaseModel):
    record_id: str
    messages: list[NormalizedMessage]
    structured_context: dict[str, Any] = Field(default_factory=dict)
    raw: dict[str, Any] = Field(default_factory=dict)
