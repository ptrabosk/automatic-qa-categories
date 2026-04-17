from __future__ import annotations

import json
from typing import Any

from src.models.io_models import ConversationRecord, NormalizedMessage
from src.utils.enums import Role

MESSAGE_CONTAINER_KEYS = ("messages", "thread", "conversation")
TOP_LEVEL_NON_CONTEXT_KEYS = set(MESSAGE_CONTAINER_KEYS) | {"record_id", "conversation_id", "id"}


def normalize_records(raw: Any) -> list[ConversationRecord]:
    if isinstance(raw, list):
        return [normalize_record(item) for item in raw]
    if isinstance(raw, dict) and isinstance(raw.get("records"), list):
        return [normalize_record(item) for item in raw["records"]]
    return [normalize_record(raw)]


def normalize_record(raw: Any) -> ConversationRecord:
    if not isinstance(raw, dict):
        raise ValueError("Each conversation record must be a JSON object")

    messages_raw = _extract_messages(raw)
    messages = [
        NormalizedMessage(
            message_id=_string_or_none(
                message.get("id") or message.get("message_id") or message.get("uuid")
            ),
            role=_normalize_role(message),
            content=_normalize_content(message),
            index=index,
            timestamp=_string_or_none(message.get("timestamp") or message.get("created_at")),
            raw=message,
        )
        for index, message in enumerate(messages_raw)
    ]
    if not messages:
        raise ValueError("Conversation record must contain at least one message")

    record_id = _string_or_none(raw.get("record_id") or raw.get("conversation_id") or raw.get("id"))
    if record_id is None:
        record_id = f"record-{abs(hash(json.dumps(raw, sort_keys=True, default=str))) % 10_000_000}"

    return ConversationRecord(
        record_id=record_id,
        messages=messages,
        structured_context=_extract_structured_context(raw),
        raw=raw,
    )


def _extract_messages(raw: dict[str, Any]) -> list[dict[str, Any]]:
    for key in MESSAGE_CONTAINER_KEYS:
        value = raw.get(key)
        if isinstance(value, list):
            return [_as_message_dict(item) for item in value]
        if isinstance(value, dict):
            for nested_key in ("messages", "thread", "turns"):
                nested = value.get(nested_key)
                if isinstance(nested, list):
                    return [_as_message_dict(item) for item in nested]
    raise ValueError("Could not find ordered conversation messages")


def _as_message_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {"role": "other", "content": str(value)}


def _normalize_role(message: dict[str, Any]) -> Role:
    raw_role = (
        message.get("role")
        or message.get("sender_type")
        or message.get("sender")
        or message.get("author_type")
        or message.get("type")
        or ""
    )
    role = str(raw_role).strip().lower()
    if any(token in role for token in ("agent", "assistant", "support", "staff", "rep")):
        return Role.AGENT
    if any(token in role for token in ("customer", "user", "shopper", "client", "visitor")):
        return Role.CUSTOMER
    if "system" in role or "internal" in role:
        return Role.SYSTEM
    return Role.OTHER


def _normalize_content(message: dict[str, Any]) -> str:
    value = (
        message.get("content")
        if "content" in message
        else message.get("text", message.get("body", message.get("message", "")))
    )
    if value is None:
        return ""
    if isinstance(value, str):
        return " ".join(value.split())
    return json.dumps(value, sort_keys=True, default=str)


def _extract_structured_context(raw: dict[str, Any]) -> dict[str, Any]:
    context = {
        key: value
        for key, value in raw.items()
        if key not in TOP_LEVEL_NON_CONTEXT_KEYS and value is not None
    }
    nested_context = raw.get("context")
    if isinstance(nested_context, dict):
        context.update({key: value for key, value in nested_context.items() if value is not None})
    return context


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
