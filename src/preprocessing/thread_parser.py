from __future__ import annotations

from src.models.io_models import NormalizedMessage
from src.utils.enums import Role


def find_final_agent_message(messages: list[NormalizedMessage]) -> NormalizedMessage:
    for message in reversed(messages):
        if message.role == Role.AGENT:
            return message
    raise ValueError("Conversation record does not contain an agent message to audit")


def find_latest_customer_before(
    messages: list[NormalizedMessage], audited_index: int
) -> NormalizedMessage | None:
    for message in reversed(messages[:audited_index]):
        if message.role == Role.CUSTOMER:
            return message
    return None


def prior_context(messages: list[NormalizedMessage], audited_index: int) -> list[NormalizedMessage]:
    return [message for message in messages if message.index < audited_index]
