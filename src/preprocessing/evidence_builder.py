from __future__ import annotations

from typing import Any

from src.models.audit_models import EvidencePacket
from src.models.io_models import ConversationRecord
from src.preprocessing.thread_parser import (
    find_final_agent_message,
    find_latest_customer_before,
    prior_context,
)

OPTIONAL_EVIDENCE_FIELDS = [
    "company_profile",
    "customer_profile",
    "notes",
    "templates",
    "product_information",
    "promo_notes",
    "checkout_page",
    "website_findings",
    "link_metadata",
    "product_views",
    "purchases",
    "workflow_flags",
]


def build_evidence_packet(record: ConversationRecord) -> EvidencePacket:
    audited_message = find_final_agent_message(record.messages)
    latest_customer = find_latest_customer_before(record.messages, audited_message.index)
    available, missing = _evidence_inventory(record.structured_context)
    return EvidencePacket(
        record_id=record.record_id,
        messages=record.messages,
        prior_messages=prior_context(record.messages, audited_message.index),
        audited_message=audited_message,
        latest_customer_message=latest_customer,
        structured_context=record.structured_context,
        available_evidence=available,
        missing_evidence=missing,
    )


def _evidence_inventory(context: dict[str, Any]) -> tuple[list[str], list[str]]:
    available: list[str] = []
    missing: list[str] = []
    for field in OPTIONAL_EVIDENCE_FIELDS:
        value = context.get(field)
        if value in (None, "", [], {}):
            missing.append(field)
        else:
            available.append(field)
    return available, missing
