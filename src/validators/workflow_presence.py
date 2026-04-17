from __future__ import annotations

from src.models.audit_models import EvidencePacket, SubcategoryScore
from src.validators.deterministic_helpers import contains_any, latest_customer_text, make_score, score_key, value_present

WORKFLOW_FIELDS = {
    "checkout_page": ("checkout_page", ["checkout", "cart", "payment", "billing"]),
    "company_profile": ("company_profile", ["company", "policy", "shipping", "returns", "hours", "store"]),
    "customer_profile": ("customer_profile", ["my profile", "my account", "my order", "my address", "my size"]),
    "notes": ("notes", ["note", "notes", "previously", "as mentioned"]),
    "product_information": ("product_information", ["product", "item", "stock", "size", "material", "restock"]),
    "promo_notes": ("promo_notes", ["discount", "promo", "coupon", "code"]),
    "templates": ("templates", ["template", "macro", "script"]),
    "website": ("website_findings", ["website", "site", "page", "link", "url"]),
}


def validate(packet: EvidencePacket, rubric: dict) -> dict[str, SubcategoryScore]:
    results: dict[str, SubcategoryScore] = {}
    conversation = make_score(
        packet,
        "workflow",
        "conversation",
        1,
        evidence_used=["messages", f"messages[{packet.audited_message.index}]"],
        rationale="The ordered conversation messages were present and the final agent message was locked for audit.",
        confidence=0.9,
    )
    results[score_key(conversation.category, conversation.subcategory)] = conversation

    for subcategory, (field, triggers) in WORKFLOW_FIELDS.items():
        result = _validate_field(packet, subcategory, field, triggers)
        results[score_key(result.category, result.subcategory)] = result
    return results


def _validate_field(
    packet: EvidencePacket, subcategory: str, field: str, triggers: list[str]
) -> SubcategoryScore:
    source_present = value_present(packet.structured_context.get(field))
    combined_text = f"{latest_customer_text(packet)} {packet.audited_message.content}"
    triggered = contains_any(combined_text, triggers)
    if not source_present and triggered:
        return make_score(
            packet,
            "workflow",
            subcategory,
            0,
            evidence_used=[f"messages[{packet.audited_message.index}]", f"missing:{field}"],
            rationale=f"The audited context references {subcategory}, but {field} is missing from the JSON.",
            failure_note=f"The audited reply addresses {subcategory.replace('_', ' ')}, but {field} is missing from the JSON evidence.",
            confidence=0.82,
            hard_fail=True,
        )
    return make_score(
        packet,
        "workflow",
        subcategory,
        1,
        evidence_used=[field if source_present else f"missing:{field}", f"messages[{packet.audited_message.index}]"],
        rationale=f"{field} is present, or the audited reply does not explicitly require that source.",
        confidence=0.68 if not source_present else 0.82,
    )
