from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any
from urllib.parse import urlparse

from src.models.audit_models import EvidencePacket, SubcategoryScore
from src.utils.enums import Method


URL_PATTERN = re.compile(r"https?://[^\s)\]}>,\"']+", re.IGNORECASE)


def score_key(category: str, subcategory: str) -> str:
    return f"{category}.{subcategory}"


def make_score(
    packet: EvidencePacket,
    category: str,
    subcategory: str,
    score: int,
    *,
    method: Method = Method.DETERMINISTIC,
    evidence_used: list[str] | None = None,
    rationale: str,
    failure_note: str | None = None,
    confidence: float = 0.9,
    hard_fail: bool = False,
    source: str = "deterministic",
) -> SubcategoryScore:
    return SubcategoryScore(
        category=category,
        subcategory=subcategory,
        score=score,
        method=method,
        evidence_used=evidence_used or [f"messages[{packet.audited_message.index}]"],
        rationale=rationale,
        failure_note=failure_note,
        confidence=confidence,
        audited_message_id=packet.audited_message.message_id,
        audited_message_index=packet.audited_message.index,
        hard_fail=hard_fail,
        source=source,
    )


def lower_text(value: str | None) -> str:
    return (value or "").casefold()


def latest_customer_text(packet: EvidencePacket) -> str:
    return packet.latest_customer_message.content if packet.latest_customer_message else ""


def contains_any(text: str, phrases: list[str]) -> bool:
    folded = lower_text(text)
    return any(phrase.casefold() in folded for phrase in phrases)


def extract_urls(text: str) -> list[str]:
    return [match.rstrip(".,;:") for match in URL_PATTERN.findall(text or "")]


def domain_for_url(url: str) -> str:
    parsed = urlparse(url)
    domain = parsed.netloc.casefold()
    return domain[4:] if domain.startswith("www.") else domain


def domains_match(candidate: str, allowed: str) -> bool:
    candidate = candidate.casefold()
    allowed = allowed.casefold()
    if candidate == allowed:
        return True
    return candidate.endswith(f".{allowed}")


def extract_domain_from_value(value: Any) -> str | None:
    if not value:
        return None
    text = str(value)
    if not text.startswith(("http://", "https://")):
        text = f"https://{text}"
    domain = domain_for_url(text)
    return domain or None


def metadata_items(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, dict)]
    if isinstance(raw, dict):
        if all(isinstance(value, dict) for value in raw.values()):
            return [
                {"url": url, **value}
                for url, value in raw.items()
                if isinstance(value, dict)
            ]
        return [raw]
    return []


def metadata_for_url(items: list[dict[str, Any]], url: str) -> dict[str, Any] | None:
    domain = domain_for_url(url)
    for item in items:
        item_url = item.get("url") or item.get("href")
        if item_url and str(item_url).rstrip("/") == url.rstrip("/"):
            return item
        item_domain = item.get("domain") or (domain_for_url(str(item_url)) if item_url else None)
        if item_domain and domains_match(domain, str(item_domain)):
            return item
    return None


def text_similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, normalize_for_compare(left), normalize_for_compare(right)).ratio()


def normalize_for_compare(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", text.casefold())).strip()


def previous_agent_message(packet: EvidencePacket) -> Any | None:
    for message in reversed(packet.prior_messages):
        if message.role == "agent":
            return message
    return None


def evidence_path(packet: EvidencePacket, field: str) -> str:
    if field == "audited_message":
        return f"messages[{packet.audited_message.index}]"
    if field == "latest_customer_turn" and packet.latest_customer_message is not None:
        return f"messages[{packet.latest_customer_message.index}]"
    return field


def context_value(packet: EvidencePacket, field: str) -> Any:
    if field == "audited_message":
        return packet.audited_message.content
    if field == "latest_customer_turn":
        return latest_customer_text(packet)
    return packet.structured_context.get(field)


def value_present(value: Any) -> bool:
    return value not in (None, "", [], {})
