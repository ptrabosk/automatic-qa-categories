from __future__ import annotations

import re

from src.models.audit_models import EvidencePacket, SubcategoryScore
from src.validators.deterministic_helpers import contains_any, latest_customer_text, make_score, score_key

OPT_OUT_PHRASES = ["unsubscribe", "opt out", "remove me", "stop texting", "stop messaging", "do not contact", "stop"]
OPT_OUT_ACK_PHRASES = ["unsubscribe", "opt out", "removed", "no longer contact", "stop messages", "won't contact"]
SENSITIVE_REQUESTS = [
    "send me your password",
    "provide your password",
    "share your password",
    "full credit card",
    "social security number",
    "ssn",
]
DEFAULT_INAPPROPRIATE_TERMS = ["damn", "hell", "idiot", "stupid"]
SSN_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
CARD_PATTERN = re.compile(r"\b(?:\d[ -]*?){13,19}\b")


def validate(packet: EvidencePacket, rubric: dict) -> dict[str, SubcategoryScore]:
    results: dict[str, SubcategoryScore] = {}
    audited = packet.audited_message.content
    latest = latest_customer_text(packet)

    if contains_any(latest, OPT_OUT_PHRASES) and not contains_any(audited, OPT_OUT_ACK_PHRASES):
        result = make_score(
            packet,
            "zero_tolerance",
            "opt_out",
            0,
            evidence_used=[f"messages[{packet.latest_customer_message.index}]", f"messages[{packet.audited_message.index}]"]
            if packet.latest_customer_message
            else [f"messages[{packet.audited_message.index}]"],
            rationale="Latest customer turn contains opt-out language and the audited reply does not acknowledge opt-out action.",
            failure_note="The customer asked to stop contact, but the audited reply did not acknowledge that opt-out request.",
            confidence=0.95,
            hard_fail=True,
        )
    else:
        result = make_score(
            packet,
            "zero_tolerance",
            "opt_out",
            1,
            evidence_used=[f"messages[{packet.audited_message.index}]"],
            rationale="No unhandled opt-out request was explicit in the JSON context.",
            confidence=0.8,
        )
    results[score_key(result.category, result.subcategory)] = result

    pi_failure = _personal_info_failure(audited)
    if pi_failure:
        result = make_score(
            packet,
            "zero_tolerance",
            "personal_information",
            0,
            rationale=pi_failure,
            failure_note=pi_failure,
            confidence=0.95,
            hard_fail=True,
        )
    else:
        result = make_score(
            packet,
            "zero_tolerance",
            "personal_information",
            1,
            rationale="The audited reply does not include an explicit sensitive-information request or disclosure pattern.",
            confidence=0.8,
        )
    results[score_key(result.category, result.subcategory)] = result

    terms = packet.structured_context.get("inappropriate_language_terms", DEFAULT_INAPPROPRIATE_TERMS)
    matched = _matched_inappropriate_term(audited, terms)
    if matched:
        result = make_score(
            packet,
            "zero_tolerance",
            "inappropriate_language",
            0,
            rationale=f"The audited reply contains inappropriate term '{matched}' from the configured lexicon.",
            failure_note=f"The audited reply contains inappropriate language: '{matched}'.",
            confidence=0.95,
            hard_fail=True,
        )
    else:
        result = make_score(
            packet,
            "zero_tolerance",
            "inappropriate_language",
            1,
            rationale="No configured inappropriate-language term appears in the audited reply.",
            confidence=0.8,
        )
    results[score_key(result.category, result.subcategory)] = result
    return results


def _personal_info_failure(text: str) -> str | None:
    if SSN_PATTERN.search(text):
        return "The audited reply includes a Social Security number pattern."
    if CARD_PATTERN.search(text):
        digits = re.sub(r"\D", "", CARD_PATTERN.search(text).group(0))  # type: ignore[union-attr]
        if len(digits) >= 13:
            return "The audited reply includes a long payment-card number pattern."
    folded = text.casefold()
    for phrase in SENSITIVE_REQUESTS:
        if phrase in folded:
            return f"The audited reply requests sensitive personal information: '{phrase}'."
    return None


def _matched_inappropriate_term(text: str, terms: object) -> str | None:
    if not isinstance(terms, list):
        terms = DEFAULT_INAPPROPRIATE_TERMS
    folded = text.casefold()
    for term in terms:
        word = str(term).casefold()
        if re.search(rf"\b{re.escape(word)}\b", folded):
            return str(term)
    return None
