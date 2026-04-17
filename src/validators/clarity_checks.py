from __future__ import annotations

import re

from src.models.audit_models import EvidencePacket, SubcategoryScore
from src.validators.deterministic_helpers import make_score, score_key


def validate(packet: EvidencePacket, rubric: dict) -> dict[str, SubcategoryScore]:
    results = {}
    for result in (
        _correct_grammar(packet),
        _no_typos(packet),
        _no_repetition(packet),
        _double_text(packet),
    ):
        results[score_key(result.category, result.subcategory)] = result
    return results


def _correct_grammar(packet: EvidencePacket) -> SubcategoryScore:
    text = packet.audited_message.content
    if _unbalanced_brackets(text):
        return make_score(
            packet,
            "clarity",
            "correct_grammar",
            0,
            rationale="The audited reply contains unbalanced brackets or parentheses.",
            failure_note="The audited reply has unbalanced brackets or parentheses.",
            confidence=0.85,
            hard_fail=True,
        )
    return make_score(packet, "clarity", "correct_grammar", 1, rationale="No deterministic grammar failure pattern was found.", confidence=0.65)


def _no_typos(packet: EvidencePacket) -> SubcategoryScore:
    text = packet.audited_message.content
    match = re.search(r"\b[a-zA-Z]*([a-zA-Z])\1{3,}[a-zA-Z]*\b", text)
    if match:
        return make_score(
            packet,
            "clarity",
            "no_typos",
            0,
            rationale="The audited reply contains a repeated-letter typo pattern.",
            failure_note=f"The audited reply contains a likely typo with repeated letters: '{match.group(0)}'.",
            confidence=0.82,
            hard_fail=True,
        )
    return make_score(packet, "clarity", "no_typos", 1, rationale="No deterministic typo pattern was found.", confidence=0.65)


def _no_repetition(packet: EvidencePacket) -> SubcategoryScore:
    text = packet.audited_message.content
    duplicate_word = re.search(r"\b(\w+)\s+\1\b", text, flags=re.IGNORECASE)
    duplicate_sentence = _duplicate_sentence(text)
    if duplicate_word or duplicate_sentence:
        repeated = duplicate_sentence or duplicate_word.group(0)  # type: ignore[union-attr]
        return make_score(
            packet,
            "clarity",
            "no_repetition",
            0,
            rationale="The audited reply contains repeated adjacent wording.",
            failure_note=f"The audited reply repeats wording: '{repeated}'.",
            confidence=0.88,
            hard_fail=True,
        )
    return make_score(packet, "clarity", "no_repetition", 1, rationale="No deterministic repetition pattern was found.", confidence=0.65)


def _double_text(packet: EvidencePacket) -> SubcategoryScore:
    from src.validators.deterministic_helpers import previous_agent_message, text_similarity

    previous = previous_agent_message(packet)
    if previous and text_similarity(previous.content, packet.audited_message.content) >= 0.92:
        return make_score(
            packet,
            "proper_resolution",
            "double_text",
            0,
            evidence_used=[f"messages[{previous.index}]", f"messages[{packet.audited_message.index}]"],
            rationale="The audited reply is nearly identical to the prior agent message.",
            failure_note="The audited final reply duplicates the prior agent message instead of adding a new response.",
            confidence=0.93,
            hard_fail=True,
        )
    return make_score(packet, "proper_resolution", "double_text", 1, rationale="The audited reply is not a near-duplicate of the prior agent message.", confidence=0.72)


def _unbalanced_brackets(text: str) -> bool:
    pairs = [("(", ")"), ("[", "]"), ("{", "}")]
    return any(text.count(left) != text.count(right) for left, right in pairs)


def _duplicate_sentence(text: str) -> str | None:
    sentences = [part.strip() for part in re.split(r"[.!?]+", text) if part.strip()]
    seen: set[str] = set()
    for sentence in sentences:
        normalized = re.sub(r"\s+", " ", sentence.casefold())
        if normalized in seen and len(normalized.split()) >= 3:
            return sentence
        seen.add(normalized)
    return None
