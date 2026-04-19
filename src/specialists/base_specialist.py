from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.adapters.base_llm import BaseLLMAdapter
from src.models.audit_models import EvidencePacket, SubcategoryScore
from src.models.llm_models import SpecialistOutput
from src.utils.enums import Method
from src.utils.files import read_text
from src.validators.deterministic_helpers import make_score, score_key


class BaseSpecialist:
    category: str

    def __init__(
        self,
        *,
        llm_adapter: BaseLLMAdapter | None,
        rubric: dict[str, Any],
        category_map: dict[str, Any],
        output_schema: dict[str, Any],
        use_llm: bool = True,
        allow_heuristic_fallback: bool = True,
    ) -> None:
        self.llm_adapter = llm_adapter
        self.rubric = rubric
        self.category_map = category_map
        self.output_schema = output_schema
        self.use_llm = use_llm
        self.allow_heuristic_fallback = allow_heuristic_fallback

    def audit(self, packet: EvidencePacket) -> dict[str, SubcategoryScore]:
        config = self.category_map["specialists"][self.category]
        subcategories = list(config["subcategories"])
        scores: dict[str, SubcategoryScore] = {}
        if self.use_llm and self.llm_adapter is not None:
            try:
                scores.update(self._audit_with_llm(packet, config, subcategories))
            except Exception as exc:
                if not self.allow_heuristic_fallback:
                    raise
                scores.update(self._fallback_scores(packet, subcategories, reason=str(exc)))
        else:
            scores.update(self._fallback_scores(packet, subcategories, reason="LLM disabled"))

        for subcategory in subcategories:
            key = score_key(self.category, subcategory)
            if key not in scores:
                scores[key] = self._fallback_score(packet, subcategory, "No specialist result returned")
        return scores

    def _audit_with_llm(
        self,
        packet: EvidencePacket,
        config: dict[str, Any],
        subcategories: list[str],
    ) -> dict[str, SubcategoryScore]:
        prompt = self._build_prompt(packet, config, subcategories)
        raw = self.llm_adapter.generate_json(prompt, self.output_schema)  # type: ignore[union-attr]
        specialist_output = SpecialistOutput.model_validate(raw)
        if specialist_output.category != self.category:
            raise ValueError(f"Specialist returned category {specialist_output.category!r}")
        scores: dict[str, SubcategoryScore] = {}
        for item in specialist_output.results:
            if item.subcategory not in subcategories:
                continue
            method = self._method_for(item.subcategory)
            score = SubcategoryScore(
                category=self.category,
                subcategory=item.subcategory,
                score=item.score,
                method=method,
                evidence_used=item.evidence_used,
                rationale=item.rationale,
                failure_note=item.failure_note,
                confidence=item.confidence,
                audited_message_id=packet.audited_message.message_id,
                audited_message_index=packet.audited_message.index,
                hard_fail=False,
                source="llm",
            )
            scores[score_key(score.category, score.subcategory)] = score
        return scores

    def _fallback_scores(
        self, packet: EvidencePacket, subcategories: list[str], reason: str
    ) -> dict[str, SubcategoryScore]:
        return {
            score_key(self.category, subcategory): self._fallback_score(packet, subcategory, reason)
            for subcategory in subcategories
        }

    def _fallback_score(
        self, packet: EvidencePacket, subcategory: str, reason: str
    ) -> SubcategoryScore:
        audited_text = packet.audited_message.content.strip()
        if not audited_text:
            return make_score(
                packet,
                self.category,
                subcategory,
                0,
                method=self._method_for(subcategory),
                rationale=f"{reason}; the audited reply is empty.",
                failure_note="The audited final agent reply is empty.",
                confidence=0.7,
                source="heuristic_specialist",
            )
        evidence_used = [f"messages[{packet.audited_message.index}]"]
        if packet.latest_customer_message is not None:
            evidence_used.append(f"messages[{packet.latest_customer_message.index}]")
        return make_score(
            packet,
            self.category,
            subcategory,
            1,
            method=self._method_for(subcategory),
            evidence_used=evidence_used,
            rationale=f"{reason}; no explicit failure was found in the provided JSON.",
            confidence=0.5,
            source="heuristic_specialist",
        )

    def _build_prompt(
        self, packet: EvidencePacket, config: dict[str, Any], subcategories: list[str]
    ) -> str:
        prompt_dir = Path(__file__).resolve().parents[1] / "prompts"
        shared = read_text(prompt_dir / "shared_system.txt")
        category_prompt = read_text(prompt_dir / config["prompt_file"])
        evidence = packet.prompt_dict(config.get("evidence_fields", []))
        rubric = self.rubric["categories"][self.category]
        payload = {
            "category": self.category,
            "subcategories": subcategories,
            "rubric": rubric,
            "evidence_packet": evidence,
        }
        return "\n\n".join(
            [
                shared,
                category_prompt,
                "Return JSON matching this schema:",
                json.dumps(self.output_schema, indent=2),
                "Audit payload:",
                json.dumps(payload, indent=2, ensure_ascii=False, default=str),
            ]
        )

    def _method_for(self, subcategory: str) -> Method:
        subconfig = self.rubric["categories"][self.category]["subcategories"][subcategory]
        method = subconfig.get("method", "hybrid")
        return Method(method)
