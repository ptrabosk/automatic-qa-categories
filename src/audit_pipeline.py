from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from src.adapters.base_llm import BaseLLMAdapter
from src.adapters.ollama_adapter import OllamaAdapter
from src.aggregation.final_scoring import build_category_rollups
from src.aggregation.merge_scores import merge_scores
from src.models.audit_models import AuditMetadata, AuditResult, EvidencePacket, SubcategoryScore
from src.preprocessing.evidence_builder import build_evidence_packet
from src.preprocessing.normalize_input import normalize_record, normalize_records
from src.specialists.clarity import ClaritySpecialist
from src.specialists.issue_identification import IssueIdentificationSpecialist
from src.specialists.proper_resolution import ProperResolutionSpecialist
from src.specialists.tone import ToneSpecialist
from src.specialists.workflow import WorkflowSpecialist
from src.specialists.zero_tolerance import ZeroToleranceSpecialist
from src.utils.files import load_yaml, read_json
from src.validators import clarity_checks, workflow_presence, zero_tolerance


SPECIALIST_CLASSES_BY_CATEGORY = {
    "issue_identification": IssueIdentificationSpecialist,
    "proper_resolution": ProperResolutionSpecialist,
    "workflow": WorkflowSpecialist,
    "clarity": ClaritySpecialist,
    "tone": ToneSpecialist,
    "zero_tolerance": ZeroToleranceSpecialist,
}

VALIDATORS_BY_CATEGORY = {
    "zero_tolerance": [zero_tolerance.validate],
    "clarity": [clarity_checks.validate],
    "workflow": [workflow_presence.validate],
}


class AuditPipeline:
    def __init__(
        self,
        config_dir: str | Path = "config",
        *,
        llm_adapter: BaseLLMAdapter | None = None,
        use_llm: bool = True,
    ) -> None:
        load_dotenv()
        self.config_dir = Path(config_dir)
        self.rubric = load_yaml(self.config_dir / "rubric.yaml")
        self.category_map = load_yaml(self.config_dir / "category_map.yaml")
        self.model_config = load_yaml(self.config_dir / "model_config.yaml")
        self.output_schema = load_yaml(self.config_dir / "output_schema.yaml")
        self.use_llm = use_llm
        self.llm_adapter = llm_adapter or (self._build_adapter() if use_llm else None)
        self.allow_heuristic_fallback = bool(self.model_config.get("allow_heuristic_fallback", True))

    def audit_file(self, path: str | Path) -> list[AuditResult]:
        return self.audit_json(read_json(path))

    def audit_json(self, raw: Any) -> list[AuditResult]:
        return [self.audit_record(record.raw) for record in normalize_records(raw)]

    def audit_record(self, raw_record: dict[str, Any]) -> AuditResult:
        started = _utc_now()
        start_time = time.perf_counter()
        record = normalize_record(raw_record)
        packet = build_evidence_packet(record)
        deterministic_scores = self._run_validators(packet)
        specialist_scores = self._run_specialists(packet)
        merged_scores = merge_scores(packet, self.rubric, deterministic_scores, specialist_scores)
        rollups, overall_score, overall_pass_fail = build_category_rollups(self.rubric, merged_scores)
        completed = _utc_now()
        duration_ms = int((time.perf_counter() - start_time) * 1000)
        metadata = AuditMetadata(
            record_id=record.record_id,
            run_id=str(uuid.uuid4()),
            model_backend=self.llm_adapter.backend_name if self.llm_adapter else "none",
            model_name=self.llm_adapter.model_name if self.llm_adapter else None,
            audited_message_id=packet.audited_message.message_id,
            audited_message_index=packet.audited_message.index,
            latest_customer_message_index=(
                packet.latest_customer_message.index if packet.latest_customer_message else None
            ),
            started_at=started,
            completed_at=completed,
            duration_ms=duration_ms,
            deterministic_fail_count=sum(
                1 for score in deterministic_scores.values() if score.score == 0 and score.hard_fail
            ),
            llm_used=self.use_llm and self.llm_adapter is not None,
            missing_evidence=packet.missing_evidence,
        )
        return AuditResult(
            record_id=record.record_id,
            audited_message_id=packet.audited_message.message_id,
            audited_message_index=packet.audited_message.index,
            subcategory_scores=merged_scores,
            category_rollups=rollups,
            final_overall_score=overall_score,
            final_overall_pass_fail=overall_pass_fail,
            metadata=metadata,
        )

    def _run_validators(self, packet: EvidencePacket) -> dict[str, SubcategoryScore]:
        scores: dict[str, SubcategoryScore] = {}
        for category in self.rubric["categories"]:
            for validator in VALIDATORS_BY_CATEGORY.get(category, []):
                scores.update(validator(packet, self.rubric))
        return scores

    def _run_specialists(self, packet: EvidencePacket) -> dict[str, SubcategoryScore]:
        scores: dict[str, SubcategoryScore] = {}
        for category in self.rubric["categories"]:
            specialist_class = SPECIALIST_CLASSES_BY_CATEGORY.get(category)
            if specialist_class is None or category not in self.category_map.get("specialists", {}):
                continue
            specialist = specialist_class(
                llm_adapter=self.llm_adapter,
                rubric=self.rubric,
                category_map=self.category_map,
                output_schema=self.output_schema,
                use_llm=self.use_llm,
                allow_heuristic_fallback=self.allow_heuristic_fallback,
            )
            scores.update(specialist.audit(packet))
        return scores

    def _build_adapter(self) -> BaseLLMAdapter:
        backend = self.model_config.get("default_backend", "ollama")
        if backend != "ollama":
            raise ValueError(f"Unsupported model backend {backend!r}; only 'ollama' is implemented")
        config = self.model_config["ollama"]
        return OllamaAdapter(
            base_url=config.get("base_url", "http://localhost:11434"),
            model=config.get("model", "llama3.1:8b-instruct-q8_0"),
            timeout_seconds=int(config.get("request_timeout_seconds", 120)),
            options=config.get("options", {"temperature": 0}),
            use_json_schema=bool(config.get("use_json_schema", True)),
        )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
