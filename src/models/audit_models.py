from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.models.io_models import NormalizedMessage
from src.utils.enums import Method, PassFail


class EvidencePacket(BaseModel):
    record_id: str
    messages: list[NormalizedMessage]
    prior_messages: list[NormalizedMessage]
    audited_message: NormalizedMessage
    latest_customer_message: NormalizedMessage | None = None
    structured_context: dict[str, Any] = Field(default_factory=dict)
    available_evidence: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)

    def prompt_dict(self, evidence_fields: list[str] | None = None) -> dict[str, Any]:
        fields = evidence_fields or []
        selected_context = {
            field: self.structured_context.get(field)
            for field in fields
            if field in self.structured_context
        }
        return {
            "record_id": self.record_id,
            "instruction": "Audit only audited_message. Prior messages are context only.",
            "audited_message": self.audited_message.model_dump(),
            "latest_customer_message": (
                self.latest_customer_message.model_dump()
                if self.latest_customer_message is not None
                else None
            ),
            "prior_messages": [message.model_dump() for message in self.prior_messages],
            "structured_context": selected_context,
            "available_evidence": self.available_evidence,
            "missing_evidence": self.missing_evidence,
        }


class ScoreProvenance(BaseModel):
    source: str
    score: int = Field(ge=0, le=1)
    hard_fail: bool = False
    rationale: str = ""


class SubcategoryScore(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    category: str
    subcategory: str
    score: int = Field(ge=0, le=1)
    method: Method
    evidence_used: list[str] = Field(default_factory=list)
    rationale: str
    failure_note: str | None = None
    confidence: float = Field(ge=0, le=1)
    audited_message_id: str | None = None
    audited_message_index: int
    hard_fail: bool = False
    source: str = "unknown"
    provenance: list[ScoreProvenance] = Field(default_factory=list)

    @field_validator("evidence_used")
    @classmethod
    def evidence_paths_not_empty(cls, value: list[str]) -> list[str]:
        return value or ["audited_message"]

    @model_validator(mode="after")
    def failure_note_rules(self) -> "SubcategoryScore":
        if self.score == 0 and not self.failure_note:
            raise ValueError("failure_note is required when score is 0")
        if self.score == 1:
            self.failure_note = None
        return self

    @property
    def key(self) -> str:
        return f"{self.category}.{self.subcategory}"


class CategoryRollup(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    category: str
    score: int = Field(ge=0, le=1)
    pass_fail: PassFail
    failed_subcategories: list[str] = Field(default_factory=list)
    required_failed_subcategories: list[str] = Field(default_factory=list)


class AuditMetadata(BaseModel):
    record_id: str
    run_id: str
    model_backend: str
    model_name: str | None = None
    audited_message_id: str | None = None
    audited_message_index: int
    latest_customer_message_index: int | None = None
    started_at: str
    completed_at: str
    duration_ms: int
    deterministic_fail_count: int
    llm_used: bool
    missing_evidence: list[str] = Field(default_factory=list)


class AuditResult(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    record_id: str
    audited_message_id: str | None = None
    audited_message_index: int
    subcategory_scores: dict[str, SubcategoryScore]
    category_rollups: dict[str, CategoryRollup]
    final_overall_score: int = Field(ge=0, le=1)
    final_overall_pass_fail: PassFail
    metadata: AuditMetadata
