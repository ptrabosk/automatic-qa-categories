from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator


class LLMRequest(BaseModel):
    prompt: str
    output_schema: dict[str, Any]
    category: str


class LLMResponse(BaseModel):
    content: dict[str, Any]
    raw_text: str | None = None
    model: str | None = None


class SpecialistSubcategoryOutput(BaseModel):
    subcategory: str
    score: int = Field(ge=0, le=1)
    evidence_used: list[str] = Field(default_factory=list)
    rationale: str
    failure_note: str | None = None
    confidence: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def require_failure_note_for_failures(self) -> "SpecialistSubcategoryOutput":
        if self.score == 0 and not self.failure_note:
            raise ValueError("failure_note is required when score is 0")
        if self.score == 1:
            self.failure_note = None
        return self


class SpecialistOutput(BaseModel):
    category: str
    results: list[SpecialistSubcategoryOutput]
