from src.audit_pipeline import AuditPipeline
from src.models.audit_models import AuditResult


def test_output_matches_pydantic_schema() -> None:
    raw = {
        "record_id": "schema",
        "messages": [
            {"role": "customer", "content": "Hello"},
            {"role": "agent", "content": "Hello, how can I help?"},
        ],
    }

    result = AuditPipeline(use_llm=False).audit_record(raw)
    validated = AuditResult.model_validate(result.model_dump())

    assert validated.record_id == "schema"
    assert validated.audited_message_index == 1
    assert validated.subcategory_scores
    assert all(score.failure_note for score in validated.subcategory_scores.values() if score.score == 0)
