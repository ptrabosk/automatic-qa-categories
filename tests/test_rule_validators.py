from src.audit_pipeline import AuditPipeline


def test_deterministic_hard_fail_overrides_specialist_pass() -> None:
    raw = {
        "record_id": "optout",
        "messages": [
            {"role": "customer", "content": "Please stop texting me."},
            {"role": "agent", "content": "You can use promo code SAVE10 today."},
        ],
        "promo_notes": [{"code": "SAVE10", "active": True}],
    }

    result = AuditPipeline(use_llm=False).audit_record(raw)
    score = result.subcategory_scores["zero_tolerance.opt_out"]

    assert score.score == 0
    assert score.hard_fail is True
    assert result.final_overall_score == 0
    assert score.provenance[0].source == "deterministic"
