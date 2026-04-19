from src.aggregation.merge_scores import merge_scores
from src.models.audit_models import EvidencePacket
from src.models.io_models import NormalizedMessage
from src.utils.enums import Method, Role
from src.validators.deterministic_helpers import make_score


def test_hard_fail_wins_over_llm_pass(minimal_rubric: dict) -> None:
    audited = NormalizedMessage(role=Role.AGENT, content="Use BAD10.", index=1)
    packet = EvidencePacket(
        record_id="r",
        messages=[
            NormalizedMessage(role=Role.CUSTOMER, content="Discount?", index=0),
            audited,
        ],
        prior_messages=[NormalizedMessage(role=Role.CUSTOMER, content="Discount?", index=0)],
        audited_message=audited,
        latest_customer_message=NormalizedMessage(role=Role.CUSTOMER, content="Discount?", index=0),
    )
    deterministic = make_score(
        packet,
        "zero_tolerance",
        "opt_out",
        0,
        rationale="The customer requested opt-out and the reply did not acknowledge it.",
        failure_note="The customer asked to stop contact, but the audited reply did not acknowledge that opt-out request.",
        hard_fail=True,
    )
    specialist = make_score(
        packet,
        "zero_tolerance",
        "opt_out",
        1,
        method=Method.HYBRID,
        rationale="LLM passed it.",
        source="llm",
    )

    merged = merge_scores(
        packet,
        minimal_rubric,
        {"zero_tolerance.opt_out": deterministic},
        {"zero_tolerance.opt_out": specialist},
    )

    assert merged["zero_tolerance.opt_out"].score == 0
    assert merged["zero_tolerance.opt_out"].source == "deterministic"
    assert merged["zero_tolerance.opt_out"].provenance[1].source == "llm"
