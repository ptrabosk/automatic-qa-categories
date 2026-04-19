from src.preprocessing.evidence_builder import build_evidence_packet
from src.preprocessing.normalize_input import normalize_record


def test_missing_optional_fields_do_not_crash_normalization() -> None:
    raw = {
        "conversation_id": "c1",
        "conversation": {
            "messages": [
                {"sender_type": "customer", "text": "Hello"},
                {"sender_type": "support_agent", "text": "Hi there"},
            ]
        },
    }

    record = normalize_record(raw)
    packet = build_evidence_packet(record)

    assert record.record_id == "c1"
    assert packet.audited_message.index == 1
    assert "company_profile" in packet.missing_evidence
    assert packet.latest_customer_message is not None
    assert packet.latest_customer_message.index == 0
