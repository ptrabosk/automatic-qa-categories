from src.preprocessing.evidence_builder import build_evidence_packet
from src.preprocessing.normalize_input import normalize_record


def test_only_final_agent_message_is_locked_for_audit() -> None:
    record = normalize_record(
        {
            "record_id": "last-agent",
            "messages": [
                {"id": "a", "role": "agent", "content": "Earlier agent text."},
                {"id": "c", "role": "customer", "content": "Can you answer the actual question?"},
                {"id": "b", "role": "agent", "content": "Final answer."},
                {"id": "s", "role": "system", "content": "Post-send delivery event."},
            ],
        }
    )

    packet = build_evidence_packet(record)

    assert packet.audited_message.message_id == "b"
    assert packet.audited_message.index == 2
    assert [message.index for message in packet.prior_messages] == [0, 1]
    assert all(message.index < packet.audited_message.index for message in packet.prior_messages)
