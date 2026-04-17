from __future__ import annotations

from typing import Any

from src.preprocessing.evidence_builder import build_evidence_packet
from src.preprocessing.normalize_input import normalize_record


def build_training_rows(labeled_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in labeled_records:
        raw_input = item.get("input") or item.get("conversation") or item.get("record") or item
        labels = item.get("labels") or item.get("expected_output") or item.get("audit_result")
        if labels is None:
            raise ValueError("Each labeled record needs labels, expected_output, or audit_result")
        record = normalize_record(raw_input)
        packet = build_evidence_packet(record)
        rows.append(
            {
                "record_id": record.record_id,
                "messages": [message.model_dump() for message in record.messages],
                "audited_message_index": packet.audited_message.index,
                "latest_customer_message_index": (
                    packet.latest_customer_message.index if packet.latest_customer_message else None
                ),
                "structured_context": record.structured_context,
                "labels": labels,
            }
        )
    return rows


def build_chat_finetune_rows(training_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in training_rows:
        rows.append(
            {
                "messages": [
                    {
                        "role": "system",
                        "content": "Audit only the final agent message using the provided JSON. Return strict JSON.",
                    },
                    {"role": "user", "content": _input_content(row)},
                    {"role": "assistant", "content": _label_content(row["labels"])},
                ]
            }
        )
    return rows


def _input_content(row: dict[str, Any]) -> str:
    import json

    return json.dumps(
        {
            "messages": row["messages"],
            "audited_message_index": row["audited_message_index"],
            "latest_customer_message_index": row["latest_customer_message_index"],
            "structured_context": row["structured_context"],
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def _label_content(labels: Any) -> str:
    import json

    return json.dumps(labels, ensure_ascii=False, sort_keys=True)
