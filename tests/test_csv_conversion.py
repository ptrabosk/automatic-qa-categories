from __future__ import annotations

import csv
import json

from src.preprocessing.evidence_builder import build_evidence_packet
from src.preprocessing.normalize_input import normalize_record
from src.training.csv_converter import convert_csv_file, convert_csv_row


def test_csv_row_converts_to_pipeline_json() -> None:
    row = {
        "SEND_ID": "send-1",
        "HAS_SHOPIFY": "false",
        "COMPANY_NAME": "Example Co",
        "COMPANY_WEBSITE": "https://example.com",
        "PERSONA": "Jordan",
        "MESSAGE_TONE": "Polished",
        "CONVERSATION_JSON": json.dumps(
            [
                {
                    "date_time": "2026-01-01 10:00:00",
                    "message_text": "Hi, do you have this?",
                    "message_type": "customer",
                },
                {
                    "date_time": "2026-01-01 10:01:00",
                    "message_id": "final-agent",
                    "message_text": "Yes, it is available.",
                    "message_type": "agent",
                },
            ]
        ),
        "LAST_5_PRODUCTS": json.dumps([{"product_name": "Widget"}]),
        "ORDERS": "",
        "COUPONS": json.dumps([{"coupon": "SAVE10", "redemption_status": "UNUSED"}]),
        "COMPANY_NOTES": "Use a concise tone.",
        "ESCALATION_TOPICS": "['Returns']",
        "BLOCKLISTED_WORDS": "['staging']",
        "COMPANY_PROMOTIONS": "",
    }

    converted = convert_csv_row(row, row_number=2)
    record = normalize_record(converted)
    packet = build_evidence_packet(record)

    assert converted["record_id"] == "send-1"
    assert converted["company_profile"]["name"] == "Example Co"
    assert converted["website_findings"]["allowed_domains"] == ["example.com"]
    assert converted["notes"] == "Use a concise tone."
    assert "templates" not in converted
    assert converted["product_information"] == [{"product_name": "Widget"}]
    assert converted["promo_notes"] == [{"coupon": "SAVE10", "redemption_status": "UNUSED"}]
    assert converted["workflow_flags"]["escalation_topics"] == ["Returns"]
    assert packet.audited_message.message_id == "final-agent"
    assert packet.audited_message.index == 1


def test_convert_csv_file_handles_embedded_json_cells(tmp_path) -> None:
    path = tmp_path / "qa.csv"
    columns = [
        "SEND_ID",
        "HAS_SHOPIFY",
        "COMPANY_NAME",
        "COMPANY_WEBSITE",
        "PERSONA",
        "MESSAGE_TONE",
        "CONVERSATION_JSON",
        "LAST_5_PRODUCTS",
        "ORDERS",
        "COUPONS",
        "COMPANY_NOTES",
        "ESCALATION_TOPICS",
        "BLOCKLISTED_WORDS",
        "COMPANY_PROMOTIONS",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerow(
            {
                "SEND_ID": "send-2",
                "HAS_SHOPIFY": "true",
                "COMPANY_NAME": "Example Co",
                "COMPANY_WEBSITE": "example.com",
                "PERSONA": "Jordan",
                "MESSAGE_TONE": "Friendly",
                "CONVERSATION_JSON": json.dumps(
                    [
                        {"message_type": "customer", "message_text": "Hello\nagain"},
                        {"message_type": "agent", "message_text": "Hi there"},
                    ],
                    indent=2,
                ),
                "LAST_5_PRODUCTS": "",
                "ORDERS": "",
                "COUPONS": "",
                "COMPANY_NOTES": "",
                "ESCALATION_TOPICS": "[]",
                "BLOCKLISTED_WORDS": "[]",
                "COMPANY_PROMOTIONS": "",
            }
        )

    payload = convert_csv_file(path)

    assert payload["record_count"] == 1
    assert payload["records"][0]["record_id"] == "send-2"
    assert payload["records"][0]["messages"][-1]["role"] == "agent"
