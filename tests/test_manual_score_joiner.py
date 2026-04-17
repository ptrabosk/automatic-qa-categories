from __future__ import annotations

import csv
import json
from pathlib import Path

from src.training.manual_score_joiner import (
    SCORE_COLUMN_MAP,
    build_chat_training_rows,
    build_joined_training_rows,
)


def test_manual_scores_join_to_chat_training_rows(tmp_path: Path) -> None:
    conversations = tmp_path / "qa_training_set.csv"
    scores = tmp_path / "qa.csv"
    templates = tmp_path / "templates.csv"
    _write_conversations_csv(conversations)
    _write_scores_csv(scores)
    _write_templates_csv(templates)

    rows = build_joined_training_rows(conversations, scores, templates_file=templates)
    chat_rows = build_chat_training_rows(rows)

    labels = rows[0]["labels"]["scores"]
    raw_scores = rows[0]["label_metadata"]["raw_scores"]
    assistant = json.loads(chat_rows[0]["messages"][2]["content"])

    assert rows[0]["send_id"] == "send-1"
    assert rows[0]["audited_message_index"] == 1
    assert rows[0]["input"]["structured_context"]["notes"] == "Company notes only."
    assert rows[0]["input"]["structured_context"]["templates"] == "Template text."
    assert rows[0]["input"]["send_id"] == "send-1"
    assert labels["issue_identification.intent_identified"] == 1
    assert raw_scores["issue_identification.intent_identified"] == 2
    assert labels["workflow.conversation"] == 0
    assert len(labels) == len(SCORE_COLUMN_MAP)
    assert assistant["send_id"] == "send-1"
    assert "record_id" not in assistant
    assert assistant["scores"] == labels
    assert set(chat_rows[0]) == {"messages"}
    assert "zero_tolerance.opt_out" in rows[0]["label_metadata"]["unlabeled_active_subcategories"]


def _write_conversations_csv(path: Path) -> None:
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
                "SEND_ID": "send-1",
                "HAS_SHOPIFY": "false",
                "COMPANY_NAME": "Example",
                "COMPANY_WEBSITE": "https://example.com",
                "PERSONA": "Jordan",
                "MESSAGE_TONE": "Polished",
                "CONVERSATION_JSON": json.dumps(
                    [
                        {"message_type": "customer", "message_text": "Can you help?"},
                        {
                            "message_type": "agent",
                            "message_id": "agent-final",
                            "message_text": "Yes, I can help.",
                        },
                    ]
                ),
                "LAST_5_PRODUCTS": "",
                "ORDERS": "",
                "COUPONS": "",
                "COMPANY_NOTES": "Company notes only.",
                "ESCALATION_TOPICS": "[]",
                "BLOCKLISTED_WORDS": "[]",
                "COMPANY_PROMOTIONS": "",
            }
        )


def _write_scores_csv(path: Path) -> None:
    columns = ["Agent Outbound Messages Send ID", *SCORE_COLUMN_MAP.keys()]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        row = {"Agent Outbound Messages Send ID": "send-1"}
        for column in SCORE_COLUMN_MAP:
            row[column] = "1"
        row["Concierge QA Audit Scores Total Intent Identified Score Count"] = "2"
        row["Concierge QA Audit Scores Total Conversation Score Count"] = "0"
        writer.writerow(row)


def _write_templates_csv(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["SEND_ID", "template_text"])
        writer.writeheader()
        writer.writerow({"SEND_ID": "send-1", "template_text": "Template text."})
