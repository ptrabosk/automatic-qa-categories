from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from src.preprocessing.evidence_builder import build_evidence_packet
from src.preprocessing.normalize_input import normalize_record
from src.training.csv_converter import convert_csv_row
from src.training.split_data import split_records
from src.training.templates_loader import load_templates_by_send_id
from src.utils.files import load_yaml


ID_COLUMN = "Agent Outbound Messages Send ID"

SCORE_COLUMN_MAP = {
    "Concierge QA Audit Scores Total Intent Identified Score Count": "issue_identification.intent_identified",
    "Concierge QA Audit Scores Total Necessary Reply Score Count": "issue_identification.necessary_reply",
    "Concierge QA Audit Scores Total Correct Escalation Score Count": "proper_resolution.correct_escalation",
    "Concierge QA Audit Scores Total Double Text Score Count": "proper_resolution.double_text",
    "Concierge QA Audit Scores Total Efficient Troubleshooting Score Count": "proper_resolution.efficient_troubleshooting",
    "Concierge QA Audit Scores Total Partial Reply Score Count": "proper_resolution.partial_reply",
    "Concierge QA Audit Scores Total Company Profile Score Count": "workflow.company_profile",
    "Concierge QA Audit Scores Total Conversation Score Count": "workflow.conversation",
    "Concierge QA Audit Scores Total Customer Profile Score Count": "workflow.customer_profile",
    "Concierge QA Audit Scores Total Notes Score Count": "workflow.notes",
    "Concierge QA Audit Scores Total Promo Notes Score Count": "workflow.promo_notes",
    "Concierge QA Audit Scores Total Templates Score Count": "workflow.templates",
    "Concierge QA Audit Scores Total Correct Grammar Score Count": "clarity.correct_grammar",
    "Concierge QA Audit Scores Total No Repetition Score Count": "clarity.no_repetition",
    "Concierge QA Audit Scores Total No Typos Score Count": "clarity.no_typos",
    "Concierge QA Audit Scores Total Understandable Message Score Count": "clarity.understandable",
    "Concierge QA Audit Scores Total Empathetic Score Count": "tone.empathetic",
    "Concierge QA Audit Scores Total Personalized Score Count": "tone.personalized",
    "Concierge QA Audit Scores Total Preferred Tone Followed Score Count": "tone.preferred_tone_followed",
}

TRAINING_SYSTEM_MESSAGE = (
    "You are a customer-support QA scoring model. Use only the supplied JSON. "
    "Audit only audited_message, which is the final agent message. Earlier messages are context only. "
    "Return strict JSON with binary scores for the requested subcategories."
)


def build_joined_training_rows(
    conversations_csv: str | Path,
    scores_csv: str | Path,
    *,
    config_dir: str | Path = "config",
    limit: int | None = None,
    skip_unmatched: bool = False,
    templates_file: str | Path | None = None,
) -> list[dict[str, Any]]:
    score_rows = _read_score_rows(scores_csv)
    templates_by_send_id = load_templates_by_send_id(templates_file)
    active_subcategories = _active_subcategories(config_dir)
    rows: list[dict[str, Any]] = []

    with Path(conversations_csv).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row_number, conversation_row in enumerate(reader, start=2):
            if limit is not None and len(rows) >= limit:
                break
            send_id = conversation_row.get("SEND_ID", "").strip()
            score_row = score_rows.get(send_id)
            if score_row is None:
                if skip_unmatched:
                    continue
                raise ValueError(f"No manual score row found for SEND_ID {send_id!r}")
            converted = convert_csv_row(
                conversation_row,
                row_number=row_number,
                templates_by_send_id=templates_by_send_id,
            )
            record = normalize_record(converted)
            packet = build_evidence_packet(record)
            labels, raw_scores = _labels_from_score_row(score_row)
            labeled_keys = set(labels)
            rows.append(
                {
                    "send_id": record.record_id,
                    "audited_message_index": packet.audited_message.index,
                    "audited_message_id": packet.audited_message.message_id,
                    "latest_customer_message_index": (
                        packet.latest_customer_message.index
                        if packet.latest_customer_message is not None
                        else None
                    ),
                    "input": {
                        "send_id": record.record_id,
                        "messages": [message.model_dump() for message in record.messages],
                        "audited_message": packet.audited_message.model_dump(),
                        "latest_customer_message": (
                            packet.latest_customer_message.model_dump()
                            if packet.latest_customer_message is not None
                            else None
                        ),
                        "structured_context": record.structured_context,
                        "available_evidence": packet.available_evidence,
                        "missing_evidence": packet.missing_evidence,
                        "requested_subcategories": sorted(labels),
                    },
                    "labels": {
                        "scores": labels,
                    },
                    "label_metadata": {
                        "source_score_file": str(scores_csv),
                        "score_mode": "positive_scores_are_pass",
                        "raw_scores": raw_scores,
                        "unlabeled_active_subcategories": sorted(
                            active_subcategories - labeled_keys
                        ),
                    },
                }
            )
    return rows


def build_chat_training_rows(
    rows: list[dict[str, Any]], *, include_metadata: bool = False
) -> list[dict[str, Any]]:
    chat_rows: list[dict[str, Any]] = []
    for row in rows:
        user_payload = {
            "task": "Score each requested subcategory as 0 or 1.",
            "send_id": row["send_id"],
            "audited_message_index": row["audited_message_index"],
            "audited_message_id": row["audited_message_id"],
            "input": row["input"],
        }
        assistant_payload = {
            "send_id": row["send_id"],
            "audited_message_index": row["audited_message_index"],
            "scores": row["labels"]["scores"],
        }
        chat_row: dict[str, Any] = {
            "messages": [
                {"role": "system", "content": TRAINING_SYSTEM_MESSAGE},
                {
                    "role": "user",
                    "content": json.dumps(user_payload, ensure_ascii=False, sort_keys=True),
                },
                {
                    "role": "assistant",
                    "content": json.dumps(
                        assistant_payload, ensure_ascii=False, sort_keys=True
                    ),
                },
            ]
        }
        if include_metadata:
            chat_row["metadata"] = {
                "send_id": row["send_id"],
                "audited_message_index": row["audited_message_index"],
                "labeled_subcategory_count": len(row["labels"]["scores"]),
                "unlabeled_active_subcategories": row["label_metadata"][
                    "unlabeled_active_subcategories"
                ],
            }
        chat_rows.append(chat_row)
    return chat_rows


def split_training_rows(
    rows: list[dict[str, Any]],
    *,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    seed: int = 13,
) -> dict[str, list[dict[str, Any]]]:
    return split_records(
        rows,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        seed=seed,
    )


def _read_score_rows(path: str | Path) -> dict[str, dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if ID_COLUMN not in (reader.fieldnames or []):
            raise ValueError(f"Manual score CSV is missing {ID_COLUMN!r}")
        missing_columns = [
            column for column in SCORE_COLUMN_MAP if column not in (reader.fieldnames or [])
        ]
        if missing_columns:
            raise ValueError(
                "Manual score CSV is missing required score columns: "
                + ", ".join(missing_columns)
            )
        rows: dict[str, dict[str, str]] = {}
        for row in reader:
            send_id = row[ID_COLUMN].strip()
            if send_id in rows:
                raise ValueError(f"Duplicate manual score row for send ID {send_id!r}")
            rows[send_id] = row
        return rows


def _labels_from_score_row(
    score_row: dict[str, str],
) -> tuple[dict[str, int], dict[str, int]]:
    labels: dict[str, int] = {}
    raw_scores: dict[str, int] = {}
    for column, label_key in SCORE_COLUMN_MAP.items():
        raw_score = _parse_score(score_row.get(column), column)
        raw_scores[label_key] = raw_score
        labels[label_key] = 1 if raw_score > 0 else 0
    return labels, raw_scores


def _parse_score(value: str | None, column: str) -> int:
    if value is None or str(value).strip() == "":
        raise ValueError(f"Missing score value for {column}")
    try:
        return int(float(str(value).strip()))
    except ValueError as exc:
        raise ValueError(f"Invalid score value {value!r} for {column}") from exc


def _active_subcategories(config_dir: str | Path) -> set[str]:
    rubric = load_yaml(Path(config_dir) / "rubric.yaml")
    active: set[str] = set()
    for category, category_config in rubric["categories"].items():
        for subcategory in category_config["subcategories"]:
            active.add(f"{category}.{subcategory}")
    return active
