from __future__ import annotations

import csv
import json
import random
from copy import deepcopy
from pathlib import Path
from typing import Any

from src.preprocessing.evidence_builder import build_evidence_packet
from src.preprocessing.normalize_input import normalize_record
from src.training.csv_converter import convert_csv_row
from src.training.split_data import split_records
from src.training.templates_loader import load_templates_by_send_id
from src.utils.files import load_yaml


ID_COLUMN = "Agent Outbound Messages Send ID"
ID_COLUMN_ALIASES = (ID_COLUMN, "Send ID", "SEND_ID", "send_id", "record_id")

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
SCORE_COLUMN_ALIASES = {
    column: (column, column.removeprefix("Concierge QA Audit Scores "))
    for column in SCORE_COLUMN_MAP
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


def filter_rows_to_category(rows: list[dict[str, Any]], category: str) -> list[dict[str, Any]]:
    return filter_rows_to_subcategories(rows, category_subcategories(category), category=category)


def filter_rows_to_subcategories(
    rows: list[dict[str, Any]], subcategories: list[str], *, category: str | None = None
) -> list[dict[str, Any]]:
    selected = set(subcategories)
    filtered_rows: list[dict[str, Any]] = []
    for row in rows:
        labels = {
            key: value
            for key, value in row["labels"]["scores"].items()
            if key in selected
        }
        if not labels:
            raise ValueError(f"No labels found for selected subcategories: {sorted(selected)}")

        filtered = deepcopy(row)
        filtered["labels"]["scores"] = labels
        filtered["input"]["requested_subcategories"] = sorted(labels)
        filtered["label_metadata"]["selected_subcategories"] = sorted(labels)
        if category is not None:
            filtered["label_metadata"]["selected_category"] = category
        raw_scores = filtered["label_metadata"].get("raw_scores")
        if isinstance(raw_scores, dict):
            filtered["label_metadata"]["raw_scores"] = {
                key: value for key, value in raw_scores.items() if key in selected
            }
        filtered_rows.append(filtered)
    return filtered_rows


def category_subcategories(category: str) -> list[str]:
    prefix = f"{category}."
    subcategories = sorted(label for label in SCORE_COLUMN_MAP.values() if label.startswith(prefix))
    if not subcategories:
        raise ValueError(f"Unknown category {category!r}")
    return subcategories


def oversample_failure_rows(
    rows: list[dict[str, Any]], *, failure_oversample_factor: int = 1
) -> list[dict[str, Any]]:
    if failure_oversample_factor < 1:
        raise ValueError("failure_oversample_factor must be at least 1")
    if failure_oversample_factor == 1:
        return list(rows)

    balanced_rows: list[dict[str, Any]] = []
    for row in rows:
        balanced_rows.append(row)
        if row_has_failure_label(row):
            balanced_rows.extend([row] * (failure_oversample_factor - 1))
    return balanced_rows


def row_has_failure_label(row: dict[str, Any]) -> bool:
    return any(score == 0 for score in row["labels"]["scores"].values())


def balance_rows_by_any_failure(
    rows: list[dict[str, Any]], *, seed: int = 13, pass_ratio: float = 1.0
) -> list[dict[str, Any]]:
    if pass_ratio < 0:
        raise ValueError("pass_ratio must be non-negative")

    failures = [row for row in rows if row_has_failure_label(row)]
    passes = [row for row in rows if not row_has_failure_label(row)]
    if not failures or not passes:
        return list(rows)

    rng = random.Random(seed)
    target_pass_count = int(round(len(failures) * pass_ratio))
    if target_pass_count <= len(passes):
        selected_passes = rng.sample(passes, target_pass_count)
    else:
        selected_passes = [rng.choice(passes) for _ in range(target_pass_count)]

    balanced = [*failures, *selected_passes]
    rng.shuffle(balanced)
    return balanced


def balance_rows_by_subcategory(
    rows: list[dict[str, Any]],
    subcategories: list[str],
    *,
    seed: int = 13,
    pass_ratio: float = 1.0,
    max_per_class: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if pass_ratio < 0:
        raise ValueError("pass_ratio must be non-negative")
    if max_per_class is not None and max_per_class < 1:
        raise ValueError("max_per_class must be at least 1")

    rng = random.Random(seed)
    selected_by_id: dict[str, dict[str, Any]] = {}
    manifest: dict[str, Any] = {}

    for subcategory in subcategories:
        failures = [row for row in rows if row["labels"]["scores"][subcategory] == 0]
        passes = [row for row in rows if row["labels"]["scores"][subcategory] == 1]
        class_size = len(failures)
        if max_per_class is not None:
            class_size = min(class_size, max_per_class)
        pass_size = int(round(class_size * pass_ratio))

        selected_failures = rng.sample(failures, class_size) if class_size < len(failures) else list(failures)
        if pass_size <= len(passes):
            selected_passes = rng.sample(passes, pass_size)
        else:
            selected_passes = [rng.choice(passes) for _ in range(pass_size)] if passes else []

        for row in [*selected_failures, *selected_passes]:
            selected_by_id.setdefault(row["send_id"], row)

        manifest[subcategory] = {
            "available": {"0": len(failures), "1": len(passes)},
            "selected": {"0": len(selected_failures), "1": len(selected_passes)},
            "send_ids": {
                "0": [row["send_id"] for row in selected_failures],
                "1": [row["send_id"] for row in selected_passes],
            },
        }

    balanced = list(selected_by_id.values())
    rng.shuffle(balanced)
    return balanced, {
        "strategy": "per_subcategory_equal_zeros_and_sampled_ones",
        "pass_ratio": pass_ratio,
        "max_per_class": max_per_class,
        "subcategories": manifest,
        "union_train_send_ids": [row["send_id"] for row in balanced],
    }


def label_distribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    subcategory_counts: dict[str, dict[str, int]] = {}
    for row in rows:
        for key, value in row["labels"]["scores"].items():
            counts = subcategory_counts.setdefault(key, {"0": 0, "1": 0})
            counts[str(int(value))] += 1
    return {
        "rows": len(rows),
        "rows_with_any_failure": sum(1 for row in rows if row_has_failure_label(row)),
        "rows_all_pass": sum(1 for row in rows if not row_has_failure_label(row)),
        "subcategory_counts": dict(sorted(subcategory_counts.items())),
    }


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
        fieldnames = reader.fieldnames or []
        id_column = _resolve_id_column(fieldnames)
        score_columns = _resolve_score_columns(fieldnames)
        rows: dict[str, dict[str, str]] = {}
        for row in reader:
            send_id = row[id_column].strip()
            if send_id in rows:
                raise ValueError(f"Duplicate manual score row for send ID {send_id!r}")
            rows[send_id] = {
                canonical_column: row[actual_column]
                for canonical_column, actual_column in score_columns.items()
            }
        return rows


def _resolve_id_column(fieldnames: list[str]) -> str:
    for candidate in ID_COLUMN_ALIASES:
        if candidate in fieldnames:
            return candidate
    raise ValueError(
        "Manual score CSV is missing a send ID column. Expected one of: "
        + ", ".join(repr(column) for column in ID_COLUMN_ALIASES)
    )


def _resolve_score_columns(fieldnames: list[str]) -> dict[str, str]:
    resolved: dict[str, str] = {}
    missing: list[str] = []
    for canonical_column, aliases in SCORE_COLUMN_ALIASES.items():
        actual_column = next((alias for alias in aliases if alias in fieldnames), None)
        if actual_column is None:
            missing.append(aliases[-1])
        else:
            resolved[canonical_column] = actual_column
    if missing:
        raise ValueError(
            "Manual score CSV is missing required score columns: "
            + ", ".join(missing)
        )
    return resolved


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
