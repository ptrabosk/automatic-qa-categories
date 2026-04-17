from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ID_CANDIDATES = (
    "SEND_ID",
    "send_id",
    "Agent Outbound Messages Send ID",
    "agent_outbound_messages_send_id",
    "record_id",
)
TEMPLATE_CANDIDATES = (
    "templates",
    "TEMPLATES",
    "template",
    "TEMPLATE",
    "template_text",
    "TEMPLATE_TEXT",
    "content",
    "CONTENT",
    "body",
    "BODY",
)


def load_templates_by_send_id(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    path = Path(path)
    if path.suffix.casefold() == ".json":
        return _load_json_templates(path)
    if path.suffix.casefold() == ".jsonl":
        return _load_jsonl_templates(path)
    return _load_csv_templates(path)


def _load_json_templates(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(data, dict):
        if "records" in data and isinstance(data["records"], list):
            return _rows_to_templates(data["records"])
        return {str(key): value for key, value in data.items()}
    if isinstance(data, list):
        return _rows_to_templates(data)
    raise ValueError("Template JSON must be an object mapping send IDs or a list of records")


def _load_jsonl_templates(path: Path) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    return _rows_to_templates(rows)


def _load_csv_templates(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return _rows_to_templates(list(csv.DictReader(handle)))


def _rows_to_templates(rows: list[Any]) -> dict[str, Any]:
    templates: dict[str, Any] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        send_id = _first_present(row, ID_CANDIDATES)
        if not send_id:
            raise ValueError("Template rows must include a send ID column")
        template_value = _first_present(row, TEMPLATE_CANDIDATES)
        if template_value is None:
            template_value = {
                key: value
                for key, value in row.items()
                if key not in ID_CANDIDATES and value not in (None, "", [], {})
            }
        templates[str(send_id)] = _parse_template_value(template_value)
    return templates


def _first_present(row: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = row.get(key)
        if value not in (None, "", [], {}):
            return value
    return None


def _parse_template_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return None
    if text[0] not in "[{\"":
        return text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text
