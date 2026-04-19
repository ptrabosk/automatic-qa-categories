from __future__ import annotations

import ast
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


CSV_COLUMNS = [
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


def convert_csv_file(
    input_path: str | Path,
    *,
    limit: int | None = None,
    skip_errors: bool = False,
    templates_by_send_id: dict[str, Any] | None = None,
) -> dict[str, Any]:
    input_path = Path(input_path)
    records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing_columns = [column for column in CSV_COLUMNS if column not in (reader.fieldnames or [])]
        if missing_columns:
            raise ValueError(f"CSV is missing required columns: {', '.join(missing_columns)}")

        for row_number, row in enumerate(reader, start=2):
            if limit is not None and len(records) >= limit:
                break
            try:
                records.append(
                    convert_csv_row(
                        row,
                        row_number=row_number,
                        templates_by_send_id=templates_by_send_id,
                    )
                )
            except Exception as exc:
                if not skip_errors:
                    raise ValueError(f"Failed to convert CSV row {row_number}: {exc}") from exc
                errors.append(
                    {
                        "row_number": row_number,
                        "send_id": row.get("SEND_ID"),
                        "error": str(exc),
                    }
                )

    return {
        "source_file": str(input_path),
        "converted_at": datetime.now(timezone.utc).isoformat(),
        "record_count": len(records),
        "errors": errors,
        "records": records,
    }


def convert_csv_row(
    row: dict[str, str],
    *,
    row_number: int | None = None,
    templates_by_send_id: dict[str, Any] | None = None,
) -> dict[str, Any]:
    conversation = _parse_required_json_list(row.get("CONVERSATION_JSON"), "CONVERSATION_JSON")
    messages = [_convert_message(message, index) for index, message in enumerate(conversation)]
    if not messages:
        raise ValueError("CONVERSATION_JSON contains no messages")
    if messages[-1]["role"] != "agent":
        raise ValueError("Final message is not an agent message")

    company_website = _empty_to_none(row.get("COMPANY_WEBSITE"))
    send_id = _empty_to_none(row.get("SEND_ID")) or f"csv-row-{row_number or 'unknown'}"
    coupons = _parse_cell(row.get("COUPONS"))
    company_promotions = _parse_cell(row.get("COMPANY_PROMOTIONS"))
    promo_notes = _combine_lists(coupons, company_promotions)
    templates = (templates_by_send_id or {}).get(send_id)

    record: dict[str, Any] = {
        "record_id": send_id,
        "messages": messages,
        "company_profile": _drop_empty(
            {
                "name": _empty_to_none(row.get("COMPANY_NAME")),
                "website": company_website,
                "has_shopify": _parse_bool(row.get("HAS_SHOPIFY")),
                "agent_persona": _empty_to_none(row.get("PERSONA")),
                "preferred_tone": _empty_to_none(row.get("MESSAGE_TONE")),
            }
        ),
        "notes": _empty_to_none(row.get("COMPANY_NOTES")),
        "templates": templates,
        "product_information": _parse_cell(row.get("LAST_5_PRODUCTS")),
        "promo_notes": promo_notes,
        "purchases": _parse_cell(row.get("ORDERS")),
        "workflow_flags": _drop_empty(
            {
                "has_shopify": _parse_bool(row.get("HAS_SHOPIFY")),
                "escalation_topics": _parse_cell(row.get("ESCALATION_TOPICS")),
            }
        ),
        "website_findings": _drop_empty(
            {
                "allowed_domains": [_domain(company_website)] if _domain(company_website) else None,
                "company_website": company_website,
            }
        ),
        "inappropriate_language_terms": _parse_cell(row.get("BLOCKLISTED_WORDS")),
        "csv_metadata": _drop_empty(
            {
                "source_row_number": row_number,
                "send_id": _empty_to_none(row.get("SEND_ID")),
            }
        ),
    }
    return _drop_empty(record)


def _convert_message(message: Any, index: int) -> dict[str, Any]:
    if not isinstance(message, dict):
        raise ValueError(f"message at index {index} is not an object")
    role = str(message.get("message_type") or message.get("role") or "other").strip().lower()
    content = message.get("message_text")
    if content is None:
        content = message.get("content", message.get("text", ""))
    converted = {
        "id": message.get("message_id") or message.get("id"),
        "role": role,
        "content": "" if content is None else str(content),
        "timestamp": message.get("date_time") or message.get("timestamp"),
        "message_media": message.get("message_media"),
        "raw_message": message,
    }
    return _drop_empty(converted)


def _parse_required_json_list(value: str | None, field_name: str) -> list[Any]:
    parsed = _parse_cell(value)
    if not isinstance(parsed, list):
        raise ValueError(f"{field_name} must parse to a JSON list")
    return parsed


def _parse_cell(value: str | None) -> Any:
    text = _empty_to_none(value)
    if text is None:
        return None
    if text[0] not in "[{\"'0123456789tfnTFN-":
        return text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    try:
        return ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return text


def _parse_bool(value: str | None) -> bool | None:
    text = _empty_to_none(value)
    if text is None:
        return None
    folded = text.casefold()
    if folded in {"true", "1", "yes", "y"}:
        return True
    if folded in {"false", "0", "no", "n"}:
        return False
    return None


def _combine_lists(*values: Any) -> list[Any] | None:
    combined: list[Any] = []
    for value in values:
        if value in (None, "", [], {}):
            continue
        if isinstance(value, list):
            combined.extend(value)
        else:
            combined.append(value)
    return combined or None


def _drop_empty(data: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in data.items() if value not in (None, "", [], {})}


def _empty_to_none(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _domain(url: str | None) -> str | None:
    if not url:
        return None
    candidate = url if url.startswith(("http://", "https://")) else f"https://{url}"
    domain = urlparse(candidate).netloc.casefold()
    return domain[4:] if domain.startswith("www.") else domain or None
