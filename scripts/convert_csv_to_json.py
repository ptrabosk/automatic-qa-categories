#!/usr/bin/env python
from __future__ import annotations

import argparse

from src.training.csv_converter import convert_csv_file
from src.training.templates_loader import load_templates_by_send_id
from src.utils.files import write_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert qa_training_set.csv rows into pipeline-ready conversation JSON."
    )
    parser.add_argument("--input", required=True, help="Path to qa_training_set.csv")
    parser.add_argument("--output", required=True, help="Path for converted JSON output")
    parser.add_argument("--limit", type=int, help="Convert only the first N records")
    parser.add_argument("--skip-errors", action="store_true", help="Collect bad rows instead of failing fast")
    parser.add_argument("--templates", help="Optional separate templates CSV/JSON/JSONL keyed by SEND_ID")
    args = parser.parse_args()

    payload = convert_csv_file(
        args.input,
        limit=args.limit,
        skip_errors=args.skip_errors,
        templates_by_send_id=load_templates_by_send_id(args.templates),
    )
    write_json(args.output, payload)
    print(f"Converted {payload['record_count']} records to {args.output}")
    if payload["errors"]:
        print(f"Skipped {len(payload['errors'])} rows with errors")


if __name__ == "__main__":
    main()
