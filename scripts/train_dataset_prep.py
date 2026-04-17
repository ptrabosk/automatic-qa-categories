#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from src.training.dataset_builder import build_chat_finetune_rows, build_training_rows
from src.training.jsonl_export import write_jsonl
from src.training.split_data import split_records
from src.utils.files import read_json, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare labeled audit data for training/eval JSONL.")
    parser.add_argument("--input", required=True, help="JSON list of labeled records.")
    parser.add_argument("--output", required=True, help="Output JSONL path for all rows.")
    parser.add_argument("--split-dir", help="Optional directory for train/validation/test JSONL files.")
    parser.add_argument("--chat-format", action="store_true", help="Export chat fine-tuning rows.")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    raw = read_json(args.input)
    if not isinstance(raw, list):
        raise ValueError("Labeled input must be a JSON list")
    rows = build_training_rows(raw)
    export_rows = build_chat_finetune_rows(rows) if args.chat_format else rows
    write_jsonl(args.output, export_rows)
    if args.split_dir:
        split_dir = Path(args.split_dir)
        splits = split_records(export_rows, seed=args.seed)
        for name, split_rows in splits.items():
            write_jsonl(split_dir / f"{name}.jsonl", split_rows)
        write_json(split_dir / "split_counts.json", {name: len(value) for name, value in splits.items()})


if __name__ == "__main__":
    main()
