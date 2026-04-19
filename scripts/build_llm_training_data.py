#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from src.training.jsonl_export import write_jsonl
from src.training.manual_score_joiner import (
    build_chat_training_rows,
    build_joined_training_rows,
    oversample_failure_rows,
    row_has_failure_label,
    split_training_rows,
)
from src.utils.files import write_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Join conversation CSV and manual QA score CSV into LLM training JSONL."
    )
    parser.add_argument("--conversations", required=True, help="Path to qa_training_set.csv")
    parser.add_argument("--scores", required=True, help="Path to score CSV")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--templates", help="Optional separate templates CSV/JSON/JSONL keyed by SEND_ID")
    parser.add_argument("--config-dir", default="config")
    parser.add_argument("--limit", type=int, help="Only export the first N matched records")
    parser.add_argument("--raw-json-output", help="Optional joined non-chat JSON output")
    parser.add_argument(
        "--include-metadata",
        action="store_true",
        help="Include top-level metadata in each JSONL row. Leave off for strict chat fine-tuning JSONL.",
    )
    parser.add_argument("--split-dir", help="Optional directory for train/validation/test JSONL")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--skip-unmatched", action="store_true")
    parser.add_argument(
        "--failure-oversample-factor",
        type=int,
        default=1,
        help=(
            "Repeat rows with at least one 0 score this many times. "
            "When --split-dir is used, this is applied only to the train split."
        ),
    )
    args = parser.parse_args()

    rows = build_joined_training_rows(
        args.conversations,
        args.scores,
        config_dir=args.config_dir,
        limit=args.limit,
        skip_unmatched=args.skip_unmatched,
        templates_file=args.templates,
    )
    output_rows = oversample_failure_rows(
        rows, failure_oversample_factor=args.failure_oversample_factor
    )
    chat_rows = build_chat_training_rows(output_rows, include_metadata=args.include_metadata)
    write_jsonl(args.output, chat_rows)

    if args.raw_json_output:
        write_json(args.raw_json_output, rows)

    if args.split_dir:
        split_dir = Path(args.split_dir)
        splits = split_training_rows(
            rows,
            train_ratio=args.train_ratio,
            validation_ratio=args.validation_ratio,
            seed=args.seed,
        )
        splits["train"] = oversample_failure_rows(
            splits["train"], failure_oversample_factor=args.failure_oversample_factor
        )
        for name, split_rows in splits.items():
            write_jsonl(
                split_dir / f"{name}.jsonl",
                build_chat_training_rows(split_rows, include_metadata=args.include_metadata),
            )
        write_json(split_dir / "split_counts.json", {name: len(items) for name, items in splits.items()})
        write_json(
            split_dir / "split_failure_counts.json",
            {name: sum(1 for item in items if row_has_failure_label(item)) for name, items in splits.items()},
        )

    print(f"Wrote {len(chat_rows)} chat training rows to {args.output}")
    if rows:
        first = rows[0]
        print(
            "First row:",
            first["send_id"],
            f"{len(first['labels']['scores'])} labeled subcategories",
        )


if __name__ == "__main__":
    main()
