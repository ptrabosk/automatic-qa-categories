#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from src.training.jsonl_export import write_jsonl
from src.training.manual_score_joiner import (
    balance_rows_by_any_failure,
    build_chat_training_rows,
    build_joined_training_rows,
    category_subcategories,
    filter_rows_to_category,
    label_distribution,
    split_training_rows,
)
from src.utils.files import write_json


def main() -> None:
    args = parse_args()

    rows = build_joined_training_rows(
        args.conversations,
        args.scores,
        config_dir=args.config_dir,
        limit=args.limit,
        skip_unmatched=args.skip_unmatched,
        templates_file=args.templates,
    )
    category_rows = filter_rows_to_category(rows, args.category)
    splits = split_training_rows(
        category_rows,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
    )

    natural_train_stats = label_distribution(splits["train"])
    if args.balance_train:
        splits["train"] = balance_rows_by_any_failure(
            splits["train"], seed=args.seed, pass_ratio=args.pass_ratio
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, split_rows in splits.items():
        write_jsonl(
            output_dir / f"{name}.jsonl",
            build_chat_training_rows(split_rows, include_metadata=args.include_metadata),
        )

    stats = {
        "category": args.category,
        "subcategories": category_subcategories(args.category),
        "balance_train": args.balance_train,
        "pass_ratio": args.pass_ratio,
        "all_rows": label_distribution(category_rows),
        "natural_train": natural_train_stats,
        "splits": {name: label_distribution(split_rows) for name, split_rows in splits.items()},
    }
    write_json(output_dir / "split_counts.json", {name: len(items) for name, items in splits.items()})
    write_json(output_dir / "label_distribution.json", stats)
    print_stats(stats, output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build chat JSONL splits for one category-specialist fine-tuning run."
    )
    parser.add_argument("--category", required=True, help="Category prefix, e.g. issue_identification")
    parser.add_argument("--conversations", default="examples/qa_training_set.csv")
    parser.add_argument("--scores", default="examples/all_qa_scores.csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--templates", help="Optional templates CSV/JSON/JSONL keyed by SEND_ID")
    parser.add_argument("--config-dir", default="config")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--skip-unmatched", action="store_true")
    parser.add_argument("--include-metadata", action="store_true")
    parser.add_argument(
        "--no-balance-train",
        dest="balance_train",
        action="store_false",
        help="Leave the train split in the natural imbalanced distribution.",
    )
    parser.add_argument(
        "--pass-ratio",
        type=float,
        default=1.0,
        help="Training all-pass rows per row with at least one category failure.",
    )
    parser.set_defaults(balance_train=True)
    return parser.parse_args()


def print_stats(stats: dict, output_dir: Path) -> None:
    print(f"Wrote category splits to {output_dir}")
    print(f"Category: {stats['category']}")
    print(f"Subcategories: {', '.join(stats['subcategories'])}")
    print("\nAll labeled rows:")
    print_distribution(stats["all_rows"])
    print("\nNatural train split before balancing:")
    print_distribution(stats["natural_train"])
    print("\nWritten splits:")
    for name, distribution in stats["splits"].items():
        print(f"[{name}]")
        print_distribution(distribution)


def print_distribution(distribution: dict) -> None:
    print(
        f"  rows={distribution['rows']} "
        f"any_failure={distribution['rows_with_any_failure']} "
        f"all_pass={distribution['rows_all_pass']}"
    )
    for subcategory, counts in distribution["subcategory_counts"].items():
        print(f"  {subcategory}: 0={counts['0']} 1={counts['1']}")


if __name__ == "__main__":
    main()
