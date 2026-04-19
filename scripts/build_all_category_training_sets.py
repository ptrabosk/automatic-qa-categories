#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.training.jsonl_export import write_jsonl
from src.training.manual_score_joiner import (
    ID_COLUMN_ALIASES,
    SCORE_COLUMN_MAP,
    balance_rows_by_subcategory,
    build_chat_training_rows,
    build_joined_training_rows,
    category_subcategories,
    filter_rows_to_category,
    label_distribution,
    split_training_rows,
)
from src.utils.files import write_json


STARTER_CATEGORY_TARGETS = {
    "issue_identification": 1500,
    "proper_resolution": 3000,
    "workflow": 3000,
    "tone": 1500,
    "clarity": None,
}
STARTER_CATEGORY_MAX_PER_CLASS = {
    "issue_identification": 375,
    "proper_resolution": 470,
    "workflow": 300,
    "tone": 330,
    "clarity": None,
}


def main() -> None:
    args = parse_args()
    categories = args.categories or all_categories()
    input_counts = input_match_counts(args.scores, args.conversations)

    rows = build_joined_training_rows(
        args.conversations,
        args.scores,
        config_dir=args.config_dir,
        limit=args.limit,
        skip_unmatched=args.skip_unmatched,
        templates_file=args.templates,
    )

    output_root = Path(args.output_root)
    summary: dict[str, object] = {
        "scores": args.scores,
        "conversations": args.conversations,
        "input_counts": input_counts | {"joined_rows": len(rows)},
        "categories": {},
    }
    all_selected_train_send_ids: set[str] = set()
    for category in categories:
        category_summary = build_category_splits(args, rows, category, output_root)
        summary["categories"][category] = category_summary
        all_selected_train_send_ids.update(category_summary["selected_train_send_ids"])

    write_text(
        output_root / "all_selected_train_send_ids.sql",
        sql_values(sorted(all_selected_train_send_ids)),
    )
    write_json(output_root / "summary.json", summary)
    print(f"Wrote all category training sets to {output_root}")
    print(
        "Input rows: "
        f"scores={input_counts['score_rows']} "
        f"conversations={input_counts['conversation_rows']} "
        f"matching_send_ids={input_counts['matching_send_ids']} "
        f"joined_rows={len(rows)}"
    )
    for category, category_summary in summary["categories"].items():
        train = category_summary["splits"]["train"]
        print(
            f"{category}: train_rows={train['rows']} "
            f"any_failure={train['rows_with_any_failure']} "
            f"all_pass={train['rows_all_pass']}"
        )
        print(f"  selected train send IDs: {output_root / category / 'selected_train_send_ids.sql'}")
        if args.print_send_ids:
            print(sql_values(category_summary["selected_train_send_ids"]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build category-specialist chat JSONL files from a score CSV, selecting balanced "
            "0/1 send IDs for every subcategory in each category."
        )
    )
    parser.add_argument("--scores", default="examples/all_qa_scores.csv")
    parser.add_argument("--conversations", default="examples/qa_training_set.csv")
    parser.add_argument("--output-root", default="examples/category_training_splits")
    parser.add_argument("--categories", nargs="*", help="Defaults to all categories.")
    parser.add_argument("--templates", help="Optional templates CSV/JSON/JSONL keyed by SEND_ID")
    parser.add_argument("--config-dir", default="config")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--skip-unmatched", action="store_true")
    parser.add_argument("--include-metadata", action="store_true")
    parser.add_argument(
        "--pass-ratio",
        type=float,
        default=1.0,
        help="Selected pass examples per selected failure example for each subcategory.",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        help="Optional cap on selected 0 examples and selected 1 examples per subcategory.",
    )
    parser.add_argument(
        "--preset",
        choices=("none", "starter"),
        default="starter",
        help=(
            "starter targets issue_identification=1500, proper_resolution=3000, "
            "workflow=3000, tone=1500, clarity=all."
        ),
    )
    parser.add_argument(
        "--print-send-ids",
        action="store_true",
        help="Print selected training send IDs for each category as SQL-style tuples.",
    )
    return parser.parse_args()


def build_category_splits(
    args: argparse.Namespace,
    rows: list[dict],
    category: str,
    output_root: Path,
) -> dict[str, object]:
    subcategories = category_subcategories(category)
    max_per_class = category_max_per_class(category, args)
    category_rows = filter_rows_to_category(rows, category)
    splits = split_training_rows(
        category_rows,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
    )
    natural_train_stats = label_distribution(splits["train"])
    splits["train"], selection_manifest = balance_rows_by_subcategory(
        splits["train"],
        subcategories,
        seed=args.seed,
        pass_ratio=args.pass_ratio,
        max_per_class=max_per_class,
    )

    output_dir = output_root / category
    output_dir.mkdir(parents=True, exist_ok=True)
    all_split_send_ids: list[str] = []
    for name, split_rows in splits.items():
        all_split_send_ids.extend(row["send_id"] for row in split_rows)
        write_jsonl(
            output_dir / f"{name}.jsonl",
            build_chat_training_rows(split_rows, include_metadata=args.include_metadata),
        )

    selected_train_send_ids = selection_manifest["union_train_send_ids"]
    write_text(output_dir / "selected_train_send_ids.sql", sql_values(selected_train_send_ids))
    write_text(output_dir / "all_split_send_ids.sql", sql_values(sorted(set(all_split_send_ids))))
    stats = {
        "category": category,
        "subcategories": subcategories,
        "selection_strategy": "per_subcategory_balanced_train_union",
        "preset": args.preset,
        "target_train_rows": category_target_train_rows(category, args),
        "pass_ratio": args.pass_ratio,
        "max_per_class": max_per_class,
        "all_rows": label_distribution(category_rows),
        "natural_train": natural_train_stats,
        "splits": {name: label_distribution(split_rows) for name, split_rows in splits.items()},
    }
    write_json(output_dir / "split_counts.json", {name: len(items) for name, items in splits.items()})
    write_json(output_dir / "label_distribution.json", stats)
    write_json(output_dir / "selected_send_ids.json", selection_manifest)
    return stats | {"selected_train_send_ids": selected_train_send_ids}


def all_categories() -> list[str]:
    return sorted({label.split(".", 1)[0] for label in SCORE_COLUMN_MAP.values()})


def category_max_per_class(category: str, args: argparse.Namespace) -> int | None:
    if args.max_per_class is not None:
        return args.max_per_class

    target_rows = category_target_train_rows(category, args)
    if target_rows is None:
        return None
    if args.preset == "starter":
        return STARTER_CATEGORY_MAX_PER_CLASS.get(category)

    subcategory_count = len(category_subcategories(category))
    examples_per_subcategory_class = target_rows // (subcategory_count * 2)
    return max(1, examples_per_subcategory_class)


def category_target_train_rows(category: str, args: argparse.Namespace) -> int | None:
    if args.preset == "starter":
        return STARTER_CATEGORY_TARGETS.get(category)
    return None


def input_match_counts(scores_path: str | Path, conversations_path: str | Path) -> dict[str, int]:
    score_ids = read_id_set(scores_path, ID_COLUMN_ALIASES)
    conversation_ids = read_id_set(conversations_path, ("SEND_ID", "send_id", "Send ID", "record_id"))
    return {
        "score_rows": len(score_ids),
        "conversation_rows": len(conversation_ids),
        "matching_send_ids": len(score_ids & conversation_ids),
        "scores_without_conversation": len(score_ids - conversation_ids),
        "conversations_without_score": len(conversation_ids - score_ids),
    }


def read_id_set(path: str | Path, aliases: tuple[str, ...]) -> set[str]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        id_column = next((alias for alias in aliases if alias in fieldnames), None)
        if id_column is None:
            raise ValueError(f"{path} is missing an ID column. Expected one of: {aliases}")
        return {row[id_column].strip() for row in reader if row.get(id_column, "").strip()}


def sql_values(send_ids: list[str]) -> str:
    values: list[str] = []
    for send_id in send_ids:
        escaped = send_id.replace("'", "''")
        values.append(f"('{escaped}')")
    return ", ".join(values)


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
