#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.training.manual_score_joiner import (
    SCORE_COLUMN_MAP,
    _labels_from_score_row,
    _read_score_rows,
    balance_rows_by_subcategory,
    category_subcategories,
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
    rows = score_rows(args.scores)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    all_selected_train_ids: set[str] = set()
    summary: dict[str, Any] = {"scores": args.scores, "categories": {}}

    for category in categories:
        max_per_class = category_max_per_class(category, args)
        category_rows = filter_score_rows_to_category(rows, category)
        splits = split_training_rows(
            category_rows,
            train_ratio=args.train_ratio,
            validation_ratio=args.validation_ratio,
            seed=args.seed,
        )
        natural_train = label_distribution(splits["train"])
        splits["train"], selection_manifest = balance_rows_by_subcategory(
            splits["train"],
            category_subcategories(category),
            seed=args.seed,
            pass_ratio=args.pass_ratio,
            max_per_class=max_per_class,
        )

        category_dir = output_root / category
        category_dir.mkdir(parents=True, exist_ok=True)
        for split_name, split_rows in splits.items():
            write_send_id_files(category_dir, split_name, [row["send_id"] for row in split_rows])

        selected_ids = selection_manifest["union_train_send_ids"]
        all_selected_train_ids.update(selected_ids)
        write_text(category_dir / "selected_train_send_ids.sql", sql_values(selected_ids))
        write_json(category_dir / "selected_send_ids.json", selection_manifest)

        stats = {
            "category": category,
            "subcategories": category_subcategories(category),
            "selection_strategy": "score_only_per_subcategory_balanced_train_union",
            "preset": args.preset,
            "target_train_rows": category_target_train_rows(category, args),
            "pass_ratio": args.pass_ratio,
            "max_per_class": max_per_class,
            "all_rows": label_distribution(category_rows),
            "natural_train": natural_train,
            "splits": {name: label_distribution(split_rows) for name, split_rows in splits.items()},
        }
        write_json(category_dir / "label_distribution.json", stats)
        summary["categories"][category] = stats

        print(
            f"{category}: train_rows={stats['splits']['train']['rows']} "
            f"any_failure={stats['splits']['train']['rows_with_any_failure']} "
            f"all_pass={stats['splits']['train']['rows_all_pass']}"
        )
        print(f"  selected train send IDs: {category_dir / 'selected_train_send_ids.sql'}")
        if args.print_send_ids:
            print(sql_values(selected_ids))

    write_text(output_root / "all_selected_train_send_ids.sql", sql_values(sorted(all_selected_train_ids)))
    write_json(output_root / "summary.json", summary)
    print(f"Wrote send ID selections to {output_root}")
    print(f"All selected train send IDs: {output_root / 'all_selected_train_send_ids.sql'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select balanced category-specialist send IDs from a score CSV only."
    )
    parser.add_argument("--scores", default="examples/all_qa_scores.csv")
    parser.add_argument("--output-root", default="examples/category_send_id_selections")
    parser.add_argument("--categories", nargs="*", help="Defaults to all categories.")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--pass-ratio", type=float, default=1.0)
    parser.add_argument("--max-per-class", type=int)
    parser.add_argument(
        "--preset",
        choices=("none", "starter"),
        default="starter",
        help=(
            "starter targets issue_identification=1500, proper_resolution=3000, "
            "workflow=3000, tone=1500, clarity=all."
        ),
    )
    parser.add_argument("--print-send-ids", action="store_true")
    return parser.parse_args()


def score_rows(scores_path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for send_id, score_row in _read_score_rows(scores_path).items():
        labels, raw_scores = _labels_from_score_row(score_row)
        rows.append(
            {
                "send_id": send_id,
                "labels": {"scores": labels},
                "label_metadata": {"raw_scores": raw_scores},
            }
        )
    return rows


def filter_score_rows_to_category(rows: list[dict[str, Any]], category: str) -> list[dict[str, Any]]:
    selected = set(category_subcategories(category))
    filtered: list[dict[str, Any]] = []
    for row in rows:
        scores = {
            key: value for key, value in row["labels"]["scores"].items() if key in selected
        }
        raw_scores = {
            key: value
            for key, value in row["label_metadata"]["raw_scores"].items()
            if key in selected
        }
        filtered.append(
            {
                "send_id": row["send_id"],
                "labels": {"scores": scores},
                "label_metadata": {"raw_scores": raw_scores},
            }
        )
    return filtered


def write_send_id_files(output_dir: Path, split_name: str, send_ids: list[str]) -> None:
    write_text(output_dir / f"{split_name}_send_ids.sql", sql_values(send_ids))
    write_text(output_dir / f"{split_name}_send_ids.txt", "\n".join(send_ids))


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


def sql_values(send_ids: list[str]) -> str:
    values: list[str] = []
    for send_id in send_ids:
        escaped = send_id.replace("'", "''")
        values.append(f"('{escaped}')")
    return ", ".join(values)


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value + "\n", encoding="utf-8")


def all_categories() -> list[str]:
    return sorted({label.split(".", 1)[0] for label in SCORE_COLUMN_MAP.values()})


if __name__ == "__main__":
    main()
