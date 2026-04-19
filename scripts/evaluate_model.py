#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import defaultdict
from typing import Any

from src.utils.files import read_json, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate QA audit predictions against labels.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    predictions = _as_list(read_json(args.predictions))
    labels = _as_list(read_json(args.labels))
    report = evaluate(predictions, labels)
    write_json(args.output, report)


def evaluate(predictions: list[dict[str, Any]], labels: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {_record_id(item): item for item in labels}
    metrics: dict[str, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "tn": 0, "fn": 0})
    category_matches = 0
    category_total = 0
    overall_matches = 0
    overall_total = 0

    for prediction in predictions:
        label = by_id.get(_record_id(prediction))
        if not label:
            continue
        pred_scores = _subcategory_scores(prediction)
        label_scores = _subcategory_scores(label)
        for key, actual in label_scores.items():
            if key not in pred_scores:
                continue
            predicted = pred_scores[key]
            _update_confusion(metrics[key], predicted, actual)
        pred_categories = _category_scores(prediction)
        label_categories = _category_scores(label)
        for key, actual in label_categories.items():
            if key in pred_categories:
                category_total += 1
                category_matches += int(pred_categories[key] == actual)
        if "final_overall_score" in prediction and "final_overall_score" in label:
            overall_total += 1
            overall_matches += int(prediction["final_overall_score"] == label["final_overall_score"])

    return {
        "subcategory_metrics": {
            key: _rates(values) | {"confusion_matrix": values}
            for key, values in sorted(metrics.items())
        },
        "category_accuracy": category_matches / category_total if category_total else None,
        "overall_accuracy": overall_matches / overall_total if overall_total else None,
        "records_evaluated": overall_total,
    }


def _update_confusion(bucket: dict[str, int], predicted: int, actual: int) -> None:
    if predicted == 1 and actual == 1:
        bucket["tp"] += 1
    elif predicted == 1 and actual == 0:
        bucket["fp"] += 1
    elif predicted == 0 and actual == 0:
        bucket["tn"] += 1
    elif predicted == 0 and actual == 1:
        bucket["fn"] += 1


def _rates(bucket: dict[str, int]) -> dict[str, float]:
    tp, fp, fn = bucket["tp"], bucket["fp"], bucket["fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def _as_list(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return value
    if isinstance(value, dict) and isinstance(value.get("records"), list):
        return value["records"]
    if isinstance(value, dict):
        return [value]
    raise ValueError("Expected object, records object, or list")


def _record_id(item: dict[str, Any]) -> str:
    return str(
        item.get("send_id")
        or item.get("record_id")
        or item.get("metadata", {}).get("send_id")
        or item.get("metadata", {}).get("record_id")
    )


def _subcategory_scores(item: dict[str, Any]) -> dict[str, int]:
    scores = item.get("scores") or item.get("subcategory_scores") or item.get("labels") or {}
    result: dict[str, int] = {}
    for key, value in scores.items():
        result[key] = int(value["score"] if isinstance(value, dict) else value)
    return result


def _category_scores(item: dict[str, Any]) -> dict[str, int]:
    rollups = item.get("category_rollups") or {}
    return {
        key: int(value["score"] if isinstance(value, dict) else value)
        for key, value in rollups.items()
    }


if __name__ == "__main__":
    main()
