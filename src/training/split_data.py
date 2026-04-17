from __future__ import annotations

import random
from typing import Any


def split_records(
    records: list[dict[str, Any]],
    *,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    seed: int = 13,
) -> dict[str, list[dict[str, Any]]]:
    if train_ratio <= 0 or validation_ratio < 0 or train_ratio + validation_ratio >= 1:
        raise ValueError("Ratios must leave a positive test split")
    shuffled = records[:]
    random.Random(seed).shuffle(shuffled)
    train_end = int(len(shuffled) * train_ratio)
    validation_end = train_end + int(len(shuffled) * validation_ratio)
    return {
        "train": shuffled[:train_end],
        "validation": shuffled[train_end:validation_end],
        "test": shuffled[validation_end:],
    }
