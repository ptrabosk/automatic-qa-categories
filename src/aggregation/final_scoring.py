from __future__ import annotations

from typing import Any

from src.models.audit_models import CategoryRollup, SubcategoryScore
from src.utils.enums import PassFail
from src.validators.deterministic_helpers import score_key


def build_category_rollups(
    rubric: dict[str, Any], scores: dict[str, SubcategoryScore]
) -> tuple[dict[str, CategoryRollup], int, PassFail]:
    rollups: dict[str, CategoryRollup] = {}
    for category, category_config in rubric["categories"].items():
        failed: list[str] = []
        required_failed: list[str] = []
        for subcategory, subconfig in category_config["subcategories"].items():
            score = scores[score_key(category, subcategory)]
            if score.score == 0:
                failed.append(subcategory)
                if subconfig.get("required_for_overall", True):
                    required_failed.append(subcategory)
        category_score = 0 if required_failed else 1
        rollups[category] = CategoryRollup(
            category=category,
            score=category_score,
            pass_fail=PassFail.PASS if category_score else PassFail.FAIL,
            failed_subcategories=failed,
            required_failed_subcategories=required_failed,
        )
    overall = 0 if any(rollup.score == 0 for rollup in rollups.values()) else 1
    return rollups, overall, PassFail.PASS if overall else PassFail.FAIL
