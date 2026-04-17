from __future__ import annotations

from typing import Any

from src.aggregation.failure_notes import normalize_failure_note
from src.models.audit_models import ScoreProvenance, SubcategoryScore
from src.utils.enums import Method
from src.validators.deterministic_helpers import make_score, score_key


def merge_scores(
    packet,
    rubric: dict[str, Any],
    deterministic_scores: dict[str, SubcategoryScore],
    specialist_scores: dict[str, SubcategoryScore],
) -> dict[str, SubcategoryScore]:
    merged: dict[str, SubcategoryScore] = {}
    max_words = int(rubric.get("note_writing", {}).get("max_words", 28))
    for category, category_config in rubric["categories"].items():
        for subcategory, subconfig in category_config["subcategories"].items():
            key = score_key(category, subcategory)
            deterministic = deterministic_scores.get(key)
            specialist = specialist_scores.get(key)
            method = Method(subconfig.get("method", "hybrid"))
            if deterministic and deterministic.score == 0 and deterministic.hard_fail:
                chosen = deterministic
            elif method == Method.DETERMINISTIC and deterministic is not None:
                chosen = deterministic
            elif specialist is not None:
                chosen = specialist
            elif deterministic is not None:
                chosen = deterministic
            else:
                chosen = make_score(
                    packet,
                    category,
                    subcategory,
                    0,
                    method=method,
                    rationale="No deterministic or specialist score was produced for this required subcategory.",
                    failure_note="The pipeline did not produce a score for this subcategory from the provided JSON.",
                    confidence=0.25,
                    source="aggregator",
                )
            chosen = normalize_failure_note(chosen.model_copy(deep=True), max_words=max_words)
            chosen.provenance = _provenance(deterministic, specialist)
            if chosen.method != method and method != Method.DETERMINISTIC and chosen.source != "deterministic":
                chosen.method = method
            merged[key] = chosen
    return merged


def _provenance(
    deterministic: SubcategoryScore | None, specialist: SubcategoryScore | None
) -> list[ScoreProvenance]:
    provenance: list[ScoreProvenance] = []
    for score in (deterministic, specialist):
        if score is not None:
            provenance.append(
                ScoreProvenance(
                    source=score.source,
                    score=score.score,
                    hard_fail=score.hard_fail,
                    rationale=score.rationale,
                )
            )
    return provenance
