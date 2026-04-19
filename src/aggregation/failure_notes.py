from __future__ import annotations

from src.models.audit_models import SubcategoryScore


def trim_note(note: str | None, max_words: int) -> str | None:
    if not note:
        return None
    words = note.split()
    if len(words) <= max_words:
        return note
    return " ".join(words[:max_words]).rstrip(".,;:") + "."


def normalize_failure_note(score: SubcategoryScore, max_words: int = 28) -> SubcategoryScore:
    if score.score == 0:
        trimmed = trim_note(score.failure_note, max_words)
        return score.model_copy(update={"failure_note": trimmed})
    return score.model_copy(update={"failure_note": None})
