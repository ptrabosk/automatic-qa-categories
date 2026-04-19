import pytest


@pytest.fixture
def minimal_rubric() -> dict:
    return {
        "note_writing": {"max_words": 28},
        "categories": {
            "zero_tolerance": {
                "subcategories": {
                    "opt_out": {
                        "method": "hybrid",
                        "required_for_overall": True,
                    }
                }
            }
        },
    }
