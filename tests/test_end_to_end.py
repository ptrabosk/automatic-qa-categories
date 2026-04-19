from src.audit_pipeline import AuditPipeline
from src.utils.files import read_json


def test_end_to_end_sample_runs_without_llm() -> None:
    result = AuditPipeline(use_llm=False).audit_file("examples/sample_input.json")[0]
    rubric_subcategory_count = sum(
        len(category["subcategories"])
        for category in AuditPipeline(use_llm=False).rubric["categories"].values()
    )

    assert result.record_id == "sample-001"
    assert result.audited_message_index == 3
    assert len(result.subcategory_scores) == rubric_subcategory_count
    assert "accuracy" not in result.category_rollups
    assert "product_sales" not in result.category_rollups
    assert not any(key.startswith("accuracy.") for key in result.subcategory_scores)
    assert not any(key.startswith("product_sales.") for key in result.subcategory_scores)
    assert result.metadata.llm_used is False

    sample = read_json("examples/sample_input.json")
    assert sample["messages"][-1]["id"] == result.audited_message_id
