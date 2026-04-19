.PHONY: install install-finetune-wsl test check-cuda single batch csv-json llm-training-jsonl llm-training-splits llm-training-sample select-category-send-ids all-category-splits category-splits fine-tune-category evaluate-category issue-identification-splits fine-tune-issue-identification evaluate-issue-identification fine-tune evaluate-lora sample clean

PYTHON ?= python3
CATEGORY ?= issue_identification
CATEGORY_SPLIT_DIR ?= examples/category_training_splits/$(CATEGORY)
CATEGORY_MODEL_DIR ?= models/qa-lora-$(CATEGORY)
SCORES ?= examples/all_qa_scores.csv
CONVERSATIONS ?= examples/qa_training_set.csv
CATEGORY_SPLIT_FLAGS ?=
SEND_ID_SELECTION_FLAGS ?=

install:
	$(PYTHON) -m pip install -e ".[dev]"

install-finetune-wsl:
	$(PYTHON) -m pip install -r requirements-finetune-wsl-cu121.txt

test:
	$(PYTHON) -m pytest

check-cuda:
	$(PYTHON) scripts/check_wsl_cuda.py

single:
	$(PYTHON) scripts/run_single_audit.py --input examples/sample_input.json --output examples/sample_output.generated.json --no-llm

batch:
	$(PYTHON) scripts/run_audit_batch.py --input-dir examples --output-dir examples/batch_outputs --no-llm

csv-json:
	$(PYTHON) scripts/convert_csv_to_json.py --input examples/qa_training_set.csv --output examples/qa_training_set.json

llm-training-jsonl:
	$(PYTHON) scripts/build_llm_training_data.py --conversations $(CONVERSATIONS) --scores $(SCORES) --output examples/qa_llm_training.jsonl

llm-training-splits:
	$(PYTHON) scripts/build_llm_training_data.py --conversations $(CONVERSATIONS) --scores $(SCORES) --output examples/qa_llm_training.jsonl --split-dir examples/llm_training_splits --failure-oversample-factor 4

llm-training-sample:
	$(PYTHON) scripts/build_llm_training_data.py --conversations $(CONVERSATIONS) --scores $(SCORES) --output examples/qa_llm_training.first10.jsonl --limit 10

select-category-send-ids:
	$(PYTHON) scripts/select_category_send_ids.py --scores $(SCORES) --output-root examples/category_send_id_selections $(SEND_ID_SELECTION_FLAGS)

all-category-splits:
	$(PYTHON) scripts/build_all_category_training_sets.py --scores $(SCORES) --conversations $(CONVERSATIONS) --output-root examples/category_training_splits --skip-unmatched $(CATEGORY_SPLIT_FLAGS)

category-splits:
	$(PYTHON) scripts/build_all_category_training_sets.py --scores $(SCORES) --conversations $(CONVERSATIONS) --output-root examples/category_training_splits --categories $(CATEGORY) --skip-unmatched $(CATEGORY_SPLIT_FLAGS)

fine-tune-category:
	PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $(PYTHON) scripts/fine_tune_lora.py --train-file $(CATEGORY_SPLIT_DIR)/train.jsonl --validation-file $(CATEGORY_SPLIT_DIR)/validation.jsonl --output-dir $(CATEGORY_MODEL_DIR) --max-seq-length 2048 --precision fp16

evaluate-category:
	PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $(PYTHON) scripts/evaluate_lora_adapter.py --input $(CATEGORY_SPLIT_DIR)/test.jsonl --adapter-dir $(CATEGORY_MODEL_DIR) --report-output reports/qa-lora-$(CATEGORY)-test-report.json --predictions-output reports/qa-lora-$(CATEGORY)-test-predictions.json --precision fp16

issue-identification-splits:
	$(MAKE) category-splits CATEGORY=issue_identification

fine-tune-issue-identification:
	$(MAKE) fine-tune-category CATEGORY=issue_identification

evaluate-issue-identification:
	$(MAKE) evaluate-category CATEGORY=issue_identification

fine-tune:
	PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $(PYTHON) scripts/fine_tune_lora.py --train-file examples/llm_training_splits/train.jsonl --validation-file examples/llm_training_splits/validation.jsonl --output-dir models/qa-lora --max-seq-length 2048 --precision fp16

evaluate-lora:
	PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $(PYTHON) scripts/evaluate_lora_adapter.py --input examples/llm_training_splits/test.jsonl --adapter-dir models/qa-lora --report-output reports/qa-lora-test-report.json --predictions-output reports/qa-lora-test-predictions.json --precision fp16

sample: single

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache *.egg-info examples/batch_outputs examples/sample_output.generated.json
