.PHONY: install test single batch csv-json llm-training-jsonl llm-training-sample sample clean

install:
	python -m pip install -e ".[dev]"

test:
	pytest

single:
	python scripts/run_single_audit.py --input examples/sample_input.json --output examples/sample_output.generated.json --no-llm

batch:
	python scripts/run_audit_batch.py --input-dir examples --output-dir examples/batch_outputs --no-llm

csv-json:
	python scripts/convert_csv_to_json.py --input examples/qa_training_set.csv --output examples/qa_training_set.json

llm-training-jsonl:
	python scripts/build_llm_training_data.py --conversations examples/qa_training_set.csv --scores examples/qa.csv --output examples/qa_llm_training.jsonl

llm-training-sample:
	python scripts/build_llm_training_data.py --conversations examples/qa_training_set.csv --scores examples/qa.csv --output examples/qa_llm_training.first10.jsonl --limit 10

sample: single

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache *.egg-info examples/batch_outputs examples/sample_output.generated.json
