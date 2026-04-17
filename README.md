# Automatic QA Categories

Local-first hybrid QA auditing for customer-support agent replies.

The core invariant is simple: the JSON record contains the full ordered thread, but the audited reply is always the final agent message in that thread. Earlier messages are context only. The pipeline never uses web lookups or outside knowledge; every score must be grounded in the JSON record and the rubric/config files.

## Architecture

The audit flow is deterministic-first, then model-assisted, then aggregated:

1. `preprocessing/` normalizes arbitrary conversation JSON into a typed internal schema.
2. `thread_parser.py` locks the last agent message as the only audited message.
3. `evidence_builder.py` records available and missing structured evidence.
4. `validators/` applies local deterministic checks for zero-tolerance risks, clarity patterns, duplicate text, and workflow evidence presence.
5. `specialists/` sends the same evidence packet, plus category slices, to category-specific LLM judges.
6. `aggregation/` applies hard-fail precedence and computes category and overall pass/fail.

Default model backend is Ollama. The model interface is `src/adapters/base_llm.py`; add LM Studio or another OpenAI-compatible local backend by implementing `generate_json(prompt, schema)`.

## Install

```bash
python3.11 -m venv .venv
source .venv/bin/activate
make install
```

Optional local model setup:

```bash
cp .env.example .env
ollama pull llama3.1:8b-instruct-q8_0
```

## Input JSON

At minimum, each record needs ordered messages:

```json
{
  "record_id": "case-123",
  "messages": [
    {"role": "customer", "content": "Do you have this in size 9?"},
    {"role": "agent", "content": "Yes, size 9 is available."}
  ]
}
```

Optional structured evidence can be supplied at the top level or under `context`, including `company_profile`, `customer_profile`, `notes`, `templates`, `product_information`, `promo_notes`, `checkout_page`, `website_findings`, `link_metadata`, `product_views`, `purchases`, and `workflow_flags`.

Missing optional fields are recorded in output metadata and do not crash the run.

## Convert CSV Input

The included converter turns `qa_training_set.csv` rows into pipeline-ready JSON:

```bash
python scripts/convert_csv_to_json.py \
  --input examples/qa_training_set.csv \
  --output examples/qa_training_set.json
```

For a quick smoke test, convert only a few rows:

```bash
python scripts/convert_csv_to_json.py \
  --input examples/qa_training_set.csv \
  --output examples/qa_training_set.sample.json \
  --limit 10
```

The converter maps `CONVERSATION_JSON` to ordered `messages`, uses `SEND_ID` as `record_id`, and maps company/products/coupons/orders/notes/escalation topics/blocklisted words into structured evidence fields. `COMPANY_NOTES` maps only to `notes`; templates are not inferred from notes.

When templates are available as a separate file, pass it with `--templates`. The file may be CSV, JSON, or JSONL keyed by `SEND_ID`, `send_id`, `Agent Outbound Messages Send ID`, or `record_id`.

## Run

Single file with Ollama:

```bash
python scripts/run_single_audit.py --input examples/sample_input.json --output output/sample.audit.json
```

Single file without LLM calls, useful for tests and dry runs:

```bash
python scripts/run_single_audit.py --input examples/sample_input.json --output output/sample.audit.json --no-llm
```

Weekly batch scoring:

```bash
python scripts/run_audit_batch.py --input-dir weekly_json --output-dir weekly_audits --glob "*.json"
```

The batch script writes one `.audit.json` file per input file.

## Output

Every subcategory returns:

- `score`: `0` or `1`
- `method`: `deterministic`, `llm`, or `hybrid`
- `evidence_used`
- `rationale`
- `failure_note` when score is `0`
- `confidence`
- audited message id/index
- provenance from deterministic and specialist layers

If any required subcategory fails, the category fails. If any required category failure exists, `final_overall_score` is `0`.

## Training Data

Join the conversation CSV and manual QA score CSV into chat-style JSONL for LLM training:

```bash
python scripts/build_llm_training_data.py \
  --conversations examples/qa_training_set.csv \
  --scores examples/qa.csv \
  --output examples/qa_llm_training.jsonl
```

If you have a separate templates file:

```bash
python scripts/build_llm_training_data.py \
  --conversations examples/qa_training_set.csv \
  --scores examples/qa.csv \
  --templates examples/templates.csv \
  --output examples/qa_llm_training.jsonl
```

For a small smoke test:

```bash
python scripts/build_llm_training_data.py \
  --conversations examples/qa_training_set.csv \
  --scores examples/qa.csv \
  --output examples/qa_llm_training.first10.jsonl \
  --limit 10
```

Manual scores are normalized to binary labels with `0 -> 0` and any positive score, including `2`, converted to `1`. The assistant training target uses `send_id` and a `scores` object, where scores come from `qa.csv`.

The default JSONL uses strict chat fine-tuning rows with only a top-level `messages` key. Add `--include-metadata` if your trainer accepts extra row metadata, or `--raw-json-output joined.json` if you want a separate auditable joined dataset with raw manual scores.

Prepare labeled JSON into JSONL:

```bash
python scripts/train_dataset_prep.py --input labeled.json --output data/audit_rows.jsonl
```

Chat fine-tuning style export:

```bash
python scripts/train_dataset_prep.py --input labeled.json --output data/chat_rows.jsonl --chat-format
```

Train/validation/test split:

```bash
python scripts/train_dataset_prep.py --input labeled.json --output data/all.jsonl --split-dir data/splits
```

Fine-tuning itself is intentionally outside this repo.

## Evaluation

```bash
python scripts/evaluate_model.py --predictions predictions.json --labels labels.json --output reports/eval.json
```

The report includes per-subcategory confusion matrices and precision/recall/F1, plus category and overall accuracy.

## Add A Subcategory

1. Add the subcategory to `config/rubric.yaml`.
2. Add it to the category entry in `config/category_map.yaml`.
3. Update the relevant prompt under `src/prompts/`.
4. Add a deterministic validator only if explicit JSON evidence can decide the case.
5. Add or update tests for precedence and output schema.

## Add A Model Backend

1. Implement `src/adapters/base_llm.py`.
2. Add backend config to `config/model_config.yaml`.
3. Update `AuditPipeline._build_adapter`.
4. Keep the adapter returning a single JSON object matching `config/output_schema.yaml`.

## Test

```bash
make test
```
