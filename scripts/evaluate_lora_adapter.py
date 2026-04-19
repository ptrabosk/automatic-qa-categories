#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def main() -> None:
    args = parse_args()
    ensure_imports()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    dtype = _compute_dtype(args.precision, torch)
    quantization_config = None
    if args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        attn_implementation=args.attn_implementation,
        device_map="auto",
        quantization_config=quantization_config,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
    )
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    model.eval()

    rows = read_jsonl(args.input)
    if args.limit is not None:
        rows = rows[: args.limit]

    predictions: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        prediction = predict_row(model, tokenizer, row, args)
        predictions.append(prediction)
        labels.append(label_from_row(row))
        if index % args.log_every == 0:
            print(f"Evaluated {index}/{len(rows)} rows")

    report = evaluate_predictions(predictions, labels)
    write_json(args.predictions_output, predictions)
    write_json(args.report_output, report)

    print(f"Wrote predictions to {args.predictions_output}")
    print(f"Wrote report to {args.report_output}")
    print_summary(report)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained LoRA adapter on chat JSONL labels.")
    parser.add_argument("--input", default="examples/llm_training_splits/test.jsonl")
    parser.add_argument("--adapter-dir", default="models/qa-lora")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--predictions-output", default="reports/qa-lora-test-predictions.json")
    parser.add_argument("--report-output", default="reports/qa-lora-test-report.json")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--precision", choices=("auto", "bf16", "fp16"), default="fp16")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--no-4bit", dest="load_in_4bit", action="store_false")
    parser.set_defaults(load_in_4bit=True)
    return parser.parse_args()


def ensure_imports() -> None:
    missing: list[str] = []
    for module in ("torch", "peft", "transformers", "bitsandbytes"):
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    if missing:
        raise SystemExit(
            "Missing evaluation dependencies: "
            + ", ".join(missing)
            + "\nInstall them with: make install-finetune-wsl"
        )


def predict_row(model: Any, tokenizer: Any, row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    import torch

    messages = row["messages"]
    assistant_index = last_assistant_index(messages)
    prompt_messages = messages[:assistant_index]
    gold = json.loads(messages[assistant_index]["content"])
    prompt = format_chat(tokenizer, prompt_messages)
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_seq_length,
        add_special_tokens=False,
    )
    encoded = {key: value.to(model.device) for key, value in encoded.items()}

    with torch.inference_mode():
        generate_kwargs: dict[str, Any] = {
            **encoded,
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.temperature > 0,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if args.temperature > 0:
            generate_kwargs["temperature"] = args.temperature
            generate_kwargs["top_p"] = args.top_p
        generated = model.generate(**generate_kwargs)

    completion_ids = generated[0, encoded["input_ids"].shape[1] :]
    raw_output = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
    parsed = parse_json_object(raw_output)
    return {
        "send_id": gold.get("send_id"),
        "audited_message_index": gold.get("audited_message_index"),
        "scores": normalize_scores(parsed.get("scores", {})) if parsed else {},
        "parse_error": parsed is None,
        "raw_output": raw_output,
    }


def label_from_row(row: dict[str, Any]) -> dict[str, Any]:
    assistant = json.loads(row["messages"][last_assistant_index(row["messages"])]["content"])
    return {
        "send_id": assistant["send_id"],
        "audited_message_index": assistant["audited_message_index"],
        "scores": normalize_scores(assistant["scores"]),
    }


def evaluate_predictions(
    predictions: list[dict[str, Any]], labels: list[dict[str, Any]]
) -> dict[str, Any]:
    labels_by_id = {item["send_id"]: item for item in labels}
    metrics: dict[str, dict[str, int]] = defaultdict(
        lambda: {"pass_tp": 0, "pass_fp": 0, "pass_fn": 0, "fail_tp": 0, "fail_fp": 0, "fail_fn": 0}
    )
    exact_matches = 0
    all_pass_exact_matches = 0
    evaluated_records = 0
    parse_errors = 0
    missing_predictions = 0
    aggregate_counts = {"fail_tp": 0, "fail_fp": 0, "fail_fn": 0, "pass_tp": 0, "pass_fp": 0, "pass_fn": 0}

    for prediction in predictions:
        label = labels_by_id.get(prediction["send_id"])
        if label is None:
            continue
        evaluated_records += 1
        parse_errors += int(bool(prediction.get("parse_error")))
        pred_scores = prediction.get("scores", {})
        label_scores = label["scores"]
        exact_matches += int(pred_scores == label_scores)
        all_pass_exact_matches += int(all(value == 1 for value in label_scores.values()))
        for key, actual in label_scores.items():
            if key not in pred_scores:
                missing_predictions += 1
                predicted = 1
            else:
                predicted = int(pred_scores[key])
            update_binary_metrics(metrics[key], predicted, int(actual))
            update_binary_metrics(aggregate_counts, predicted, int(actual))

    return {
        "records_evaluated": evaluated_records,
        "exact_record_match_rate": exact_matches / evaluated_records if evaluated_records else 0.0,
        "all_pass_exact_match_baseline": (
            all_pass_exact_matches / evaluated_records if evaluated_records else 0.0
        ),
        "parse_errors": parse_errors,
        "missing_subcategory_predictions": missing_predictions,
        "aggregate_metrics": metric_rates(aggregate_counts) | {"counts": aggregate_counts},
        "subcategory_metrics": {
            key: metric_rates(bucket) | {"counts": bucket} for key, bucket in sorted(metrics.items())
        },
    }


def update_binary_metrics(bucket: dict[str, int], predicted: int, actual: int) -> None:
    if predicted == 1 and actual == 1:
        bucket["pass_tp"] += 1
    elif predicted == 1 and actual == 0:
        bucket["pass_fp"] += 1
        bucket["fail_fn"] += 1
    elif predicted == 0 and actual == 0:
        bucket["fail_tp"] += 1
    elif predicted == 0 and actual == 1:
        bucket["fail_fp"] += 1
        bucket["pass_fn"] += 1


def metric_rates(bucket: dict[str, int]) -> dict[str, float]:
    pass_precision = divide(bucket["pass_tp"], bucket["pass_tp"] + bucket["pass_fp"])
    pass_recall = divide(bucket["pass_tp"], bucket["pass_tp"] + bucket["pass_fn"])
    fail_precision = divide(bucket["fail_tp"], bucket["fail_tp"] + bucket["fail_fp"])
    fail_recall = divide(bucket["fail_tp"], bucket["fail_tp"] + bucket["fail_fn"])
    return {
        "pass_precision": pass_precision,
        "pass_recall": pass_recall,
        "pass_f1": f1(pass_precision, pass_recall),
        "failure_precision": fail_precision,
        "failure_recall": fail_recall,
        "failure_f1": f1(fail_precision, fail_recall),
        "support_pass": bucket["pass_tp"] + bucket["pass_fn"],
        "support_failure": bucket["fail_tp"] + bucket["fail_fn"],
    }


def print_summary(report: dict[str, Any]) -> None:
    print(f"Records evaluated: {report['records_evaluated']}")
    print(f"Exact record match rate: {report['exact_record_match_rate']:.3f}")
    print(f"All-pass exact match baseline: {report['all_pass_exact_match_baseline']:.3f}")
    print(f"Parse errors: {report['parse_errors']}")
    aggregate = report["aggregate_metrics"]
    print(
        "Aggregate failure metrics: "
        f"f1={aggregate['failure_f1']:.3f}, "
        f"recall={aggregate['failure_recall']:.3f}, "
        f"precision={aggregate['failure_precision']:.3f}"
    )
    print("Worst failure F1 by subcategory:")
    rows = sorted(
        report["subcategory_metrics"].items(),
        key=lambda item: (item[1]["failure_f1"], item[1]["support_failure"]),
    )
    for key, values in rows[:10]:
        print(
            f"  {key}: failure_f1={values['failure_f1']:.3f}, "
            f"failure_recall={values['failure_recall']:.3f}, "
            f"failure_precision={values['failure_precision']:.3f}, "
            f"support_failure={values['support_failure']}"
        )


def parse_json_object(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for start, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def normalize_scores(scores: dict[str, Any]) -> dict[str, int]:
    normalized: dict[str, int] = {}
    for key, value in scores.items():
        if isinstance(value, dict):
            value = value.get("score")
        try:
            normalized[key] = 1 if int(value) > 0 else 0
        except (TypeError, ValueError):
            continue
    return normalized


def format_chat(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return "\n\n".join(f"{item['role'].title()}:\n{item['content']}" for item in messages) + "\n\nAssistant:\n"


def last_assistant_index(messages: list[dict[str, Any]]) -> int:
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].get("role") == "assistant":
            return index
    raise ValueError("No assistant message found")


def _compute_dtype(precision: str, torch: Any) -> Any:
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)


def divide(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


if __name__ == "__main__":
    main()
