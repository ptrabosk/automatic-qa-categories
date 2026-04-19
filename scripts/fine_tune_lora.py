#!/usr/bin/env python
from __future__ import annotations

import argparse
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def main() -> None:
    args = parse_args()
    ensure_training_imports()

    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
    )
    completion_trainer = completion_only_trainer_class(Trainer)

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available. Run `make check-cuda` before fine-tuning.")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    compute_dtype = _compute_dtype(args.precision, torch)
    quantization_config = None
    if args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        attn_implementation=args.attn_implementation,
        device_map="auto",
        quantization_config=quantization_config,
        torch_dtype=compute_dtype,
        trust_remote_code=args.trust_remote_code,
    )
    model.config.use_cache = False

    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=args.target_modules.split(","),
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_rows = read_jsonl(args.train_file)
    validation_rows = read_jsonl(args.validation_file) if args.validation_file else []
    if args.max_train_samples is not None:
        train_rows = train_rows[: args.max_train_samples]
    if args.max_validation_samples is not None:
        validation_rows = validation_rows[: args.max_validation_samples]
    train_dataset = Dataset.from_list(train_rows).map(
        TokenizeChatRows(tokenizer, args.max_seq_length),
        remove_columns=list(train_rows[0].keys()),
    )
    eval_dataset = None
    if validation_rows:
        eval_dataset = Dataset.from_list(validation_rows).map(
            TokenizeChatRows(tokenizer, args.max_seq_length),
            remove_columns=list(validation_rows[0].keys()),
        )

    training_args = TrainingArguments(**_training_arguments_kwargs(args, compute_dtype, eval_dataset))

    trainer = completion_trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForCausalLM(tokenizer.pad_token_id),
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a local chat model with LoRA/QLoRA.")
    parser.add_argument("--train-file", required=True, help="Chat JSONL training file.")
    parser.add_argument("--validation-file", help="Optional chat JSONL validation file.")
    parser.add_argument("--output-dir", default="models/qa-lora", help="Adapter output directory.")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-validation-samples", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--lr-scheduler-type", default="cosine")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated module names for LoRA adapters.",
    )
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--report-to", default="none")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument(
        "--precision",
        choices=("auto", "bf16", "fp16"),
        default="auto",
        help="Training precision. Use fp16 if BF16 causes CUDA driver issues on WSL.",
    )
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--no-4bit", dest="load_in_4bit", action="store_false")
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")
    parser.set_defaults(load_in_4bit=True, gradient_checkpointing=True)
    return parser.parse_args()


def ensure_training_imports() -> None:
    missing: list[str] = []
    for module in ("torch", "datasets", "peft", "transformers", "bitsandbytes"):
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    if missing:
        raise SystemExit(
            "Missing fine-tuning dependencies: "
            + ", ".join(missing)
            + "\nInstall them with: make install-finetune-wsl"
        )


def _compute_dtype(precision: str, torch: Any) -> Any:
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def _training_arguments_kwargs(
    args: argparse.Namespace, compute_dtype: Any, eval_dataset: Any | None
) -> dict[str, Any]:
    values: dict[str, Any] = {
        "output_dir": args.output_dir,
        "num_train_epochs": args.epochs,
        "max_steps": args.max_steps,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "warmup_ratio": args.warmup_ratio,
        "lr_scheduler_type": args.lr_scheduler_type,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "eval_steps": args.eval_steps if eval_dataset is not None else None,
        "save_total_limit": args.save_total_limit,
        "bf16": str(compute_dtype) == "torch.bfloat16",
        "fp16": str(compute_dtype) == "torch.float16",
        "optim": "paged_adamw_8bit" if args.load_in_4bit else "adamw_torch",
        "report_to": args.report_to,
        "remove_unused_columns": False,
    }
    evaluation_key = _training_arguments_evaluation_key()
    values[evaluation_key] = "steps" if eval_dataset is not None else "no"
    return values


def _training_arguments_evaluation_key() -> str:
    try:
        from transformers import TrainingArguments
    except ImportError:
        return "eval_strategy"
    parameters = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in parameters:
        return "eval_strategy"
    return "evaluation_strategy"


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


@dataclass
class TokenizeChatRows:
    tokenizer: Any
    max_seq_length: int

    def __call__(self, row: dict[str, Any]) -> dict[str, list[int]]:
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) < 2:
            raise ValueError("Each training row must contain a chat-style messages list")

        assistant_index = _last_assistant_index(messages)
        prompt_messages = messages[:assistant_index]
        full_text = _format_chat(self.tokenizer, messages, add_generation_prompt=False)
        prompt_text = _format_chat(self.tokenizer, prompt_messages, add_generation_prompt=True)

        full_ids = self.tokenizer(full_text, add_special_tokens=False)["input_ids"]
        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        prompt_length = min(len(prompt_ids), len(full_ids))
        completion_ids = full_ids[prompt_length:]
        if not completion_ids:
            raise ValueError("Could not find assistant completion tokens in chat row")

        if len(completion_ids) >= self.max_seq_length:
            completion_ids = completion_ids[-(self.max_seq_length - 1) :]
        max_prompt_length = self.max_seq_length - len(completion_ids)
        prompt_ids = prompt_ids[-max_prompt_length:]

        input_ids = prompt_ids + completion_ids
        labels = [-100] * len(prompt_ids) + completion_ids
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
            "completion_length": len(completion_ids),
        }


def _last_assistant_index(messages: list[dict[str, str]]) -> int:
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].get("role") == "assistant":
            return index
    raise ValueError("Each training row needs an assistant target message")


def _format_chat(tokenizer: Any, messages: list[dict[str, str]], add_generation_prompt: bool) -> str:
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    chunks: list[str] = []
    for message in messages:
        role = message.get("role", "user").title()
        chunks.append(f"{role}:\n{message.get('content', '')}")
    if add_generation_prompt:
        chunks.append("Assistant:\n")
    return "\n\n".join(chunks)


@dataclass
class DataCollatorForCausalLM:
    pad_token_id: int

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, Any]:
        import torch

        max_length = max(len(feature["input_ids"]) for feature in features)
        batch: dict[str, list[list[int]]] = {"input_ids": [], "attention_mask": [], "labels": []}
        completion_lengths: list[int] = []
        for feature in features:
            pad_length = max_length - len(feature["input_ids"])
            batch["input_ids"].append([self.pad_token_id] * pad_length + feature["input_ids"])
            batch["attention_mask"].append([0] * pad_length + feature["attention_mask"])
            batch["labels"].append([-100] * pad_length + feature["labels"])
            completion_lengths.append(int(feature["completion_length"]))
        tensors = {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}
        tensors["completion_length"] = torch.tensor(completion_lengths, dtype=torch.long)
        return tensors


def completion_only_trainer_class(base_trainer: Any) -> Any:
    class CompletionOnlyTrainer(base_trainer):
        def compute_loss(
            self,
            model: Any,
            inputs: dict[str, Any],
            return_outputs: bool = False,
            num_items_in_batch: Any | None = None,
        ) -> Any:
            import torch.nn.functional as F

            labels = inputs.pop("labels")
            completion_lengths = inputs.pop("completion_length")
            logits_to_keep = int(completion_lengths.max().item()) + 1

            outputs = model(**inputs, logits_to_keep=logits_to_keep)
            logits = outputs.logits

            shifted_labels = F.pad(labels, (0, 1), value=-100)[:, -logits_to_keep:].contiguous()
            shifted_labels = shifted_labels.to(logits.device)
            loss = F.cross_entropy(
                logits.float().reshape(-1, logits.size(-1)),
                shifted_labels.reshape(-1),
                ignore_index=-100,
            )
            return (loss, outputs) if return_outputs else loss

    return CompletionOnlyTrainer


if __name__ == "__main__":
    main()
