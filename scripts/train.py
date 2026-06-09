#!/usr/bin/env python3
"""QLoRA SFT training for Mythograph Atelier strict-JSON outputs."""

from __future__ import annotations

import argparse
import inspect
import os
from typing import Any, Dict, List

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

try:
    from trl import DataCollatorForCompletionOnlyLM
except ImportError:  # pragma: no cover - depends on TRL version
    DataCollatorForCompletionOnlyLM = None


DEFAULT_BASE_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Nemotron 3 Nano 4B with QLoRA.")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--train-file", default="data/train.jsonl")
    parser.add_argument("--validation-file", default="data/validation.jsonl")
    parser.add_argument("--output-dir", default="outputs/mythograph-nemotron-lora")
    parser.add_argument("--hub-model-id", default=None, help="Optional adapter repo id to push, e.g. user/mythograph-nemotron-lora")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit QLoRA loading.")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--report-to", default="none", help='Use "trackio" or "tensorboard" if configured.')
    return parser.parse_args()


def build_sft_config(args: argparse.Namespace) -> SFTConfig:
    supported = set(inspect.signature(SFTConfig).parameters)
    kwargs: Dict[str, Any] = {
        "output_dir": args.output_dir,
        "num_train_epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "gradient_checkpointing": True,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "eval_steps": args.eval_steps,
        "save_strategy": "steps",
        "bf16": torch.cuda.is_available(),
        "fp16": False,
        "max_steps": args.max_steps,
        "push_to_hub": args.push_to_hub,
        "hub_model_id": args.hub_model_id,
        "report_to": args.report_to,
        "packing": False,
    }

    if "eval_strategy" in supported:
        kwargs["eval_strategy"] = "steps"
    elif "evaluation_strategy" in supported:
        kwargs["evaluation_strategy"] = "steps"

    if "max_length" in supported:
        kwargs["max_length"] = args.max_seq_length
    elif "max_seq_length" in supported:
        kwargs["max_seq_length"] = args.max_seq_length

    if "assistant_only_loss" in supported:
        kwargs["assistant_only_loss"] = True
    if "dataset_text_field" in supported:
        kwargs["dataset_text_field"] = "text"

    return SFTConfig(**{k: v for k, v in kwargs.items() if k in supported and v is not None})


def format_example(example: Dict[str, Any], tokenizer: AutoTokenizer) -> str:
    return tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)


def main() -> None:
    args = parse_args()
    if args.push_to_hub and not args.hub_model_id:
        raise ValueError("--hub-model-id is required when --push-to-hub is set")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if not args.no_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        quantization_config=quantization_config,
    )
    model.config.use_cache = False

    dataset = load_dataset(
        "json",
        data_files={"train": args.train_file, "validation": args.validation_file},
    )

    def add_text(example: Dict[str, Any]) -> Dict[str, str]:
        return {"text": format_example(example, tokenizer)}

    dataset = dataset.map(add_text, remove_columns=dataset["train"].column_names)

    data_collator = None
    if DataCollatorForCompletionOnlyLM is not None:
        sentinel = "MYTHOGRAPH_ASSISTANT_SENTINEL"
        rendered_assistant = tokenizer.apply_chat_template(
            [{"role": "assistant", "content": sentinel}],
            tokenize=False,
            add_generation_prompt=False,
        )
        response_template = rendered_assistant.split(sentinel, 1)[0]
        if tokenizer.bos_token and response_template.startswith(tokenizer.bos_token):
            response_template = response_template[len(tokenizer.bos_token) :]
        if response_template:
            data_collator = DataCollatorForCompletionOnlyLM(response_template=response_template, tokenizer=tokenizer)

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )

    trainer_kwargs = {
        "model": model,
        "args": build_sft_config(args),
        "train_dataset": dataset["train"],
        "eval_dataset": dataset["validation"],
        "peft_config": lora_config,
        "tokenizer": tokenizer,
        "dataset_text_field": "text",
    }
    trainer_supported = set(inspect.signature(SFTTrainer).parameters)
    if "processing_class" in trainer_supported:
        trainer_kwargs["processing_class"] = tokenizer
        trainer_kwargs.pop("tokenizer", None)
    if "dataset_text_field" not in trainer_supported:
        trainer_kwargs.pop("dataset_text_field", None)
    if data_collator is not None:
        trainer_kwargs["data_collator"] = data_collator

    trainer = SFTTrainer(**{k: v for k, v in trainer_kwargs.items() if k in trainer_supported})
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    if args.push_to_hub:
        trainer.push_to_hub()

    print(f"Saved LoRA adapter to {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
