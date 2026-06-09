#!/usr/bin/env python3
"""Merge a Mythograph LoRA adapter into the Nemotron BF16 base model."""

from __future__ import annotations

import argparse
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_BASE_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge a PEFT LoRA adapter into the base model.")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter", default="outputs/mythograph-nemotron-lora")
    parser.add_argument("--output-dir", default="outputs/mythograph-nemotron-merged")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hub-model-id", default=None)
    args = parser.parse_args()

    if args.push_to_hub and not args.hub_model_id:
        raise ValueError("--hub-model-id is required when --push-to-hub is set")

    tokenizer = AutoTokenizer.from_pretrained(args.adapter, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, args.adapter)
    merged = model.merge_and_unload()

    merged.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)

    if args.push_to_hub:
        merged.push_to_hub(args.hub_model_id, safe_serialization=True)
        tokenizer.push_to_hub(args.hub_model_id)

    print(f"Saved merged model to {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
