#!/usr/bin/env python3
"""Modal workflow for Mythograph Nemotron fine-tuning and GGUF publishing.

Authentication:
  1. Run locally once: python -m modal setup
  2. Create a Modal secret named "huggingface" with key HF_TOKEN.

Common runs:
  modal run --detach scripts/modal_workflow.py --action train
  modal run --detach scripts/modal_workflow.py --action merge --adapter-repo /vol/outputs/mythograph-nemotron-lora
  modal run --detach scripts/modal_workflow.py --action gguf
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Optional

import modal


APP_NAME = "mythograph-nemotron-ft"
VOLUME_NAME = "mythograph-nemotron-ft-cache"
HF_SECRET_NAME = "huggingface"
BASE_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"

REMOTE_ROOT = PurePosixPath("/workspace/mythograph-nemotron-ft")
VOLUME_ROOT = PurePosixPath("/vol")
TRAIN_FILE = REMOTE_ROOT / "data/train.jsonl"
VALIDATION_FILE = REMOTE_ROOT / "data/validation.jsonl"
ADAPTER_DIR = VOLUME_ROOT / "outputs/mythograph-nemotron-lora"
MERGED_DIR = VOLUME_ROOT / "outputs/mythograph-nemotron-merged"
GGUF_DIR = VOLUME_ROOT / "outputs/gguf"


volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("bash", "build-essential", "cmake", "curl", "git", "ninja-build")
    .pip_install("torch==2.6.0", index_url="https://download.pytorch.org/whl/cu124")
    .pip_install("pip>=26.1.2", "wheel", "packaging", "ninja")
    .pip_install(
        "transformers==4.52.4",
        "datasets>=2.20.0",
        "accelerate>=0.33.0",
        "peft==0.15.2",
        "trl==0.17.0",
        "bitsandbytes>=0.43.0",
        "safetensors>=0.4.3",
        "sentencepiece>=0.2.0",
        "protobuf>=4.25.0",
        "einops>=0.8.0",
        "huggingface_hub>=0.24.0",
    )
    .pip_install(
        "causal-conv1d==1.5.0.post8",
        "mamba-ssm==2.2.4",
        extra_options="--no-build-isolation --no-deps",
        env={"MAX_JOBS": "4", "MAMBA_FORCE_CXX11_ABI": "FALSE"},
        gpu="A100",
    )
    .env(
        {
            "HF_HOME": str(VOLUME_ROOT / "hf-cache"),
            "TRANSFORMERS_CACHE": str(VOLUME_ROOT / "hf-cache"),
            "HF_XET_HIGH_PERFORMANCE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
)

app = modal.App(APP_NAME, image=image, volumes={str(VOLUME_ROOT): volume})


def _as_path(path: PurePosixPath | str) -> Path:
    return Path(str(path))


def _run(cmd: list[str], cwd: PurePosixPath = REMOTE_ROOT) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _print_spawned_call(action: str, call: modal.FunctionCall) -> None:
    print(
        json.dumps(
            {
                "action": action,
                "status": "spawned",
                "function_call_id": call.object_id,
                "dashboard_url": call.get_dashboard_url(),
            },
            indent=2,
        )
    )


def _write_jsonl_data(train_jsonl: str, validation_jsonl: str) -> None:
    _as_path(REMOTE_ROOT / "data").mkdir(parents=True, exist_ok=True)
    _as_path(TRAIN_FILE).write_text(train_jsonl, encoding="utf-8")
    _as_path(VALIDATION_FILE).write_text(validation_jsonl, encoding="utf-8")


def _load_jsonl(path: PurePosixPath) -> list[dict[str, Any]]:
    rows = []
    with _as_path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _validate_source_data() -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for split, path in [("train", TRAIN_FILE), ("validation", VALIDATION_FILE)]:
        rows = _load_jsonl(path)
        counts = {"conversation_turn": 0, "final_art_recipe": 0}
        for row in rows:
            assistant_obj = json.loads(row["messages"][-1]["content"])
            if "controls" in assistant_obj or "assistant_message" in assistant_obj:
                counts["conversation_turn"] += 1
            elif "image_prompt" in assistant_obj or "friend_explanation" in assistant_obj:
                counts["final_art_recipe"] += 1
            else:
                raise ValueError(f"Unknown assistant schema in {split}")
        summary[split] = {"rows": len(rows), **counts}
    return summary


def _sanitize_generation_config(model: Any) -> None:
    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None and not getattr(generation_config, "do_sample", False):
        generation_config.top_p = None
        generation_config.temperature = None


@app.function(
    gpu="A100",
    cpu=8,
    memory=65536,
    timeout=6 * 60 * 60,
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
)
def train_lora(
    train_jsonl: str,
    validation_jsonl: str,
    adapter_repo: str,
    base_model: str = BASE_MODEL,
    epochs: float = 3.0,
    learning_rate: float = 1e-4,
    max_seq_length: int = 4096,
) -> dict[str, Any]:
    import inspect

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    _write_jsonl_data(train_jsonl, validation_jsonl)
    summary = _validate_source_data()
    print(json.dumps(summary, indent=2))

    _as_path(ADAPTER_DIR).mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.config.use_cache = False

    dataset = load_dataset("json", data_files={"train": str(TRAIN_FILE), "validation": str(VALIDATION_FILE)})

    def add_text(example: Dict[str, Any]) -> Dict[str, str]:
        return {
            "text": tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
        }

    dataset = dataset.map(add_text, remove_columns=dataset["train"].column_names)

    supported = set(inspect.signature(SFTConfig).parameters)
    config_kwargs: Dict[str, Any] = {
        "output_dir": str(ADAPTER_DIR),
        "num_train_epochs": epochs,
        "learning_rate": learning_rate,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "gradient_checkpointing": True,
        "logging_steps": 10,
        "save_steps": 100,
        "eval_steps": 100,
        "save_strategy": "steps",
        "bf16": True,
        "push_to_hub": True,
        "hub_model_id": adapter_repo,
        "report_to": "none",
        "packing": False,
    }
    if "eval_strategy" in supported:
        config_kwargs["eval_strategy"] = "steps"
    elif "evaluation_strategy" in supported:
        config_kwargs["evaluation_strategy"] = "steps"
    if "max_length" in supported:
        config_kwargs["max_length"] = max_seq_length
    elif "max_seq_length" in supported:
        config_kwargs["max_seq_length"] = max_seq_length
    if "assistant_only_loss" in supported:
        config_kwargs["assistant_only_loss"] = True
    if "dataset_text_field" in supported:
        config_kwargs["dataset_text_field"] = "text"

    trainer_kwargs: Dict[str, Any] = {
        "model": model,
        "args": SFTConfig(**{k: v for k, v in config_kwargs.items() if k in supported}),
        "train_dataset": dataset["train"],
        "eval_dataset": dataset["validation"],
        "peft_config": LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
        ),
        "tokenizer": tokenizer,
        "dataset_text_field": "text",
    }
    trainer_supported = set(inspect.signature(SFTTrainer).parameters)
    if "processing_class" in trainer_supported:
        trainer_kwargs["processing_class"] = tokenizer
        trainer_kwargs.pop("tokenizer", None)
    if "dataset_text_field" not in trainer_supported:
        trainer_kwargs.pop("dataset_text_field", None)
    trainer = SFTTrainer(**{k: v for k, v in trainer_kwargs.items() if k in trainer_supported})
    trainer.train()
    trainer.save_model(str(ADAPTER_DIR))
    tokenizer.save_pretrained(str(ADAPTER_DIR))
    trainer.push_to_hub()
    volume.commit()
    return {"adapter_dir": str(ADAPTER_DIR), "adapter_repo": adapter_repo, "data": summary}


@app.function(
    gpu="A100",
    cpu=8,
    memory=65536,
    timeout=3 * 60 * 60,
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
)
def merge_lora(
    adapter_repo_or_path: str,
    merged_repo: Optional[str] = None,
    base_model: str = BASE_MODEL,
) -> dict[str, str]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter = adapter_repo_or_path or str(ADAPTER_DIR)
    _as_path(MERGED_DIR).mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(adapter, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, adapter)
    merged = model.merge_and_unload()
    _sanitize_generation_config(merged)
    merged.save_pretrained(str(MERGED_DIR), safe_serialization=True)
    tokenizer.save_pretrained(str(MERGED_DIR))

    if merged_repo:
        merged.push_to_hub(merged_repo, safe_serialization=True)
        tokenizer.push_to_hub(merged_repo)

    volume.commit()
    return {"merged_dir": str(MERGED_DIR), "merged_repo": merged_repo or ""}


@app.function(cpu=12, memory=65536, timeout=4 * 60 * 60, secrets=[modal.Secret.from_name(HF_SECRET_NAME)])
def build_gguf(gguf_repo: Optional[str] = None, prefix: str = "mythograph-nemotron") -> dict[str, str]:
    llama_cpp_dir = VOLUME_ROOT / "third_party/llama.cpp"
    _as_path(GGUF_DIR).mkdir(parents=True, exist_ok=True)

    if not _as_path(llama_cpp_dir).exists():
        _run(["git", "clone", "https://github.com/ggml-org/llama.cpp.git", str(llama_cpp_dir)], cwd=VOLUME_ROOT)

    _run(["python", "-m", "pip", "install", "-r", str(llama_cpp_dir / "requirements.txt")], cwd=llama_cpp_dir)

    f16 = GGUF_DIR / f"{prefix}-F16.gguf"
    if not _as_path(f16).exists():
        _run(
            [
                "python",
                str(llama_cpp_dir / "convert_hf_to_gguf.py"),
                str(MERGED_DIR),
                "--outfile",
                str(f16),
                "--outtype",
                "f16",
            ],
            cwd=llama_cpp_dir,
        )

    if not _as_path(llama_cpp_dir / "build/bin/llama-quantize").exists():
        _run(["cmake", "-S", str(llama_cpp_dir), "-B", str(llama_cpp_dir / "build"), "-DCMAKE_BUILD_TYPE=Release"], cwd=llama_cpp_dir)
        _run(["cmake", "--build", str(llama_cpp_dir / "build"), "--config", "Release", "-j", "--target", "llama-quantize"], cwd=llama_cpp_dir)

    quantize = llama_cpp_dir / "build/bin/llama-quantize"
    if not _as_path(quantize).exists():
        quantize = llama_cpp_dir / "build/bin/quantize"
    if not _as_path(quantize).exists():
        raise FileNotFoundError(f"Could not find llama.cpp quantizer under {llama_cpp_dir / 'build'}")

    q4 = GGUF_DIR / f"{prefix}-Q4_K_M.gguf"
    q5 = GGUF_DIR / f"{prefix}-Q5_K_M.gguf"
    if not _as_path(q4).exists():
        _run([str(quantize), str(f16), str(q4), "Q4_K_M"], cwd=llama_cpp_dir)
    if not _as_path(q5).exists():
        _run([str(quantize), str(f16), str(q5), "Q5_K_M"], cwd=llama_cpp_dir)

    readme = GGUF_DIR / "README.md"
    if not _as_path(readme).exists():
        _as_path(readme).write_text(
            """---
base_model: nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16
library_name: llama.cpp
license: other
language:
- en
tags:
- gguf
- llama.cpp
- q4_k_m
- q5_k_m
- build-small-hackathon
- mythograph-atelier
- json-generation
---

# Mythograph Nemotron 3 Nano 4B GGUF

This model is a fine-tuned GGUF version of NVIDIA Nemotron 3 Nano 4B for Mythograph Atelier.
It is intended to run with llama.cpp / llama-cpp-python.

Recommended file: `mythograph-nemotron-Q5_K_M.gguf`
Lightweight file: `mythograph-nemotron-Q4_K_M.gguf`

The base model is `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` and is governed by the NVIDIA Nemotron Open Model License.
""",
            encoding="utf-8",
        )

    if gguf_repo:
        from huggingface_hub import HfApi, create_repo

        api = HfApi()
        repo_id = gguf_repo
        if "/" not in repo_id:
            repo_id = f"{api.whoami()['name']}/{repo_id}"
        create_repo(repo_id, repo_type="model", exist_ok=True)
        print(f"Uploading GGUF files to https://huggingface.co/{repo_id}")
        for path in [q4, q5, readme]:
            api.upload_file(
                path_or_fileobj=str(path),
                path_in_repo=path.name,
                repo_id=repo_id,
                repo_type="model",
            )
        gguf_repo = repo_id

    volume.commit()
    return {"f16": str(f16), "q4": str(q4), "q5": str(q5), "gguf_repo": gguf_repo or ""}


@app.local_entrypoint()
def main(
    action: str = "train",
    gpu: str = "A100",
    adapter_repo: str = "mythograph-nemotron-3-nano-4b-lora",
    merged_repo: str = "",
    gguf_repo: str = "mythograph-nemotron-3-nano-4b-gguf",
    epochs: float = 3.0,
    max_seq_length: int = 4096,
) -> None:
    train_jsonl = Path("data/train.jsonl").read_text(encoding="utf-8")
    validation_jsonl = Path("data/validation.jsonl").read_text(encoding="utf-8")

    if action == "train":
        call = train_lora.with_options(gpu=gpu).spawn(
            train_jsonl,
            validation_jsonl,
            adapter_repo,
            BASE_MODEL,
            epochs,
            1e-4,
            max_seq_length,
        )
        _print_spawned_call(action, call)
        return
    elif action == "merge":
        call = merge_lora.with_options(gpu=gpu).spawn(adapter_repo or str(ADAPTER_DIR), merged_repo or None, BASE_MODEL)
        _print_spawned_call(action, call)
        return
    elif action == "gguf":
        call = build_gguf.spawn(gguf_repo, "mythograph-nemotron")
        _print_spawned_call(action, call)
        return
    elif action == "all":
        train_result = train_lora.with_options(gpu=gpu).remote(
            train_jsonl,
            validation_jsonl,
            adapter_repo,
            BASE_MODEL,
            epochs,
            1e-4,
            max_seq_length,
        )
        print(json.dumps(train_result, indent=2))
        merge_result = merge_lora.with_options(gpu=gpu).remote(adapter_repo, merged_repo or None, BASE_MODEL)
        print(json.dumps(merge_result, indent=2))
        result = build_gguf.remote(gguf_repo, "mythograph-nemotron")
    else:
        raise ValueError("action must be one of: train, merge, gguf, all")

    print(json.dumps(result, indent=2))
