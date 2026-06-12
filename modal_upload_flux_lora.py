#!/usr/bin/env python3
"""Upload the trained FLUX.2 Klein LoRA from Modal storage to Hugging Face."""

from __future__ import annotations

from io import BytesIO

import modal


APP_NAME = "mythograph-flux2-klein-lora-upload"
VOLUME_NAME = "mythograph-flux-lora"
HF_SECRET_NAME = "huggingface"

DEFAULT_REPO_ID = "kasimakpinar/mythograph-atelier-flux2-klein-lora"
CHECKPOINT_PATH = "/workspace/outputs/mythograph_flux2_klein_lora/mythograph_flux2_klein_lora.safetensors"
OUTPUT_DIR = "/workspace/outputs/mythograph_flux2_klein_lora"

MODEL_CARD = """---
base_model: black-forest-labs/FLUX.2-klein-base-4B
library_name: diffusers
license: apache-2.0
tags:
- flux
- flux-2
- lora
- text-to-image
- mythograph-atelier
---

# Mythograph Atelier FLUX.2 Klein LoRA

This is a style LoRA for Mythograph Atelier, trained on 38 curated public-domain abstract images with synthetic captions.

## Usage

- Training base model: `black-forest-labs/FLUX.2-klein-base-4B`
- Recommended inference model: `black-forest-labs/FLUX.2-klein-4B`
- Trigger word: `MYTHABS1`
- Recommended negative prompt: `text, words, letters, signature, watermark, photorealism`
- Recommended inference steps: `4`
- Recommended guidance scale: start with `1.0`

The LoRA is intended to make FLUX render Mythograph Atelier prompts in a soft, abstract, worn, emotionally legible visual language.

## Training Summary

- Dataset: `flux_lora_dataset`, prepared as image/text-caption pairs
- Images used: 38
- Training steps: 1,800
- LoRA rank/alpha: linear 128/64, conv 64/32
- Optimizer: AdamW 8-bit
- Precision: bf16
- Training-time transformer quantization was enabled only to reduce GPU memory use; the uploaded artifact is the LoRA checkpoint, not GGUF and not a quantized model export.
"""


app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "huggingface_hub",
    "hf_transfer",
)


@app.function(
    image=image,
    volumes={"/workspace": volume},
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
    timeout=30 * 60,
)
def upload(repo_id: str, checkpoint_path_in_repo: str) -> str:
    import os
    from pathlib import Path

    from huggingface_hub import HfApi, create_repo

    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(f"Modal secret {HF_SECRET_NAME!r} did not provide HF_TOKEN")

    checkpoint = Path(CHECKPOINT_PATH)
    output_dir = Path(OUTPUT_DIR)
    samples_dir = output_dir / "samples"
    config_path = output_dir / "config.yaml"

    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    create_repo(repo_id, repo_type="model", exist_ok=True, token=token)
    api = HfApi(token=token)

    api.upload_file(
        path_or_fileobj=str(checkpoint),
        path_in_repo=checkpoint_path_in_repo,
        repo_id=repo_id,
        repo_type="model",
    )
    api.upload_file(
        path_or_fileobj=BytesIO(MODEL_CARD.encode("utf-8")),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
    )
    if config_path.exists():
        api.upload_file(
            path_or_fileobj=str(config_path),
            path_in_repo="training/config.yaml",
            repo_id=repo_id,
            repo_type="model",
        )

    if samples_dir.exists():
        for sample in sorted(samples_dir.glob("*__000001800_*.jpg")):
            api.upload_file(
                path_or_fileobj=str(sample),
                path_in_repo=f"samples/final/{sample.name}",
                repo_id=repo_id,
                repo_type="model",
            )

    return f"https://huggingface.co/{repo_id}"


@app.local_entrypoint()
def main(
    repo_id: str = DEFAULT_REPO_ID,
    checkpoint_path_in_repo: str = "mythograph_flux2_klein_lora_final_1800.safetensors",
) -> None:
    url = upload.remote(repo_id, checkpoint_path_in_repo)
    print(f"Uploaded FLUX LoRA to {url}")
