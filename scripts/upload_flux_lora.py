#!/usr/bin/env python3
"""Upload a selected FLUX LoRA checkpoint and metadata to Hugging Face."""

from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path

from huggingface_hub import HfApi, create_repo


DEFAULT_MODEL_CARD = """---
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

This is a style LoRA for Mythograph Atelier, trained on curated abstract public-domain images with short visual captions.

## Usage

- Training base model: `black-forest-labs/FLUX.2-klein-base-4B`
- Recommended inference model: `black-forest-labs/FLUX.2-klein-4B`
- Trigger word: `MYTHABS1`
- Recommended negative prompt: `text, words, letters, signature, watermark, photorealism`
- Recommended inference steps: `4`
- Recommended guidance scale: start with `1.0`

The LoRA is intended to make FLUX render Mythograph Atelier prompts in a soft, abstract, worn, emotionally legible visual language.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload a selected FLUX LoRA checkpoint to Hugging Face.")
    parser.add_argument("--checkpoint", required=True, help="Path to the selected .safetensors LoRA checkpoint.")
    parser.add_argument("--repo-id", default="kasimakpinar/mythograph-atelier-flux2-klein-lora")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--readme", default=None, help="Optional README/model card to upload.")
    parser.add_argument("--samples-dir", default=None, help="Optional directory of selected sample images.")
    parser.add_argument("--path-in-repo", default=None, help="Optional checkpoint filename in the repo.")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    if checkpoint.suffix != ".safetensors":
        raise ValueError(f"Expected a .safetensors checkpoint, got: {checkpoint}")

    create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    api = HfApi()

    checkpoint_name = args.path_in_repo or checkpoint.name
    api.upload_file(
        path_or_fileobj=str(checkpoint),
        path_in_repo=checkpoint_name,
        repo_id=args.repo_id,
        repo_type="model",
    )

    if args.readme:
        readme = Path(args.readme)
        if not readme.exists():
            raise FileNotFoundError(readme)
        api.upload_file(path_or_fileobj=str(readme), path_in_repo="README.md", repo_id=args.repo_id, repo_type="model")
    else:
        api.upload_file(
            path_or_fileobj=BytesIO(DEFAULT_MODEL_CARD.encode("utf-8")),
            path_in_repo="README.md",
            repo_id=args.repo_id,
            repo_type="model",
        )

    if args.samples_dir:
        samples_dir = Path(args.samples_dir)
        if not samples_dir.exists():
            raise FileNotFoundError(samples_dir)
        for path in sorted(samples_dir.iterdir()):
            if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
                api.upload_file(
                    path_or_fileobj=str(path),
                    path_in_repo=f"samples/{path.name}",
                    repo_id=args.repo_id,
                    repo_type="model",
                )

    print(f"Uploaded {checkpoint_name} to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
