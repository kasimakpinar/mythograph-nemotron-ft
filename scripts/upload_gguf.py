#!/usr/bin/env python3
import argparse
from pathlib import Path
from huggingface_hub import HfApi, create_repo


DEFAULT_README = """---
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
"""


def main():
    parser = argparse.ArgumentParser(description="Upload GGUF files to a Hugging Face model repo.")
    parser.add_argument("--repo-id", required=True, help="Example: build-small-hackathon/mythograph-nemotron-3-nano-4b-gguf")
    parser.add_argument("--folder", default="outputs/gguf")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.exists():
        raise FileNotFoundError(folder)

    readme = folder / "README.md"
    if not readme.exists():
        readme.write_text(DEFAULT_README, encoding="utf-8")

    create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    api = HfApi()

    for path in sorted(folder.glob("*.gguf")):
        print(f"Uploading {path}...")
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=path.name,
            repo_id=args.repo_id,
            repo_type="model",
        )

    api.upload_file(
        path_or_fileobj=str(readme),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="model",
    )

    print("Done.")


if __name__ == "__main__":
    main()
