#!/usr/bin/env python3
"""Validate a folder-style image LoRA dataset with sidecar captions."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
BANNED_TERMS = [
    "abstract expressionism",
    "abstract expressionist",
    "minimalism",
    "minimalist",
    "bauhaus",
    "cubism",
    "cubist",
    "kandinsky",
    "rothko",
    "pollock",
    "modern art style",
    "painting style",
    "digital art style",
]


def normalize_caption(caption: str, trigger: str) -> str:
    caption = caption.strip()
    pattern = rf"^({re.escape(trigger)}[\.\:\-\s]*)+"
    caption = re.sub(pattern, "", caption, flags=re.IGNORECASE).strip()
    if caption:
        caption = caption[0].upper() + caption[1:]
    return f"{trigger}. {caption}"


def matching_image(root: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTS:
        candidate = root / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate image/caption pairs for FLUX LoRA training.")
    parser.add_argument("--dataset-dir", default="flux_lora_dataset")
    parser.add_argument("--trigger", default="MYTHABS1")
    parser.add_argument("--fix", action="store_true", help="Normalize duplicate/malformed trigger prefixes in captions.")
    parser.add_argument("--min-words", type=int, default=20)
    parser.add_argument("--max-words", type=int, default=55)
    args = parser.parse_args()

    root = Path(args.dataset_dir)
    if not root.exists():
        raise FileNotFoundError(root)

    images = {p.stem: p for p in root.iterdir() if p.suffix.lower() in IMAGE_EXTS}
    txts = {p.stem: p for p in root.glob("*.txt")}

    errors: list[str] = []
    warnings: list[str] = []

    for stem, image in sorted(images.items()):
        if stem not in txts:
            errors.append(f"Missing caption for image: {image.name}")

    for stem, txt in sorted(txts.items()):
        if stem not in images and matching_image(root, stem) is None:
            warnings.append(f"Caption without image: {txt.name}")

        caption = txt.read_text(encoding="utf-8").strip()
        normalized = normalize_caption(caption, args.trigger)
        trigger_count = len(re.findall(rf"\b{re.escape(args.trigger)}\b", caption, flags=re.IGNORECASE))

        if caption != normalized or trigger_count != 1:
            msg = f"Needs trigger normalization: {txt.name}"
            if args.fix:
                txt.write_text(normalized + "\n", encoding="utf-8")
                warnings.append(msg + " - fixed")
                caption = normalized
            else:
                warnings.append(msg)

        if not caption.startswith(f"{args.trigger}."):
            errors.append(f"Caption must start with `{args.trigger}.`: {txt.name}")

        lower_caption = caption.lower()
        for term in BANNED_TERMS:
            if term in lower_caption:
                warnings.append(f"Banned/style term `{term}` found in {txt.name}")

        word_count = len(caption.split())
        if word_count < args.min_words:
            warnings.append(f"Caption may be too short ({word_count} words): {txt.name}")
        if word_count > args.max_words:
            warnings.append(f"Caption may be too long ({word_count} words): {txt.name}")

    print("Dataset validation summary")
    print(f"Dataset: {root}")
    print(f"Images: {len(images)}")
    print(f"Captions: {len(txts)}")
    print(f"Errors: {len(errors)}")
    print(f"Warnings: {len(warnings)}")

    if errors:
        print("\nErrors:")
        for error in errors:
            print(f"- {error}")

    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"- {warning}")

    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
