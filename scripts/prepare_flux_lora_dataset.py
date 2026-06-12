#!/usr/bin/env python3
"""Create a resized training copy of the FLUX LoRA dataset."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from PIL import Image


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def resize_image(source: Path, target: Path, max_long_edge: int, jpeg_quality: int) -> dict[str, object]:
    with Image.open(source) as image:
        image = image.convert("RGB")
        original_size = image.size
        long_edge = max(original_size)
        if long_edge > max_long_edge:
            scale = max_long_edge / long_edge
            new_size = (max(1, round(original_size[0] * scale)), max(1, round(original_size[1] * scale)))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        else:
            new_size = original_size

        target = target.with_suffix(".jpg")
        image.save(target, format="JPEG", quality=jpeg_quality, optimize=True)

    return {
        "source": source.name,
        "target": target.name,
        "original_size": list(original_size),
        "prepared_size": list(new_size),
        "resized": original_size != new_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a normalized FLUX LoRA dataset copy.")
    parser.add_argument("--source-dir", default="flux_lora_dataset")
    parser.add_argument("--output-dir", default="flux_lora_dataset_prepared")
    parser.add_argument("--max-long-edge", type=int, default=2048)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument(
        "--skip-below-long-edge",
        type=int,
        default=0,
        help="Skip images whose long edge is below this size. Default keeps all images.",
    )
    args = parser.parse_args()

    source_root = Path(args.source_dir)
    output_root = Path(args.output_dir)
    if not source_root.exists():
        raise FileNotFoundError(source_root)

    output_root.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []

    for image_path in sorted(p for p in source_root.iterdir() if p.suffix.lower() in IMAGE_EXTS):
        caption_path = source_root / f"{image_path.stem}.txt"
        if not caption_path.exists():
            raise FileNotFoundError(f"Missing caption for {image_path.name}: {caption_path}")

        with Image.open(image_path) as probe:
            width, height = probe.size
        if args.skip_below_long_edge and max(width, height) < args.skip_below_long_edge:
            skipped.append({"source": image_path.name, "size": [width, height], "reason": "below_long_edge_threshold"})
            continue

        target_image = output_root / f"{image_path.stem}.jpg"
        record = resize_image(image_path, target_image, args.max_long_edge, args.jpeg_quality)
        shutil.copy2(caption_path, output_root / f"{image_path.stem}.txt")
        manifest.append(record)

    review_path = source_root / "captions_review.jsonl"
    if review_path.exists():
        shutil.copy2(review_path, output_root / "captions_review.jsonl")

    (output_root / "prepared_manifest.json").write_text(
        json.dumps(
            {
                "source_dir": str(source_root),
                "output_dir": str(output_root),
                "max_long_edge": args.max_long_edge,
                "jpeg_quality": args.jpeg_quality,
                "images": manifest,
                "skipped": skipped,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print("Prepared FLUX LoRA dataset")
    print(f"Source: {source_root}")
    print(f"Output: {output_root}")
    print(f"Images written: {len(manifest)}")
    print(f"Images skipped: {len(skipped)}")
    print(f"Resized: {sum(1 for item in manifest if item['resized'])}")


if __name__ == "__main__":
    main()
