#!/usr/bin/env python3
"""Upload a local dataset folder to a Modal Volume."""

from __future__ import annotations

import argparse
from pathlib import Path

import modal


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload a dataset directory to a Modal Volume.")
    parser.add_argument("--local-dataset-dir", default="flux_lora_dataset_prepared")
    parser.add_argument("--volume-name", default="mythograph-flux-lora")
    parser.add_argument("--remote-dir", default="/datasets/flux_lora_dataset")
    args = parser.parse_args()

    local_dir = Path(args.local_dataset_dir)
    if not local_dir.exists():
        raise FileNotFoundError(local_dir)

    volume = modal.Volume.from_name(args.volume_name, create_if_missing=True)
    with volume.batch_upload() as batch:
        batch.put_directory(str(local_dir), args.remote_dir)

    print(f"Uploaded {local_dir} to Modal volume {args.volume_name}:{args.remote_dir}")


if __name__ == "__main__":
    main()
