#!/usr/bin/env python3
"""Modal entrypoint for training the Mythograph FLUX.2 Klein style LoRA."""

from __future__ import annotations

from pathlib import Path

import modal


APP_NAME = "mythograph-flux2-klein-lora-train"
VOLUME_NAME = "mythograph-flux-lora"
HF_SECRET_NAME = "huggingface"

LOCAL_CONFIG = Path("configs/mythograph_flux2_klein_lora.yaml")
REMOTE_CONFIG = "/workspace/configs/mythograph_flux2_klein_lora.yaml"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install("pip>=26.1.2", "wheel", "packaging")
    .pip_install("torch", "torchvision", "torchaudio", index_url="https://download.pytorch.org/whl/cu124")
    .pip_install(
        "accelerate",
        "transformers",
        "diffusers",
        "peft",
        "safetensors",
        "sentencepiece",
        "protobuf",
        "einops",
        "omegaconf",
        "pillow",
        "huggingface_hub",
        "hf_transfer",
    )
    .run_commands(
        "cd /root && git clone https://github.com/ostris/ai-toolkit.git",
        "cd /root/ai-toolkit && git submodule update --init --recursive",
        "cd /root/ai-toolkit && pip install -r requirements.txt",
    )
    .env(
        {
            "HF_HOME": "/workspace/model_cache/huggingface",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
)


@app.function(
    image=image,
    gpu="L4",
    cpu=8,
    memory=32768,
    timeout=2 * 60 * 60,
    volumes={"/workspace": volume},
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
)
def train(config_text: str) -> dict[str, str]:
    import os
    import subprocess
    from pathlib import Path

    dataset_dir = Path("/workspace/datasets/flux_lora_dataset")
    output_dir = Path("/workspace/outputs/mythograph_flux2_klein_lora")
    config_path = Path(REMOTE_CONFIG)

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset not found on Modal volume: {dataset_dir}")

    config_path.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path.write_text(config_text, encoding="utf-8")

    env = os.environ.copy()
    if "HF_TOKEN" in env:
        env.setdefault("HUGGING_FACE_HUB_TOKEN", env["HF_TOKEN"])

    cmd = ["python", "run.py", str(config_path)]
    subprocess.run(cmd, cwd="/root/ai-toolkit", check=True, env=env)

    volume.commit()
    return {
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "config_path": str(config_path),
    }


@app.local_entrypoint()
def main(gpu: str = "L4") -> None:
    config_text = LOCAL_CONFIG.read_text(encoding="utf-8")
    result = train.with_options(gpu=gpu).remote(config_text)
    print(result)
