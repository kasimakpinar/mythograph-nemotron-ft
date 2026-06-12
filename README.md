# Mythograph Model Fine-Tuning Project

This folder contains Mythograph model fine-tuning workflows: Nemotron text/JSON fine-tuning, FLUX.2 Klein image style LoRA fine-tuning, evaluation helpers, Modal cloud jobs, upload tooling, and a Gradio demo for the fine-tuned Mythograph Atelier model.

## Project Files

```text
app.py                        # Gradio demo using llama-cpp-python
requirements.txt              # training/eval/app dependencies
reports/                      # fine-tuning result reports and run summaries
README_QUANTIZATION.md        # conversion and quantization instructions
nemotron_dataset/train.jsonl  # source Nemotron training data, chat JSONL
nemotron_dataset/validation.jsonl # source Nemotron validation data, chat JSONL
flux_lora_dataset/            # source FLUX style LoRA images and captions
scripts/prepare_dataset.py    # validate source chat JSONL without changing it
scripts/train.py              # QLoRA SFT with TRL/PEFT/bitsandbytes
scripts/merge_lora.py         # merge LoRA adapter into the BF16 base model
scripts/eval_json.py          # strict JSON/schema evaluator
scripts/convert_to_gguf.sh    # merged HF model -> F16 GGUF
scripts/quantize_gguf.sh      # F16 GGUF -> Q4_K_M and Q5_K_M
scripts/upload_gguf.py        # upload GGUF files to Hugging Face
scripts/validate_lora_dataset.py # validate FLUX image/caption pairs
scripts/prepare_flux_lora_dataset.py # create resized FLUX training copy
scripts/upload_dataset_to_modal.py # upload prepared FLUX dataset to Modal
scripts/upload_flux_lora.py  # upload selected FLUX LoRA checkpoint to Hugging Face
modal_train_flux_lora.py      # Modal FLUX.2 Klein LoRA training entrypoint
modal_upload_flux_lora.py     # upload trained FLUX LoRA from Modal volume to Hugging Face
configs/mythograph_flux2_klein_lora.yaml # ai-toolkit FLUX LoRA config
```

## Latest Results

The latest fine-tuning runs are summarized in:

- [reports/2026-06-13-flux2-klein-lora-results-report.md](reports/2026-06-13-flux2-klein-lora-results-report.md)
- [reports/2026-06-10-finetuning-results-report.md](reports/2026-06-10-finetuning-results-report.md)

Published Hugging Face artifacts:

- FLUX.2 Klein style LoRA: https://huggingface.co/kasimakpinar/mythograph-atelier-flux2-klein-lora
- LoRA adapter: https://huggingface.co/kasimakpinar/mythograph-nemotron-3-nano-4b-lora
- GGUF Q4/Q5 model: https://huggingface.co/kasimakpinar/mythograph-nemotron-3-nano-4b-gguf

## FLUX.2 Klein LoRA Workflow

The FLUX style LoRA dataset lives in `flux_lora_dataset/`. Captions use the trigger word `MYTHABS1`.

Validate the source dataset:

```bash
python scripts/validate_lora_dataset.py \
  --dataset-dir flux_lora_dataset \
  --trigger MYTHABS1
```

Create a resized training copy. This keeps all images, converts them to JPG, and downsizes only images above a 2048 px long edge:

```bash
python scripts/prepare_flux_lora_dataset.py \
  --source-dir flux_lora_dataset \
  --output-dir flux_lora_dataset_prepared \
  --max-long-edge 2048
```

Upload the prepared dataset to Modal:

```bash
python scripts/upload_dataset_to_modal.py \
  --local-dataset-dir flux_lora_dataset_prepared \
  --volume-name mythograph-flux-lora \
  --remote-dir /datasets/flux_lora_dataset
```

Train on Modal:

```bash
python -m modal run --detach modal_train_flux_lora.py --gpu A100
```

For FLUX.2 Klein, train the LoRA on `black-forest-labs/FLUX.2-klein-base-4B`, then use the selected LoRA checkpoint at inference with `black-forest-labs/FLUX.2-klein-4B`. Do not GGUF-convert or quantize this model. Pick the best checkpoint by reviewing generated samples, usually around steps 750-1500 rather than blindly choosing the final checkpoint.

After choosing the best `.safetensors` checkpoint, upload it under the same Hugging Face namespace used for the Nemotron artifacts. If local Hugging Face auth is available, use:

```bash
python scripts/upload_flux_lora.py \
  --checkpoint path/to/best_checkpoint.safetensors \
  --repo-id kasimakpinar/mythograph-atelier-flux2-klein-lora
```

If the checkpoint is already on the Modal volume and `HF_TOKEN` is configured as the Modal `huggingface` secret, upload directly from Modal:

```bash
python -m modal run modal_upload_flux_lora.py
```

## Validate the Data

The source JSONL files are chat-style examples and are not rewritten by default.

```bash
python scripts/prepare_dataset.py \
  --train-file nemotron_dataset/train.jsonl \
  --validation-file nemotron_dataset/validation.jsonl
```

## Train the LoRA Adapter

```bash
python scripts/train.py \
  --base-model nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 \
  --train-file nemotron_dataset/train.jsonl \
  --validation-file nemotron_dataset/validation.jsonl \
  --output-dir outputs/mythograph-nemotron-lora \
  --hub-model-id mythograph-nemotron-3-nano-4b-lora \
  --push-to-hub
```

The defaults use 3 epochs, learning rate `1e-4`, LoRA rank `16`, alpha `32`, dropout `0.05`, max sequence length `4096`, BF16 when CUDA is available, and 4-bit QLoRA. The trainer formats the existing chat messages with the model chat template and uses assistant-only loss when supported by the installed TRL version.

## Merge the LoRA Adapter

```bash
python scripts/merge_lora.py \
  --base-model nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 \
  --adapter outputs/mythograph-nemotron-lora \
  --output-dir outputs/mythograph-nemotron-merged
```

## Evaluate a Merged Transformers Model

```bash
python scripts/eval_json.py \
  --backend transformers \
  --model outputs/mythograph-nemotron-merged \
  --validation-file nemotron_dataset/validation.jsonl \
  --out-dir outputs/eval_merged \
  --temperature 0.2 \
  --max-new-tokens 900
```

## Evaluate a Local GGUF Model

```bash
python scripts/eval_json.py \
  --backend llama_cpp \
  --gguf outputs/gguf/mythograph-nemotron-Q5_K_M.gguf \
  --validation-file nemotron_dataset/validation.jsonl \
  --out-dir outputs/eval_q5 \
  --temperature 0.2 \
  --max-new-tokens 900
```

## Run the llama.cpp App Locally

```bash
MODEL_PATH=outputs/gguf/mythograph-nemotron-Q5_K_M.gguf python app.py
```

Run from Hugging Face Hub:

```bash
MODEL_REPO=mythograph-nemotron-3-nano-4b-gguf \
MODEL_FILE=mythograph-nemotron-Q5_K_M.gguf \
python app.py
```

The Hugging Face Space should use the same `app.py`, `requirements.txt`, and a README that states the demo runs a fine-tuned Nemotron 3 Nano 4B GGUF model through llama.cpp via llama-cpp-python.
