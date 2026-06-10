# Mythograph Nemotron Fine-Tuning Project

This folder contains the data, QLoRA fine-tuning scripts, JSON/schema evaluation, GGUF conversion helpers, upload tooling, and a Gradio demo for the fine-tuned Mythograph Atelier model.

## Project Files

```text
app.py                        # Gradio demo using llama-cpp-python
requirements.txt              # training/eval/app dependencies
reports/                      # fine-tuning result reports and run summaries
README_QUANTIZATION.md        # conversion and quantization instructions
data/train.jsonl              # source training data, chat JSONL
data/validation.jsonl         # source validation data, chat JSONL
scripts/prepare_dataset.py    # validate source chat JSONL without changing it
scripts/train.py              # QLoRA SFT with TRL/PEFT/bitsandbytes
scripts/merge_lora.py         # merge LoRA adapter into the BF16 base model
scripts/eval_json.py          # strict JSON/schema evaluator
scripts/convert_to_gguf.sh    # merged HF model -> F16 GGUF
scripts/quantize_gguf.sh      # F16 GGUF -> Q4_K_M and Q5_K_M
scripts/upload_gguf.py        # upload GGUF files to Hugging Face
```

## Latest Results

The latest fine-tuning run is summarized in:

- [reports/2026-06-10-finetuning-results-report.md](reports/2026-06-10-finetuning-results-report.md)

Published Hugging Face artifacts:

- LoRA adapter: https://huggingface.co/kasimakpinar/mythograph-nemotron-3-nano-4b-lora
- GGUF Q4/Q5 model: https://huggingface.co/kasimakpinar/mythograph-nemotron-3-nano-4b-gguf

## Validate the Data

The source JSONL files are chat-style examples and are not rewritten by default.

```bash
python scripts/prepare_dataset.py \
  --train-file data/train.jsonl \
  --validation-file data/validation.jsonl
```

## Train the LoRA Adapter

```bash
python scripts/train.py \
  --base-model nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 \
  --train-file data/train.jsonl \
  --validation-file data/validation.jsonl \
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
  --validation-file data/validation.jsonl \
  --out-dir outputs/eval_merged \
  --temperature 0.2 \
  --max-new-tokens 900
```

## Evaluate a Local GGUF Model

```bash
python scripts/eval_json.py \
  --backend llama_cpp \
  --gguf outputs/gguf/mythograph-nemotron-Q5_K_M.gguf \
  --validation-file data/validation.jsonl \
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
