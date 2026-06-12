# GGUF Conversion and Quantization

This project fine-tunes the Hugging Face BF16 model first, then converts the merged model to GGUF for llama.cpp inference.

Recommended output files:

- `mythograph-nemotron-Q5_K_M.gguf` - recommended default for JSON reliability.
- `mythograph-nemotron-Q4_K_M.gguf` - smaller/faster llama.cpp-compatible variant.
- `mythograph-nemotron-Q8_0.gguf` - optional reference-quality variant.

## 1. Merge the LoRA Adapter First

You should already have a merged model folder such as:

```bash
outputs/mythograph-nemotron-merged
```

If you need to create it locally:

```bash
python scripts/merge_lora.py \
  --base-model nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 \
  --adapter outputs/mythograph-nemotron-lora \
  --output-dir outputs/mythograph-nemotron-merged
```

The merged folder should contain normal Hugging Face model files, tokenizer files, and config files.

## 2. Convert Merged HF Model to F16 GGUF

```bash
bash scripts/convert_to_gguf.sh \
  outputs/mythograph-nemotron-merged \
  outputs/gguf/mythograph-nemotron-F16.gguf
```

The helper uses `LLAMA_CPP_DIR=third_party/llama.cpp` by default. Override it if you already have llama.cpp cloned elsewhere:

```bash
LLAMA_CPP_DIR=/path/to/llama.cpp bash scripts/convert_to_gguf.sh \
  outputs/mythograph-nemotron-merged \
  outputs/gguf/mythograph-nemotron-F16.gguf
```

## 3. Quantize to Q4_K_M and Q5_K_M

```bash
bash scripts/quantize_gguf.sh \
  outputs/gguf/mythograph-nemotron-F16.gguf \
  outputs/gguf \
  mythograph-nemotron
```

This creates:

```text
outputs/gguf/mythograph-nemotron-Q4_K_M.gguf
outputs/gguf/mythograph-nemotron-Q5_K_M.gguf
```

## 4. Evaluate Each Quantization

```bash
python scripts/eval_json.py \
  --backend llama_cpp \
  --gguf outputs/gguf/mythograph-nemotron-Q5_K_M.gguf \
  --validation-file nemotron_dataset/validation.jsonl \
  --out-dir outputs/eval_q5 \
  --temperature 0.2 \
  --max-new-tokens 900

python scripts/eval_json.py \
  --backend llama_cpp \
  --gguf outputs/gguf/mythograph-nemotron-Q4_K_M.gguf \
  --validation-file nemotron_dataset/validation.jsonl \
  --out-dir outputs/eval_q4 \
  --temperature 0.2 \
  --max-new-tokens 900
```

Compare these files:

```text
outputs/eval_q5/eval_report.json
outputs/eval_q4/eval_report.json
outputs/eval_q5/errors.jsonl
outputs/eval_q4/errors.jsonl
```

Choose the default Space model based on `schema_valid_rate`, not only size or speed. Use `Q5_K_M` by default unless `Q4_K_M` is nearly identical on schema validity.

## 5. Upload GGUF Files

```bash
python scripts/upload_gguf.py \
  --repo-id build-small-hackathon/mythograph-nemotron-3-nano-4b-gguf \
  --folder outputs/gguf
```

If `outputs/gguf/README.md` does not exist, the upload script writes a default model card with `llama.cpp` metadata, the recommended Q5_K_M file, the lightweight Q4_K_M file, and the NVIDIA Nemotron Open Model License note.

## 6. Run the Space App Locally

```bash
MODEL_PATH=outputs/gguf/mythograph-nemotron-Q5_K_M.gguf python app.py
```

Or from Hugging Face Hub:

```bash
MODEL_REPO=build-small-hackathon/mythograph-nemotron-3-nano-4b-gguf \
MODEL_FILE=mythograph-nemotron-Q5_K_M.gguf \
python app.py
```
