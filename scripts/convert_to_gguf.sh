#!/usr/bin/env bash
set -euo pipefail

# Convert a merged Hugging Face model folder to F16 GGUF.
# Usage:
#   bash scripts/convert_to_gguf.sh outputs/mythograph-nemotron-merged outputs/gguf/mythograph-nemotron-F16.gguf

MERGED_MODEL_DIR="${1:-outputs/mythograph-nemotron-merged}"
OUTFILE="${2:-outputs/gguf/mythograph-nemotron-F16.gguf}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-third_party/llama.cpp}"

mkdir -p "$(dirname "$OUTFILE")" "$(dirname "$LLAMA_CPP_DIR")"

if [ ! -d "$LLAMA_CPP_DIR" ]; then
  git clone https://github.com/ggml-org/llama.cpp.git "$LLAMA_CPP_DIR"
fi

python -m pip install -r "$LLAMA_CPP_DIR/requirements.txt"

python "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" "$MERGED_MODEL_DIR" \
  --outfile "$OUTFILE" \
  --outtype f16

echo "Created: $OUTFILE"
