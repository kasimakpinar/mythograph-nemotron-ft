#!/usr/bin/env bash
set -euo pipefail

# Quantize an F16 GGUF to Q4_K_M and Q5_K_M by default.
# Usage:
#   bash scripts/quantize_gguf.sh outputs/gguf/mythograph-nemotron-F16.gguf outputs/gguf mythograph-nemotron

F16_GGUF="${1:-outputs/gguf/mythograph-nemotron-F16.gguf}"
OUT_DIR="${2:-outputs/gguf}"
PREFIX="${3:-mythograph-nemotron}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-third_party/llama.cpp}"

mkdir -p "$OUT_DIR" "$(dirname "$LLAMA_CPP_DIR")"

if [ ! -d "$LLAMA_CPP_DIR" ]; then
  git clone https://github.com/ggml-org/llama.cpp.git "$LLAMA_CPP_DIR"
fi

cmake -S "$LLAMA_CPP_DIR" -B "$LLAMA_CPP_DIR/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$LLAMA_CPP_DIR/build" --config Release -j --target llama-quantize llama-cli llama-server

for candidate in \
  "$LLAMA_CPP_DIR/build/bin/llama-quantize" \
  "$LLAMA_CPP_DIR/build/bin/llama-quantize.exe" \
  "$LLAMA_CPP_DIR/build/bin/Release/llama-quantize.exe" \
  "$LLAMA_CPP_DIR/build/bin/quantize" \
  "$LLAMA_CPP_DIR/build/bin/quantize.exe" \
  "$LLAMA_CPP_DIR/build/bin/Release/quantize.exe"
do
  if [ -x "$candidate" ]; then
    QUANTIZE_BIN="$candidate"
    break
  fi
done

if [ -z "${QUANTIZE_BIN:-}" ]; then
  echo "Could not find llama.cpp quantize binary under $LLAMA_CPP_DIR/build" >&2
  exit 1
fi

"$QUANTIZE_BIN" "$F16_GGUF" "$OUT_DIR/${PREFIX}-Q4_K_M.gguf" Q4_K_M
"$QUANTIZE_BIN" "$F16_GGUF" "$OUT_DIR/${PREFIX}-Q5_K_M.gguf" Q5_K_M

# Optional reference quality quant. Uncomment if you want it.
# "$QUANTIZE_BIN" "$F16_GGUF" "$OUT_DIR/${PREFIX}-Q8_0.gguf" Q8_0

echo "Created quantized files in: $OUT_DIR"
