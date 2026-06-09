---
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
