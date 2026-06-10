# Mythograph Nemotron Fine-Tuning Results Report

Date: 2026-06-10  
Project: Mythograph Atelier synthetic dataset fine-tuning  
Base model: `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16`  
Delivery formats: LoRA adapter and GGUF quantized models  

## Executive Summary

This run fine-tuned NVIDIA Nemotron 3 Nano 4B for the Mythograph Atelier assistant behavior: warm conversational art guidance, strict JSON responses, final abstract art recipes, and conversation starter generation.

The fine-tuning completed successfully on Modal using the new dataset in `data/train.jsonl` and `data/validation.jsonl`. The LoRA adapter was uploaded to Hugging Face, then merged into the BF16 base model on Modal, converted to GGUF, quantized to Q4_K_M and Q5_K_M, and uploaded to Hugging Face.

Final training metrics were healthy for this type of supervised fine-tuning run:

- Final train loss: `2.8923`
- Final eval loss: `0.2711`
- Final eval mean token accuracy: `0.9209`
- Training runtime: about `29m 46s`
- Total optimizer steps: `711`

The low validation loss and high token accuracy indicate that the model learned the structure and style of the validation examples well. These metrics do not, by themselves, prove real-world quality, strict JSON reliability under every prompt, or creative usefulness. Those should be confirmed with task-level evaluation and manual testing.

## Hugging Face Outputs

LoRA adapter:

https://huggingface.co/kasimakpinar/mythograph-nemotron-3-nano-4b-lora

Verified files:

- `adapter_model.safetensors`
- `adapter_config.json`
- `tokenizer.json`
- `tokenizer_config.json`
- `special_tokens_map.json`
- `chat_template.jinja`
- `training_args.bin`
- `README.md`

GGUF model:

https://huggingface.co/kasimakpinar/mythograph-nemotron-3-nano-4b-gguf

Verified files:

- `mythograph-nemotron-Q4_K_M.gguf`
- `mythograph-nemotron-Q5_K_M.gguf`
- `README.md`

Recommended runtime file:

- Use `mythograph-nemotron-Q5_K_M.gguf` when quality matters and memory allows.
- Use `mythograph-nemotron-Q4_K_M.gguf` when memory or speed is more important.

## GitHub Changes

Repository:

https://github.com/kasimakpinar/mythograph-nemotron-ft

Commits pushed to `main`:

- `848c220` - `Update synthetic training dataset`
- `fb8aaa0` - `Accept conversation starters in Modal workflow`

The second commit was needed because the new dataset includes a third task type, `conversation_starters`. The previous Modal workflow only counted `conversation_turn` and `final_art_recipe`, so the validator was updated to accept all three schemas.

## Dataset Summary

Source files:

- `data/train.jsonl`
- `data/validation.jsonl`

Total examples:

- Training: `1,890`
- Validation: `210`
- Combined: `2,100`

Task distribution:

| Split | conversation_turn | final_art_recipe | conversation_starters | Total |
| --- | ---: | ---: | ---: | ---: |
| Train | 1,321 | 380 | 189 | 1,890 |
| Validation | 149 | 40 | 21 | 210 |
| Total | 1,470 | 420 | 210 | 2,100 |

Each row is a chat-format JSONL example with:

- one `system` message,
- one `user` message,
- one `assistant` message.

The assistant message content is itself strict JSON for one of the supported task schemas.

## Target Behavior

The model was fine-tuned to support three Mythograph Atelier tasks:

1. `conversation_turn`

   The assistant responds conversationally, reflects the user's emotional/artistic theme, asks one useful next question, and optionally provides one dynamic UI control.

2. `final_art_recipe`

   The assistant produces a structured abstract-art recipe with title, main idea, visual style, palette, symbols, composition, image prompt, negative prompt, and a plain-language explanation.

3. `conversation_starters`

   The assistant produces starter prompts for users who need help beginning the creative conversation.

The key behavioral goal is not generic chat ability. The target is a narrow product behavior: warm, personal, schema-constrained responses for an abstract art companion.

## Training Configuration

Modal app:

- App name: `mythograph-nemotron-ft`
- Modal volume: `mythograph-nemotron-ft-cache`
- Secret used: `huggingface` with `HF_TOKEN`

Training job:

- Modal function call ID: `fc-01KTRY3ZKSC78BFG7GMTBAC4R4`
- GPU: `A100`
- CPU: `8`
- Memory: `65,536 MB`
- Timeout: `6 hours`

Model and libraries:

- Base model: `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16`
- Training approach: LoRA supervised fine-tuning with TRL `SFTTrainer`
- Precision: BF16
- Tokenization: model chat template via `tokenizer.apply_chat_template`
- Packing: disabled
- Gradient checkpointing: enabled
- Reporting: disabled, `report_to="none"`

Hyperparameters:

| Setting | Value |
| --- | --- |
| Epochs | `3.0` |
| Learning rate | `1e-4` |
| Max sequence length | `4096` |
| Per-device train batch size | `1` |
| Per-device eval batch size | `1` |
| Gradient accumulation steps | `8` |
| Save steps | `100` |
| Eval steps | `100` |
| LoRA rank | `16` |
| LoRA alpha | `32` |
| LoRA dropout | `0.05` |
| LoRA target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `up_proj`, `down_proj` |

Effective batch size:

- Per-device train batch size `1` x gradient accumulation `8` = effective batch size of `8` examples per optimizer step.

## Training Results

Final training summary:

| Metric | Value |
| --- | ---: |
| Train runtime | `1785.6338s` |
| Train samples/sec | `3.175` |
| Train steps/sec | `0.398` |
| Train loss | `2.8922907875` |
| Mean token accuracy | `0.9257738590` |
| Epoch | `3.0` |
| Optimizer steps | `711` |
| Tokens processed | `7,439,223` |

Final validation checkpoint observed near the end:

| Metric | Value |
| --- | ---: |
| Eval loss | `0.2711057961` |
| Eval runtime | `15.5345s` |
| Eval samples/sec | `13.518` |
| Eval steps/sec | `13.518` |
| Eval mean token accuracy | `0.9209078712` |
| Eval tokens | `7,331,112` |
| Epoch | `2.96` |

Training progression highlights:

| Step | Epoch | Loss | Mean Token Accuracy |
| ---: | ---: | ---: | ---: |
| 20 | 0.08 | `14.2874` | `0.6142` |
| 30 | 0.13 | `9.1205` | `0.7394` |
| 40 | 0.17 | `5.3550` | `0.8468` |
| 50 | 0.21 | `3.5752` | `0.8940` |
| 80 | 0.34 | `3.1091` | `0.8961` |
| 300 | 1.27 | eval loss `0.2879` | eval `0.9170` |
| 700 | 2.96 | eval loss `0.2711` | eval `0.9209` |
| 710 | 3.00 | `2.1765` | `0.9231` |

Interpretation:

- The model quickly learned the schema and response style, visible in the early drop from loss `14.2874` to `3.5752`.
- Validation loss continued improving from `0.2879` at epoch `1.27` to `0.2711` near epoch `2.96`.
- Validation token accuracy around `0.921` suggests strong next-token agreement with held-out synthetic examples.
- The final training loss is higher than validation loss because logged training loss includes harder/longer mixed examples and is averaged differently than the final validation pass. This is not automatically a problem.

## Merge and GGUF Conversion

Merge job:

- Modal function call ID: `fc-01KTS00DDKZJ2C90BETHFDABQA`
- Input adapter: `kasimakpinar/mythograph-nemotron-3-nano-4b-lora`
- Base model: `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16`
- Output path on Modal volume: `/vol/outputs/mythograph-nemotron-merged`

Verified merged files on Modal volume:

- `model-00001-of-00002.safetensors`
- `model-00002-of-00002.safetensors`
- `model.safetensors.index.json`
- `config.json`
- `generation_config.json`
- tokenizer files
- `chat_template.jinja`

GGUF job:

- Modal function call ID: `fc-01KTS0ESHK49NGA1Z33P93E21C`
- Output path on Modal volume: `/vol/outputs/gguf`
- Conversion tool: `llama.cpp` `convert_hf_to_gguf.py`
- Quantization tool: `llama-quantize`

Generated GGUF files:

- `mythograph-nemotron-F16.gguf`
- `mythograph-nemotron-Q4_K_M.gguf`
- `mythograph-nemotron-Q5_K_M.gguf`

Only Q4_K_M and Q5_K_M were uploaded to the public Hugging Face GGUF repo.

## What This Result Means

This fine-tune should be better than the base model at:

- returning the expected JSON-only format,
- following Mythograph Atelier's three task schemas,
- producing warmer and more personal art-companion language,
- turning emotional themes into abstract visual recipes,
- avoiding generic prompt-builder behavior,
- creating structured outputs that are easier for the app to parse.

This fine-tune does not guarantee:

- perfect JSON validity under every possible user input,
- better general reasoning than the base model,
- better multilingual performance,
- robust behavior outside the trained Mythograph task frame,
- no hallucinated fields,
- no style drift under long or adversarial prompts.

The model should be evaluated in the app with representative prompts before being treated as production-ready.

## Recommended Next Validation

Run task-level evaluation with `scripts/eval_json.py`:

```bash
python scripts/eval_json.py \
  --backend llama_cpp \
  --repo-id kasimakpinar/mythograph-nemotron-3-nano-4b-gguf \
  --filename mythograph-nemotron-Q5_K_M.gguf \
  --validation-file data/validation.jsonl \
  --out-dir outputs/eval_q5 \
  --temperature 0.2 \
  --max-new-tokens 900
```

Important metrics to review:

- strict JSON valid rate,
- salvage JSON valid rate,
- schema valid rate,
- exact task match rate,
- error examples in `outputs/eval_q5/errors.jsonl`.

Recommended manual tests:

- short emotional theme,
- vague first message,
- user asks for colors directly,
- user provides a memory/object,
- final recipe generation,
- conversation starter generation,
- malformed or off-topic input,
- very long personal description.

## Likely Questions and Prepared Answers

### What model did we fine-tune?

We fine-tuned `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` using LoRA supervised fine-tuning. The final user-facing outputs were uploaded as a Hugging Face LoRA adapter and as GGUF quantized files.

### Did we fine-tune the GGUF model directly?

No. The trainable base was the BF16 Hugging Face model. After training, the LoRA adapter was merged into the BF16 model and then converted/quantized to GGUF. GGUF is a deployment format, not the format used for the training step.

### What data was used?

The run used `2,100` synthetic chat-format examples: `1,890` training rows and `210` validation rows. The examples cover `conversation_turn`, `final_art_recipe`, and `conversation_starters`.

### Why is there both a LoRA repo and a GGUF repo?

The LoRA repo contains the small adapter produced by training. It is useful for further merging, continued training, or transformer-based inference. The GGUF repo contains quantized deployment files for llama.cpp or llama-cpp-python.

### Which GGUF file should be used?

Use `mythograph-nemotron-Q5_K_M.gguf` for better quality when memory allows. Use `mythograph-nemotron-Q4_K_M.gguf` when lower memory use or faster loading is more important.

### Are the metrics good?

They are promising. The validation loss reached `0.2711`, and validation mean token accuracy reached about `92.1%`. That indicates the model learned the validation distribution well. However, the real test is JSON/schema reliability and qualitative behavior in the app.

### Does low validation loss mean the model is production-ready?

Not by itself. It means the model predicts held-out synthetic examples well. Production readiness should also include strict JSON evaluation, app-level testing, and manual review of generated responses.

### Was there any sign of overfitting?

There was no obvious metric warning during the observed run: validation loss improved from `0.2879` around epoch `1.27` to `0.2711` near epoch `2.96`. But because the dataset is synthetic and task-specific, practical overfitting should be checked through prompts that differ from the training patterns.

### What was changed in the training workflow?

The Modal validation helper was updated to recognize `conversation_starters`, because the new dataset includes that task type. Without that patch, the training workflow would reject valid examples from the new dataset.

### Where are the final artifacts?

LoRA adapter:

https://huggingface.co/kasimakpinar/mythograph-nemotron-3-nano-4b-lora

GGUF quantized files:

https://huggingface.co/kasimakpinar/mythograph-nemotron-3-nano-4b-gguf

### What should we do next?

Run `scripts/eval_json.py` against the Q5 GGUF model, inspect schema failures, and test the model manually in the Mythograph Atelier app. If JSON reliability is not high enough, add targeted examples for the failing cases and fine-tune another version.

## Known Caveats

- This report is based on Modal training logs and verified Hugging Face file listings, not a full post-training behavioral benchmark.
- The dataset is synthetic, so it may not fully represent real user phrasing.
- The model was trained for a narrow app behavior; it should not be judged as a general-purpose assistant.
- The GGUF files were uploaded after conversion/quantization, but runtime quality should still be checked in the exact app environment.
- The current report does not include a strict JSON/schema evaluation score for generated outputs. That should be produced as a follow-up evaluation run.

## One-Sentence Summary

We successfully fine-tuned Nemotron 3 Nano 4B on 2,100 Mythograph Atelier synthetic examples, achieved strong validation metrics, uploaded the LoRA adapter and Q4/Q5 GGUF models to Hugging Face, and now need task-level JSON/schema evaluation plus manual app testing to confirm production readiness.
