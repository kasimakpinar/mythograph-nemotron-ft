# FLUX.2 Klein LoRA Fine-Tuning Results Report

Date: 2026-06-13

## Summary

Fine-tuned a Mythograph Atelier style LoRA for `black-forest-labs/FLUX.2-klein-base-4B` using the prepared abstract image dataset in `flux_lora_dataset/`.

Published model:

- Hugging Face repo: https://huggingface.co/kasimakpinar/mythograph-atelier-flux2-klein-lora
- Uploaded checkpoint: `mythograph_flux2_klein_lora_final_1800.safetensors`
- Recommended inference base: `black-forest-labs/FLUX.2-klein-4B`
- Trigger word: `MYTHABS1`

This is a LoRA-only artifact. It was not converted to GGUF and no quantized model export was produced.

## Dataset

- Source folder: `flux_lora_dataset/`
- Prepared training folder: `flux_lora_dataset_prepared/`
- Modal volume path: `mythograph-flux-lora:/datasets/flux_lora_dataset`
- Images: 38
- Captions: 38
- Validation: every image had a matching caption and captions contained the trigger word `MYTHABS1`
- Image preparation: kept all images; resized only images above a 2048 px long edge
- Large images resized: 16

The dataset is small, so the result should be treated as a style LoRA rather than a broad concept model. Small images were kept because the training config uses multiple resolution buckets and the subject matter is abstract style transfer, where discarding several examples would have reduced stylistic coverage.

## Training Setup

- Platform: Modal
- Modal volume: `mythograph-flux-lora`
- Modal app: `mythograph-flux2-klein-lora-train`
- Successful run app id: `ap-tp5OmIe8CZXVrDhgTEPBFV`
- GPU used: A100
- Training entrypoint: `modal_train_flux_lora.py`
- Toolkit: `ostris/ai-toolkit`
- Config: `configs/mythograph_flux2_klein_lora.yaml`

Key training settings:

- Base model: `black-forest-labs/FLUX.2-klein-base-4B`
- Architecture: `flux2_klein_4b`
- Steps: 1,800
- Batch size: 1
- Gradient accumulation: 1
- Precision: bf16
- Optimizer: AdamW 8-bit
- Learning rate: 0.0001
- Noise scheduler: flowmatch
- Train UNet/transformer: true
- Train text encoder: false
- LoRA linear rank/alpha: 128/64
- LoRA conv rank/alpha: 64/32
- Sample interval: every 250 steps
- Save interval: every 250 steps

Training-time transformer quantization was enabled in the ai-toolkit config to reduce GPU memory use. This does not mean the uploaded artifact is a quantized model; the published file is a standard LoRA `.safetensors` checkpoint.

## Training Metrics and Learning Rate

Unlike the Nemotron text fine-tune, this FLUX LoRA run did not produce a held-out validation loss or token accuracy metric. The useful numeric training signal available from the Modal logs is the per-step training loss reported by ai-toolkit, together with the learning-rate value used at each step. For image LoRA training, these loss values are helpful for spotting obvious instability, but visual sample review is still the main quality check.

Learning-rate summary:

| Metric | Value |
| --- | ---: |
| Configured learning rate | `0.0001` |
| Scheduler behavior observed in logs | constant |
| First logged learning rate | `0.0001` |
| Final logged learning rate | `0.0001` |
| Minimum logged learning rate | `0.0001` |
| Maximum logged learning rate | `0.0001` |

The learning rate did not decay or warm down during this run. Every parsed training-progress line reported `lr: 1.0e-04`, so the effective schedule was a flat `1e-4` learning rate for all 1,800 optimizer steps.

Loss summary from parsed Modal logs:

| Metric | Value |
| --- | ---: |
| Parsed optimizer steps | `1,800` |
| First logged loss | `1.0520` |
| Final logged loss | `0.6265` |
| Minimum logged loss | `0.1926` |
| Maximum logged loss | `1.2570` |
| Mean logged loss | `0.8269` |
| Median logged loss | `0.8165` |

Loss by training interval:

| Steps | Parsed Steps | Mean Loss | Median Loss | Min Loss | Max Loss |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0-249 | `250` | `0.8374` | `0.8321` | `0.2043` | `1.2420` |
| 250-499 | `250` | `0.8283` | `0.8224` | `0.2065` | `1.2330` |
| 500-749 | `250` | `0.8408` | `0.8158` | `0.1951` | `1.2430` |
| 750-999 | `250` | `0.8153` | `0.8135` | `0.1984` | `1.2180` |
| 1000-1249 | `250` | `0.8385` | `0.8402` | `0.1940` | `1.2570` |
| 1250-1499 | `250` | `0.8338` | `0.8260` | `0.2095` | `1.2400` |
| 1500-1749 | `250` | `0.7992` | `0.7755` | `0.1926` | `1.2530` |
| 1750-1799 | `50` | `0.8017` | `0.7627` | `0.2512` | `1.2010` |

Selected checkpoint-step losses:

| Step | Learning Rate | Logged Loss |
| ---: | ---: | ---: |
| 0 | `0.0001` | `1.0520` |
| 250 | `0.0001` | `0.8288` |
| 500 | `0.0001` | `0.8761` |
| 750 | `0.0001` | `0.9523` |
| 1000 | `0.0001` | `0.8441` |
| 1250 | `0.0001` | `0.7655` |
| 1500 | `0.0001` | `0.8194` |
| 1750 | `0.0001` | `0.7878` |
| 1799 | `0.0001` | `0.6265` |

Interpretation:

- The flat `1e-4` learning rate was stable for the full run; there were no obvious loss spikes, divergence events, or NaN/inf failures in the parsed progress logs.
- Loss moved within a relatively narrow range for most of training, with the mean dropping from about `0.84` in the first 250 steps to about `0.80` in the last 300 steps.
- The loss curve alone does not prove that the final checkpoint is best. The visual samples showed the late checkpoints had the strongest Mythograph Atelier style, which is why the final 1,800-step LoRA was uploaded.
- If a future run looks over-stylized, the first thing to test is not necessarily a lower learning rate; try inference at lower LoRA strength and compare the 1,250 or 1,500-step checkpoints first.

## Checkpoints

Saved checkpoints on the Modal volume:

- `mythograph_flux2_klein_lora_000000250.safetensors`
- `mythograph_flux2_klein_lora_000000500.safetensors`
- `mythograph_flux2_klein_lora_000000750.safetensors`
- `mythograph_flux2_klein_lora_000001000.safetensors`
- `mythograph_flux2_klein_lora_000001250.safetensors`
- `mythograph_flux2_klein_lora_000001500.safetensors`
- `mythograph_flux2_klein_lora_000001750.safetensors`
- `mythograph_flux2_klein_lora.safetensors`

The final checkpoint was selected for upload as `mythograph_flux2_klein_lora_final_1800.safetensors`.

## Sample Review

Generated samples were reviewed from steps 0, 250, 500, 750, 1000, 1250, 1500, 1750, and 1800.

Observed progression:

- Step 0: base-model output already produced simple abstract compositions for the sample prompts.
- Steps 500-1250: style became more painterly and began matching the dataset language more clearly.
- Steps 1500-1800: strongest Mythograph Atelier visual identity, with heavier abstract shapes, warmer accents, and more decisive texture.

The final samples are strongly stylized and minimal. This is acceptable for the intended abstract style LoRA, but it is worth testing inference prompts with several LoRA strengths. If outputs become too heavy or too simple, lower the LoRA scale before retraining.

Uploaded sample files:

- `samples/final/1781305630025__000001800_0.jpg`
- `samples/final/1781305647394__000001800_1.jpg`
- `samples/final/1781305664788__000001800_2.jpg`
- `samples/final/1781305682191__000001800_3.jpg`

## Hugging Face Upload

Uploaded to:

https://huggingface.co/kasimakpinar/mythograph-atelier-flux2-klein-lora

Verified files:

- `README.md`
- `mythograph_flux2_klein_lora_final_1800.safetensors` - 369,644,448 bytes
- `training/config.yaml`
- four final sample images under `samples/final/`

The local machine did not have Hugging Face authentication available, so upload was performed from Modal using the existing `huggingface` secret with `HF_TOKEN`.

## Recommended Usage

Use this LoRA with `black-forest-labs/FLUX.2-klein-4B`, not with the GGUF/Nemotron workflow.

Suggested prompt pattern:

```text
MYTHABS1. A quiet abstract painting about [emotion or scene], with [simple shapes], [limited colors], [texture], and no text.
```

Suggested negative prompt:

```text
text, words, letters, signature, watermark, photorealism
```

Recommended initial inference settings:

- Steps: 4
- Guidance scale: 1.0
- LoRA strength: start around 0.6-0.8, then increase if the style is too weak

## Notes for Future Runs

- For faster iteration, A100 is appropriate for this project budget and dataset size.
- If the final checkpoint feels over-stylized in real inference, test the 1250 or 1500 checkpoint before changing the dataset.
- If captions are expanded later, keep the trigger word exact: `MYTHABS1`.
- Do not GGUF-convert or quantize FLUX LoRA outputs.
