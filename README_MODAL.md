# Modal Workflow

Use Modal for the GPU-heavy parts of this project: QLoRA fine-tuning, LoRA merge, GGUF conversion, quantization, and GGUF upload.

## 1. Authenticate Modal

Install the Modal client locally and authenticate:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install modal
python -m modal token new
```

The token flow opens a browser and writes a Modal profile to your local machine.

## 2. Add the Hugging Face Secret

Remote Modal jobs need a Hugging Face token with write access to the target organization.

Create a Modal secret named `huggingface` with a key named `HF_TOKEN`:

```powershell
python -m modal secret create huggingface HF_TOKEN=hf_your_token_here
```

You can also create this secret in the Modal dashboard. The important details are:

- Secret name: `huggingface`
- Key: `HF_TOKEN`
- Value: a Hugging Face write token

## 3. Train on Modal

```powershell
python -m modal run scripts/modal_workflow.py `
  --action train `
  --gpu A100 `
  --adapter-repo build-small-hackathon/mythograph-nemotron-3-nano-4b-lora
```

The training job writes checkpoints and the final adapter to the Modal volume `mythograph-nemotron-ft-cache`, then pushes the adapter to Hugging Face.

## 4. Merge the Adapter

```powershell
python -m modal run scripts/modal_workflow.py `
  --action merge `
  --gpu A100 `
  --adapter-repo build-small-hackathon/mythograph-nemotron-3-nano-4b-lora `
  --merged-repo build-small-hackathon/mythograph-nemotron-3-nano-4b-merged
```

The merged model is saved in the Modal volume and optionally pushed to Hugging Face.

## 5. Build and Upload GGUF

```powershell
python -m modal run scripts/modal_workflow.py `
  --action gguf `
  --gguf-repo build-small-hackathon/mythograph-nemotron-3-nano-4b-gguf
```

This converts the merged model to F16 GGUF, quantizes `Q4_K_M` and `Q5_K_M`, and uploads them with a model card.

## 6. One-Shot Flow

After the Hugging Face secret is configured, you can run train, merge, and GGUF upload in one command:

```powershell
python -m modal run scripts/modal_workflow.py `
  --action all `
  --gpu A100 `
  --adapter-repo build-small-hackathon/mythograph-nemotron-3-nano-4b-lora `
  --merged-repo build-small-hackathon/mythograph-nemotron-3-nano-4b-merged `
  --gguf-repo build-small-hackathon/mythograph-nemotron-3-nano-4b-gguf
```

For the first run, prefer separate `train`, `merge`, and `gguf` commands so failures are easier to inspect.
