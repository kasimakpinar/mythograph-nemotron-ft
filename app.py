import json
import os
from typing import Any, Dict, List

import gradio as gr
from llama_cpp import Llama

MODEL_REPO = os.getenv("MODEL_REPO", "build-small-hackathon/mythograph-nemotron-3-nano-4b-gguf")
MODEL_FILE = os.getenv("MODEL_FILE", "mythograph-nemotron-Q5_K_M.gguf")
MODEL_PATH = os.getenv("MODEL_PATH")  # Optional local path. If set, overrides MODEL_REPO/MODEL_FILE.

N_CTX = int(os.getenv("N_CTX", "4096"))
N_THREADS = int(os.getenv("N_THREADS", str(os.cpu_count() or 4)))
N_GPU_LAYERS = int(os.getenv("N_GPU_LAYERS", "0"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
TOP_P = float(os.getenv("TOP_P", "0.9"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "900"))

SYSTEM_PROMPT = """You are Mythograph Atelier, a creative interview director for an abstract image app.

Return valid JSON only.
Do not use markdown.
Do not include explanations outside JSON.
For conversation turns, return exactly one control.
For final art recipes, include all required recipe fields.
"""

EXAMPLE_PAYLOAD = """{
  "task": "conversation_turn",
  "user_state": {
    "seed_text": "I feel like a lighthouse in fog",
    "answers_so_far": []
  }
}"""


def load_llm() -> Llama:
    common_kwargs = {
        "n_ctx": N_CTX,
        "n_threads": N_THREADS,
        "n_gpu_layers": N_GPU_LAYERS,
        "verbose": False,
    }
    if MODEL_PATH:
        return Llama(model_path=MODEL_PATH, **common_kwargs)
    return Llama.from_pretrained(repo_id=MODEL_REPO, filename=MODEL_FILE, **common_kwargs)


llm = load_llm()


def pretty_or_raw(text: str) -> str:
    try:
        parsed = json.loads(text)
        return json.dumps(parsed, ensure_ascii=False, indent=2)
    except Exception:
        return text


def generate(user_payload: str, temperature: float, max_tokens: int) -> str:
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_payload.strip()},
    ]

    result: Dict[str, Any] = llm.create_chat_completion(
        messages=messages,
        temperature=temperature,
        top_p=TOP_P,
        max_tokens=max_tokens,
    )

    text = result["choices"][0]["message"]["content"].strip()
    return pretty_or_raw(text)


with gr.Blocks(title="Mythograph Atelier - Nemotron GGUF") as demo:
    gr.Markdown(
        """
# Mythograph Atelier - Nemotron 3 Nano GGUF

Fine-tuned NVIDIA Nemotron 3 Nano 4B running through **llama.cpp / llama-cpp-python**.
The model is trained to produce strict JSON for adaptive creative UI turns and final image recipes.
"""
    )

    with gr.Row():
        with gr.Column(scale=1):
            user_payload = gr.Textbox(
                label="Input payload",
                value=EXAMPLE_PAYLOAD,
                lines=18,
            )
            temperature = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=TEMPERATURE,
                step=0.05,
                label="Temperature",
            )
            max_tokens = gr.Slider(
                minimum=128,
                maximum=1600,
                value=MAX_TOKENS,
                step=64,
                label="Max tokens",
            )
            button = gr.Button("Generate JSON", variant="primary")

        with gr.Column(scale=1):
            output = gr.Code(label="Model output", language="json", lines=24)

    button.click(generate, inputs=[user_payload, temperature, max_tokens], outputs=output)


if __name__ == "__main__":
    demo.launch()
