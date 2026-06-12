#!/usr/bin/env python3
"""
Evaluate Mythograph Atelier fine-tuned models for strict JSON and schema validity.

Supports three backends:
  1. transformers: local/HF model id, optionally with a PEFT adapter
  2. llama_cpp: local GGUF file, or repo_id + filename via Llama.from_pretrained
  3. file: pre-generated predictions JSONL

Examples:
  python scripts/eval_json.py \
    --backend transformers \
    --model outputs/mythograph-nemotron-merged \
    --validation-file nemotron_dataset/validation.jsonl \
    --out-dir outputs/eval_merged

  python scripts/eval_json.py \
    --backend llama_cpp \
    --gguf outputs/gguf/mythograph-nemotron-Q5_K_M.gguf \
    --validation-file nemotron_dataset/validation.jsonl \
    --out-dir outputs/eval_q5

  python scripts/eval_json.py \
    --backend llama_cpp \
    --repo-id build-small-hackathon/mythograph-nemotron-3-nano-4b-gguf \
    --filename mythograph-nemotron-Q5_K_M.gguf \
    --validation-file nemotron_dataset/validation.jsonl \
    --out-dir outputs/eval_q5_hub
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


CONVERSATION_REQUIRED = {
    "assistant_message": str,
    "progress_label": str,
    "reason": str,
    "is_ready": bool,
    "controls": list,
}

RECIPE_REQUIRED = {
    "title": str,
    "main_idea": str,
    "visual_style": str,
    "palette": list,
    "symbols": list,
    "composition": str,
    "image_prompt": str,
    "negative_prompt": str,
    "friend_explanation": str,
}

ALLOWED_CONTROL_KINDS = {
    "choice_cards",
    "multi_choice_cards",
    "slider_group",
    "swatch_picker",
    "ready_button",
    "text_refinement",
}

HEX_RE = re.compile(r"^#[0-9A-Fa-f]{6}$")


@dataclass
class ExampleResult:
    index: int
    task: str
    strict_json_valid: bool
    salvage_json_valid: bool
    schema_valid: bool
    exact_task_match: bool
    latency_seconds: Optional[float]
    output_chars: int
    error: Optional[str]
    prediction_text: str
    prediction_json: Optional[Dict[str, Any]]
    expected_json: Dict[str, Any]


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def expected_obj(row: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return json.loads(row["messages"][-1]["content"])
    except Exception as exc:
        raise ValueError("Expected assistant message is not valid JSON") from exc


def infer_task(obj: Dict[str, Any]) -> str:
    if "controls" in obj or "assistant_message" in obj:
        return "conversation_turn"
    if "image_prompt" in obj or "friend_explanation" in obj:
        return "final_art_recipe"
    return "unknown"


def strict_parse(text: str) -> Tuple[bool, Optional[Any], Optional[str]]:
    try:
        return True, json.loads(text.strip()), None
    except Exception as exc:
        return False, None, str(exc)


def salvage_parse(text: str) -> Tuple[bool, Optional[Any], Optional[str]]:
    """Try to recover a JSON object from text with accidental surrounding text/code fences."""
    cleaned = text.strip()
    cleaned = cleaned.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    ok, obj, err = strict_parse(cleaned)
    if ok:
        return ok, obj, err

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        candidate = cleaned[start : end + 1]
        return strict_parse(candidate)

    return False, None, err


def check_required_types(obj: Dict[str, Any], required: Dict[str, type]) -> List[str]:
    errors = []
    for key, typ in required.items():
        if key not in obj:
            errors.append(f"missing key: {key}")
        elif not isinstance(obj[key], typ):
            errors.append(f"key {key!r} expected {typ.__name__}, got {type(obj[key]).__name__}")
    return errors


def validate_control(control: Any, index: int) -> List[str]:
    errors = []
    if not isinstance(control, dict):
        return [f"control[{index}] is not an object"]

    for key in ["kind", "label", "prompt", "options", "sliders"]:
        if key not in control:
            errors.append(f"control[{index}] missing key: {key}")

    if "kind" in control:
        if not isinstance(control["kind"], str):
            errors.append(f"control[{index}].kind is not a string")
        elif control["kind"] not in ALLOWED_CONTROL_KINDS:
            errors.append(f"control[{index}].kind unknown: {control['kind']}")

    for key in ["label", "prompt"]:
        if key in control and not isinstance(control[key], str):
            errors.append(f"control[{index}].{key} is not a string")

    if "options" in control:
        if not isinstance(control["options"], list):
            errors.append(f"control[{index}].options is not a list")
        elif not all(isinstance(x, str) for x in control["options"]):
            errors.append(f"control[{index}].options must contain only strings")

    if "sliders" in control:
        if not isinstance(control["sliders"], list):
            errors.append(f"control[{index}].sliders is not a list")
        else:
            for s_i, slider in enumerate(control["sliders"]):
                if not isinstance(slider, dict):
                    errors.append(f"control[{index}].sliders[{s_i}] is not an object")
                    continue
                for s_key in ["label", "min", "max", "value"]:
                    # The training data uses either numeric min/max sliders or
                    # labelled left/right sliders. Accept both shapes.
                    if s_key not in slider and not (
                        s_key in {"min", "max"} and {"left_label", "right_label"}.issubset(slider)
                    ):
                        errors.append(f"control[{index}].sliders[{s_i}] missing key: {s_key}")
                if "key" in slider and not isinstance(slider["key"], str):
                    errors.append(f"control[{index}].sliders[{s_i}].key is not a string")

    return errors


def validate_conversation_turn(obj: Dict[str, Any]) -> List[str]:
    errors = check_required_types(obj, CONVERSATION_REQUIRED)

    # Keep strictness high: the app expects a single next UI control.
    controls = obj.get("controls")
    if isinstance(controls, list):
        if len(controls) != 1:
            errors.append(f"controls must contain exactly 1 item, got {len(controls)}")
        for i, control in enumerate(controls):
            errors.extend(validate_control(control, i))

    if isinstance(obj.get("assistant_message"), str) and not obj["assistant_message"].strip():
        errors.append("assistant_message is empty")
    if isinstance(obj.get("progress_label"), str) and not obj["progress_label"].strip():
        errors.append("progress_label is empty")
    if isinstance(obj.get("reason"), str) and not obj["reason"].strip():
        errors.append("reason is empty")

    return errors


def validate_final_art_recipe(obj: Dict[str, Any]) -> List[str]:
    errors = check_required_types(obj, RECIPE_REQUIRED)

    palette = obj.get("palette")
    if isinstance(palette, list):
        if not 2 <= len(palette) <= 8:
            errors.append(f"palette should contain 2-8 colors, got {len(palette)}")
        for i, color in enumerate(palette):
            if not isinstance(color, str):
                errors.append(f"palette[{i}] is not a string")
            elif not HEX_RE.match(color):
                errors.append(f"palette[{i}] is not a #RRGGBB hex color: {color}")

    symbols = obj.get("symbols")
    if isinstance(symbols, list):
        if len(symbols) < 1:
            errors.append("symbols must contain at least 1 item")
        for i, symbol in enumerate(symbols):
            if not isinstance(symbol, dict):
                errors.append(f"symbols[{i}] is not an object")
                continue
            for key in ["visual", "meaning"]:
                if key not in symbol:
                    errors.append(f"symbols[{i}] missing key: {key}")
                elif not isinstance(symbol[key], str):
                    errors.append(f"symbols[{i}].{key} is not a string")
                elif not symbol[key].strip():
                    errors.append(f"symbols[{i}].{key} is empty")

    for key in [
        "title",
        "main_idea",
        "visual_style",
        "composition",
        "image_prompt",
        "negative_prompt",
        "friend_explanation",
    ]:
        if isinstance(obj.get(key), str) and not obj[key].strip():
            errors.append(f"{key} is empty")

    return errors


def validate_schema(obj: Any, expected_task: str) -> Tuple[bool, List[str], str]:
    if not isinstance(obj, dict):
        return False, ["top-level JSON is not an object"], "unknown"

    predicted_task = infer_task(obj)
    if expected_task == "conversation_turn":
        errors = validate_conversation_turn(obj)
    elif expected_task == "final_art_recipe":
        errors = validate_final_art_recipe(obj)
    else:
        errors = ["unknown expected task"]

    if predicted_task != expected_task:
        errors.append(f"task mismatch: expected {expected_task}, got {predicted_task}")

    return len(errors) == 0, errors, predicted_task


def build_prompt_messages(row: Dict[str, Any]) -> List[Dict[str, str]]:
    return row["messages"][:-1]


class Generator:
    def generate(self, messages: List[Dict[str, str]]) -> str:  # pragma: no cover
        raise NotImplementedError


class TransformersGenerator(Generator):
    def __init__(
        self,
        model_id: str,
        adapter_id: Optional[str],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        do_sample: bool,
        device_map: str,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map=device_map,
        )

        if adapter_id:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter_id)

        self.model.eval()

    def _apply_chat_template(self, messages: List[Dict[str, str]]) -> str:
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

    def generate(self, messages: List[Dict[str, str]]) -> str:
        prompt = self._apply_chat_template(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_len = inputs["input_ids"].shape[-1]

        generation_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
            "do_sample": self.do_sample,
        }
        if self.do_sample:
            generation_kwargs.update({"temperature": self.temperature, "top_p": self.top_p})

        with self.torch.no_grad():
            out = self.model.generate(**inputs, **generation_kwargs)

        new_tokens = out[0][input_len:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


class LlamaCppGenerator(Generator):
    def __init__(
        self,
        gguf: Optional[str],
        repo_id: Optional[str],
        filename: Optional[str],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        n_ctx: int,
        n_threads: int,
        n_gpu_layers: int,
    ) -> None:
        from llama_cpp import Llama

        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        common_kwargs = {
            "n_ctx": n_ctx,
            "n_threads": n_threads,
            "n_gpu_layers": n_gpu_layers,
            "verbose": False,
        }

        if gguf:
            self.llm = Llama(model_path=gguf, **common_kwargs)
        else:
            if not repo_id or not filename:
                raise ValueError("For llama_cpp backend, provide either --gguf or both --repo-id and --filename")
            self.llm = Llama.from_pretrained(repo_id=repo_id, filename=filename, **common_kwargs)

    def generate(self, messages: List[Dict[str, str]]) -> str:
        result = self.llm.create_chat_completion(
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_new_tokens,
        )
        return result["choices"][0]["message"]["content"].strip()


class FileGenerator(Generator):
    def __init__(self, predictions_file: str | Path) -> None:
        rows = load_jsonl(predictions_file)
        self.predictions = []
        for row in rows:
            if "prediction_text" in row:
                self.predictions.append(row["prediction_text"])
            elif "output" in row:
                self.predictions.append(row["output"])
            elif "text" in row:
                self.predictions.append(row["text"])
            else:
                raise ValueError("Predictions file rows must contain prediction_text, output, or text")
        self.index = 0

    def generate(self, messages: List[Dict[str, str]]) -> str:
        if self.index >= len(self.predictions):
            raise IndexError("Not enough predictions in predictions file")
        text = self.predictions[self.index]
        self.index += 1
        return text


def make_generator(args: argparse.Namespace) -> Generator:
    if args.backend == "transformers":
        return TransformersGenerator(
            model_id=args.model,
            adapter_id=args.adapter,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=args.do_sample,
            device_map=args.device_map,
        )
    if args.backend == "llama_cpp":
        return LlamaCppGenerator(
            gguf=args.gguf,
            repo_id=args.repo_id,
            filename=args.filename,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            n_ctx=args.n_ctx,
            n_threads=args.n_threads,
            n_gpu_layers=args.n_gpu_layers,
        )
    if args.backend == "file":
        return FileGenerator(args.predictions_file)
    raise ValueError(f"Unknown backend: {args.backend}")


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    rows = load_jsonl(args.validation_file)
    if args.limit:
        rows = rows[: args.limit]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    generator = make_generator(args)
    results: List[ExampleResult] = []

    for i, row in enumerate(rows):
        expected = expected_obj(row)
        task = infer_task(expected)
        messages = build_prompt_messages(row)

        start = time.perf_counter()
        try:
            prediction_text = generator.generate(messages)
            latency = time.perf_counter() - start
            strict_ok, strict_obj, strict_err = strict_parse(prediction_text)
            salvage_ok, salvage_obj, salvage_err = salvage_parse(prediction_text)
            parsed_obj = strict_obj if strict_ok else salvage_obj

            if salvage_ok:
                schema_ok, schema_errors, predicted_task = validate_schema(parsed_obj, task)
            else:
                schema_ok = False
                schema_errors = [strict_err or salvage_err or "invalid JSON"]
                predicted_task = "unknown"

            error = None if schema_ok else "; ".join(schema_errors[:10])
            results.append(
                ExampleResult(
                    index=i,
                    task=task,
                    strict_json_valid=strict_ok,
                    salvage_json_valid=salvage_ok,
                    schema_valid=schema_ok,
                    exact_task_match=predicted_task == task,
                    latency_seconds=latency,
                    output_chars=len(prediction_text),
                    error=error,
                    prediction_text=prediction_text,
                    prediction_json=parsed_obj if isinstance(parsed_obj, dict) else None,
                    expected_json=expected,
                )
            )
        except Exception as exc:
            latency = time.perf_counter() - start
            results.append(
                ExampleResult(
                    index=i,
                    task=task,
                    strict_json_valid=False,
                    salvage_json_valid=False,
                    schema_valid=False,
                    exact_task_match=False,
                    latency_seconds=latency,
                    output_chars=0,
                    error=repr(exc),
                    prediction_text="",
                    prediction_json=None,
                    expected_json=expected,
                )
            )

        if args.print_every and (i + 1) % args.print_every == 0:
            print(f"Evaluated {i + 1}/{len(rows)} examples...")

    def rate(attr: str) -> float:
        return round(sum(1 for r in results if getattr(r, attr)) / max(len(results), 1), 4)

    by_task: Dict[str, Dict[str, Any]] = {}
    for task in sorted(set(r.task for r in results)):
        task_results = [r for r in results if r.task == task]
        by_task[task] = {
            "count": len(task_results),
            "strict_json_valid_rate": round(sum(r.strict_json_valid for r in task_results) / len(task_results), 4),
            "salvage_json_valid_rate": round(sum(r.salvage_json_valid for r in task_results) / len(task_results), 4),
            "schema_valid_rate": round(sum(r.schema_valid for r in task_results) / len(task_results), 4),
            "exact_task_match_rate": round(sum(r.exact_task_match for r in task_results) / len(task_results), 4),
        }

    latencies = [r.latency_seconds for r in results if r.latency_seconds is not None]
    report = {
        "backend": args.backend,
        "model": args.model,
        "adapter": args.adapter,
        "gguf": args.gguf,
        "repo_id": args.repo_id,
        "filename": args.filename,
        "validation_file": args.validation_file,
        "count": len(results),
        "strict_json_valid_rate": rate("strict_json_valid"),
        "salvage_json_valid_rate": rate("salvage_json_valid"),
        "schema_valid_rate": rate("schema_valid"),
        "exact_task_match_rate": rate("exact_task_match"),
        "avg_latency_seconds": round(statistics.mean(latencies), 4) if latencies else None,
        "median_latency_seconds": round(statistics.median(latencies), 4) if latencies else None,
        "avg_output_chars": round(statistics.mean([r.output_chars for r in results]), 2) if results else None,
        "by_task": by_task,
    }

    with open(out_dir / "eval_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    with open(out_dir / "predictions.jsonl", "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

    with open(out_dir / "errors.jsonl", "w", encoding="utf-8") as f:
        for r in results:
            if not r.schema_valid:
                f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate strict JSON/schema quality for Mythograph Atelier models.")
    parser.add_argument("--backend", choices=["transformers", "llama_cpp", "file"], required=True)
    parser.add_argument("--validation-file", default="nemotron_dataset/validation.jsonl")
    parser.add_argument("--out-dir", default="outputs/eval")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--print-every", type=int, default=10)

    # Transformers backend
    parser.add_argument("--model", default=None, help="HF model id or local merged model path.")
    parser.add_argument("--adapter", default=None, help="Optional PEFT adapter id/path.")
    parser.add_argument("--device-map", default="auto")

    # llama.cpp backend
    parser.add_argument("--gguf", default=None, help="Local GGUF path.")
    parser.add_argument("--repo-id", default=None, help="HF repo id for GGUF.")
    parser.add_argument("--filename", default=None, help="GGUF filename in repo-id.")
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument("--n-threads", type=int, default=max(os.cpu_count() or 4, 4))
    parser.add_argument("--n-gpu-layers", type=int, default=0)

    # File backend
    parser.add_argument("--predictions-file", default=None)

    # Generation
    parser.add_argument("--max-new-tokens", type=int, default=900)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--do-sample", action="store_true", help="Use sampling for Transformers backend. Default is greedy.")

    args = parser.parse_args()

    if args.backend == "transformers" and not args.model:
        parser.error("--model is required for --backend transformers")
    if args.backend == "llama_cpp" and not (args.gguf or (args.repo_id and args.filename)):
        parser.error("For --backend llama_cpp, provide --gguf or both --repo-id and --filename")
    if args.backend == "file" and not args.predictions_file:
        parser.error("--predictions-file is required for --backend file")

    return args


if __name__ == "__main__":
    report = evaluate(parse_args())
    print(json.dumps(report, ensure_ascii=False, indent=2))
