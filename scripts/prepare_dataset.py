#!/usr/bin/env python3
"""Validate and optionally materialize chat-formatted SFT JSONL data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} is not valid JSON: {exc}") from exc
    return rows


def validate_row(row: Dict[str, Any], path: Path, index: int) -> str:
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 3:
        raise ValueError(f"{path}:{index + 1} must contain at least system, user, and assistant messages")

    for message in messages:
        if not isinstance(message, dict):
            raise ValueError(f"{path}:{index + 1} has a non-object message")
        if message.get("role") not in {"system", "user", "assistant"}:
            raise ValueError(f"{path}:{index + 1} has invalid role {message.get('role')!r}")
        if not isinstance(message.get("content"), str):
            raise ValueError(f"{path}:{index + 1} has non-string message content")

    assistant_messages = [m for m in messages if m["role"] == "assistant"]
    if not assistant_messages:
        raise ValueError(f"{path}:{index + 1} has no assistant message")

    try:
        assistant_json = json.loads(assistant_messages[-1]["content"])
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}:{index + 1} assistant content is not strict JSON: {exc}") from exc

    if not isinstance(assistant_json, dict):
        raise ValueError(f"{path}:{index + 1} assistant JSON must be an object")

    if "controls" in assistant_json or "assistant_message" in assistant_json:
        return "conversation_turn"
    if "image_prompt" in assistant_json or "friend_explanation" in assistant_json:
        return "final_art_recipe"
    raise ValueError(f"{path}:{index + 1} assistant JSON does not match a known task schema")


def write_jsonl(rows: Iterable[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Mythograph chat JSONL data for SFT.")
    parser.add_argument("--train-file", default="data/train.jsonl")
    parser.add_argument("--validation-file", default="data/validation.jsonl")
    parser.add_argument("--out-dir", default=None, help="Optional directory for validated train/validation JSONL copies.")
    args = parser.parse_args()

    summary: Dict[str, Dict[str, int]] = {}
    loaded: Dict[str, List[Dict[str, Any]]] = {}

    for split, file_name in [("train", args.train_file), ("validation", args.validation_file)]:
        path = Path(file_name)
        rows = load_jsonl(path)
        counts = {"conversation_turn": 0, "final_art_recipe": 0}
        for index, row in enumerate(rows):
            task = validate_row(row, path, index)
            counts[task] += 1
        summary[split] = {"rows": len(rows), **counts}
        loaded[split] = rows

    if args.out_dir:
        out_dir = Path(args.out_dir)
        write_jsonl(loaded["train"], out_dir / "train.jsonl")
        write_jsonl(loaded["validation"], out_dir / "validation.jsonl")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
