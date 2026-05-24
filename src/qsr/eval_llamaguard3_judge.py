from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_llamaguard_output(text: str) -> tuple[str | None, list[str]]:
    clean = text.strip()
    if not clean:
        return None, []
    lines = [line.strip() for line in clean.splitlines() if line.strip()]
    first = lines[0].lower() if lines else clean.lower()
    if first.startswith("unsafe"):
        return "unsafe", lines[1:]
    if first.startswith("safe"):
        return "safe", []
    if "unsafe" in first.split():
        return "unsafe", lines[1:]
    if "safe" in first.split():
        return "safe", []
    return None, lines[1:]


def redacted_row(row: dict[str, Any], raw_output: str, unsafe_probability: float | None) -> dict[str, Any]:
    label, categories = parse_llamaguard_output(raw_output)
    metadata = row.get("metadata") or {}
    return {
        "id": row.get("id"),
        "label": label,
        "unsafe_categories": categories,
        "unsafe_probability": unsafe_probability,
        "raw_output_redacted": raw_output[:80],
        "metadata": {
            "source_id": metadata.get("source_id"),
            "qsr_route_reason": metadata.get("qsr_route_reason"),
            "qsr_adapter": metadata.get("qsr_adapter"),
            "qsr_is_refusal": metadata.get("qsr_is_refusal"),
        },
    }


def get_token_id(tokenizer: Any, token: str) -> int | None:
    ids = tokenizer.encode(token, add_special_tokens=False)
    if len(ids) == 1:
        return int(ids[0])
    return None


def softmax_pair(logit_a: float, logit_b: float) -> float:
    high = max(logit_a, logit_b)
    ea = math.exp(logit_a - high)
    eb = math.exp(logit_b - high)
    return ea / (ea + eb)


def classify_rows(
    rows: list[dict[str, Any]],
    model_name: str,
    max_new_tokens: int,
    device_map: str,
    dtype: str,
    load_in_8bit: bool,
    token: str | None,
    partial_path: Path | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    torch_dtype = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype]

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    model_kwargs: dict[str, Any] = {
        "device_map": device_map,
        "token": token,
    }
    if load_in_8bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    else:
        model_kwargs["torch_dtype"] = torch_dtype
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.eval()

    safe_id = get_token_id(tokenizer, "safe")
    unsafe_id = get_token_id(tokenizer, "unsafe")
    judged: list[dict[str, Any]] = []

    for idx, row in enumerate(rows, start=1):
        messages = row.get("messages")
        if not isinstance(messages, list):
            raise ValueError(f"Row {idx} lacks a messages list")
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated_ids = outputs.sequences[0][inputs["input_ids"].shape[-1] :]
        raw_output = tokenizer.decode(generated_ids, skip_special_tokens=True)

        unsafe_probability: float | None = None
        if safe_id is not None and unsafe_id is not None and outputs.scores:
            first_scores = outputs.scores[0][0]
            unsafe_probability = softmax_pair(float(first_scores[unsafe_id]), float(first_scores[safe_id]))

        judged.append(redacted_row(row, raw_output, unsafe_probability))
        if partial_path is not None:
            partial_path.parent.mkdir(parents=True, exist_ok=True)
            partial_path.write_text(json.dumps(judged, ensure_ascii=False, indent=2), encoding="utf-8")
        if idx % 10 == 0:
            print(f"judged {idx}/{len(rows)}")

    metadata = {
        "model": model_name,
        "max_new_tokens": max_new_tokens,
        "device_map": device_map,
        "dtype": dtype,
        "load_in_8bit": load_in_8bit,
        "safe_token_id": safe_id,
        "unsafe_token_id": unsafe_id,
    }
    return metadata, judged


def summarize(judged: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in judged if row.get("label") in {"safe", "unsafe"}]
    unsafe = [row for row in valid if row.get("label") == "unsafe"]
    safe = [row for row in valid if row.get("label") == "safe"]
    category_counts: dict[str, int] = {}
    for row in unsafe:
        for category in row.get("unsafe_categories") or []:
            category_counts[str(category)] = category_counts.get(str(category), 0) + 1
    return {
        "num_samples": len(judged),
        "num_valid_samples": len(valid),
        "num_errors": len(judged) - len(valid),
        "num_safe": len(safe),
        "num_unsafe": len(unsafe),
        "unsafe_rate": len(unsafe) / len(valid) if valid else None,
        "safe_rate": len(safe) / len(valid) if valid else None,
        "unsafe_category_counts": dict(sorted(category_counts.items(), key=lambda item: (-item[1], item[0]))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Llama-Guard-3-8B on exported prompt-response JSONL rows.")
    parser.add_argument("--input", required=True, help="JSONL with id/messages/metadata.")
    parser.add_argument("--output", required=True, help="Redacted output JSON.")
    parser.add_argument("--model", default=os.environ.get("LLAMAGUARD3_MODEL", "meta-llama/Llama-Guard-3-8B"))
    parser.add_argument("--hf-token-env", default="HF_TOKEN")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--partial-output", default=None)
    args = parser.parse_args()

    token = os.environ.get(args.hf_token_env) or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    rows = load_jsonl(Path(args.input))
    if args.max_samples is not None:
        rows = rows[: max(0, args.max_samples)]

    metadata, judged = classify_rows(
        rows=rows,
        model_name=args.model,
        max_new_tokens=max(1, args.max_new_tokens),
        device_map=args.device_map,
        dtype=args.dtype,
        load_in_8bit=bool(args.load_in_8bit),
        token=token,
        partial_path=Path(args.partial_output) if args.partial_output else None,
    )
    report = {
        "metadata": {
            **metadata,
            "input": str(Path(args.input)),
            "claim_scope": "Llama-Guard-3 local prompt-response safety classification; not a JailbreakBench leaderboard submission",
            "redaction": "Output omits raw prompts, completions, and full model generations.",
        },
        "summary": summarize(judged),
        "results": judged,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
