from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any
from urllib import error, request


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
    tokens = first.replace(".", " ").replace(",", " ").split()
    if "unsafe" in tokens:
        return "unsafe", lines[1:]
    if "safe" in tokens:
        return "safe", []
    return None, lines[1:]


def redacted_row(row: dict[str, Any], raw_output: str, error_text: str | None = None) -> dict[str, Any]:
    label, categories = parse_llamaguard_output(raw_output)
    metadata = row.get("metadata") or {}
    out = {
        "id": row.get("id"),
        "label": label,
        "unsafe_categories": categories,
        "raw_output_redacted": raw_output[:80],
        "metadata": {
            "source_id": metadata.get("source_id"),
            "qsr_route_reason": metadata.get("qsr_route_reason"),
            "qsr_adapter": metadata.get("qsr_adapter"),
            "qsr_is_refusal": metadata.get("qsr_is_refusal"),
        },
    }
    if error_text:
        out["judge_error"] = error_text[:240]
    return out


def normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def chat_completion(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    timeout_seconds: int,
    referer: str | None,
    title: str | None,
) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if referer:
        headers["HTTP-Referer"] = referer
    if title:
        headers["X-Title"] = title

    req = request.Request(
        f"{normalize_base_url(base_url)}/chat/completions",
        data=body,
        headers=headers,
        method="POST",
    )
    with request.urlopen(req, timeout=timeout_seconds) as resp:
        raw = resp.read().decode("utf-8")
    data = json.loads(raw)
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError("Provider response had no choices")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if content is None:
        raise RuntimeError("Provider response choice had no message.content")
    return str(content)


def classify_rows(
    rows: list[dict[str, Any]],
    *,
    base_url: str,
    api_key: str,
    model: str,
    max_tokens: int,
    timeout_seconds: int,
    sleep_seconds: float,
    max_retries: int,
    partial_path: Path | None,
    referer: str | None,
    title: str | None,
) -> list[dict[str, Any]]:
    judged: list[dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        messages = row.get("messages")
        if not isinstance(messages, list):
            raise ValueError(f"Row {idx} lacks a messages list")
        last_error: str | None = None
        for attempt in range(max_retries + 1):
            try:
                raw = chat_completion(
                    base_url=base_url,
                    api_key=api_key,
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    timeout_seconds=timeout_seconds,
                    referer=referer,
                    title=title,
                )
                judged.append(redacted_row(row, raw))
                last_error = None
                break
            except error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:500]
                last_error = f"HTTP {exc.code}: {detail}"
            except Exception as exc:
                last_error = str(exc)

            if attempt >= max_retries:
                judged.append(redacted_row(row, "", last_error or "unknown provider error"))
                break
            wait = sleep_seconds * (2**attempt)
            print(
                f"item {idx}/{len(rows)} failed on attempt {attempt + 1}; retrying in {wait:.1f}s: {last_error}",
                file=sys.stderr,
            )
            time.sleep(wait)

        if last_error is None and sleep_seconds > 0 and idx < len(rows):
            time.sleep(sleep_seconds)
        if partial_path is not None:
            partial_path.parent.mkdir(parents=True, exist_ok=True)
            partial_path.write_text(json.dumps(judged, ensure_ascii=False, indent=2), encoding="utf-8")
        if idx % 10 == 0:
            print(f"judged {idx}/{len(rows)}")
    return judged


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
    parser = argparse.ArgumentParser(description="Run provider-hosted LlamaGuard on exported prompt-response JSONL rows.")
    parser.add_argument("--input", required=True, help="JSONL with id/messages/metadata.")
    parser.add_argument("--output", required=True, help="Redacted output JSON.")
    parser.add_argument("--base-url", default=os.environ.get("LLAMAGUARD_PROVIDER_BASE_URL", "https://llm.wavespeed.ai/v1"))
    parser.add_argument("--api-key-env", default=os.environ.get("LLAMAGUARD_PROVIDER_API_KEY_ENV", "WAVESPEED_API_KEY"))
    parser.add_argument("--model", default=os.environ.get("LLAMAGUARD_PROVIDER_MODEL", "meta-llama/llama-guard-3-8b"))
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--sleep-seconds", type=float, default=0.25)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--partial-output", default=None)
    parser.add_argument("--http-referer", default=os.environ.get("LLAMAGUARD_PROVIDER_HTTP_REFERER"))
    parser.add_argument("--x-title", default=os.environ.get("LLAMAGUARD_PROVIDER_X_TITLE", "Q-SR safety audit"))
    args = parser.parse_args()

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(
            f"Missing API key environment variable: {args.api_key_env}. "
            "Set it in the current shell; do not write provider keys into scripts or result files."
        )

    rows = load_jsonl(Path(args.input))
    if args.max_samples is not None:
        rows = rows[: max(0, args.max_samples)]
    judged = classify_rows(
        rows,
        base_url=args.base_url,
        api_key=api_key,
        model=args.model,
        max_tokens=max(1, args.max_tokens),
        timeout_seconds=max(1, args.timeout_seconds),
        sleep_seconds=max(0.0, args.sleep_seconds),
        max_retries=max(0, args.max_retries),
        partial_path=Path(args.partial_output) if args.partial_output else None,
        referer=args.http_referer,
        title=args.x_title,
    )
    report = {
        "metadata": {
            "provider_base_url": args.base_url,
            "model": args.model,
            "input": str(Path(args.input)),
            "api_key_env": args.api_key_env,
            "claim_scope": "Provider-hosted LlamaGuard prompt-response safety classification; not a JailbreakBench leaderboard submission.",
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
