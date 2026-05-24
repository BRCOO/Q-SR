from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
JBB_SRC = PROJECT_ROOT / "data" / "jailbreakbench" / "src"
if str(JBB_SRC) not in sys.path:
    sys.path.insert(0, str(JBB_SRC))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _label_key(judge: str) -> str:
    if judge == "jailbreak":
        return "is_jailbroken"
    if judge == "refusal":
        return "is_refusal"
    raise ValueError(f"Unsupported judge: {judge}")


def classify_with_official_jbb_source(rows: list[dict[str, Any]], judge: str, api_key: str) -> list[dict[str, Any]]:
    from jailbreakbench.classifier import Llama3JailbreakJudge, Llama3RefusalJudge

    prompts = [str(row["prompt"]) for row in rows]
    responses = [str(row["completion"]) for row in rows]
    if judge == "jailbreak":
        classifier = Llama3JailbreakJudge(api_key)
        labels = classifier(prompts, responses)
    elif judge == "refusal":
        classifier = Llama3RefusalJudge(api_key)
        labels = classifier(prompts, responses)
    else:
        raise ValueError(f"Unsupported judge: {judge}")

    label_key = _label_key(judge)
    out = []
    for row, label in zip(rows, labels):
        judged = dict(row)
        judged[label_key] = bool(label)
        out.append(judged)
    return out


def classify_with_prompt_compatible_model(
    rows: list[dict[str, Any]],
    judge: str,
    api_key: str,
    model: str,
    batch_size: int,
    sleep_seconds: float,
    max_retries: int,
    partial_path: Path | None = None,
) -> list[dict[str, Any]]:
    import litellm
    from jailbreakbench.classifier import Llama3JailbreakJudge, Llama3RefusalJudge

    if judge == "jailbreak":
        user_prompt = Llama3JailbreakJudge.USER_PROMPT
        positive_answer = "unsafe"
    elif judge == "refusal":
        user_prompt = Llama3RefusalJudge.USER_PROMPT
        positive_answer = "yes"
    else:
        raise ValueError(f"Unsupported judge: {judge}")

    label_key = _label_key(judge)
    all_classifier_inputs = [
        [{"role": "user", "content": user_prompt.format(goal=str(row["prompt"]), response=str(row["completion"]))}]
        for row in rows
    ]

    judged_rows = []
    for idx, classifier_input in enumerate(all_classifier_inputs, start=1):
        last_error: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                output = litellm.completion(
                    model=model,
                    messages=classifier_input,
                    api_key=api_key,
                    temperature=0,
                    max_tokens=16,
                    extra_body={"thinking": {"type": "disabled"}} if model.startswith("deepseek/") else None,
                )
                if isinstance(output, Exception):
                    raise output
                content = str(output["choices"][0]["message"].content).strip().lower()
                row = rows[idx - 1]
                judged = dict(row)
                if judge == "jailbreak":
                    judged[label_key] = content.startswith(positive_answer) or positive_answer in content.split()
                else:
                    judged[label_key] = content.startswith(positive_answer)
                judged["judge_raw_output"] = content
                judged_rows.append(judged)
                break
            except Exception as exc:
                last_error = exc
                if attempt >= max_retries:
                    if partial_path is None:
                        raise
                    error_row = dict(rows[idx - 1])
                    error_row[label_key] = None
                    error_row["judge_raw_output"] = None
                    error_row["judge_error"] = str(exc)
                    judged_rows.append(error_row)
                    break
                wait = sleep_seconds * (2**attempt)
                print(f"item {idx}/{len(all_classifier_inputs)} failed on attempt {attempt + 1}; retrying in {wait:.1f}s: {exc}", file=sys.stderr)
                time.sleep(wait)
        if last_error is None and sleep_seconds > 0 and idx < len(all_classifier_inputs):
            time.sleep(sleep_seconds)
        if partial_path is not None:
            partial_path.parent.mkdir(parents=True, exist_ok=True)
            partial_path.write_text(json.dumps(judged_rows, ensure_ascii=False, indent=2), encoding="utf-8")
        if idx % 10 == 0:
            print(f"judged {idx}/{len(all_classifier_inputs)}")
    return judged_rows


def summarize(rows: list[dict[str, Any]], judge: str) -> dict[str, Any]:
    if judge == "jailbreak":
        valid_rows = [row for row in rows if row.get("is_jailbroken") is not None]
        jailbroken = sum(bool(row["is_jailbroken"]) for row in valid_rows)
        return {
            "judge": "JailbreakBench Llama3JailbreakJudge",
            "num_samples": len(rows),
            "num_valid_samples": len(valid_rows),
            "num_errors": len(rows) - len(valid_rows),
            "jailbreak_rate": jailbroken / len(valid_rows) if valid_rows else None,
            "non_jailbreak_rate": 1 - (jailbroken / len(valid_rows)) if valid_rows else None,
        }
    valid_rows = [row for row in rows if row.get("is_refusal") is not None]
    refusals = sum(bool(row["is_refusal"]) for row in valid_rows)
    return {
        "judge": "JailbreakBench Llama3RefusalJudge",
        "num_samples": len(rows),
        "num_valid_samples": len(valid_rows),
        "num_errors": len(rows) - len(valid_rows),
        "refusal_rate": refusals / len(valid_rows) if valid_rows else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run official-source JailbreakBench Llama3 judges on exported Q-SR JSONL.")
    parser.add_argument("--input", required=True, help="JSONL file with prompt/completion fields.")
    parser.add_argument("--output", required=True, help="Output JSON path.")
    parser.add_argument("--judge", required=True, choices=["jailbreak", "refusal"])
    parser.add_argument("--api-key-env", default="TOGETHER_API_KEY")
    parser.add_argument(
        "--model",
        default=None,
        help="Optional LiteLLM model string. If set, uses the JBB Llama3 judge prompt with this model instead of the exact JBB Together model constants.",
    )
    parser.add_argument(
        "--model-env",
        default=None,
        help="Optional environment variable containing a LiteLLM model string. Ignored if --model is set.",
    )
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size for alternative LiteLLM provider/model runs.")
    parser.add_argument("--sleep-seconds", type=float, default=1.0, help="Sleep between batches for alternative provider/model runs.")
    parser.add_argument("--max-retries", type=int, default=3, help="Retries per batch for alternative provider/model runs.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap for probe runs.")
    parser.add_argument("--partial-output", default=None, help="Optional partial judged-row JSON path for alternative provider/model runs.")
    args = parser.parse_args()

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"Missing API key environment variable: {args.api_key_env}")

    input_path = Path(args.input)
    rows = load_jsonl(input_path)
    if args.max_samples is not None:
        rows = rows[: max(0, args.max_samples)]
    model_override = args.model or (os.environ.get(args.model_env) if args.model_env else None)
    if model_override:
        judged_rows = classify_with_prompt_compatible_model(
            rows,
            args.judge,
            api_key,
            model_override,
            max(1, args.batch_size),
            max(0.0, args.sleep_seconds),
            max(0, args.max_retries),
            Path(args.partial_output) if args.partial_output else None,
        )
        claim_scope = "JailbreakBench official-source judge prompt with alternative LiteLLM provider/model; not exact original Together JBB model and not leaderboard submission"
    else:
        judged_rows = classify_with_official_jbb_source(rows, args.judge, api_key)
        claim_scope = "official-source JailbreakBench judge path; not leaderboard submission"

    report = {
        "metadata": {
            "source": "data/jailbreakbench/src/jailbreakbench/classifier.py",
            "input": str(input_path),
            "judge": args.judge,
            "api_key_env": args.api_key_env,
            "model": model_override,
            "model_env": args.model_env,
            "claim_scope": claim_scope,
        },
        "summary": summarize(judged_rows, args.judge),
        "results": judged_rows,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
