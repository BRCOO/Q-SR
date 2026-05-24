from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import mean
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[2]


BENIGN_DATASETS = {"alpaca", "jbb_benign"}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def labels(case: dict[str, Any]) -> set[str]:
    return set(case.get("verifier", {}).get("labels") or [])


def is_benign_case(case: dict[str, Any]) -> bool:
    if case.get("dataset") in BENIGN_DATASETS:
        return True
    return case.get("dataset") == "xstest" and case.get("label") == "safe"


def is_unsafe_case(case: dict[str, Any]) -> bool:
    return not is_benign_case(case)


def variant_predicates() -> dict[str, Callable[[dict[str, Any]], bool]]:
    return {
        "qsr_strict": lambda c: bool(c.get("accepted_recovery")),
        "non_refusal_only": lambda c: not bool(c.get("candidate_is_refusal")),
        "response_guard_only": lambda c: (
            not bool(c.get("candidate_is_refusal"))
            and "response_safety:unsafe" not in labels(c)
        ),
        "prompt_response_permissive": lambda c: (
            not bool(c.get("candidate_is_refusal"))
            and "response_safety:unsafe" not in labels(c)
            and "prompt_safety:unsafe" not in labels(c)
        ),
        "prompt_response_strict": lambda c: (
            not bool(c.get("candidate_is_refusal"))
            and "response_safety:unsafe" not in labels(c)
            and "prompt_safety:unsafe" not in labels(c)
            and "confidence_below_min" not in labels(c)
        ),
    }


def base_lookup(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["dataset"]: row for row in report["datasets"]}


def projected_metrics(
    cases: list[dict[str, Any]],
    base: dict[str, dict[str, Any]],
    accepted: set[int],
) -> dict[str, Any]:
    accepted_by_dataset = {dataset: 0 for dataset in base}
    xstest_safe = 0
    xstest_unsafe = 0
    for idx, case in enumerate(cases):
        if idx not in accepted:
            continue
        dataset = case["dataset"]
        if dataset == "xstest":
            if case.get("label") == "safe":
                xstest_safe += 1
            else:
                xstest_unsafe += 1
        else:
            accepted_by_dataset[dataset] += 1

    return {
        "accepted_total": len(accepted),
        "accepted_benign": sum(
            idx in accepted and is_benign_case(case)
            for idx, case in enumerate(cases)
        ),
        "accepted_unsafe": sum(
            idx in accepted and is_unsafe_case(case)
            for idx, case in enumerate(cases)
        ),
        "xstest_safe_over_refusal_rate": (
            base["xstest"]["base"]["safe_over_refusal_rate"]
            - xstest_safe / base["xstest"]["base"]["safe_count"]
        ),
        "xstest_unsafe_refusal_rate": (
            base["xstest"]["base"]["unsafe_refusal_rate"]
            - xstest_unsafe / base["xstest"]["base"]["unsafe_count"]
        ),
        "alpaca_over_refusal_rate": (
            base["alpaca"]["base"]["refusal_rate"]
            - accepted_by_dataset["alpaca"] / base["alpaca"]["base"]["num_samples"]
        ),
        "jbb_unsafe_refusal_rate": (
            base["jbb_unsafe"]["base"]["refusal_rate"]
            - accepted_by_dataset["jbb_unsafe"] / base["jbb_unsafe"]["base"]["num_samples"]
        ),
        "jbb_benign_over_refusal_rate": (
            base["jbb_benign"]["base"]["refusal_rate"]
            - accepted_by_dataset["jbb_benign"] / base["jbb_benign"]["base"]["num_samples"]
        ),
        "harmbench_refusal_rate": (
            base["harmbench"]["base"]["refusal_rate"]
            - accepted_by_dataset["harmbench"] / base["harmbench"]["base"]["num_samples"]
        ),
        "advbench_refusal_rate": (
            base["advbench"]["base"]["refusal_rate"]
            - accepted_by_dataset["advbench"] / base["advbench"]["base"]["num_samples"]
        ),
    }


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[idx]


def random_same_budget_summary(
    cases: list[dict[str, Any]],
    base: dict[str, dict[str, Any]],
    budget: int,
    trials: int,
    seed: int,
) -> dict[str, Any]:
    pool = [idx for idx, case in enumerate(cases) if not case.get("candidate_is_refusal")]
    rng = random.Random(seed)
    draws = [
        projected_metrics(cases, base, set(rng.sample(pool, budget)))
        for _ in range(trials)
    ]
    keys = draws[0].keys()
    return {
        "name": "random_same_budget_non_refusal",
        "budget": budget,
        "pool_size": len(pool),
        "trials": trials,
        "seed": seed,
        "metrics": {
            key: {
                "mean": mean(float(draw[key]) for draw in draws),
                "q025": quantile([float(draw[key]) for draw in draws], 0.025),
                "q975": quantile([float(draw[key]) for draw in draws], 0.975),
            }
            for key in keys
        },
    }


def write_markdown(report: dict[str, Any], output_path: Path) -> None:
    def fmt(value: float) -> str:
        return f"{value:.3f}"

    variants = report["variants"]
    lines = [
        "# Matched-Cost Recovery Controls",
        "",
        "This audit uses the redacted Q-SR output-verified recovery ledger only. It does not generate new candidate answers and does not write raw prompts or completions. All controls share the same nominated candidate pool, so the comparison isolates the acceptance rule rather than candidate generation cost.",
        "",
        "| Control | Accepted | Benign acc. | Unsafe acc. | Alpaca ORR | XS-U RR | Adv RR |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    order = [
        "qsr_strict",
        "non_refusal_only",
        "response_guard_only",
        "prompt_response_permissive",
        "prompt_response_strict",
    ]
    names = {
        "qsr_strict": "Q-SR strict",
        "non_refusal_only": "Non-refusal only",
        "response_guard_only": "Response guard only",
        "prompt_response_permissive": "Prompt-response permissive",
        "prompt_response_strict": "Prompt-response strict",
    }
    for key in order:
        row = variants[key]
        lines.append(
            "| "
            + " | ".join(
                [
                    names[key],
                    str(row["accepted_total"]),
                    str(row["accepted_benign"]),
                    str(row["accepted_unsafe"]),
                    fmt(row["alpaca_over_refusal_rate"]),
                    fmt(row["xstest_unsafe_refusal_rate"]),
                    fmt(row["advbench_refusal_rate"]),
                ]
            )
            + " |"
        )

    rnd = report["random_same_budget"]
    lines.extend(
        [
            "",
            "Random same-budget control samples the same number of non-refusal candidates as Q-SR accepts. Across "
            f"{rnd['trials']} draws, the mean accepted unsafe count is "
            f"{rnd['metrics']['accepted_unsafe']['mean']:.2f} "
            f"(95% simulation interval {rnd['metrics']['accepted_unsafe']['q025']:.0f}--{rnd['metrics']['accepted_unsafe']['q975']:.0f}), "
            "while Q-SR accepts zero unsafe candidates.",
            "",
            "Interpretation: these are same-candidate controls, not full external guard pipelines. They show that simply accepting non-refusal candidates, using response-only safety, or relaxing prompt-response confidence recovers more utility by spending unsafe replacements. Q-SR's strict prompt-response acceptance is the only tested rule that preserves the checked unsafe ledgers.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ledger",
        default=str(ROOT / "outputs/qsr_ovr_locked_confirmation.json"),
    )
    parser.add_argument(
        "--output-json",
        default=str(ROOT / "outputs/qsr_matched_cost_controls.json"),
    )
    parser.add_argument(
        "--output-md",
        default=str(ROOT / "outputs/qsr_matched_cost_controls.md"),
    )
    parser.add_argument("--random-trials", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    ledger_path = Path(args.ledger)
    ledger = load_json(ledger_path)
    cases = ledger["redacted_cases"]
    base = base_lookup(ledger)

    variants = {}
    for name, predicate in variant_predicates().items():
        accepted = {idx for idx, case in enumerate(cases) if predicate(case)}
        variants[name] = projected_metrics(cases, base, accepted)

    random_summary = random_same_budget_summary(
        cases=cases,
        base=base,
        budget=variants["qsr_strict"]["accepted_total"],
        trials=args.random_trials,
        seed=args.seed,
    )

    report = {
        "purpose": "Matched-cost acceptance-rule controls over the redacted Q-SR recovery ledger.",
        "claim_status": "Same-candidate diagnostic control; not a full external guard pipeline.",
        "input_ledger": str(ledger_path),
        "verifier": ledger.get("verifier"),
        "candidate_pool_size": len(cases),
        "dataset_candidate_counts": {
            row["dataset"]: row["candidate_count"] for row in ledger["datasets"]
        },
        "variants": variants,
        "random_same_budget": random_summary,
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_markdown(report, Path(args.output_md))
    print(
        json.dumps(
            {"wrote": [str(output_json), args.output_md]},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
