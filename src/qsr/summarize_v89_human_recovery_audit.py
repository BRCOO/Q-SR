from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]

FIELDS = [
    "safe_0_1",
    "partially_helpful_0_1",
    "fully_helpful_0_1",
    "overall_accept_0_1",
]


def read_csv(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return {str(row["audit_id"]): row for row in reader}


def parse_binary(value: str, *, audit_id: str, field: str, annotator: str) -> int:
    text = str(value).strip()
    if text == "":
        raise ValueError(f"Missing {field} for {audit_id} in annotator {annotator}")
    try:
        parsed = int(text)
    except ValueError as exc:
        raise ValueError(f"Non-integer {field}={text!r} for {audit_id} in annotator {annotator}") from exc
    if parsed not in {0, 1}:
        raise ValueError(f"Out-of-range {field}={parsed} for {audit_id} in annotator {annotator}")
    return parsed


def cohen_kappa(a: list[int], b: list[int]) -> float | None:
    n = len(a)
    if n == 0:
        return None
    observed = sum(x == y for x, y in zip(a, b)) / n
    ca = Counter(a)
    cb = Counter(b)
    expected = sum((ca[label] / n) * (cb[label] / n) for label in [0, 1])
    if expected == 1.0:
        return 1.0 if observed == 1.0 else None
    return (observed - expected) / (1.0 - expected)


def summarize_one(rows: dict[str, dict[str, int]]) -> dict[str, Any]:
    n = len(rows)
    out: dict[str, Any] = {"n": n}
    for field in FIELDS:
        count = sum(row[field] == 1 for row in rows.values())
        out[field.replace("_0_1", "")] = count
        out[field.replace("_0_1", "") + "_rate"] = count / max(1, n)
    return out


def write_disagreements(path: Path, disagreements: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["audit_id", "field", "annotator_a", "annotator_b", "notes_a", "notes_b"]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in disagreements:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_md(report: dict[str, Any], path: Path) -> None:
    a = report["annotator_a"]
    b = report["annotator_b"]
    c = report["conservative_consensus"]
    k = report["agreement"]["kappa"]
    lines = [
        "# Q-SR 160-token Accepted-Recovery Human Audit",
        "",
        "This report summarizes two human annotations over the exact 29 benign replacements accepted by the paper-facing 160-token Q-SR recovery ledger. Raw prompts, answers, and annotator notes are kept in private CSV files; this report contains aggregate counts only.",
        "",
        "| Quantity | Annotator A | Annotator B | Conservative consensus |",
        "| --- | ---: | ---: | ---: |",
        f"| Judged recoveries | {a['n']} | {b['n']} | {c['n']} |",
        f"| Safe | {a['safe']} | {b['safe']} | {c['safe_both']} |",
        f"| Partially helpful | {a['partially_helpful']} | {b['partially_helpful']} | {c['partially_helpful_both']} |",
        f"| Fully helpful | {a['fully_helpful']} | {b['fully_helpful']} | {c['fully_helpful_both']} |",
        f"| Overall accepted | {a['overall_accept']} | {b['overall_accept']} | {c['overall_accept_both']} |",
        "",
        "| Agreement metric | Value |",
        "| --- | ---: |",
    ]
    for field in FIELDS:
        value = k[field]
        shown = "NA" if value is None else f"{value:.3f}"
        lines.append(f"| Cohen kappa, {field} | {shown} |")
    lines.extend(
        [
            f"| Any-label disagreement cells | {report['agreement']['disagreement_cells']} |",
            "",
            "Interpretation boundary: this is a small accepted-recovery audit, not a full benchmark judge and not a population-level answer-quality guarantee. It should be used only to support the scoped claim that Q-SR reduces measured over-refusal while accepted-answer quality remains separately audited.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annotator-a",
        default=str(ROOT / "outputs/human_audit/annotator_A.csv"),
    )
    parser.add_argument(
        "--annotator-b",
        default=str(ROOT / "outputs/human_audit/annotator_B.csv"),
    )
    parser.add_argument(
        "--output-json",
        default=str(ROOT / "outputs/human_audit/human_audit_summary.json"),
    )
    parser.add_argument(
        "--output-md",
        default=str(ROOT / "outputs/human_audit/human_audit_summary.md"),
    )
    parser.add_argument(
        "--disagreement-csv",
        default=str(ROOT / "outputs/human_audit/human_audit_disagreements.csv"),
    )
    args = parser.parse_args()

    raw_a = read_csv(Path(args.annotator_a))
    raw_b = read_csv(Path(args.annotator_b))
    ids_a = set(raw_a)
    ids_b = set(raw_b)
    if ids_a != ids_b:
        raise SystemExit(f"Annotator files have different audit ids: only_a={sorted(ids_a - ids_b)}, only_b={sorted(ids_b - ids_a)}")

    scored_a: dict[str, dict[str, int]] = {}
    scored_b: dict[str, dict[str, int]] = {}
    disagreements: list[dict[str, Any]] = []
    for audit_id in sorted(ids_a):
        scored_a[audit_id] = {}
        scored_b[audit_id] = {}
        for field in FIELDS:
            va = parse_binary(raw_a[audit_id].get(field, ""), audit_id=audit_id, field=field, annotator="A")
            vb = parse_binary(raw_b[audit_id].get(field, ""), audit_id=audit_id, field=field, annotator="B")
            scored_a[audit_id][field] = va
            scored_b[audit_id][field] = vb
            if va != vb:
                disagreements.append(
                    {
                        "audit_id": audit_id,
                        "field": field,
                        "annotator_a": va,
                        "annotator_b": vb,
                        "notes_a": raw_a[audit_id].get("notes", ""),
                        "notes_b": raw_b[audit_id].get("notes", ""),
                    }
                )

    consensus = {"n": len(scored_a)}
    for field in FIELDS:
        name = field.replace("_0_1", "")
        consensus[name + "_both"] = sum(
            1 for audit_id in scored_a
            if scored_a[audit_id][field] == 1 and scored_b[audit_id][field] == 1
        )
    kappa = {
        field: cohen_kappa(
            [scored_a[audit_id][field] for audit_id in sorted(ids_a)],
            [scored_b[audit_id][field] for audit_id in sorted(ids_a)],
        )
        for field in FIELDS
    }
    report = {
        "annotator_a": summarize_one(scored_a),
        "annotator_b": summarize_one(scored_b),
        "conservative_consensus": consensus,
        "agreement": {
            "kappa": kappa,
            "disagreement_cells": len(disagreements),
        },
        "status": "human_labels_summarized",
    }

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_disagreements(Path(args.disagreement_csv), disagreements)
    write_md(report, Path(args.output_md))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
