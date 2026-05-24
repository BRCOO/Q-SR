from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time

from qsr.router_v7 import (
    make_calibrator_feature_rows,
    save_v7_bundle,
    score_with_base_router,
    train_calibrator,
)


@dataclass
class RouterV7TrainConfig:
    train_jsonl: str
    router_v6_path: str
    output_dir: str
    test_size: float = 0.1
    seed: int = 42
    label_mode: str = "route_target"
    model_type: str = "extra_trees"
    feature_mode: str = "full"


def load_config(path: str | Path) -> RouterV7TrainConfig:
    with open(path, "r", encoding="utf-8") as f:
        return RouterV7TrainConfig(**json.load(f))


def load_rows(path: str | Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def build_route_labels(rows: list[dict]) -> list[int]:
    labels = []
    for row in rows:
        source = row.get("source", "")
        risk_label = int(row["risk_label"])
        ambiguity_label = int(row.get("ambiguity_label", 0))
        if ambiguity_label == 1:
            labels.append(0)
        elif source in {
            "safe_ambiguity_template",
            "fictional_safe_template",
            "safe_target_template",
            "figurative_safe_template",
            "safe_context_template",
            "definition_safe_template",
            "historical_safe_template",
        }:
            labels.append(0)
        else:
            labels.append(risk_label)
    return labels


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    started = time.time()
    rows = load_rows(cfg.train_jsonl)
    texts = [row["text"] for row in rows]
    if cfg.label_mode == "risk_label":
        labels = [int(row["risk_label"]) for row in rows]
    elif cfg.label_mode == "route_target":
        labels = build_route_labels(rows)
    else:
        raise ValueError(f"Unsupported label_mode: {cfg.label_mode}")
    risk_scores, ambiguity_scores, base_bundle = score_with_base_router(cfg.router_v6_path, texts)
    feature_rows = make_calibrator_feature_rows(
        texts,
        risk_scores,
        ambiguity_scores,
        feature_mode=cfg.feature_mode,
    )
    calibrator = train_calibrator(
        feature_rows=feature_rows,
        labels=labels,
        test_size=cfg.test_size,
        seed=cfg.seed,
        model_type=cfg.model_type,
    )
    metadata = {
        "train_jsonl": cfg.train_jsonl,
        "router_v6_path": cfg.router_v6_path,
        "num_rows": len(rows),
        "positive_rows": sum(labels),
        "test_size": cfg.test_size,
        "seed": cfg.seed,
        "elapsed_seconds": time.time() - started,
        "xstest_used_for_training": False,
        "method": "learned_target_validity_calibrator",
        "label_mode": cfg.label_mode,
        "model_type": cfg.model_type,
        "feature_mode": cfg.feature_mode,
    }
    bundle = {
        "calibrator": calibrator["pipeline"],
        "base_router_metadata": base_bundle.get("metadata"),
        "base_router_metrics": base_bundle.get("metrics"),
        "metrics": {"calibrator": calibrator["metrics"]},
        "metadata": metadata,
    }
    save_v7_bundle(cfg.output_dir, bundle)
    print(f"Saved v7 router bundle to {cfg.output_dir}")
    print(json.dumps({"metadata": metadata, "metrics": bundle["metrics"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
