from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time

from qsr.router_v6 import save_v6_router_bundle, train_binary_pipeline


@dataclass
class RouterV6TrainConfig:
    train_jsonl: str
    output_dir: str
    test_size: float = 0.1
    seed: int = 42
    max_features: int = 80000
    use_features: bool = True


def load_config(path: str | Path) -> RouterV6TrainConfig:
    with open(path, "r", encoding="utf-8") as f:
        return RouterV6TrainConfig(**json.load(f))


def load_rows(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    started = time.time()
    rows = load_rows(cfg.train_jsonl)
    risk_bundle = train_binary_pipeline(
        rows=rows,
        text_key="text",
        label_key="risk_label",
        test_size=cfg.test_size,
        seed=cfg.seed,
        max_features=cfg.max_features,
        use_features=cfg.use_features,
    )
    ambiguity_bundle = train_binary_pipeline(
        rows=rows,
        text_key="text",
        label_key="ambiguity_label",
        test_size=cfg.test_size,
        seed=cfg.seed + 1,
        max_features=cfg.max_features,
        use_features=cfg.use_features,
    )
    metadata = {
        "train_jsonl": cfg.train_jsonl,
        "num_rows": len(rows),
        "risk_positive": sum(int(row["risk_label"]) for row in rows),
        "ambiguity_positive": sum(int(row["ambiguity_label"]) for row in rows),
        "test_size": cfg.test_size,
        "seed": cfg.seed,
        "max_features": cfg.max_features,
        "use_features": cfg.use_features,
        "elapsed_seconds": time.time() - started,
        "xstest_used_for_training": False,
    }
    bundle = {
        "risk_pipeline": risk_bundle["pipeline"],
        "ambiguity_pipeline": ambiguity_bundle["pipeline"],
        "metrics": {
            "risk": risk_bundle["metrics"],
            "ambiguity": ambiguity_bundle["metrics"],
        },
        "metadata": metadata,
    }
    save_v6_router_bundle(cfg.output_dir, bundle)
    print(f"Saved v6 router bundle to {cfg.output_dir}")
    print(json.dumps({"metadata": metadata, "metrics": bundle["metrics"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
