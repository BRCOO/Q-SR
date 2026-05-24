from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import time

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from qsr.router_v7 import score_with_base_router
from qsr.train_utility_calibrator import load_training_rows
from qsr.utility_calibrator import TextAndDictVectorizer, build_utility_features


@dataclass
class RecoveryVetoTrainConfig:
    router_v6_path: str
    output_dir: str
    test_size: float = 0.1
    seed: int = 44
    max_safe: int = 9000
    max_unsafe: int = 9000
    target_unsafe_recall: float = 0.985
    include_harmbench_text_test_negatives: bool = True
    max_hard_negative: int = 2000
    include_synthetic_utility_positives: bool = True
    include_expanded_synthetic_utility_positives: bool = True
    include_category_synthetic_utility_positives: bool = True
    include_synthetic_hard_negatives: bool = True
    include_expanded_synthetic_hard_negatives: bool = True
    include_abstract_boundary_hard_negatives: bool = True


def load_config(path: str | Path) -> RecoveryVetoTrainConfig:
    return RecoveryVetoTrainConfig(**json.loads(Path(path).read_text(encoding="utf-8")))


def choose_threshold(y_true: np.ndarray, probability: np.ndarray, target_recall: float) -> tuple[float, dict]:
    candidates = np.linspace(0.02, 0.98, 97)
    feasible = []
    for threshold in candidates:
        pred = (probability >= threshold).astype(int)
        recall = recall_score(y_true, pred, zero_division=0)
        precision = precision_score(y_true, pred, zero_division=0)
        f1 = f1_score(y_true, pred, zero_division=0)
        false_positive_rate = float(((pred == 1) & (y_true == 0)).sum() / max(1, int((y_true == 0).sum())))
        feasible.append(
            {
                "threshold": float(threshold),
                "recall": float(recall),
                "precision": float(precision),
                "f1": float(f1),
                "false_positive_rate": false_positive_rate,
            }
        )
    valid = [row for row in feasible if row["recall"] >= target_recall]
    if valid:
        # Highest threshold keeps the veto as narrow as possible while preserving unsafe recall on validation.
        chosen = max(valid, key=lambda row: (row["threshold"], row["precision"]))
    else:
        chosen = max(feasible, key=lambda row: (row["recall"], row["precision"]))
    return float(chosen["threshold"]), chosen


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    started = time.time()
    rows = load_training_rows(
        cfg.seed,
        cfg.max_safe,
        cfg.max_unsafe,
        include_harmbench_text_test_negatives=cfg.include_harmbench_text_test_negatives,
        max_hard_negative=cfg.max_hard_negative,
        include_synthetic_utility_positives=cfg.include_synthetic_utility_positives,
        include_expanded_synthetic_utility_positives=cfg.include_expanded_synthetic_utility_positives,
        include_category_synthetic_utility_positives=cfg.include_category_synthetic_utility_positives,
        include_synthetic_hard_negatives=cfg.include_synthetic_hard_negatives,
        include_expanded_synthetic_hard_negatives=cfg.include_expanded_synthetic_hard_negatives,
        include_abstract_boundary_hard_negatives=cfg.include_abstract_boundary_hard_negatives,
    )
    texts = [row["text"] for row in rows]
    # Veto positive label means "do not recover this refusal".
    labels = np.array([0 if int(row["label"]) == 1 else 1 for row in rows], dtype=int)
    risk_scores, ambiguity_scores, base_bundle = score_with_base_router(cfg.router_v6_path, texts)
    feature_rows = [
        build_utility_features(text, risk, amb)
        for text, risk, amb in zip(texts, risk_scores, ambiguity_scores)
    ]
    x_train, x_val, y_train, y_val = train_test_split(
        feature_rows,
        labels,
        test_size=cfg.test_size,
        random_state=cfg.seed,
        stratify=labels,
    )
    pipeline = Pipeline(
        steps=[
            ("vec", TextAndDictVectorizer()),
            (
                "clf",
                LogisticRegression(
                    max_iter=3000,
                    class_weight={0: 1.0, 1: 2.5},
                    solver="liblinear",
                    random_state=cfg.seed,
                ),
            ),
        ]
    )
    pipeline.fit(x_train, y_train)
    probability = pipeline.predict_proba(x_val)[:, 1]
    threshold, threshold_metrics = choose_threshold(y_val, probability, cfg.target_unsafe_recall)
    pred_default = pipeline.predict(x_val)
    pred_threshold = (probability >= threshold).astype(int)

    source_counts = Counter(row["source"] for row in rows)
    metrics = {
        "default_0p5": {
            "accuracy": accuracy_score(y_val, pred_default),
            "precision": precision_score(y_val, pred_default, zero_division=0),
            "recall": recall_score(y_val, pred_default, zero_division=0),
            "f1": f1_score(y_val, pred_default, zero_division=0),
        },
        "selected_threshold": {
            "threshold": threshold,
            "target_unsafe_recall": cfg.target_unsafe_recall,
            **threshold_metrics,
            "accuracy": accuracy_score(y_val, pred_threshold),
        },
        "val_positive_rate": float((y_val == 1).mean()),
        "val_prob_mean": float(probability.mean()),
    }
    metadata = {
        "router_v6_path": cfg.router_v6_path,
        "num_rows": len(rows),
        "safe_rows": int((labels == 0).sum()),
        "unsafe_rows": int((labels == 1).sum()),
        "source_counts": dict(source_counts),
        "test_size": cfg.test_size,
        "seed": cfg.seed,
        "elapsed_seconds": time.time() - started,
        "xstest_used_for_training": False,
        "alpacaeval_used_for_training": False,
        "jbb_harmbench_val_advbench_used_for_training": False,
        "harmbench_text_test_used_as_hard_negative": cfg.include_harmbench_text_test_negatives,
        "method": "learned_recovery_safety_veto",
        "model_type": "text_logistic",
    }
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = {
        "calibrator": pipeline,
        "recommended_threshold": threshold,
        "base_router_metadata": base_bundle.get("metadata"),
        "base_router_metrics": base_bundle.get("metrics"),
        "metrics": {"recovery_veto": metrics},
        "metadata": metadata,
    }
    joblib.dump(bundle, output_dir / "recovery_veto.joblib")
    (output_dir / "metrics.json").write_text(json.dumps(bundle["metrics"], ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved recovery veto to {output_dir}")
    print(json.dumps({"metadata": metadata, "metrics": bundle["metrics"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
