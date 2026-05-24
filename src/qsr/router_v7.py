from __future__ import annotations

from pathlib import Path
import json
from typing import Iterable

import joblib
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from qsr.features_v6 import extract_feature_flags
from qsr.router_v6 import load_v6_router_bundle, make_router_records


def build_calibrator_features(
    text: str,
    risk_score: float,
    ambiguity_score: float,
    feature_mode: str = "full",
) -> dict[str, float]:
    flags = extract_feature_flags(text)
    if feature_mode == "score_only":
        flags = {key: 0.0 for key in flags}
    elif feature_mode not in {"full", "no_ambiguity"}:
        raise ValueError(f"Unsupported feature_mode: {feature_mode}")

    features = dict(flags)
    features["risk_score"] = float(risk_score)
    if feature_mode != "no_ambiguity":
        features["ambiguity_score"] = float(ambiguity_score)
        features["risk_minus_ambiguity"] = float(risk_score - ambiguity_score)
        features["risk_x_ambiguity"] = float(risk_score * ambiguity_score)
    features["risk_x_risk_feature_count"] = float(risk_score * flags["risk_feature_count"])
    if feature_mode != "no_ambiguity":
        features["ambiguity_x_benign_feature_count"] = float(ambiguity_score * flags["benign_feature_count"])
    features["target_validity_hint"] = float(
        flags["risk_feature_count"] - flags["benign_feature_count"]
    )
    features["benign_context_hint"] = float(
        flags["privacy_fictional_context"]
        or flags["flag__nonhuman_or_object_group"]
        or (flags["benign_feature_count"] > 0 and flags["risk_feature_count"] == 0)
    )
    features["risk_context_hint"] = float(
        flags["privacy_real_person"]
        or flags["harmful_history_or_discr"]
        or flags["flag__jailbreak_harmful_request"]
    )
    return features


def make_calibrator_feature_rows(
    texts: Iterable[str],
    risk_scores: Iterable[float],
    ambiguity_scores: Iterable[float],
    feature_mode: str = "full",
) -> list[dict[str, float]]:
    return [
        build_calibrator_features(text, risk_score, ambiguity_score, feature_mode=feature_mode)
        for text, risk_score, ambiguity_score in zip(texts, risk_scores, ambiguity_scores)
    ]


def train_calibrator(
    feature_rows: list[dict[str, float]],
    labels: list[int],
    test_size: float,
    seed: int,
    model_type: str = "extra_trees",
) -> dict:
    x_train, x_val, y_train, y_val = train_test_split(
        feature_rows,
        labels,
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )
    if model_type == "logistic":
        vectorizer = DictVectorizer(sparse=True)
        clf = LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            solver="liblinear",
            random_state=seed,
        )
    elif model_type == "extra_trees":
        vectorizer = DictVectorizer(sparse=False)
        clf = ExtraTreesClassifier(
            n_estimators=400,
            max_depth=10,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unsupported calibrator model_type: {model_type}")
    pipeline = Pipeline(steps=[("vec", vectorizer), ("clf", clf)])
    pipeline.fit(x_train, y_train)
    pred = pipeline.predict(x_val)
    prob = pipeline.predict_proba(x_val)[:, 1]
    metrics = {
        "train_size": len(x_train),
        "val_size": len(x_val),
        "accuracy": accuracy_score(y_val, pred),
        "precision": precision_score(y_val, pred, zero_division=0),
        "recall": recall_score(y_val, pred, zero_division=0),
        "f1": f1_score(y_val, pred, zero_division=0),
        "val_positive_rate": float((pred == 1).mean()),
        "val_prob_mean": float(prob.mean()),
    }
    return {"pipeline": pipeline, "metrics": metrics}


def score_with_base_router(router_v6_path: str | Path, texts: list[str]) -> tuple[list[float], list[float], dict]:
    bundle = load_v6_router_bundle(router_v6_path)
    records = make_router_records(texts)
    risk_scores = bundle["risk_pipeline"].predict_proba(records)[:, 1].tolist()
    ambiguity_scores = bundle["ambiguity_pipeline"].predict_proba(records)[:, 1].tolist()
    return risk_scores, ambiguity_scores, bundle


def save_v7_bundle(output_dir: str | Path, bundle: dict) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output_dir / "router_v7.joblib")
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(bundle["metrics"], f, ensure_ascii=False, indent=2)
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(bundle["metadata"], f, ensure_ascii=False, indent=2)


def load_v7_bundle(path: str | Path) -> dict:
    return joblib.load(path)
