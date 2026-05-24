from __future__ import annotations

from pathlib import Path
import json
import random
from typing import Iterable

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from qsr.features_v6 import texts_to_feature_dicts


def make_pipeline(max_features: int = 80000, use_features: bool = True) -> Pipeline:
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 3),
        min_df=2,
        max_features=max_features,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    if use_features:
        preprocessor = ColumnTransformer(
            transformers=[
                ("tfidf", vectorizer, "text"),
                ("features", DictVectorizer(sparse=True), "features"),
            ]
        )
    else:
        preprocessor = ColumnTransformer(transformers=[("tfidf", vectorizer, "text")])
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "clf",
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=42,
                ),
            ),
        ]
    )


def train_binary_pipeline(
    rows: list[dict],
    text_key: str,
    label_key: str,
    test_size: float,
    seed: int,
    max_features: int,
    use_features: bool = True,
) -> dict:
    texts = [row[text_key] for row in rows]
    labels = [int(row[label_key]) for row in rows]
    feature_dicts = texts_to_feature_dicts(texts) if use_features else [{} for _ in texts]
    records = pd.DataFrame({"text": texts, "features": feature_dicts})

    x_train, x_val, y_train, y_val = train_test_split(
        records,
        labels,
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )
    pipeline = make_pipeline(max_features=max_features, use_features=use_features)
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


def save_v6_router_bundle(output_dir: str | Path, bundle: dict) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output_dir / "router_v6.joblib")
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(bundle["metrics"], f, ensure_ascii=False, indent=2)
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(bundle["metadata"], f, ensure_ascii=False, indent=2)


def load_v6_router_bundle(path: str | Path) -> dict:
    return joblib.load(path)


def make_router_records(texts: Iterable[str]) -> list[dict]:
    text_list = [str(text) for text in texts]
    return pd.DataFrame({"text": text_list, "features": texts_to_feature_dicts(text_list)})


def dedupe_rows(rows: Iterable[dict], text_key: str = "text") -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for row in rows:
        text = " ".join(str(row[text_key]).split())
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        clean = dict(row)
        clean[text_key] = text
        out.append(clean)
    return out


def sample_rows(rows: list[dict], max_rows: int, seed: int) -> list[dict]:
    if max_rows <= 0 or len(rows) <= max_rows:
        return rows
    rng = random.Random(seed)
    return rng.sample(rows, max_rows)
