from __future__ import annotations

import json
from pathlib import Path

import joblib
from scipy.sparse import hstack
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from qsr.router_v7 import build_calibrator_features


def build_utility_features(text: str, risk_score: float, ambiguity_score: float) -> dict[str, float]:
    features = build_calibrator_features(text, risk_score, ambiguity_score, feature_mode="full")
    features["__text__"] = str(text)
    features["utility_safe_prior"] = float(
        features.get("flag__benign_daily_utility", 0.0)
        or features.get("flag__benign_sports_photo_art", 0.0)
        or features.get("flag__benign_objects_pests_food", 0.0)
        or features.get("flag__benign_software", 0.0)
        or features.get("flag__benign_legal_business", 0.0)
        or features.get("flag__definition_or_education", 0.0)
    )
    features["strong_harmful_intent_prior"] = float(
        features.get("flag__misinformation_persuasion", 0.0)
        or features.get("flag__self_harm_persuasion", 0.0)
        or features.get("privacy_real_person", 0.0)
        or features.get("harmful_history_or_discr", 0.0)
        or features.get("flag__jailbreak_harmful_request", 0.0)
        or features.get("flag__targeted_harm_intent", 0.0)
        or features.get("flag__hard_drug_procurement", 0.0)
        or features.get("flag__discrimination_harm", 0.0)
    )
    return features


class TextAndDictVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, max_text_features: int = 25000) -> None:
        self.max_text_features = max_text_features
        self.text_vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            max_features=max_text_features,
            sublinear_tf=True,
        )
        self.dict_vectorizer = DictVectorizer(sparse=True)

    def fit(self, records, y=None):
        texts = [str(record.get("__text__", "")) for record in records]
        dict_records = [
            {key: value for key, value in record.items() if key != "__text__"}
            for record in records
        ]
        self.text_vectorizer.fit(texts)
        self.dict_vectorizer.fit(dict_records)
        return self

    def transform(self, records):
        texts = [str(record.get("__text__", "")) for record in records]
        dict_records = [
            {key: value for key, value in record.items() if key != "__text__"}
            for record in records
        ]
        text_matrix = self.text_vectorizer.transform(texts)
        dict_matrix = self.dict_vectorizer.transform(dict_records)
        return hstack([text_matrix, dict_matrix], format="csr")


def train_utility_calibrator(
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
    if model_type == "extra_trees":
        pipeline = Pipeline(
            steps=[
                ("vec", DictVectorizer(sparse=False)),
                (
                    "clf",
                    ExtraTreesClassifier(
                        n_estimators=500,
                        max_depth=12,
                        min_samples_leaf=3,
                        class_weight="balanced",
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
    elif model_type == "text_logistic":
        pipeline = Pipeline(
            steps=[
                ("vec", TextAndDictVectorizer()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=3000,
                        class_weight="balanced",
                        solver="liblinear",
                        random_state=seed,
                    ),
                ),
            ]
        )
    else:
        raise ValueError(f"Unsupported utility calibrator model_type: {model_type}")
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


def save_utility_bundle(output_dir: str | Path, bundle: dict) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output_dir / "utility_calibrator.joblib")
    (output_dir / "metrics.json").write_text(json.dumps(bundle["metrics"], ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "metadata.json").write_text(json.dumps(bundle["metadata"], ensure_ascii=False, indent=2), encoding="utf-8")


def load_utility_bundle(path: str | Path) -> dict:
    return joblib.load(path)
