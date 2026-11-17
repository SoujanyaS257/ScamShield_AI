"""TF-IDF + logistic regression pipeline."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from src.config import LABELS, MODEL_PATH, RANDOM_STATE
from src.preprocessing.text import normalize


def _normalize_batch(texts):
    return [normalize(str(text)) for text in texts]


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("prep", FunctionTransformer(_normalize_batch, validate=False)),
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=1,
                    sublinear_tf=True,
                    max_features=8000,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=4.0,
                    class_weight="balanced",
                    solver="lbfgs",
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


class TextClassifier:
    """Loaded scikit-learn pipeline. Probabilities come from predict_proba."""

    def __init__(self, pipeline: Pipeline, metrics: dict | None = None) -> None:
        self.pipeline = pipeline
        self.metrics = metrics or {}

    def predict(self, text: str) -> dict:
        probabilities_row = self.pipeline.predict_proba([text])[0]
        classes = [str(label) for label in self.pipeline.named_steps["clf"].classes_]
        probabilities = {label: float(value) for label, value in zip(classes, probabilities_row)}
        missing = [label for label in LABELS if label not in probabilities]
        if missing:
            raise RuntimeError(f"Model is missing classes: {', '.join(missing)}")
        label = max(LABELS, key=lambda name: probabilities[name])
        return {
            "label": label,
            "confidence": float(probabilities[label]),
            "probabilities": {name: probabilities[name] for name in LABELS},
        }

    def contributions(self, text: str, label: str, limit: int = 6) -> list[dict]:
        """Linear TF-IDF contributions for one class. Intercept is excluded."""
        classifier = self.pipeline.named_steps["clf"]
        vectorizer = self.pipeline.named_steps["tfidf"]
        class_index = list(classifier.classes_).index(label)
        coefficients = classifier.coef_[class_index]
        matrix = vectorizer.transform([normalize(text)])
        names = vectorizer.get_feature_names_out()
        scored: list[tuple[str, float]] = []
        row = matrix.tocoo()
        for index, value in zip(row.col, row.data):
            scored.append((str(names[index]), float(value * coefficients[index])))
        scored.sort(key=lambda item: abs(item[1]), reverse=True)
        results = []
        for feature, contribution in scored[:limit]:
            results.append(
                {
                    "feature": feature,
                    "contribution": round(contribution, 4),
                    "direction": "toward this class" if contribution > 0 else "against this class",
                }
            )
        return results


def save_classifier(pipeline: Pipeline, metrics: dict, model_path: Path = MODEL_PATH) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "metrics": metrics}, model_path)


def load_saved(model_path: Path = MODEL_PATH) -> TextClassifier:
    payload = joblib.load(model_path)
    return TextClassifier(payload["pipeline"], payload.get("metrics") or {})


@lru_cache(maxsize=1)
def get_classifier() -> TextClassifier:
    if not MODEL_PATH.exists():
        from src.ml.train import train_and_save

        train_and_save()
    classifier = load_saved()
    metrics_path = MODEL_PATH.with_name("metrics.json")
    if metrics_path.exists() and not classifier.metrics:
        classifier.metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return classifier


def clear_classifier_cache() -> None:
    get_classifier.cache_clear()
