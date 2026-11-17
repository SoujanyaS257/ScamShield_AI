"""Train the text classifier and write held-out metrics."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import CORPUS_CSV, LABELS, METRICS_PATH, MODEL_PATH, RANDOM_STATE, TEST_FRACTION
from src.ml.classifier import build_pipeline, clear_classifier_cache, save_classifier


def load_corpus() -> pd.DataFrame:
    from data.corpus import records

    frame = pd.DataFrame(records())
    if frame.empty:
        raise RuntimeError("The training corpus is empty.")
    unknown = sorted(set(frame["label"]) - set(LABELS))
    if unknown:
        raise RuntimeError(f"Unexpected labels in corpus: {unknown}")
    counts = Counter(frame["label"])
    missing = [label for label in LABELS if counts[label] < 20]
    if missing:
        raise RuntimeError(f"Need at least 20 examples for: {', '.join(missing)}")
    CORPUS_CSV.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(CORPUS_CSV, index=False)
    return frame


def train_and_save(model_path: Path = MODEL_PATH) -> dict:
    frame = load_corpus()
    train_frame, test_frame = train_test_split(
        frame,
        test_size=TEST_FRACTION,
        random_state=RANDOM_STATE,
        stratify=frame["label"],
    )
    pipeline = build_pipeline()
    pipeline.fit(train_frame["text"].tolist(), train_frame["label"].tolist())
    predictions = pipeline.predict(test_frame["text"].tolist())
    truth = test_frame["label"].tolist()
    report = classification_report(truth, predictions, labels=list(LABELS), output_dict=True, zero_division=0)
    metrics = {
        "algorithm": "tfidf_logistic_regression",
        "vectorizer": "word tf-idf, unigrams and bigrams, sublinear tf",
        "preprocessing": "lowercase, strip zero-width characters, replace urls emails and long numbers",
        "random_state": RANDOM_STATE,
        "test_fraction": TEST_FRACTION,
        "train_size": int(len(train_frame)),
        "test_size": int(len(test_frame)),
        "label_counts": {label: int((frame["label"] == label).sum()) for label in LABELS},
        "accuracy": round(float(accuracy_score(truth, predictions)), 4),
        "macro_f1": round(float(f1_score(truth, predictions, labels=list(LABELS), average="macro")), 4),
        "classification_report": report,
        "dataset": (
            "Original synthetic messages in data/corpus.py, written for ScamShield AI. "
            "They are not real user reports, not the Enron corpus, and not a production fraud feed."
        ),
        "limitations": [
            "The model memorizes patterns in a small synthetic set and will miss paraphrases it has not seen.",
            "Logistic regression probabilities are not perfectly calibrated.",
            "Bag-of-words features ignore word order beyond bigrams and do not understand sarcasm or images.",
            "Channel is not a model feature. SMS, email, chat, and call text share one classifier.",
            "URL, QR, OCR, and audio modules depend on this classifier only after text is extracted.",
        ],
    }
    save_classifier(pipeline, metrics, model_path)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    clear_classifier_cache()
    return metrics


def main() -> None:
    metrics = train_and_save()
    print(
        f"Saved {MODEL_PATH.name}. Held-out accuracy {metrics['accuracy']:.3f}, "
        f"macro F1 {metrics['macro_f1']:.3f} "
        f"({metrics['train_size']} train / {metrics['test_size']} test)."
    )


if __name__ == "__main__":
    main()
