"""Paths and tunable constants for ScamShield AI."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
CORPUS_CSV = DATA_DIR / "training_data.csv"
MODEL_PATH = MODEL_DIR / "text_clf.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"
HISTORY_PATH = Path(os.getenv("HISTORY_FALLBACK_PATH", str(DATA_DIR / "history.json")))
if not HISTORY_PATH.is_absolute():
    HISTORY_PATH = ROOT / HISTORY_PATH

MONGODB_URI = os.getenv("MONGODB_URI", "").strip()
MONGODB_DB = os.getenv("MONGODB_DB", "scamshield").strip() or "scamshield"
MONGODB_COLLECTION = os.getenv("MONGODB_COLLECTION", "detections").strip() or "detections"

LABELS = ("SAFE", "SUSPICIOUS", "FRAUD")

# Blended risk = MODEL_BLEND * model_risk + RULE_BLEND * rule_risk.
MODEL_BLEND = 0.62
RULE_BLEND = 0.38
MODEL_RISK_SAFE = 8.0
MODEL_RISK_SUSPICIOUS = 58.0
MODEL_RISK_FRAUD = 100.0

# Below this confidence, the model label is not trusted on its own.
MODEL_CONFIDENCE_FLOOR = 0.55
RISK_SUSPICIOUS = 35.0
RISK_FRAUD = 70.0
RULE_RISK_CAP = 100.0

MAX_TEXT_CHARS = 8000
MAX_URL_CHARS = 2000
MAX_UPLOAD_BYTES = 8_000_000

RANDOM_STATE = 42
TEST_FRACTION = 0.25
