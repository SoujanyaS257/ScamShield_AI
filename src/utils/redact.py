"""Redact identifiers before anything is written to history."""

from __future__ import annotations

import hashlib
import re

_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_CARD = re.compile(r"\b(?:\d[ -]?){13,19}\b")
_LABELED_SECRET = re.compile(
    r"\b(aadhaar|ssn|cvv|cvc|otp|pin|password|passcode)\b\s*[:#-]?\s*[\w-]{3,}",
    re.IGNORECASE,
)


def redact(text: str) -> str:
    value = text or ""
    value = _LABELED_SECRET.sub(lambda match: f"{match.group(1)} [redacted]", value)
    value = _SSN.sub("[redacted-id]", value)
    value = _CARD.sub("[redacted-number]", value)
    return value


def preview(text: str, limit: int = 400) -> str:
    cleaned = " ".join(redact(text).split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1].rstrip() + "…"


def content_hash(text: str) -> str:
    digest = hashlib.sha256((text or "").encode("utf-8")).hexdigest()
    return digest[:16]
