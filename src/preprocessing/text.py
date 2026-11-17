"""Normalize text before TF-IDF. Rules still see the raw message."""

from __future__ import annotations

import re
import unicodedata

_ZERO_WIDTH = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060\ufeff]")
_URL = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
_EMAIL = re.compile(r"\b[\w.+-]+@[\w.-]+\.\w+\b")
_LONG_NUMBER = re.compile(r"\b\d{5,}\b")
_NON_LETTER = re.compile(r"[^a-z\s]+")
_SPACES = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Lowercase, strip obfuscation characters, and collapse contact tokens.

    URLs, email addresses, and long digit runs become placeholder tokens so
    the linear model learns that contact data is present without memorizing
    a specific domain. Domain quality is scored separately by URL rules.
    """
    value = unicodedata.normalize("NFKC", text or "")
    value = _ZERO_WIDTH.sub("", value)
    value = value.lower()
    value = _URL.sub(" url ", value)
    value = _EMAIL.sub(" email ", value)
    value = _LONG_NUMBER.sub(" number ", value)
    value = _NON_LETTER.sub(" ", value)
    value = _SPACES.sub(" ", value).strip()
    return value
