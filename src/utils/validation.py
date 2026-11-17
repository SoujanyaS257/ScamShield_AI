"""Input checks shared by the scanners."""

from __future__ import annotations

from pathlib import Path

from src.config import MAX_TEXT_CHARS, MAX_UPLOAD_BYTES, MAX_URL_CHARS


def require_text(text: str | None, field: str = "Text") -> str:
    value = "" if text is None else str(text).strip()
    if not value:
        raise ValueError(f"{field} is empty.")
    if len(value) > MAX_TEXT_CHARS:
        raise ValueError(f"{field} is longer than {MAX_TEXT_CHARS:,} characters.")
    return value


def require_url(url: str | None) -> str:
    value = "" if url is None else str(url).strip()
    if not value:
        raise ValueError("Enter a URL.")
    if len(value) > MAX_URL_CHARS:
        raise ValueError(f"URL is longer than {MAX_URL_CHARS:,} characters.")
    return value


def check_upload(name: str, size: int, allowed: set[str]) -> str:
    suffix = Path(name or "").suffix.lower()
    if suffix not in allowed:
        allowed_list = ", ".join(sorted(allowed))
        raise ValueError(f"Unsupported file type '{suffix or 'unknown'}'. Allowed: {allowed_list}.")
    if size <= 0:
        raise ValueError("The uploaded file is empty.")
    if size > MAX_UPLOAD_BYTES:
        raise ValueError("File is larger than 8 MB.")
    return suffix
