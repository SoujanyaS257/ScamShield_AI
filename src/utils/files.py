"""Temporary files for uploads."""

from __future__ import annotations

import tempfile
from pathlib import Path

from src.utils.validation import check_upload


def save_upload(name: str, data: bytes, allowed: set[str]) -> Path:
    suffix = check_upload(name, len(data), allowed)
    handle = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    handle.write(data)
    handle.close()
    return Path(handle.name)
