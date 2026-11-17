"""Optional dependency probes for the sidebar. Imports are attempted and failures are reported."""

from __future__ import annotations

from src.audio.transcriber import available_backends as speech_backends
from src.ocr.reader import available_backends as ocr_backends


def probe(storage_mode: str) -> list[dict[str, str]]:
    rows = [
        {"component": "History storage", "status": storage_mode, "ready": "yes" if storage_mode else "no"},
        {"component": "Text model", "status": "tf-idf logistic regression", "ready": "yes"},
        {"component": "QR decode", "status": "OpenCV, then pyzbar if installed", "ready": "yes"},
    ]
    for name, status in ocr_backends().items():
        rows.append(
            {
                "component": f"OCR {name}",
                "status": status,
                "ready": "yes" if status == "installed" else "no",
            }
        )
    for name, status in speech_backends().items():
        rows.append(
            {
                "component": f"Speech {name}",
                "status": status,
                "ready": "yes" if status.startswith("installed") else "no",
            }
        )
    return rows
