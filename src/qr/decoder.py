"""Decode QR images locally. OpenCV is the default; pyzbar is used when it imports."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _load_image(path: Path):
    import cv2

    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        raise ValueError("The image file is empty.")
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("The file could not be decoded as an image.")
    return image


def _opencv_decode(image) -> str:
    import cv2

    detector = cv2.QRCodeDetector()
    value, _points, _straight = detector.detectAndDecode(image)
    if value:
        return value.strip()
    try:
        ok, decoded, _points, _straight = detector.detectAndDecodeMulti(image)
    except Exception:
        ok, decoded = False, []
    if ok and decoded is not None:
        texts = [item.strip() for item in decoded if item and item.strip()]
        if texts:
            return texts[0]
    return ""


def _pyzbar_decode(image) -> str:
    try:
        from pyzbar.pyzbar import decode as zbar_decode
    except Exception:
        return ""
    import cv2

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    try:
        found = zbar_decode(gray)
    except Exception:
        return ""
    for item in found:
        text = item.data.decode("utf-8", errors="replace").strip()
        if text:
            return text
    return ""


def decode_qr(path: str | Path) -> tuple[str, str]:
    image_path = Path(path)
    if not image_path.is_file():
        raise ValueError("QR image was not found.")
    image = _load_image(image_path)
    value = _opencv_decode(image)
    if value:
        return value, "opencv"
    value = _pyzbar_decode(image)
    if value:
        return value, "pyzbar"
    raise ValueError("No QR code was found in that image. Use a clear, uncropped QR image.")


def write_qr(path: str | Path, payload: str) -> None:
    import qrcode

    image = qrcode.make(payload)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination)
