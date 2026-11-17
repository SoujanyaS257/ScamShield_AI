"""Extract text from images with EasyOCR or Tesseract, whichever is installed."""

from __future__ import annotations

from pathlib import Path

_EASYOCR_READER = None


def write_text_image(path: str | Path, text: str) -> None:
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", (1100, 280), "white")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 28)
    except OSError:
        font = ImageFont.load_default()
    draw.multiline_text((24, 40), text, fill="black", font=font, spacing=8)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination)


def _easyocr(path: Path) -> str:
    global _EASYOCR_READER
    import numpy as np
    from PIL import Image

    import easyocr

    if _EASYOCR_READER is None:
        local_model_dir = Path(__file__).resolve().parents[2] / "data" / "models" / "easyocr" / "model"
        if local_model_dir.is_dir():
            user_dir = local_model_dir.parent / "user_network"
            user_dir.mkdir(parents=True, exist_ok=True)
            _EASYOCR_READER = easyocr.Reader(
                ["en"],
                gpu=False,
                model_storage_directory=str(local_model_dir),
                user_network_directory=str(user_dir),
                download_enabled=False,
                verbose=False,
            )
        else:
            _EASYOCR_READER = easyocr.Reader(["en"], gpu=False, verbose=False)
    image = np.array(Image.open(path).convert("RGB"))
    lines = _EASYOCR_READER.readtext(image, detail=0, paragraph=True)
    return "\n".join(str(line) for line in lines).strip()


def _tesseract(path: Path) -> str:
    import pytesseract
    from PIL import Image

    return pytesseract.image_to_string(Image.open(path)).strip()


def extract_text(path: str | Path) -> tuple[str, str]:
    image_path = Path(path)
    if not image_path.is_file():
        raise ValueError("Image file was not found.")
    errors: list[str] = []
    for backend, reader in (("easyocr", _easyocr), ("tesseract", _tesseract)):
        try:
            text = reader(image_path)
        except Exception as exc:
            errors.append(f"{backend}: {exc.__class__.__name__}: {exc}")
            continue
        if text:
            return text, backend
        errors.append(f"{backend}: no text found")
    joined = " | ".join(errors)
    raise RuntimeError(
        "OCR could not read this image. Install Tesseract and keep it on PATH, "
        "or install EasyOCR from requirements-optional.txt. Details: " + joined
    )


def available_backends() -> dict[str, str]:
    status = {"easyocr": "not installed", "tesseract": "not installed"}
    try:
        import easyocr  # noqa: F401

        status["easyocr"] = "installed"
    except Exception as exc:
        status["easyocr"] = f"unavailable ({exc.__class__.__name__})"
    try:
        import pytesseract

        pytesseract.get_tesseract_version()
        status["tesseract"] = "installed"
    except Exception as exc:
        status["tesseract"] = f"unavailable ({exc.__class__.__name__})"
    return status
