"""Run every detection module once and print the real scores."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import samples
from src.detection.engine import (
    analyze_audio,
    analyze_email,
    analyze_image,
    analyze_qr_image,
    analyze_sms,
    analyze_transcript,
    analyze_url,
    analyze_whatsapp,
)
from src.ml.train import train_and_save
from src.qr.decoder import write_qr


def _line(name: str, result) -> None:
    confidence = "n/a" if result.model_confidence is None else f"{result.model_confidence:.1%}"
    print(
        f"{name:16} {result.classification:11} risk {result.risk_score:5.1f}  "
        f"model {result.model_label or '—':11} confidence {confidence}"
    )


def main() -> None:
    metrics = train_and_save()
    print(f"Held-out accuracy {metrics['accuracy']:.3f}  macro F1 {metrics['macro_f1']:.3f}")
    _line("sms safe", analyze_sms(samples.SMS_SAFE))
    _line("sms fraud", analyze_sms(samples.SMS_FRAUD))
    _line("whatsapp", analyze_whatsapp(samples.WHATSAPP_FRAUD, forwarded=True))
    _line(
        "email",
        analyze_email(
            samples.EMAIL_BODY,
            sender=samples.EMAIL_SENDER,
            reply_to="refunds@mail.example",
            subject=samples.EMAIL_SUBJECT,
            attachment="invoice.html",
        ),
    )
    _line("url safe", analyze_url(samples.URL_SAFE))
    _line("url fraud", analyze_url(samples.URL_FRAUD))
    _line("transcript", analyze_transcript(samples.TRANSCRIPT_FRAUD))
    with tempfile.TemporaryDirectory() as directory:
        qr_path = Path(directory) / "sample.png"
        write_qr(qr_path, samples.QR_PAYLOAD)
        _line("qr", analyze_qr_image(qr_path))
        try:
            from src.ocr.reader import write_text_image

            image_path = Path(directory) / "note.png"
            write_text_image(image_path, samples.IMAGE_TEXT)
            _line("image", analyze_image(image_path))
        except Exception as exc:
            print(f"{'image':16} SKIPPED  {exc}")
        try:
            from src.audio.transcriber import synthesize_sample

            audio_path = Path(directory) / "call.wav"
            synthesize_sample(audio_path, samples.VOICE_SCRIPT)
            _line("voice", analyze_audio(audio_path))
        except Exception as exc:
            print(f"{'voice':16} SKIPPED  {exc}")


if __name__ == "__main__":
    main()
