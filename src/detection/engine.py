"""Public detection API used by the dashboard and the tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from src.config import LABELS
from src.detection.fusion import POLICY_SUMMARY, fuse
from src.detection.indicators import Indicator, dedupe
from src.detection.rules import email_header_indicators, text_indicators
from src.detection.urls import score_url
from src.ml.classifier import get_classifier
from src.preprocessing.text import normalize
from src.utils.redact import content_hash, preview
from src.utils.validation import require_text, require_url


@dataclass
class DetectionResult:
    input_type: str
    classification: str
    risk_score: float
    model_label: str | None
    model_confidence: float | None
    class_probabilities: dict[str, float] | None
    model_risk: float | None
    rule_risk: float
    indicators: list[Indicator]
    feature_contributions: list[dict]
    policy_notes: list[str]
    explanation: str
    extracted_text: str | None = None
    normalized_text: str | None = None
    metadata: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "input_type": self.input_type,
            "classification": self.classification,
            "risk_score": self.risk_score,
            "model_label": self.model_label,
            "model_confidence": self.model_confidence,
            "class_probabilities": self.class_probabilities,
            "model_risk": self.model_risk,
            "rule_risk": self.rule_risk,
            "indicators": [item.as_dict() for item in self.indicators],
            "feature_contributions": self.feature_contributions,
            "policy_notes": self.policy_notes,
            "explanation": self.explanation,
            "extracted_text": self.extracted_text,
            "normalized_text": self.normalized_text,
            "metadata": self.metadata,
            "policy_summary": POLICY_SUMMARY,
        }


def analyze_text(
    text: str,
    input_type: str,
    *,
    extra_indicators: list[Indicator] | None = None,
    metadata: dict | None = None,
    forwarded: bool = False,
    field_name: str = "Text",
) -> DetectionResult:
    raw = require_text(text, field_name)
    prediction = get_classifier().predict(raw)
    indicators = text_indicators(raw, forwarded=forwarded)
    if extra_indicators:
        indicators = dedupe(indicators + extra_indicators)
    decision = fuse(
        indicators,
        model_label=prediction["label"],
        probabilities=prediction["probabilities"],
    )
    contributions = get_classifier().contributions(raw, prediction["label"])
    return DetectionResult(
        input_type=input_type,
        classification=decision.classification,
        risk_score=decision.risk_score,
        model_label=decision.model_label,
        model_confidence=decision.model_confidence,
        class_probabilities=decision.class_probabilities,
        model_risk=decision.model_risk,
        rule_risk=decision.rule_risk,
        indicators=indicators,
        feature_contributions=contributions,
        policy_notes=decision.policy_notes,
        explanation=decision.explanation,
        extracted_text=raw,
        normalized_text=normalize(raw),
        metadata=metadata or {},
    )


def analyze_sms(text: str) -> DetectionResult:
    return analyze_text(text, "sms", field_name="SMS message")


def analyze_whatsapp(text: str, *, forwarded: bool = False) -> DetectionResult:
    return analyze_text(text, "whatsapp", forwarded=forwarded, field_name="WhatsApp message")


def analyze_transcript(text: str) -> DetectionResult:
    return analyze_text(text, "transcript", field_name="Call transcript")


def analyze_email(
    body: str,
    *,
    sender: str = "",
    reply_to: str = "",
    subject: str = "",
    attachment: str = "",
) -> DetectionResult:
    message = "\n".join(
        [
            f"From: {sender.strip()}",
            f"Reply-To: {reply_to.strip()}",
            f"Subject: {subject.strip()}",
            f"Attachment: {attachment.strip()}",
            "",
            body,
        ]
    )
    return analyze_text(
        message,
        "email",
        extra_indicators=email_header_indicators(sender, reply_to, attachment),
        metadata={
            "sender": sender.strip(),
            "reply_to": reply_to.strip(),
            "subject": subject.strip(),
            "attachment": attachment.strip(),
        },
        field_name="Email body",
    )


def analyze_url(url: str, input_type: str = "url") -> DetectionResult:
    raw = require_url(url)
    assessment = score_url(raw)
    notes = [
        "The text classifier was not applied. Opening the address would expose the server to untrusted network requests, so only the URL string was scored."
    ]
    if assessment.classification == "FRAUD" and any(item.severity == "critical" for item in assessment.indicators):
        notes.append("A critical URL finding set the classification to FRAUD.")
    explanation = (
        f"Final classification: {assessment.classification}. "
        f"Structural risk score: {assessment.risk_score:.1f} out of 100, the capped sum of URL findings. "
        + ("Indicators: " + "; ".join(f"{item.name} ({item.weight:g})" for item in assessment.indicators) + ". " if assessment.indicators else "No URL findings fired. ")
        + " ".join(notes)
        + " A URL score is a decision-support signal, not proof that the destination is safe or fraudulent."
    )
    return DetectionResult(
        input_type=input_type,
        classification=assessment.classification,
        risk_score=assessment.risk_score,
        model_label=None,
        model_confidence=None,
        class_probabilities=None,
        model_risk=None,
        rule_risk=assessment.risk_score,
        indicators=assessment.indicators,
        feature_contributions=[],
        policy_notes=notes,
        explanation=explanation,
        extracted_text=assessment.normalized_url,
        normalized_text=None,
        metadata={"url_findings": assessment.findings},
    )


def analyze_qr_image(path: str | Path) -> DetectionResult:
    from src.qr.decoder import decode_qr

    payload, backend = decode_qr(path)
    if _looks_like_url(payload):
        result = analyze_url(payload, input_type="qr")
    else:
        result = analyze_text(payload, "qr", field_name="QR payload")
    result.metadata = {**result.metadata, "qr_decoder": backend, "filename": Path(path).name}
    result.extracted_text = payload
    return result


def analyze_image(path: str | Path) -> DetectionResult:
    from src.ocr.reader import extract_text

    text, backend = extract_text(path)
    result = analyze_text(text, "image", field_name="OCR text")
    result.metadata = {**result.metadata, "ocr_backend": backend, "filename": Path(path).name}
    result.extracted_text = text
    return result


def analyze_audio(path: str | Path) -> DetectionResult:
    from src.audio.transcriber import transcribe

    text, backend = transcribe(path)
    result = analyze_text(text, "voice", field_name="Transcript")
    result.metadata = {**result.metadata, "speech_backend": backend, "filename": Path(path).name}
    result.extracted_text = text
    return result


def to_history_record(result: DetectionResult, source_text: str, filename: str | None = None) -> dict:
    if result.classification not in LABELS:
        raise ValueError("Refusing to store an unknown classification.")
    stored = source_text or result.extracted_text or ""
    return {
        "id": str(uuid4()),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input_type": result.input_type,
        "content_preview": preview(stored),
        "content_sha256": content_hash(stored),
        "classification": result.classification,
        "risk_score": result.risk_score,
        "model_label": result.model_label,
        "model_confidence": None if result.model_confidence is None else round(result.model_confidence, 4),
        "model_risk": result.model_risk,
        "rule_risk": result.rule_risk,
        "explanation": result.explanation,
        "indicators": [item.as_dict() for item in result.indicators],
        "filename": filename,
    }


def _looks_like_url(payload: str) -> bool:
    text = payload.strip().lower()
    return text.startswith(("http://", "https://", "javascript:", "data:", "www.")) or (
        " " not in text and "." in text and "/" in text
    )
