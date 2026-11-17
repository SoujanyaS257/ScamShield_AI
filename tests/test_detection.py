"""Unit tests for scoring, fusion, storage, and QR decoding."""

from __future__ import annotations

from pathlib import Path

import pytest

from src import samples
from src.database import open_store
from src.detection.engine import analyze_email, analyze_sms, analyze_url
from src.detection.fusion import fuse
from src.detection.indicators import Indicator
from src.detection.rules import text_indicators
from src.ml.classifier import clear_classifier_cache
from src.ml.train import train_and_save
from src.preprocessing.text import normalize
from src.utils.redact import redact


@pytest.fixture(scope="session", autouse=True)
def trained_model():
    train_and_save()
    clear_classifier_cache()
    yield
    clear_classifier_cache()


def test_normalize_replaces_links_and_keeps_words():
    assert normalize("Pay at https://evil.example/login now") == "pay at url now"


def test_redact_labeled_secret_and_card():
    cleaned = redact("otp 123456 and card 4111 1111 1111 1111")
    assert "123456" not in cleaned
    assert "4111" not in cleaned


def test_safe_denial_is_not_a_critical_rule():
    indicators = text_indicators("We will never ask for your password, OTP, or a gift card.")
    assert not any(item.severity == "critical" for item in indicators)


def test_otp_request_is_critical_even_after_do_not_hang_up():
    indicators = text_indicators("Do not hang up. Read me the one-time passcode we just sent.")
    assert any(item.name == "otp_harvest" for item in indicators)


def test_fusion_uses_the_documented_formula():
    probabilities = {"SAFE": 0.2, "SUSPICIOUS": 0.3, "FRAUD": 0.5}
    indicators = [Indicator("otp_harvest", "critical", 36, "asks for a code")]
    decision = fuse(indicators, model_label="FRAUD", probabilities=probabilities)
    model_component = 100 * 0.5 + 58 * 0.3 + 8 * 0.2
    expected = round(0.62 * model_component + 0.38 * 36, 1)
    assert decision.risk_score == expected
    assert decision.classification == "FRAUD"
    assert abs(sum(decision.class_probabilities.values()) - 1) < 1e-4


def test_low_confidence_uses_risk_band_without_critical_rules():
    probabilities = {"SAFE": 0.34, "SUSPICIOUS": 0.33, "FRAUD": 0.33}
    indicators = [Indicator("urgency", "info", 8, "act now")]
    decision = fuse(indicators, model_label="SAFE", probabilities=probabilities)
    assert decision.model_confidence < 0.55
    assert decision.classification in {"SAFE", "SUSPICIOUS", "FRAUD"}
    assert "below" in decision.explanation


def test_obvious_messages_and_urls():
    fraud = analyze_sms(samples.SMS_FRAUD)
    assert fraud.classification == "FRAUD"
    assert fraud.model_confidence is not None
    assert abs(sum(fraud.class_probabilities.values()) - 1) < 1e-4
    safe = analyze_sms(samples.SMS_SAFE)
    assert safe.classification != "FRAUD"
    assert not any(item.severity == "critical" for item in safe.indicators)
    official = analyze_url("https://www.paypal.com/signin")
    assert official.classification == "SAFE"
    assert official.model_confidence is None
    lookalike = analyze_url(samples.URL_FRAUD)
    assert lookalike.classification == "FRAUD"
    assert any(item.name == "brand_lookalike" for item in lookalike.indicators)
    short = analyze_url(samples.URL_SHORT)
    assert short.classification == "SUSPICIOUS"
    docs = analyze_url(samples.URL_SAFE)
    assert docs.classification == "SAFE"


def test_email_display_name_impersonation():
    result = analyze_email(
        samples.EMAIL_BODY,
        sender=samples.EMAIL_SENDER,
        reply_to="refunds@mail.example",
        subject=samples.EMAIL_SUBJECT,
        attachment="invoice.html",
    )
    names = {item.name for item in result.indicators}
    assert "display_name_impersonation" in names
    assert "dangerous_attachment_name" in names
    assert "reply_to_mismatch" in names
    assert result.classification == "FRAUD"


def test_empty_input_is_rejected():
    with pytest.raises(ValueError):
        analyze_sms("   ")


def test_json_history_roundtrip(tmp_path: Path):
    store = open_store(mongo_uri="", fallback_path=tmp_path / "history.json")
    assert store.mode == "local-json"
    store.add({"id": "1", "created_at": "2026-01-01T00:00:00+00:00", "classification": "SAFE"})
    store.add({"id": "2", "created_at": "2026-01-02T00:00:00+00:00", "classification": "FRAUD"})
    rows = store.list_recent()
    assert [row["id"] for row in rows] == ["2", "1"]
    store.clear()
    assert store.list_recent() == []


def test_holdout_metrics_are_real():
    from src.config import METRICS_PATH
    import json

    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    assert metrics["train_size"] > metrics["test_size"] > 10
    assert 0 < metrics["accuracy"] <= 1
    assert metrics["macro_f1"] > 0.4


def test_qr_roundtrip(tmp_path: Path):
    pytest.importorskip("cv2")
    from src.qr.decoder import decode_qr, write_qr
    from src.detection.engine import analyze_qr_image

    path = tmp_path / "code.png"
    write_qr(path, samples.QR_PAYLOAD)
    payload, backend = decode_qr(path)
    assert payload == samples.QR_PAYLOAD
    assert backend == "opencv"
    result = analyze_qr_image(path)
    assert result.input_type == "qr"
    assert result.classification == "FRAUD"
    assert result.metadata["qr_decoder"] == "opencv"


def test_model_contributions_exist():
    from src.ml.classifier import get_classifier

    prediction = get_classifier().predict(samples.SMS_FRAUD)
    rows = get_classifier().contributions(samples.SMS_FRAUD, prediction["label"])
    assert rows
    assert {"feature", "contribution", "direction"} <= set(rows[0])


def test_whatsapp_forwarded_flag_adds_indicator():
    from src.detection.engine import analyze_whatsapp

    res = analyze_whatsapp(samples.WHATSAPP_FRAUD, forwarded=True)
    assert res.classification == "FRAUD"
    names = {item.name for item in res.indicators}
    assert "forwarded_blast" in names or any("forwarded" in item.detail.lower() for item in res.indicators)


def test_phone_transcript_threat_detection():
    from src.detection.engine import analyze_transcript

    res = analyze_transcript(samples.TRANSCRIPT_FRAUD)
    assert res.classification == "FRAUD"
    assert res.risk_score >= 60.0
    names = {item.name for item in res.indicators}
    assert "gift_card_payment" in names or "authority_threat" in names


def test_image_ocr_detection(tmp_path: Path):
    from src.detection.engine import analyze_image
    from src.ocr.reader import write_text_image

    img_path = tmp_path / "fraud_memo.png"
    write_text_image(img_path, samples.IMAGE_TEXT)
    res = analyze_image(img_path)
    assert res.input_type == "image"
    assert res.classification == "FRAUD"
    assert res.extracted_text is not None


def test_voice_audio_transcription_detection(tmp_path: Path):
    from src.detection.engine import analyze_audio
    from src.audio.transcriber import synthesize_sample

    audio_path = tmp_path / "fraud_call.wav"
    synthesize_sample(audio_path, samples.VOICE_SCRIPT)
    res = analyze_audio(audio_path)
    assert res.input_type == "voice"
    assert res.classification == "FRAUD"
    assert res.extracted_text is not None
