"""Dashboard pages."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from frontend.components import show_result
from src import samples
from src.config import METRICS_PATH
from src.detection.engine import (
    analyze_audio,
    analyze_email,
    analyze_image,
    analyze_qr_image,
    analyze_sms,
    analyze_text,
    analyze_transcript,
    analyze_url,
    analyze_whatsapp,
)
from src.detection.fusion import POLICY_SUMMARY
from src.utils.capabilities import probe
from src.utils.files import save_upload

IMAGE_TYPES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
AUDIO_TYPES = {".wav", ".flac", ".aiff", ".aif", ".mp3", ".m4a", ".ogg"}


def render_home(store) -> None:
    st.markdown("<div class='hero-kicker'>ScamShield AI</div>", unsafe_allow_html=True)
    st.title("Intelligent multi-modal scam detection")
    st.write(
        "Paste a message, email, call transcript, or URL. Upload a QR code, a screenshot, or a voice note. "
        "Every text scan uses a TF-IDF logistic regression model plus written rules. "
        "Risk scores are calculated. They are never sampled at random."
    )
    rows = store.list_recent(limit=300)
    fraud = sum(row["classification"] == "FRAUD" for row in rows)
    suspicious = sum(row["classification"] == "SUSPICIOUS" for row in rows)
    safe = sum(row["classification"] == "SAFE" for row in rows)
    one, two, three, four = st.columns(4)
    one.metric("Stored scans", len(rows))
    two.metric("Fraud", fraud)
    three.metric("Suspicious", suspicious)
    four.metric("Safe", safe)

    st.subheader("Quick text scan")
    channel = st.selectbox(
        "Treat the text as",
        ["sms", "whatsapp", "email", "transcript"],
        format_func=lambda value: value.capitalize(),
    )
    st.session_state.setdefault("quick_text", "")
    if st.button("Load a fraud example", key="quick-sample"):
        st.session_state.quick_text = samples.SMS_FRAUD
        st.rerun()
    text = st.text_area("Content", key="quick_text", height=160, placeholder="Paste a message to score")
    if st.button("Analyze text", type="primary", key="quick-go"):
        _scan(store, "quick", text, lambda raw: analyze_text(raw, channel, field_name="Content"))
    _show_saved("quick")

    st.subheader("Scanners")
    cards = [
        ("sms", "SMS and WhatsApp", "Message patterns, forwarded blasts, and suspicious links."),
        ("email", "Email", "Body text plus display-name, reply-to, and attachment-name checks."),
        ("url", "Website and URL", "Structural phishing signals. The site is not opened."),
        ("qr", "QR code", "Decode a QR image, then score the URL or text inside it."),
        ("image", "Image OCR", "Read text from a screenshot, then run the text engine."),
        ("voice", "Voice", "Transcribe a recording, then score the transcript."),
        ("transcript", "Phone transcript", "Impersonation, payment pressure, and call-control language."),
        ("analytics", "Analytics", "Charts from the scans stored in history."),
    ]
    left, right = st.columns(2)
    for index, (page, title, copy) in enumerate(cards):
        column = left if index % 2 == 0 else right
        with column.container(border=True):
            st.markdown(f"**{title}**")
            st.write(copy)
            if st.button("Open", key=f"open-{page}"):
                st.query_params["page"] = page
                st.rerun()


def render_sms(store) -> None:
    st.title("SMS and message scanner")
    st.write("SMS and WhatsApp share the text model. WhatsApp can add a forwarded-message indicator.")
    tab_sms, tab_whatsapp = st.tabs(["SMS", "WhatsApp"])
    with tab_sms:
        _sample_buttons("sms_text", [("Ordinary message", samples.SMS_SAFE), ("Fraud message", samples.SMS_FRAUD)])
        text = st.text_area("SMS message", key="sms_text", height=180)
        if st.button("Analyze SMS", type="primary"):
            _scan(store, "sms", text, analyze_sms)
        _show_saved("sms")
    with tab_whatsapp:
        _sample_buttons("wa_text", [("Forwarded fee demand", samples.WHATSAPP_FRAUD)])
        forwarded = st.checkbox("Sender marked this message as forwarded")
        text = st.text_area("WhatsApp message", key="wa_text", height=180)
        if st.button("Analyze WhatsApp", type="primary"):
            _scan(store, "whatsapp", text, lambda raw: analyze_whatsapp(raw, forwarded=forwarded))
        _show_saved("whatsapp")


def render_email(store) -> None:
    st.title("Email scanner")
    st.write("The model scores the subject and body. Header rules catch display-name spoofing, a mismatched reply-to, and risky attachment names.")
    if st.button("Load impersonation example"):
        st.session_state.email_sender = samples.EMAIL_SENDER
        st.session_state.email_subject = samples.EMAIL_SUBJECT
        st.session_state.email_body = samples.EMAIL_BODY
        st.session_state.email_reply = "refunds@mail.example"
        st.session_state.email_attachment = "invoice.html"
        st.rerun()
    sender = st.text_input("From", key="email_sender", placeholder="PayPal Security <alert@example.com>")
    reply_to = st.text_input("Reply-To", key="email_reply", placeholder="optional")
    subject = st.text_input("Subject", key="email_subject")
    attachment = st.text_input("Attachment filename", key="email_attachment", placeholder="invoice.pdf")
    body = st.text_area("Body", key="email_body", height=180)
    if st.button("Analyze email", type="primary"):
        _scan(
            store,
            "email",
            body,
            lambda raw: analyze_email(raw, sender=sender, reply_to=reply_to, subject=subject, attachment=attachment),
        )
    _show_saved("email")


def render_url(store) -> None:
    st.title("Website and URL scanner")
    st.write(
        "The address is parsed and scored on this machine. ScamShield does not connect to it. "
        "A missing scheme is treated as http, which itself adds the no-HTTPS finding."
    )
    _sample_buttons(
        "url_text",
        [
            ("Official-style documentation site", samples.URL_SAFE),
            ("Look-alike host", samples.URL_FRAUD),
            ("Short link", samples.URL_SHORT),
        ],
    )
    url = st.text_input("URL", key="url_text", placeholder="https://example.com/path")
    if st.button("Analyze URL", type="primary"):
        _scan(store, "url", url, analyze_url)
    _show_saved("url")


def render_qr(store) -> None:
    st.title("QR code scanner")
    st.write("The image is decoded locally with OpenCV. If OpenCV finds nothing, pyzbar is tried when it is installed. The payload is then scored as a URL or as text.")
    upload = st.file_uploader("QR image", type=["png", "jpg", "jpeg", "webp", "bmp"])
    if st.button("Analyze a generated sample QR"):
        _scan_generated_qr(store)
    if st.button("Analyze uploaded QR", type="primary", disabled=upload is None):
        _scan_file(store, "qr-upload", upload, IMAGE_TYPES, analyze_qr_image)
    _show_saved("qr-sample")
    _show_saved("qr-upload")


def render_image(store) -> None:
    st.title("Image scanner")
    st.write("OCR reads the picture, then the text engine scores what was read. EasyOCR is preferred when installed. Otherwise Tesseract is used if the binary is on PATH.")
    from src.ocr.reader import available_backends

    status = available_backends()
    st.caption("EasyOCR: " + status["easyocr"] + " · Tesseract: " + status["tesseract"])
    upload = st.file_uploader("Screenshot or photo", type=["png", "jpg", "jpeg", "webp", "bmp"], key="image-upload")
    if st.button("Analyze a generated sample image"):
        _scan_generated_image(store)
    if st.button("Analyze uploaded image", type="primary", disabled=upload is None):
        _scan_file(store, "image-upload", upload, IMAGE_TYPES, analyze_image)
    _show_saved("image-sample")
    _show_saved("image-upload")


def render_voice(store) -> None:
    st.title("Voice scanner")
    st.write(
        "Audio is transcribed, then the transcript is scored with the same text engine. "
        "Whisper is used when it is installed. Otherwise SpeechRecognition sends WAV, FLAC, or AIFF audio to Google's recognizer, which needs a network connection. "
        "The synthesized sample button uses the Windows voice, then tries that transcription path. If transcription fails, the scan stops. The spoken script is not labeled as a transcript."
    )
    from src.audio.transcriber import available_backends

    status = available_backends()
    st.caption("Whisper: " + status["whisper"] + " · SpeechRecognition: " + status["speech_recognition"])
    upload = st.file_uploader("Recording", type=["wav", "flac", "aiff", "aif", "mp3", "m4a", "ogg"], key="voice-upload")
    if st.button("Create a spoken sample and analyze it"):
        _scan_generated_voice(store)
    if st.button("Analyze uploaded audio", type="primary", disabled=upload is None):
        _scan_file(store, "voice-upload", upload, AUDIO_TYPES, analyze_audio)
    _show_saved("voice-sample")
    _show_saved("voice-upload")
    st.info(
        "If speech recognition is unavailable, open Phone Transcript Scanner and paste the words. "
        f"Sample script, for that fallback only: {samples.VOICE_SCRIPT}"
    )


def render_transcript(store) -> None:
    st.title("Phone transcript scanner")
    st.write("Paste a call transcript. Rules look for authority threats, gift-card demands, remote-access tools, and instructions not to hang up.")
    _sample_buttons("call_text", [("Tax impersonation", samples.TRANSCRIPT_FRAUD)])
    text = st.text_area("Transcript", key="call_text", height=200)
    if st.button("Analyze transcript", type="primary"):
        _scan(store, "transcript", text, analyze_transcript)
    _show_saved("transcript")


def render_history(store) -> None:
    st.title("Detection history")
    st.caption(f"Storage backend: {store.mode}. Previews are truncated and obvious secrets are redacted. Audio and images are not stored.")
    if store.warning:
        st.info(store.warning)
    rows = store.list_recent()
    if not rows:
        st.info("No scans yet. Run a scanner and the record will appear here.")
        return
    frame = pd.DataFrame(rows)
    columns = [
        "created_at",
        "input_type",
        "classification",
        "risk_score",
        "model_label",
        "model_confidence",
        "content_preview",
    ]
    existing = [column for column in columns if column in frame.columns]
    st.dataframe(frame[existing], width="stretch", hide_index=True)
    if st.checkbox("I want to delete every stored scan"):
        if st.button("Clear history"):
            store.clear()
            st.rerun()


def render_analytics(store) -> None:
    st.title("Analytics")
    rows = store.list_recent(limit=1000)
    if not rows:
        st.info("Analytics uses stored scans. Run a few scanners first.")
        return
    frame = pd.DataFrame(rows)
    labels = frame["classification"].value_counts().rename_axis("classification").reset_index(name="count")
    types = frame["input_type"].value_counts().rename_axis("input_type").reset_index(name="count")
    left, right = st.columns(2)
    with left:
        figure = px.pie(labels, names="classification", values="count", color="classification", color_discrete_map={"SAFE": "#34d399", "SUSPICIOUS": "#fbbf24", "FRAUD": "#fb7185"}, title="Classifications")
        figure.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#e5e7eb", height=360)
        st.plotly_chart(figure, width="stretch")
    with right:
        figure = px.bar(types, x="input_type", y="count", title="Scans by input type")
        figure.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e5e7eb", height=360)
        st.plotly_chart(figure, width="stretch")
    frame["created_at"] = pd.to_datetime(frame["created_at"], utc=True, errors="coerce")
    dated = frame.dropna(subset=["created_at"]).copy()
    dated["day"] = dated["created_at"].dt.date
    daily = dated.groupby("day").size().reset_index(name="scans")
    trend = px.line(daily, x="day", y="scans", markers=True, title="Scans by day")
    trend.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e5e7eb", height=340)
    st.plotly_chart(trend, width="stretch")
    spread = px.histogram(frame, x="risk_score", nbins=12, title="Blended risk scores")
    spread.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e5e7eb", height=340)
    st.plotly_chart(spread, width="stretch")


def render_about(store) -> None:
    st.title("About the project")
    st.write(
        "ScamShield AI is a defensive review tool. It classifies content as SAFE, SUSPICIOUS, or FRAUD "
        "and shows the evidence behind the label. It is built to be explained in an engineering interview, "
        "not to replace a bank, an email gateway, or a human analyst."
    )
    st.subheader("How a text decision is made")
    st.write(POLICY_SUMMARY)
    st.subheader("Held-out training metrics")
    if METRICS_PATH.exists():
        import json

        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        one, two, three = st.columns(3)
        one.metric("Held-out accuracy", f"{metrics['accuracy']:.1%}")
        two.metric("Held-out macro F1", f"{metrics['macro_f1']:.1%}")
        three.metric("Test messages", metrics["test_size"])
        st.write(metrics["dataset"])
        st.markdown("**Limitations**")
        for item in metrics["limitations"]:
            st.write(f"- {item}")
        report = metrics.get("classification_report", {})
        rows = []
        for label in ("SAFE", "SUSPICIOUS", "FRAUD"):
            stats = report.get(label, {})
            rows.append(
                {
                    "label": label,
                    "precision": round(float(stats.get("precision", 0)), 3),
                    "recall": round(float(stats.get("recall", 0)), 3),
                    "f1": round(float(stats.get("f1-score", 0)), 3),
                    "support": stats.get("support", 0),
                }
            )
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    else:
        st.info("The model has not been trained in this folder yet. The first scan trains it.")
    st.subheader("What each module does")
    st.markdown(
        """
- **SMS, WhatsApp, email body, transcript, OCR text, and speech text** share one classifier. Channel-specific rules are added around it.
- **URL and QR** use structural rules only. Destinations are not fetched.
- **Image** depends on EasyOCR or Tesseract. If neither can read the file, the scan stops with the installer error.
- **Voice** depends on Whisper or SpeechRecognition. A failed transcription is not replaced with a guessed transcript.
- **History** is MongoDB when `MONGODB_URI` answers a ping. Otherwise the same records go to `data/history.json`.
        """
    )
    st.subheader("Runtime capabilities")
    st.dataframe(pd.DataFrame(probe(store.mode)), width="stretch", hide_index=True)
    st.subheader("Layout")
    st.code(
        "\n".join(
            [
                "app.py                     Streamlit entry point",
                "src/detection              engines, rules, URL scoring, fusion",
                "src/ml                     TF-IDF logistic regression",
                "src/preprocessing          text normalization",
                "src/ocr src/qr src/audio   extraction tools",
                "src/database               MongoDB and JSON history",
                "data/corpus.py             synthetic training messages",
                "frontend                   theme, widgets, pages",
                "tests                      unit tests",
            ]
        ),
        language="text",
    )


def _sample_buttons(key: str, choices: list[tuple[str, str]]) -> None:
    st.session_state.setdefault(key, "")
    columns = st.columns(len(choices))
    for column, (label, value) in zip(columns, choices):
        with column:
            if st.button(label, key=f"sample-{key}-{label}"):
                st.session_state[key] = value
                st.rerun()


def _scan(store, state_key: str, source: str, analyzer) -> None:
    try:
        with st.spinner("Scoring with the text model and the rule set…"):
            result = analyzer(source)
    except Exception as exc:
        st.session_state.pop(f"result-{state_key}", None)
        st.error(str(exc))
        return
    _keep(store, state_key, result, source, None)


def _show_saved(state_key: str) -> None:
    result = st.session_state.get(f"result-{state_key}")
    if result is None:
        return
    note = st.session_state.get(f"save-note-{state_key}")
    if note:
        st.caption(note)
    show_result(result)


def _scan_file(store, state_key: str, upload, allowed: set[str], analyzer) -> None:
    path = None
    try:
        path = save_upload(upload.name, upload.getvalue(), allowed)
        with st.spinner("Reading the file and scoring the extracted content…"):
            result = analyzer(path)
    except Exception as exc:
        st.session_state.pop(f"result-{state_key}", None)
        st.error(str(exc))
        return
    finally:
        if path is not None:
            path.unlink(missing_ok=True)
    _keep(store, state_key, result, result.extracted_text or "", upload.name)


def _scan_generated_qr(store) -> None:
    from src.qr.decoder import write_qr

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "sample-qr.png"
        try:
            write_qr(path, samples.QR_PAYLOAD)
            with st.spinner("Decoding the sample QR…"):
                result = analyze_qr_image(path)
        except Exception as exc:
            st.error(str(exc))
            return
    _keep(store, "qr-sample", result, result.extracted_text or "", "sample-qr.png")


def _scan_generated_image(store) -> None:
    from src.ocr.reader import write_text_image

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "sample-note.png"
        try:
            write_text_image(path, samples.IMAGE_TEXT)
            with st.spinner("Running OCR on the sample image…"):
                result = analyze_image(path)
        except Exception as exc:
            st.error(str(exc))
            return
    _keep(store, "image-sample", result, result.extracted_text or "", "sample-note.png")


def _scan_generated_voice(store) -> None:
    from src.audio.transcriber import synthesize_sample

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "sample-call.wav"
        try:
            with st.spinner("Speaking the sample, then transcribing it…"):
                synthesize_sample(path, samples.VOICE_SCRIPT)
                result = analyze_audio(path)
        except Exception as exc:
            st.error(str(exc))
            return
    _keep(store, "voice-sample", result, result.extracted_text or "", "sample-call.wav")


def _keep(store, state_key: str, result, source: str, filename: str | None) -> None:
    from src.detection.engine import to_history_record

    st.session_state[f"result-{state_key}"] = result
    st.session_state[f"source-{state_key}"] = source
    if st.session_state.get(f"saved-{state_key}") == id(result):
        return
    try:
        store.add(to_history_record(result, source, filename))
        st.session_state[f"saved-{state_key}"] = id(result)
        st.session_state[f"save-note-{state_key}"] = f"Saved in history ({store.mode})."
    except Exception as exc:
        st.session_state[f"save-note-{state_key}"] = f"The scan is shown, but history was not saved: {exc}"
