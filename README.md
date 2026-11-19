# ScamShield AI

ScamShield AI reviews SMS, email, WhatsApp, call transcripts, URLs, QR codes, images, and voice notes. Each scan is labeled **SAFE**, **SUSPICIOUS**, or **FRAUD**, with a blended risk score, the text model's class probabilities, and the rules that fired.

The scores are computed. Nothing in the app draws a random label or a random confidence value.

## How a decision is made

Text scans (SMS, WhatsApp, email, transcripts, OCR text, and speech transcripts) do two things:

1. A **TF-IDF logistic regression** predicts SAFE, SUSPICIOUS, or FRAUD. Confidence is the highest `predict_proba` value.
2. **Rules** look for high-precision patterns: one-time-code requests, gift-card demands, remote-access tools, look-alike links, spoofed display names, and similar signals.

Blended risk:

```
model risk = 100×P(FRAUD) + 58×P(SUSPICIOUS) + 8×P(SAFE)
rule risk  = min(100, sum of indicator weights)
blended    = 0.62×model risk + 0.38×rule risk
```

If model confidence is at least 55%, the model label is the starting classification. Below that, the blended score uses bands: under 35 SAFE, under 70 SUSPICIOUS, otherwise FRAUD. Any **critical** indicator forces FRAUD. URL scans skip the text model and publish the structural score only.

URL scoring does **not** open the address. Fetching user-supplied URLs would be server-side request forgery.

## Dataset and limits

Training text lives in `data/corpus.py` and is exported to `data/training_data.csv` when you train. The messages were written for this project. They are not real customer reports, not the Enron corpus, and not a production fraud feed.

The held-out accuracy and macro F1 are written to `models/metrics.json` and shown on the About page. Read those numbers from the file after training. Do not memorize a claim from this README, because a future corpus edit changes them.

Limits worth saying out loud:

- A small synthetic set does not represent real inboxes. Paraphrases can be missed.
- Logistic regression probabilities are not perfectly calibrated. That is why confidence and the blended risk score are shown separately.
- Features are unigrams and bigrams. Word order, sarcasm, and images of text are outside the linear model until OCR extracts text.
- One classifier serves every channel. Channel differences are rules, not separate models.
- Grammar and spelling are not scored.
- OCR needs EasyOCR or a Tesseract binary. Voice needs Whisper, or SpeechRecognition plus a network call to Google for WAV, FLAC, or AIFF. If those tools fail, the module stops. It does not invent a transcript.
- This is decision support for a demo and an interview explanation. It is not a mail gateway, a bank control, or proof of fraud.

## Project layout

```
app.py                  Streamlit entry point
frontend/               Theme, widgets, and pages
src/detection/          Engines, rules, URL scoring, fusion
src/ml/                 Training and the saved scikit-learn pipeline
src/preprocessing/      Normalization used inside the model
src/ocr/ src/qr/ src/audio/
src/database/           MongoDB, or data/history.json when MongoDB is down
data/corpus.py          Labeled synthetic messages
tests/                  Unit tests
```

## Install and run (Windows)

```powershell
cd ScamShield-AI
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m src.ml.train
streamlit run app.py
```

Open http://localhost:8501.

Train once if you want metrics before the first click. The app also trains on startup when `models/text_clf.joblib` is missing.

### Tests

```powershell
python -m pytest -q
python scripts/smoke.py
```

### MongoDB

Copy `.env.example` to `.env`. Set `MONGODB_URI` to a reachable server, for example `mongodb://localhost:27017`. If the URI is blank or the ping fails, history is stored in `data/history.json` and the sidebar says so. Stored rows keep a short redacted preview, not uploaded audio or images.

### Optional OCR, speech, and pyzbar

```powershell
python -m pip install -r requirements-optional.txt
```

- **EasyOCR** is large because it installs PyTorch. The image scanner uses it when import works.
- **Tesseract** is the lighter OCR path. Install the Windows binary from the UB Mannheim build, add it to `PATH`, and keep `pytesseract` from the main requirements file. The image page shows which backend is available.
- **openai-whisper** lets the voice page transcribe more than WAV/FLAC/AIFF. Without it, SpeechRecognition uses Google's web recognizer and needs internet.
- **pyzbar** is a second QR decoder. OpenCV is the one the tests use.

The voice page can synthesize a sample with the Windows text-to-speech voice (`pyttsx3`) and then try to transcribe that file. If transcription fails, you still get the error. Use Phone Transcript Scanner to score the words directly.
