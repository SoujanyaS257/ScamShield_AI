"""Transcribe speech with Whisper when installed, otherwise SpeechRecognition."""

from __future__ import annotations

from pathlib import Path

_WHISPER = None
_SPEECH_AUDIO_SUFFIXES = {".wav", ".aiff", ".aif", ".flac"}


def transcribe(path: str | Path) -> tuple[str, str]:
    audio_path = Path(path)
    if not audio_path.is_file():
        raise ValueError("Audio file was not found.")
    errors: list[str] = []
    whisper_text = _try_whisper(audio_path, errors)
    if whisper_text:
        return whisper_text, "whisper"
    if audio_path.suffix.lower() not in _SPEECH_AUDIO_SUFFIXES:
        raise RuntimeError(
            "This format needs Whisper. Install openai-whisper from requirements-optional.txt, "
            "or upload a WAV/FLAC file. SpeechRecognition reads WAV, AIFF, and FLAC only. "
            + " ".join(errors)
        )
    speech_text, backend = _try_speech_recognition(audio_path, errors)
    if speech_text:
        return speech_text, backend
    raise RuntimeError(
        "Could not transcribe the recording. Paste the words into Phone Transcript Scanner if you need a result now. "
        + " | ".join(errors)
    )


def _load_audio_array(path: Path):
    """Load audio file into 16kHz float32 mono numpy array without requiring ffmpeg."""
    try:
        import numpy as np
        from scipy.io import wavfile
        from scipy.signal import resample

        sr, data = wavfile.read(str(path))
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            data = data.astype(np.float32) / 2147483648.0
        elif data.dtype == np.uint8:
            data = (data.astype(np.float32) - 128.0) / 128.0
        else:
            data = data.astype(np.float32)

        if data.ndim > 1:
            data = data.mean(axis=1)

        target_sr = 16000
        if sr != target_sr:
            num_samples = int(len(data) * target_sr / sr)
            data = resample(data, num_samples).astype(np.float32)

        return data
    except Exception:
        return None


def _try_whisper(path: Path, errors: list[str]) -> str:
    global _WHISPER
    try:
        import whisper
    except Exception as exc:
        errors.append(f"whisper unavailable ({exc.__class__.__name__})")
        return ""
    try:
        if _WHISPER is None:
            local_model = Path(__file__).resolve().parents[2] / "data" / "models" / "whisper" / "tiny.pt"
            if local_model.is_file():
                _WHISPER = whisper.load_model(str(local_model))
            else:
                _WHISPER = whisper.load_model("tiny")

        audio_input = _load_audio_array(path)
        if audio_input is not None:
            result = _WHISPER.transcribe(audio_input, fp16=False)
        else:
            result = _WHISPER.transcribe(str(path), fp16=False)
    except Exception as exc:
        errors.append(f"whisper failed ({exc.__class__.__name__}: {exc})")
        return ""
    return str(result.get("text") or "").strip()


def _try_speech_recognition(path: Path, errors: list[str]) -> tuple[str, str]:
    try:
        import speech_recognition as sr
    except Exception as exc:
        errors.append(f"SpeechRecognition unavailable ({exc.__class__.__name__})")
        return "", ""
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(str(path)) as source:
            audio = recognizer.record(source)
    except Exception as exc:
        errors.append(f"could not read audio ({exc.__class__.__name__}: {exc})")
        return "", ""
    try:
        text = recognizer.recognize_google(audio)
        if str(text).strip():
            return str(text).strip(), "speech_recognition.google"
        errors.append("Google speech recognition returned empty text")
    except Exception as exc:
        errors.append(f"Google speech recognition failed ({exc.__class__.__name__}: {exc})")
    if hasattr(recognizer, "recognize_sphinx"):
        try:
            text = recognizer.recognize_sphinx(audio)
            if str(text).strip():
                return str(text).strip(), "speech_recognition.sphinx"
        except Exception as exc:
            errors.append(f"Sphinx failed ({exc.__class__.__name__}: {exc})")
    return "", ""


def available_backends() -> dict[str, str]:
    status = {"whisper": "not installed", "speech_recognition": "not installed"}
    try:
        import whisper  # noqa: F401

        status["whisper"] = "installed"
    except Exception as exc:
        status["whisper"] = f"unavailable ({exc.__class__.__name__})"
    try:
        import speech_recognition  # noqa: F401

        status["speech_recognition"] = "installed (Google recognizer needs a network; WAV/FLAC/AIFF only)"
    except Exception as exc:
        status["speech_recognition"] = f"unavailable ({exc.__class__.__name__})"
    return status


def synthesize_sample(path: str | Path, text: str) -> None:
    """Write a WAV with the operating-system voice. Raises if TTS is unavailable."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()

    # Try native Windows SAPI first if available (avoids comtypes permission errors)
    try:
        import win32com.client

        speaker = win32com.client.Dispatch("SAPI.SpVoice")
        stream = win32com.client.Dispatch("SAPI.SpFileStream")
        stream.Open(str(destination.resolve()), 3, False)  # 3 = SSFMCreateForWrite
        speaker.AudioOutputStream = stream
        speaker.Speak(text)
        stream.Close()
        if destination.exists() and destination.stat().st_size > 0:
            return
    except Exception:
        pass

    import pyttsx3

    engine = pyttsx3.init()
    engine.save_to_file(text, str(destination))
    engine.runAndWait()
    if not destination.exists() or destination.stat().st_size == 0:
        raise RuntimeError("The text-to-speech engine did not write an audio file.")
