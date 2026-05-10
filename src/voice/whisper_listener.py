from __future__ import annotations

import io
import logging
import queue
import threading
from pathlib import Path
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)


class WhisperListener:
    """
    High-quality STT using OpenAI Whisper (local model).
    Streams audio from microphone in chunks and transcribes continuously.
    Falls back to SpeechRecognition if Whisper is unavailable.
    """

    MODEL_SIZE = "base"    # tiny/base/small/medium/large

    def __init__(self, model_size: str = MODEL_SIZE, language: str = "fr"):
        self.language    = language
        self.model_size  = model_size
        self._model      = None
        self._running    = False
        self._q: queue.Queue[str] = queue.Queue()
        self._load_model()

    def _load_model(self) -> None:
        try:
            import whisper
            self._model = whisper.load_model(self.model_size)
            logger.info("Whisper model '%s' loaded", self.model_size)
        except Exception as e:
            logger.debug("Whisper unavailable: %s — falling back to SpeechRecognition", e)

    def transcribe_file(self, audio_path: str | Path) -> str:
        if self._model is None:
            return ""
        try:
            result = self._model.transcribe(
                str(audio_path),
                language=self.language,
                fp16=False,
            )
            return result.get("text", "").strip()
        except Exception as e:
            logger.debug("Whisper transcribe error: %s", e)
            return ""

    def transcribe_bytes(self, audio_bytes: bytes, sample_rate: int = 16000) -> str:
        if self._model is None:
            return self._fallback_transcribe(audio_bytes, sample_rate)
        try:
            audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            result   = self._model.transcribe(audio_np, language=self.language, fp16=False)
            return result.get("text", "").strip()
        except Exception as e:
            logger.debug("Whisper bytes error: %s", e)
            return ""

    def start_streaming(self, callback: Callable[[str], None], chunk_seconds: int = 3) -> None:
        self._running = True
        t = threading.Thread(
            target=self._stream_loop, args=(callback, chunk_seconds), daemon=True
        )
        t.start()

    def stop_streaming(self) -> None:
        self._running = False

    def listen_once(self, timeout: int = 10) -> str:
        try:
            import speech_recognition as sr
            r   = sr.Recognizer()
            mic = sr.Microphone(sample_rate=16000)
            with mic as source:
                r.adjust_for_ambient_noise(source, duration=0.3)
                audio = r.listen(source, timeout=timeout, phrase_time_limit=15)
            wav = audio.get_wav_data()
            if self._model:
                return self.transcribe_bytes(wav)
            return r.recognize_google(audio, language=self.language + "-" + self.language.upper())
        except Exception as e:
            logger.debug("listen_once error: %s", e)
            return ""

    # ── Private ───────────────────────────────────────────────────────

    def _stream_loop(self, callback: Callable[[str], None], chunk_seconds: int) -> None:
        try:
            import sounddevice as sd
            sample_rate = 16000
            chunk_size  = sample_rate * chunk_seconds
            buffer: list[np.ndarray] = []

            def audio_callback(indata, frames, time, status):
                buffer.append(indata.copy().flatten())
                if len(buffer) * frames >= chunk_size:
                    chunk = np.concatenate(buffer)
                    buffer.clear()
                    text  = self.transcribe_bytes((chunk * 32768).astype(np.int16).tobytes())
                    if text:
                        callback(text)

            with sd.InputStream(samplerate=sample_rate, channels=1, callback=audio_callback):
                while self._running:
                    import time
                    time.sleep(0.1)
        except Exception as e:
            logger.debug("Stream loop error: %s", e)

    def _fallback_transcribe(self, audio_bytes: bytes, sample_rate: int) -> str:
        try:
            import speech_recognition as sr
            r = sr.Recognizer()
            audio = sr.AudioData(audio_bytes, sample_rate, 2)
            return r.recognize_google(audio, language=self.language)
        except Exception:
            return ""
