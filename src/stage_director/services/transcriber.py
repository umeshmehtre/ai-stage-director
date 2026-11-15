"""Speech-to-text using faster-whisper (audio decoded with ffmpeg, not PyAV)."""

import logging
import subprocess
from typing import Protocol

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000  # Whisper expects 16 kHz mono audio


class Transcriber(Protocol):
    def transcribe(self, audio: bytes) -> str: ...


def decode_audio(audio: bytes):
    """Decode any audio container (webm, wav, mp3...) to 16 kHz mono float32."""
    import numpy as np

    result = subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", "pipe:0",
         "-f", "f32le", "-ac", "1", "-ar", str(SAMPLE_RATE), "pipe:1"],
        input=audio, capture_output=True, timeout=30,
    )
    if result.returncode != 0:
        raise ValueError(
            f"ffmpeg could not decode the audio: {result.stderr.decode(errors='ignore')[:300]}"
        )
    return np.frombuffer(result.stdout, dtype=np.float32)


class WhisperTranscriber:
    def __init__(
        self,
        model_size: str = "small",
        language: str | None = "en",
        task: str = "transcribe",
    ) -> None:
        from faster_whisper import WhisperModel

        from stage_director.services.hardware import gpu_available

        use_gpu = gpu_available()
        device = "cuda" if use_gpu else "cpu"
        compute_type = "float16" if use_gpu else "int8"

        self._language = language  # None = auto-detect
        self._task = task  # "transcribe" or "translate" (to English)
        logger.info(
            "Loading Whisper %s on %s (language=%s, task=%s) ...",
            model_size, device, language or "auto", task,
        )
        self._model = WhisperModel(model_size, device=device, compute_type=compute_type)
        logger.info("Whisper loaded")

    def transcribe(self, audio: bytes) -> str:
        samples = decode_audio(audio)
        segments, info = self._model.transcribe(
            samples, beam_size=5, language=self._language, task=self._task
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
        logger.info("Transcribed (%s): %s", info.language, text)
        return text