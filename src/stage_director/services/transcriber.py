"""Speech-to-text using faster-whisper."""

import io
import logging
from typing import Protocol

logger = logging.getLogger(__name__)


class Transcriber(Protocol):
    def transcribe(self, audio: bytes) -> str: ...


class WhisperTranscriber:
    def __init__(self, model_size: str = "small") -> None:
        from faster_whisper import WhisperModel

        from stage_director.services.hardware import gpu_available

        use_gpu = gpu_available()
        device = "cuda" if use_gpu else "cpu"
        compute_type = "float16" if use_gpu else "int8"

        logger.info("Loading Whisper %s on %s ...", model_size, device)
        self._model = WhisperModel(model_size, device=device, compute_type=compute_type)
        logger.info("Whisper loaded")

    def transcribe(self, audio: bytes) -> str:
        segments, info = self._model.transcribe(io.BytesIO(audio), beam_size=5)
        text = " ".join(segment.text.strip() for segment in segments).strip()
        logger.info("Transcribed (%s): %s", info.language, text)
        return text