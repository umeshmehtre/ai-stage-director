"""Text-to-speech using Kokoro."""

import io
import logging
from typing import Protocol

logger = logging.getLogger(__name__)


class Synthesizer(Protocol):
    def synthesize(self, text: str) -> bytes:
        """Return a WAV file (as bytes) speaking the given text."""
        ...


class KokoroSynthesizer:
    SAMPLE_RATE = 24000  # Kokoro always outputs 24 kHz audio

    def __init__(self, voice: str = "af_heart", lang_code: str = "a") -> None:
        # Heavy import lives here so machines without Kokoro still run.
        from kokoro import KPipeline

        logger.info("Loading Kokoro (voice=%s) ...", voice)
        self._pipeline = KPipeline(lang_code=lang_code)  # "a" = American English
        self._voice = voice
        logger.info("Kokoro loaded")

    def synthesize(self, text: str) -> bytes:
        import numpy as np
        import soundfile as sf

        chunks = []
        for result in self._pipeline(text, voice=self._voice):
            # Different Kokoro versions yield an object or a tuple.
            audio = result.audio if hasattr(result, "audio") else result[2]
            if hasattr(audio, "detach"):  # torch tensor -> numpy
                audio = audio.detach().cpu().numpy()
            chunks.append(np.asarray(audio, dtype="float32"))

        if not chunks:
            raise RuntimeError("Kokoro produced no audio")

        buffer = io.BytesIO()
        sf.write(buffer, np.concatenate(chunks), self.SAMPLE_RATE, format="WAV")
        return buffer.getvalue()