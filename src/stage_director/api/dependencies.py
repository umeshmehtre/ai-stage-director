"""Factories that build the heavy objects exactly once."""

import logging
from functools import lru_cache

from stage_director.config import get_settings
from stage_director.services.director import Director, RuleBasedDirector
from stage_director.services.synthesizer import Synthesizer
from stage_director.services.transcriber import Transcriber

logger = logging.getLogger(__name__)


@lru_cache
def get_director() -> Director:
    settings = get_settings()
    if settings.director_backend == "llm":
        from stage_director.services.llm_director import LLMDirector

        return LLMDirector(settings.llm_model_id)
    return RuleBasedDirector()


@lru_cache
def get_transcriber() -> Transcriber:
    from stage_director.services.transcriber import WhisperTranscriber

    settings = get_settings()
    return WhisperTranscriber(
        settings.whisper_model_size,
        settings.whisper_language or None,
        settings.whisper_task,
    )


@lru_cache
def get_synthesizer() -> Synthesizer | None:
    """Return the TTS engine, or None if disabled or if it failed to load."""
    settings = get_settings()
    if settings.tts_backend != "kokoro":
        return None
    try:
        from stage_director.services.synthesizer import KokoroSynthesizer

        return KokoroSynthesizer(settings.tts_voice)
    except Exception:
        logger.exception("TTS failed to load. Continuing without audio.")
        return None