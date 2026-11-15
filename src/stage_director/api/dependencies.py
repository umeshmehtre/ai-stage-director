"""Factories that build the heavy objects exactly once."""

from functools import lru_cache

from stage_director.config import get_settings
from stage_director.services.director import Director, RuleBasedDirector
from stage_director.services.transcriber import Transcriber


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

    return WhisperTranscriber(get_settings().whisper_model_size)