"""Application settings, loaded from environment variables."""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All configurable values live here, never hardcoded elsewhere."""

    app_name: str = "AI Stage Director"
    app_version: str = "0.1.0"
    log_level: str = "INFO"
    cors_origins: list[str] = ["*"]  # In production, list your real domain.

    director_backend: Literal["rules", "llm"] = "rules"
    llm_model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"

    whisper_model_size: str = "small"
    whisper_language: str = "en"  # empty string = auto-detect
    whisper_task: Literal["transcribe", "translate"] = "transcribe"
    max_upload_mb: int = 10

    tts_backend: Literal["none", "kokoro"] = "none"
    tts_voice: str = "af_heart"

    model_config = SettingsConfigDict(env_file=".env", env_prefix="STAGE_")


@lru_cache
def get_settings() -> Settings:
    """Create Settings once and reuse it (cached)."""
    return Settings()