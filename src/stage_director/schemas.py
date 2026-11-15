"""Request and response shapes (the 'contract' of our API)."""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class LightingPreset(str, Enum):
    """Only these values are allowed. Typos become impossible."""

    NEUTRAL = "neutral"
    RED_ALERT = "red_alert"
    BLUE_CALM = "blue_calm"
    SPOTLIGHT = "spotlight"


class DirectRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    command: str = Field(
        ...,
        min_length=1,
        max_length=500,
        examples=["Turn the lights red"],
        description="The director's command as text.",
    )


class DirectResponse(BaseModel):
    received_command: str
    dialogue: str
    lighting_preset: LightingPreset


class HealthResponse(BaseModel):
    status: str
    version: str
    gpu_available: bool


class LLMOutput(BaseModel):
    """The exact JSON shape we demand from the LLM."""

    dialogue: str = Field(..., min_length=1, max_length=300)
    lighting_preset: LightingPreset


class TranscribeResponse(BaseModel):
    transcript: str


class VoiceDirectResponse(BaseModel):
    transcript: str
    direction: DirectResponse
    latency_ms: int