"""Request and response shapes (the 'contract' of our API)."""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class LightingPreset(str, Enum):
    """Only these values are allowed. Typos become impossible."""

    NEUTRAL = "neutral"
    RED_ALERT = "red_alert"


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