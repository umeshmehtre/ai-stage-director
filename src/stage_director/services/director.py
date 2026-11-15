"""The 'director' decides how to react to a command.

Two implementations share the same shape (the Director protocol):
- RuleBasedDirector: simple keyword rules (no GPU needed).
- LLMDirector (llm_director.py): Qwen-powered.
"""

import logging
from typing import Protocol

from stage_director.schemas import DirectResponse, LightingPreset

logger = logging.getLogger(__name__)


class Director(Protocol):
    """Anything with a direct() method is a valid director."""

    def direct(self, command: str) -> DirectResponse: ...


class RuleBasedDirector:
    """Placeholder brain using keyword rules."""

    def direct(self, command: str) -> DirectResponse:
        logger.info("Received command: %s", command)

        if "red" in command.lower():
            preset = LightingPreset.RED_ALERT
        else:
            preset = LightingPreset.NEUTRAL

        return DirectResponse(
            received_command=command,
            dialogue=f"Copy that, director. Executing: {command}",
            lighting_preset=preset,
        )