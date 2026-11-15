"""The 'director' decides how to react to a command.

Today: simple rules. Milestone 2: replace with Whisper + Qwen.
Keeping this separate means the API layer never changes when the brain does.
"""

import logging

from stage_director.schemas import DirectResponse, LightingPreset

logger = logging.getLogger(__name__)


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