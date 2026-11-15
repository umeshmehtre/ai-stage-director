"""The 'director' decides how to react to a command.

Two implementations share the same shape (the Director protocol):
- RuleBasedDirector: keyword rules. No GPU, no model, works offline.
- LLMDirector (llm_director.py): fine-tuned Qwen.
"""

import logging
import re
from typing import Protocol

from stage_director.schemas import DirectResponse, LightingPreset

logger = logging.getLogger(__name__)


class Director(Protocol):
    """Anything with a direct() method is a valid director."""

    def direct(self, command: str) -> DirectResponse: ...


# Order matters: a spotlight request wins over a mood word (same rule as the hard test set).
_RULES: list[tuple[LightingPreset, tuple[str, ...], str]] = [
    (LightingPreset.SPOTLIGHT, ("spotlight", "spot", "highlight", "focus", "beam"), "Spotlight on!"),
    (LightingPreset.RED_ALERT, ("red", "alert", "emergency", "danger", "alarm", "siren"), "Switching to red alert!"),
    (LightingPreset.BLUE_CALM, ("blue", "calm", "relax", "peaceful", "serene"), "Setting a calm blue mood."),
]
_NEUTRAL_REPLY = "Understood, keeping the lighting neutral."


def _has_word(words: tuple[str, ...], text: str) -> bool:
    """Whole-word match, so 'centered' does not trigger 'red'."""
    return re.search(r"\b(" + "|".join(words) + r")\b", text) is not None


class RuleBasedDirector:
    """Offline fallback brain using keyword rules."""

    def direct(self, command: str) -> DirectResponse:
        logger.info("Received command: %s", command)
        text = command.lower()

        for preset, words, reply in _RULES:
            if _has_word(words, text):
                return DirectResponse(
                    received_command=command, dialogue=reply, lighting_preset=preset
                )

        return DirectResponse(
            received_command=command,
            dialogue=_NEUTRAL_REPLY,
            lighting_preset=LightingPreset.NEUTRAL,
        )