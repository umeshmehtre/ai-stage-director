"""Turn raw LLM text into a validated DirectResponse.

Kept separate from the model code so it can be tested without a GPU.
"""

import json
import re

from pydantic import ValidationError

from stage_director.schemas import DirectResponse, LLMOutput

_JSON_OBJECT = re.compile(r"\{.*?\}", re.DOTALL)


class DirectorOutputError(Exception):
    """Raised when the LLM output cannot be turned into a valid response."""


def parse_llm_output(raw: str, command: str) -> DirectResponse:
    match = _JSON_OBJECT.search(raw)
    if match is None:
        raise DirectorOutputError(f"No JSON object found in: {raw!r}")

    try:
        parsed = LLMOutput(**json.loads(match.group()))
    except (json.JSONDecodeError, ValidationError, TypeError) as exc:
        raise DirectorOutputError(f"Invalid LLM output: {raw!r}") from exc

    return DirectResponse(
        received_command=command,
        dialogue=parsed.dialogue,
        lighting_preset=parsed.lighting_preset,
    )