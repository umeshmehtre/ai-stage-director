import pytest

from stage_director.schemas import LightingPreset
from stage_director.services.parsing import DirectorOutputError, parse_llm_output


def test_parses_clean_json():
    raw = '{"dialogue": "Going red!", "lighting_preset": "red_alert"}'
    result = parse_llm_output(raw, "make it red")
    assert result.lighting_preset == LightingPreset.RED_ALERT
    assert result.received_command == "make it red"


def test_parses_json_surrounded_by_text():
    raw = 'Sure! {"dialogue": "Hi", "lighting_preset": "neutral"} Hope that helps.'
    assert parse_llm_output(raw, "hi").lighting_preset == LightingPreset.NEUTRAL


def test_rejects_unknown_preset():
    raw = '{"dialogue": "Hi", "lighting_preset": "purple_rain"}'
    with pytest.raises(DirectorOutputError):
        parse_llm_output(raw, "hi")


def test_rejects_no_json():
    with pytest.raises(DirectorOutputError):
        parse_llm_output("I cannot do that.", "hi")