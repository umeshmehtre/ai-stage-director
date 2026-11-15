import pytest

from stage_director.services.director import RuleBasedDirector


@pytest.mark.parametrize(
    "command, expected",
    [
        ("Turn the lights red", "red_alert"),
        ("Make it calm and blue", "blue_calm"),
        ("Spotlight on the dancer", "spotlight"),
        ("Hello there", "neutral"),
        ("The centered actor", "neutral"),
        ("Keep it calm but spotlight the singer", "spotlight"),
    ],
)
def test_rule_based_director(command, expected):
    result = RuleBasedDirector().direct(command)
    assert result.lighting_preset.value == expected