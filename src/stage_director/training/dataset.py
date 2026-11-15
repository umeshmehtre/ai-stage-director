"""Synthetic dataset builder for fine-tuning. Pure Python, no ML libraries needed.

How leakage is avoided: we hold out whole BASE PHRASES for the test set, so the
test commands use wording the model never saw in training (not just new
prefixes/suffixes on the same phrase).
"""

import json
import random
from pathlib import Path

PREFIXES = ["", "Please ", "Hey director, ", "Quick: ", "Director, ", "Okay, "]
SUFFIXES = ["", " now", " please", "!", " right away", " for this scene"]

BASES: dict[str, list[str]] = {
    "red_alert": [
        "Make the lights red", "Switch to red lighting", "Set the stage to red",
        "Give me red lights", "Go red", "Trigger the red alert", "Warning lights on",
        "Flood the stage with red", "Bathe the stage in red",
        "We need emergency lighting", "Code red on stage",
        "Make it look like an alarm", "Switch on the siren lights",
        "Crimson lights over the stage", "Light it up red", "Alert mode lighting",
        "Everything red", "Paint the stage red", "Emergency lights",
        "Sound the alarm with red lights",
    ],
    "blue_calm": [
        "Switch to blue lighting", "Set the stage to calm blue",
        "Give me soft blue lights", "Make the lights blue", "Go blue",
        "Cool blue ambience", "Bathe the stage in blue",
        "Create a relaxing atmosphere", "Set a serene underwater feel",
        "Night time calm lighting", "Gentle moonlight on stage",
        "Quiet and peaceful lighting", "Soothing blue glow",
        "Dim and cool the lights", "Lullaby scene lighting", "Tranquil blue tones",
        "Make the stage feel like deep water", "Slow and calm mood",
        "Light it up blue", "Chill blue vibe",
    ],
    "spotlight": [
        "Spotlight on the dancer", "Shine a spotlight on the speaker",
        "Focus the light on the performer", "Light up the soloist",
        "Single beam on center stage", "Highlight the main character",
        "Follow spot on the actress", "Narrow beam on the guitarist",
        "Isolate the performer in light", "Light only the lead",
        "Pin light on the host", "Aim the spotlight at the stage center",
        "Spotlight", "Highlight the speaker", "Put the light on the lead singer",
        "Bright focus on the star", "Illuminate only the narrator",
        "Draw attention to the performer", "Shine a light on me alone",
        "Spotlight the magician",
    ],
    "neutral": [
        "Hello there", "Hi director", "What's your name", "How are you today",
        "Thank you", "Good job everyone", "What time is it", "Tell me a joke",
        "Reset the lights to normal", "Reset to default lighting",
        "Lights back to standard", "Turn off the effects", "Who are you",
        "Let's start rehearsal", "Take five everyone", "Check the microphone",
        "Is the show ready", "Good morning", "Clear all lighting effects",
        "Keep the lights as they are",
    ],
}

DIALOGUES: dict[str, list[str]] = {
    "red_alert": [
        "Switching to red alert!", "Red alert activated.",
        "Emergency lighting is on.", "Going red, stay alert!",
    ],
    "blue_calm": [
        "Setting a calm blue mood.", "Cool blue lighting, nice and peaceful.",
        "Blue ambience coming up.", "Easing into a calm blue glow.",
    ],
    "spotlight": [
        "Spotlight on!", "Putting the spotlight on the performer.",
        "Focusing the light, you're on.", "All eyes on stage center.",
    ],
    "neutral": [
        "Understood, keeping the lighting neutral.",
        "Standing by for your next command.",
        "Okay, lights stay as they are.", "Got it, director.",
    ],
}

# Written by hand. NEVER used for training. This is the honest check.
HANDWRITTEN: list[dict[str, str]] = [
    {"command": "Turn the lights red", "preset": "red_alert"},
    {"command": "Spotlight on the lead actor", "preset": "spotlight"},
    {"command": "Make it calm and blue", "preset": "blue_calm"},
    {"command": "Back to normal lighting", "preset": "neutral"},
    {"command": "Red alert!", "preset": "red_alert"},
    {"command": "Focus light on center stage", "preset": "spotlight"},
    {"command": "I want a peaceful ocean mood", "preset": "blue_calm"},
    {"command": "Danger, emergency lighting", "preset": "red_alert"},
    {"command": "Hello, how are you?", "preset": "neutral"},
    {"command": "Highlight the singer in the center", "preset": "spotlight"},
]

TRAIN_FRACTION = 0.8


def _variants(bases: list[str], banned: set[str]) -> list[str]:
    """Expand base phrases with prefixes and suffixes into unique commands."""
    commands: set[str] = set()
    for base in bases:
        for prefix in PREFIXES:
            body = prefix + base[0].lower() + base[1:] if prefix else base
            for suffix in SUFFIXES:
                command = (body + suffix).strip()
                if command.lower() not in banned:
                    commands.add(command)
    return sorted(commands)  # sorted = deterministic before sampling


def _sample_rows(
    preset: str, bases: list[str], count: int, rng: random.Random, banned: set[str]
) -> list[dict[str, str]]:
    variants = _variants(bases, banned)
    chosen = rng.sample(variants, min(count, len(variants)))
    return [
        {"command": c, "preset": preset, "dialogue": rng.choice(DIALOGUES[preset])}
        for c in chosen
    ]


def build_splits(
    train_per_class: int = 80, test_per_class: int = 20, seed: int = 42
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Return (train_rows, test_rows). Same seed gives the same data."""
    rng = random.Random(seed)
    banned = {row["command"].lower() for row in HANDWRITTEN}
    train: list[dict[str, str]] = []
    test: list[dict[str, str]] = []

    for preset, bases in BASES.items():
        shuffled = sorted(bases)
        rng.shuffle(shuffled)
        cut = int(len(shuffled) * TRAIN_FRACTION)
        train += _sample_rows(preset, shuffled[:cut], train_per_class, rng, banned)
        test += _sample_rows(preset, shuffled[cut:], test_per_class, rng, banned)

    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]