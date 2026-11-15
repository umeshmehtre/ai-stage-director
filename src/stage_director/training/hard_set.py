"""A hand-written HARD test set. Never used for training.

Labeling rule for mixed commands: if a specific performer is named
(singer, villain, lead...), 'spotlight' wins. Otherwise the mood word wins.
"""

HARD_TEST: list[dict[str, str]] = [
    # Typos
    {"command": "turn teh lights rd", "preset": "red_alert"},
    {"command": "spotligth on the dancr", "preset": "spotlight"},
    {"command": "make it calm n bluee", "preset": "blue_calm"},
    {"command": "plz reset lites to normall", "preset": "neutral"},
    # Slang and indirect wording
    {"command": "Make it spooky and red", "preset": "red_alert"},
    {"command": "Gimme some chill ocean vibes", "preset": "blue_calm"},
    {"command": "All eyes on the drummer", "preset": "spotlight"},
    {"command": "Somebody's in danger, show it with the lights", "preset": "red_alert"},
    {"command": "I want everyone to relax", "preset": "blue_calm"},
    {"command": "Just the singer, nothing else lit", "preset": "spotlight"},
    {"command": "Lights as they were before", "preset": "neutral"},
    {"command": "Sirens! Panic mode!", "preset": "red_alert"},
    {"command": "It should feel like a quiet night by the sea", "preset": "blue_calm"},
    {"command": "Hit the guitarist with a beam", "preset": "spotlight"},
    # Negations and traps (look like lighting commands, are not)
    {"command": "Don't change the lights", "preset": "neutral"},
    {"command": "Is the red light on the console working?", "preset": "neutral"},
    {"command": "What color is the sky", "preset": "neutral"},
    {"command": "Tell me about spotlights", "preset": "neutral"},
    {"command": "Not red, keep it standard", "preset": "neutral"},
    # Hinglish
    {"command": "lights ko red kar do", "preset": "red_alert"},
    {"command": "singer par spotlight daalo", "preset": "spotlight"},
    {"command": "thoda shaant neela mahaul banao", "preset": "blue_calm"},
    {"command": "sab normal kar do", "preset": "neutral"},
    # Multi-intent
    {"command": "Keep it calm but spotlight the singer", "preset": "spotlight"},
    {"command": "Red alert and put a beam on the villain", "preset": "spotlight"},
    {"command": "Go calm blue, and tell everyone good job", "preset": "blue_calm"},
    {"command": "Thanks! Now make it red", "preset": "red_alert"},
    {"command": "Dim everything cool blue except the lead who needs a light", "preset": "spotlight"},
    # Long and noisy
    {"command": "uh okay so like, can you maybe turn the whole stage red, thanks", "preset": "red_alert"},
    {"command": "hey um so the scene is sad so can we go blue", "preset": "blue_calm"},
]