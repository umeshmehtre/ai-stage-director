from stage_director.schemas import LightingPreset
from stage_director.training.dataset import HANDWRITTEN, build_splits
from stage_director.training.hard_set import HARD_TEST


def test_hard_set_labels_are_valid():
    valid = {p.value for p in LightingPreset}
    assert len(HARD_TEST) >= 20
    assert all(item["preset"] in valid for item in HARD_TEST)


def test_hard_set_has_no_duplicates():
    commands = [item["command"].lower() for item in HARD_TEST]
    assert len(commands) == len(set(commands))


def test_hard_set_never_overlaps_other_data():
    train, test = build_splits()
    seen = {row["command"].lower() for row in train + test}
    seen |= {item["command"].lower() for item in HANDWRITTEN}
    hard = {item["command"].lower() for item in HARD_TEST}
    assert hard.isdisjoint(seen)