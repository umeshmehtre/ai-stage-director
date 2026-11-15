from stage_director.schemas import LightingPreset
from stage_director.training.dataset import HANDWRITTEN, build_splits


def test_default_sizes():
    train, test = build_splits()
    assert len(train) == 320
    assert len(test) == 80


def test_all_presets_present_in_train():
    train, _ = build_splits()
    assert {row["preset"] for row in train} == {p.value for p in LightingPreset}


def test_labels_are_valid_presets():
    train, test = build_splits()
    valid = {p.value for p in LightingPreset}
    assert all(row["preset"] in valid for row in train + test)


def test_no_overlap_between_train_and_test():
    train, test = build_splits()
    train_cmds = {row["command"] for row in train}
    test_cmds = {row["command"] for row in test}
    assert train_cmds.isdisjoint(test_cmds)


def test_handwritten_never_in_training_or_test():
    train, test = build_splits()
    hand = {item["command"].lower() for item in HANDWRITTEN}
    generated = {row["command"].lower() for row in train + test}
    assert hand.isdisjoint(generated)


def test_split_is_reproducible():
    assert build_splits(seed=7) == build_splits(seed=7)