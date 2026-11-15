"""Generate train/test/handwritten/hard JSONL files.

Run from the repo root:  python scripts/generate_dataset.py
"""

import argparse
from pathlib import Path

from stage_director.training.dataset import HANDWRITTEN, build_splits, write_jsonl
from stage_director.training.hard_set import HARD_TEST


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("data"))
    parser.add_argument("--train-per-class", type=int, default=80)
    parser.add_argument("--test-per-class", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train, test = build_splits(args.train_per_class, args.test_per_class, args.seed)
    write_jsonl(args.out_dir / "train.jsonl", train)
    write_jsonl(args.out_dir / "test.jsonl", test)
    write_jsonl(args.out_dir / "handwritten_test.jsonl", HANDWRITTEN)
    write_jsonl(args.out_dir / "hard_test.jsonl", HARD_TEST)

    print(
        f"train: {len(train)} | test: {len(test)} | "
        f"handwritten: {len(HANDWRITTEN)} | hard: {len(HARD_TEST)}"
    )
    print(f"Files written to {args.out_dir}/")


if __name__ == "__main__":
    main()