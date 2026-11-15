"""Measure a model on a JSONL test set: JSON validity and preset accuracy.

Example:
    python scripts/evaluate.py --model-id Qwen/Qwen2.5-1.5B-Instruct \
        --data data/test.jsonl --label baseline_test
"""

import argparse
import json
import logging
import time
from pathlib import Path

from stage_director.services.llm_director import LLMDirector
from stage_director.services.parsing import DirectorOutputError, parse_llm_output
from stage_director.training.dataset import load_jsonl

logging.basicConfig(level=logging.WARNING)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True, help="HF model id or local path")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--label", required=True, help="Name for the results file")
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    rows = load_jsonl(args.data)
    director = LLMDirector(args.model_id)

    valid = correct = 0
    latencies_ms: list[float] = []
    failures: list[dict[str, str]] = []

    for row in rows:
        started = time.perf_counter()
        raw = director.generate_raw(row["command"])
        latencies_ms.append((time.perf_counter() - started) * 1000)

        try:
            result = parse_llm_output(raw, row["command"])
        except DirectorOutputError:
            failures.append(
                {"command": row["command"], "expected": row["preset"], "got": "INVALID_JSON", "raw": raw}
            )
            continue

        valid += 1
        if result.lighting_preset.value == row["preset"]:
            correct += 1
        else:
            failures.append(
                {"command": row["command"], "expected": row["preset"], "got": result.lighting_preset.value, "raw": raw}
            )

    total = len(rows)
    summary = {
        "label": args.label,
        "model_id": args.model_id,
        "data": str(args.data),
        "examples": total,
        "json_valid_pct": round(100 * valid / total, 1),
        "preset_accuracy_pct": round(100 * correct / total, 1),
        "avg_latency_ms": round(sum(latencies_ms) / total),
        "failures": failures,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_file = args.out_dir / f"{args.label}.json"
    out_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n=== {args.label} ({total} examples) ===")
    print(f"JSON valid:      {summary['json_valid_pct']}%")
    print(f"Preset accuracy: {summary['preset_accuracy_pct']}%")
    print(f"Avg latency:     {summary['avg_latency_ms']} ms")
    print(f"Saved to {out_file}")
    for item in failures[:8]:
        print(f"  MISS: {item['command']!r} expected={item['expected']} got={item['got']}")


if __name__ == "__main__":
    main()