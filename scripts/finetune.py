"""LoRA fine-tuning of Qwen on stage-director commands.

Run on a GPU (Colab T4):  python scripts/finetune.py
Output: a merged model folder usable via STAGE_LLM_MODEL_ID.
"""

import argparse
import json
import logging
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

from stage_director.services.llm_director import SYSTEM_PROMPT
from stage_director.training.dataset import load_jsonl

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("finetune")

IGNORE_INDEX = -100  # PyTorch skips these label positions when computing loss


def build_example(row: dict[str, str], tokenizer, max_len: int = 256) -> dict[str, list[int]]:
    """Turn one row into token ids, with the loss computed on the answer only."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": row["command"]},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    answer = (
        json.dumps(
            {"dialogue": row["dialogue"], "lighting_preset": row["preset"]},
            ensure_ascii=False,
        )
        + "<|im_end|>"  # Qwen's end-of-turn token: teaches the model when to stop
    )

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]

    input_ids = (prompt_ids + answer_ids)[:max_len]
    labels = ([IGNORE_INDEX] * len(prompt_ids) + answer_ids)[:max_len]
    return {"input_ids": input_ids, "labels": labels}


class PadCollator:
    """Pads a batch of different-length examples to the same length."""

    def __init__(self, pad_token_id: int) -> None:
        self._pad = pad_token_id

    def __call__(self, batch: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        longest = max(len(item["input_ids"]) for item in batch)
        input_ids, labels, attention = [], [], []
        for item in batch:
            gap = longest - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [self._pad] * gap)
            labels.append(item["labels"] + [IGNORE_INDEX] * gap)
            attention.append([1] * len(item["input_ids"]) + [0] * gap)
        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(attention),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--train-file", type=Path, default=Path("data/train.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/qwen-stage-director"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("No GPU found. Use a GPU runtime (Colab: Runtime > T4 GPU).")

    set_seed(args.seed)

    logger.info("Loading tokenizer and base model: %s", args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.float16)
    model.to("cuda")

    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)

    # fp16 training needs the small trainable LoRA weights in float32.
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.float()
    model.print_trainable_parameters()

    rows = load_jsonl(args.train_file)
    examples = [build_example(row, tokenizer) for row in rows]
    logger.info("Training examples: %d", len(examples))

    training_args = TrainingArguments(
        output_dir=str(args.output_dir / "checkpoints"),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=2,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        fp16=True,
        logging_steps=5,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=examples,
        data_collator=PadCollator(tokenizer.pad_token_id),
    )
    trainer.train()

    logger.info("Merging LoRA layers into the base model ...")
    merged = model.merge_and_unload()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info("Saved merged model to %s", args.output_dir)


if __name__ == "__main__":
    main()