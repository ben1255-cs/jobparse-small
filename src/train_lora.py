"""
Step 4: Fine-tune a small open model with LoRA to turn adverts into JSON.

Needs a GPU - run it on Google Colab or Kaggle (see notebooks/train_colab.ipynb).
A free Colab T4 GPU takes roughly 20-40 minutes for the default settings.
"""
import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

from schema import build_messages, to_target


def load_split(path):
    """Convert {text, label} rows into TRL's conversational prompt/completion format,
    so the loss is only computed on the JSON answer, not on the advert."""
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            rows.append({
                "prompt": build_messages(r["text"]),
                "completion": [{"role": "assistant", "content": to_target(r["label"])}],
            })
    return Dataset.from_list(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--output-dir", default="outputs/lora-jobparse")
    p.add_argument("--epochs", type=float, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--rank", type=int, default=16)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("No GPU found. Run this on Colab/Kaggle with a GPU runtime.")

    # Newer GPUs (A100, L4) support bf16; the free T4 needs fp16 mixed precision.
    use_bf16 = torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float32

    train_ds = load_split("data/train.jsonl")
    val_ds = load_split("data/val.jsonl")
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    peft_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank * 2,
        lora_dropout=0.05,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )

    config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=0.05,  # a float < 1 means "5% of total steps"
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=4,
        gradient_checkpointing=True,
        max_length=1024,
        bf16=use_bf16,
        fp16=not use_bf16,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        model_init_kwargs={"dtype": dtype},
        seed=42,
    )

    trainer = SFTTrainer(
        model=args.base_model,
        args=config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=peft_config,
    )
    trainer.model.print_trainable_parameters()
    trainer.train()

    trainer.save_model(args.output_dir)
    print(f"Saved LoRA adapter to {args.output_dir}")


if __name__ == "__main__":
    main()
