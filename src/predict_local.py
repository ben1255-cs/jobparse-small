"""
Step 5a: Run the small model (with or without the LoRA adapter) on a test file.

Examples (on a GPU):
  python src/predict_local.py --input data/test.jsonl --out results/preds_base.jsonl
  python src/predict_local.py --input data/test.jsonl --out results/preds_finetuned.jsonl --adapter outputs/lora-jobparse
"""
import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from schema import build_messages, parse_json


def load_model(base_model, adapter=None):
    tok = AutoTokenizer.from_pretrained(base_model)
    tok.padding_side = "left"  # needed for batched generation
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    if not torch.cuda.is_available():
        dtype = torch.float32
    model = AutoModelForCausalLM.from_pretrained(base_model, dtype=dtype, device_map="auto")
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload()  # faster inference
    model.eval()
    return tok, model


@torch.no_grad()
def extract(tok, model, texts, max_new_tokens=300):
    prompts = [tok.apply_chat_template(build_messages(t), tokenize=False, add_generation_prompt=True) for t in texts]
    batch = tok(prompts, return_tensors="pt", padding=True).to(model.device)
    out = model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
    new_tokens = out[:, batch["input_ids"].shape[1]:]
    return tok.batch_decode(new_tokens, skip_special_tokens=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/test.jsonl")
    p.add_argument("--out", required=True)
    p.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--adapter", default=None, help="path to LoRA adapter; omit for the un-tuned base model")
    p.add_argument("--batch-size", type=int, default=8)
    args = p.parse_args()

    rows = [json.loads(l) for l in Path(args.input).read_text(encoding="utf-8").splitlines() if l.strip()]
    tok, model = load_model(args.base_model, args.adapter)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with open(args.out, "w", encoding="utf-8") as f:
        for i in range(0, len(rows), args.batch_size):
            chunk = rows[i: i + args.batch_size]
            raws = extract(tok, model, [r["text"] for r in chunk])
            for r, raw in zip(chunk, raws):
                f.write(json.dumps({"text": r["text"], "label": r["label"], "raw": raw,
                                    "pred": parse_json(raw)}, ensure_ascii=False) + "\n")
            print(f"  {min(i + args.batch_size, len(rows))}/{len(rows)}")
    secs = time.time() - start
    print(f"Done in {secs:.0f}s ({secs / len(rows):.2f}s per advert). Saved to {args.out}")


if __name__ == "__main__":
    main()
