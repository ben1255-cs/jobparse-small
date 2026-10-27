"""
Step 5b: Baseline - run a large model through an API on the same test file,
with exactly the same prompt as the small model.

Example:
  python src/predict_api.py --input data/test.jsonl --out results/preds_large.jsonl
"""
import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from schema import build_messages, parse_json

load_dotenv()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/test.jsonl")
    p.add_argument("--out", required=True)
    p.add_argument("--model", default=None, help="defaults to LLM_BASELINE_MODEL or LLM_MODEL in .env")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    model = args.model or os.environ.get("LLM_BASELINE_MODEL") or os.environ["LLM_MODEL"]
    client = OpenAI(api_key=os.environ["LLM_API_KEY"], base_url=os.environ.get("LLM_BASE_URL") or None)
    rows = [json.loads(l) for l in Path(args.input).read_text(encoding="utf-8").splitlines() if l.strip()]

    def call(row):
        for attempt in range(5):
            try:
                resp = client.chat.completions.create(
                    model=model, messages=build_messages(row["text"]), temperature=0, max_tokens=400)
                return resp.choices[0].message.content
            except Exception:
                time.sleep(2 ** attempt)
        return None

    start = time.time()
    with ThreadPoolExecutor(args.workers) as pool:
        raws = list(pool.map(call, rows))
    secs = time.time() - start

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r, raw in zip(rows, raws):
            f.write(json.dumps({"text": r["text"], "label": r["label"], "raw": raw,
                                "pred": parse_json(raw)}, ensure_ascii=False) + "\n")
    print(f"{model}: {len(rows)} adverts in {secs:.0f}s. Saved to {args.out}")


if __name__ == "__main__":
    main()
