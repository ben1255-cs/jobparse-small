"""
Step 2: Turn each structured record into a messy, realistic job advert using an LLM.

Works with any OpenAI-compatible API (OpenAI, Groq, Together, OpenRouter,
Google Gemini's OpenAI-compatible endpoint, a local Ollama server, ...).
Configure it in a .env file - see .env.example.

The script is resumable: re-running it skips records that are already done.
"""
import argparse
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

STYLES = [
    "a formal corporate advert with headed sections",
    "a short, casual post like you'd see on LinkedIn, with a couple of emojis",
    "a recruiter's email with a friendly tone and a few small typos",
    "a bullet-pointed listing on a job board",
    "a startup advert full of buzzwords and enthusiasm",
    "a terse listing of two or three sentences",
    "a long advert that buries the key details in the middle of company waffle",
]

SALARY_FORMATS = [
    "like '£45,000 - £50,000 per annum'",
    "in shorthand like '45-50k'",
    "like '£45000 to £50000 a year'",
    "in a casual way, e.g. 'paying 45 to 50 grand'",
]

INSTRUCTIONS = """Write {style} for the job described by this JSON:

{record}

Rules:
- Mention EVERY non-null field naturally somewhere in the advert.
- Fields that are null must NOT be mentioned or hinted at at all.
- If a salary is present, write it {salary_format}, adapted to the real numbers and currency.
  If salary_min equals salary_max, state a single figure. An hourly rate must clearly be per hour.
- experience_years_min of 0 means no experience is needed - say so in your own words.
- Work the skills into sentences or bullets rather than copying the list.
- Do not invent other concrete facts such as a different salary, city or contract type.
- Output only the advert text, nothing else."""


def make_client():
    return OpenAI(
        api_key=os.environ["LLM_API_KEY"],
        base_url=os.environ.get("LLM_BASE_URL") or None,
    )


def write_ad(client, model, record, rng):
    prompt = INSTRUCTIONS.format(
        style=rng.choice(STYLES),
        record=json.dumps(record, indent=2),
        salary_format=rng.choice(SALARY_FORMATS),
    )
    for attempt in range(5):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=1.0,
                max_tokens=700,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:  # rate limits, timeouts, etc.
            wait = 2 ** attempt
            print(f"  API error ({e.__class__.__name__}), retrying in {wait}s")
            time.sleep(wait)
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--records", default="data/records.jsonl")
    p.add_argument("--out", default="data/ads.jsonl")
    p.add_argument("--limit", type=int, default=None, help="only do the first N records (cheap test run)")
    p.add_argument("--workers", type=int, default=4, help="parallel API requests")
    args = p.parse_args()

    model = os.environ.get("LLM_GENERATOR_MODEL") or os.environ["LLM_MODEL"]
    client = make_client()

    lines = Path(args.records).read_text(encoding="utf-8").splitlines()
    records = [json.loads(l) for l in lines if l.strip()]
    if args.limit:
        records = records[: args.limit]

    out = Path(args.out)
    done = set()
    if out.exists():
        done = {json.loads(l)["id"] for l in out.read_text(encoding="utf-8").splitlines() if l.strip()}
    todo = [(i, r) for i, r in enumerate(records) if i not in done]
    print(f"{len(done)} already done, generating {len(todo)} ads with {model}")

    with out.open("a", encoding="utf-8") as f, ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(write_ad, client, model, r, random.Random(i)): (i, r) for i, r in todo}
        for n, fut in enumerate(as_completed(futures), 1):
            i, r = futures[fut]
            text = fut.result()
            if text:
                f.write(json.dumps({"id": i, "text": text, "label": r}, ensure_ascii=False) + "\n")
                f.flush()
            if n % 50 == 0:
                print(f"  {n}/{len(todo)}")
    print(f"Done. Ads saved to {out}")


if __name__ == "__main__":
    main()
