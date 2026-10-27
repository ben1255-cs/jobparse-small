"""
Step 3: Quality-check the generated adverts and split them into train / val / test.

The generator LLM sometimes breaks the rules (e.g. forgets the company, or
invents a location that should be null). Those examples would teach the
model the wrong thing, so we drop them and report how many were removed.
"""
import json
import random
import re
from collections import Counter
from pathlib import Path

from make_records import COMPANIES, LOCATIONS

random.seed(42)

KNOWN_COMPANIES = [c for c in COMPANIES if c]
KNOWN_CITIES = [c for c in LOCATIONS if c]
MONEY = re.compile(r"[£€$]\s?\d|\b\d+\s?k\b|\bper (hour|annum)\b", re.IGNORECASE)


def problems(text, label):
    """Return a list of reasons this example looks unreliable (empty list = keep)."""
    t = text.lower()
    issues = []
    if label["job_title"].lower() not in t:
        issues.append("job_title missing")
    if label["company"] and label["company"].lower() not in t:
        issues.append("company missing")
    if not label["company"] and any(c.lower() in t for c in KNOWN_COMPANIES):
        issues.append("company leaked")
    if label["location"] and label["location"].lower() not in t:
        issues.append("location missing")
    if not label["location"] and any(c.lower() in t for c in KNOWN_CITIES):
        issues.append("location leaked")
    if label["salary_min"] is None and MONEY.search(text):
        issues.append("salary leaked")
    return issues


def main():
    rows = [json.loads(l) for l in Path("data/ads.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    kept, reasons = [], Counter()
    for r in rows:
        issues = problems(r["text"], r["label"])
        if issues:
            reasons.update(issues)
        else:
            kept.append(r)

    print(f"Kept {len(kept)} of {len(rows)} adverts ({len(rows) - len(kept)} removed)")
    for reason, n in reasons.most_common():
        print(f"  {reason}: {n}")

    random.shuffle(kept)
    n = len(kept)
    splits = {
        "train": kept[: int(n * 0.8)],
        "val": kept[int(n * 0.8): int(n * 0.9)],
        "test": kept[int(n * 0.9):],
    }
    for name, data in splits.items():
        path = Path(f"data/{name}.jsonl")
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in data), encoding="utf-8")
        print(f"{name}: {len(data)} -> {path}")


if __name__ == "__main__":
    main()
