"""
Print examples a model got wrong, to understand *how* it fails.

Example:
  python src/show_errors.py results/preds_finetuned.jsonl --n 5
  python src/show_errors.py results/preds_finetuned.jsonl --field salary_max
"""
import argparse
import json
import random
from pathlib import Path

from evaluate import field_correct
from schema import FIELDS


def main():
    p = argparse.ArgumentParser()
    p.add_argument("preds")
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--field", default=None, help="only show mistakes on this field")
    args = p.parse_args()

    rows = [json.loads(l) for l in Path(args.preds).read_text(encoding="utf-8").splitlines() if l.strip()]
    fields = [args.field] if args.field else FIELDS
    errors = []
    for r in rows:
        if r["pred"] is None:
            errors.append((r, ["<invalid JSON>"]))
            continue
        wrong = [f for f in fields if not field_correct(f, r["pred"].get(f), r["label"].get(f))]
        if wrong:
            errors.append((r, wrong))

    print(f"{len(errors)} of {len(rows)} adverts have mistakes\n")
    random.seed(0)
    for r, wrong in random.sample(errors, min(args.n, len(errors))):
        print("=" * 80)
        print(r["text"][:800] + ("..." if len(r["text"]) > 800 else ""))
        print("-" * 80)
        if wrong == ["<invalid JSON>"]:
            print("INVALID OUTPUT:", (r["raw"] or "")[:300])
            continue
        for f in wrong:
            print(f"{f:22} predicted={r['pred'].get(f)!r:30} correct={r['label'].get(f)!r}")
    print("=" * 80)


if __name__ == "__main__":
    main()
