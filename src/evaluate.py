"""
Step 6: Score one or more prediction files and print a comparison table.

Example:
  python src/evaluate.py results/preds_base.jsonl results/preds_finetuned.jsonl results/preds_large.jsonl

Metrics
- valid_json:   output could be parsed as a JSON object
- schema_ok:    JSON has exactly the expected keys
- field_acc:    share of the 11 fields that are exactly right (averaged over all adverts)
- skills_f1:    overlap between predicted and true skills
- exact_match:  every single field is right
Failed outputs count as wrong for every field, so models can't score well by refusing.
"""
import json
import sys
from pathlib import Path

from schema import FIELDS


def norm(v):
    if isinstance(v, str):
        v = v.strip().lower()
        return v or None
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    return v


def field_correct(field, pred, gold):
    if field == "skills":
        return {norm(s) for s in (pred or [])} == {norm(s) for s in (gold or [])}
    return norm(pred) == norm(gold)


def skills_f1(pred, gold):
    p = {norm(s) for s in (pred or []) if isinstance(s, str)}
    g = {norm(s) for s in (gold or [])}
    if not p and not g:
        return 1.0
    tp = len(p & g)
    if tp == 0:
        return 0.0
    prec, rec = tp / len(p), tp / len(g)
    return 2 * prec * rec / (prec + rec)


def score(path):
    rows = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    n = len(rows)
    totals = {"valid_json": 0, "schema_ok": 0, "field_acc": 0.0, "skills_f1": 0.0, "exact_match": 0}
    per_field = {f: 0 for f in FIELDS}
    for r in rows:
        pred, gold = r["pred"], r["label"]
        if pred is None:
            continue
        totals["valid_json"] += 1
        totals["schema_ok"] += set(pred) == set(FIELDS)
        correct = [field_correct(f, pred.get(f), gold.get(f)) for f in FIELDS]
        for f, ok in zip(FIELDS, correct):
            per_field[f] += ok
        totals["field_acc"] += sum(correct) / len(FIELDS)
        totals["skills_f1"] += skills_f1(pred.get("skills"), gold.get("skills"))
        totals["exact_match"] += all(correct)
    summary = {k: v / n for k, v in totals.items()}
    return n, summary, {f: c / n for f, c in per_field.items()}


def main():
    paths = sys.argv[1:]
    if not paths:
        raise SystemExit("Usage: python src/evaluate.py results/preds_a.jsonl results/preds_b.jsonl ...")

    results = {Path(p).stem: score(p) for p in paths}
    metrics = ["valid_json", "schema_ok", "field_acc", "skills_f1", "exact_match"]

    lines = ["| Model | N | " + " | ".join(metrics) + " |",
             "|---|---|" + "---|" * len(metrics)]
    for name, (n, s, _) in results.items():
        lines.append(f"| {name} | {n} | " + " | ".join(f"{s[m]:.1%}" for m in metrics) + " |")

    lines += ["", "**Per-field accuracy**", "",
              "| Field | " + " | ".join(results) + " |",
              "|---|" + "---|" * len(results)]
    for f in FIELDS:
        lines.append(f"| {f} | " + " | ".join(f"{pf[f]:.1%}" for _, _, pf in results.values()) + " |")

    table = "\n".join(lines)
    print(table)
    out = Path("results/results.md")
    out.parent.mkdir(exist_ok=True)
    out.write_text(table + "\n", encoding="utf-8")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
