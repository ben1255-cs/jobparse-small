"""
Build the real-world test set by hand-labelling genuine job adverts.

1. Copy real adverts into data/real_ads/ as .txt files (one advert per file).
2. Run:  python src/label_real.py
3. For each advert, type each field (press Enter to leave it null).

Labels are saved to data/real_test.jsonl. Already-labelled files are skipped,
so you can stop and continue later.
"""
import json
from pathlib import Path

ADS_DIR = Path("data/real_ads")
OUT = Path("data/real_test.jsonl")

QUESTIONS = [
    ("job_title", str, "Job title"),
    ("company", str, "Company"),
    ("location", str, "Location (city)"),
    ("work_mode", str, "Work mode [onsite/hybrid/remote]"),
    ("contract_type", str, "Contract [permanent/contract/internship]"),
    ("experience_years_min", int, "Minimum years of experience (0 if none needed)"),
    ("skills", list, "Skills, comma-separated"),
    ("salary_min", float, "Salary min (number only, e.g. 45000)"),
    ("salary_max", float, "Salary max"),
    ("salary_currency", str, "Currency [GBP/EUR/USD]"),
    ("salary_period", str, "Salary period [year/hour]"),
]


def ask(prompt, kind):
    while True:
        raw = input(f"  {prompt}: ").strip()
        if not raw:
            return [] if kind is list else None
        try:
            if kind is list:
                return sorted(s.strip() for s in raw.split(",") if s.strip())
            if kind is float:
                v = float(raw.replace(",", ""))
                return int(v) if v.is_integer() else v
            return kind(raw)
        except ValueError:
            print("  Couldn't read that, try again.")


def main():
    ADS_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(ADS_DIR.glob("*.txt"))
    if not files:
        raise SystemExit(f"Put real adverts as .txt files in {ADS_DIR}/ first.")

    done = set()
    if OUT.exists():
        done = {json.loads(l)["source"] for l in OUT.read_text(encoding="utf-8").splitlines() if l.strip()}

    todo = [f for f in files if f.name not in done]
    print(f"{len(done)} labelled, {len(todo)} to go. Press Enter to leave a field null.\n")
    for f in todo:
        text = f.read_text(encoding="utf-8")
        print("=" * 80 + f"\n{f.name}\n" + "=" * 80 + f"\n{text}\n" + "-" * 80)
        label = {key: ask(prompt, kind) for key, kind, prompt in QUESTIONS}
        with OUT.open("a", encoding="utf-8") as out:
            out.write(json.dumps({"source": f.name, "text": text, "label": label}, ensure_ascii=False) + "\n")
        print("  Saved.\n")


if __name__ == "__main__":
    main()
