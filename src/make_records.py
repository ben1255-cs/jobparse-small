"""
Step 1: Generate random, structured job records (the "ground truth").

Later, an LLM will turn each record into a messy, realistic job ad.
Because we create the JSON first, every training label is correct by construction.
"""
import json
import random
from pathlib import Path

random.seed(42)  # makes the dataset reproducible

TITLES = ["Data Scientist", "Machine Learning Engineer", "Software Engineer",
          "Data Analyst", "Backend Developer", "AI Research Intern",
          "Frontend Developer", "DevOps Engineer", "Product Analyst",
          "NLP Engineer", "Cloud Engineer", "QA Engineer"]
COMPANIES = ["Northwind Labs", "Bluepeak Analytics", "Harbour Tech",
             "Quillon Systems", "Aster Health", "Fennick Finance",
             "Larkspur AI", "Tidewell Retail", "Corvid Robotics", None]
LOCATIONS = ["London", "Newcastle", "Manchester", "Edinburgh", "Leeds",
             "Bristol", "Glasgow", "Birmingham", "Dublin", "Berlin", None]
WORK_MODES = ["onsite", "hybrid", "remote", None]
CONTRACTS = ["permanent", "contract", "internship", None]
SKILLS = ["Python", "SQL", "PyTorch", "TensorFlow", "AWS", "Docker",
          "Kubernetes", "React", "TypeScript", "Java", "Spark", "Git",
          "scikit-learn", "Tableau", "Azure", "Go", "C++", "Airflow"]


def make_salary(contract):
    """Return salary fields, sometimes missing, sometimes hourly."""
    if random.random() < 0.25:  # 25% of ads don't mention pay
        return {"salary_min": None, "salary_max": None,
                "salary_currency": None, "salary_period": None}
    if contract == "contract" and random.random() < 0.6:
        low = random.randrange(30, 90, 5)
        return {"salary_min": low, "salary_max": low + random.randrange(0, 30, 5),
                "salary_currency": "GBP", "salary_period": "hour"}
    low = random.randrange(22000, 90000, 1000)
    high = low + random.choice([0, 5000, 10000, 15000, 20000])
    currency = random.choices(["GBP", "EUR", "USD"], weights=[8, 1, 1])[0]
    return {"salary_min": low, "salary_max": high,
            "salary_currency": currency, "salary_period": "year"}


def make_record():
    contract = random.choice(CONTRACTS)
    record = {
        "job_title": random.choice(TITLES),
        "company": random.choice(COMPANIES),
        "location": random.choice(LOCATIONS),
        "work_mode": random.choice(WORK_MODES),
        "contract_type": contract,
        "experience_years_min": random.choice([None, 0, 1, 2, 3, 5, 7]),
        "skills": sorted(random.sample(SKILLS, k=random.randint(1, 6))),
    }
    record.update(make_salary(contract))
    return record


if __name__ == "__main__":
    out = Path("data/records.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for _ in range(2000):
            f.write(json.dumps(make_record()) + "\n")
    print(f"Wrote 2000 records to {out}")
