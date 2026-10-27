"""
Shared definitions used by every script: the output schema, the prompt,
and a robust JSON parser for model outputs.
"""
import json
import re

FIELDS = [
    "job_title", "company", "location", "work_mode", "contract_type",
    "experience_years_min", "skills",
    "salary_min", "salary_max", "salary_currency", "salary_period",
]

SYSTEM_PROMPT = """You extract structured data from job adverts.
Return ONLY a JSON object with exactly these keys:
- job_title: string
- company: string or null
- location: city name as a string, or null
- work_mode: one of "onsite", "hybrid", "remote", or null
- contract_type: one of "permanent", "contract", "internship", or null
- experience_years_min: integer or null (0 if no experience is required)
- skills: list of strings, sorted alphabetically
- salary_min: number or null (e.g. 45000 for "£45k")
- salary_max: number or null
- salary_currency: one of "GBP", "EUR", "USD", or null
- salary_period: one of "year", "hour", or null
Use null for anything the advert does not state. Do not add extra keys or any text outside the JSON."""


def build_messages(text):
    """Chat messages for one extraction request."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Job advert:\n\n{text}"},
    ]


def parse_json(raw):
    """Pull the first JSON object out of a model response. Returns a dict or None."""
    if raw is None:
        return None
    raw = re.sub(r"```(?:json)?", "", raw).strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        obj = json.loads(raw[start:end + 1])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def to_target(label):
    """The exact JSON string the model is trained to produce."""
    return json.dumps({k: label.get(k) for k in FIELDS}, ensure_ascii=False)
