"""
Interactive demo: paste a job advert, get structured JSON back.

Run locally (a GPU is recommended but not required):
  python demo/app.py --adapter outputs/lora-jobparse

It can also be deployed as a free Hugging Face Space - see the README.
"""
import argparse
import json
import sys
from pathlib import Path

import gradio as gr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from predict_local import extract, load_model  # noqa: E402
from schema import parse_json  # noqa: E402

EXAMPLE = """🚀 We're hiring! Larkspur AI is looking for a Machine Learning Engineer to join our hybrid team in Leeds.
You'll need 2+ years building models in Python and PyTorch, and be comfy shipping things with Docker.
Permanent role, paying 55-65k. DM me if interested!"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--adapter", default="outputs/lora-jobparse",
                   help="local folder or Hugging Face Hub repo id of the LoRA adapter")
    args = p.parse_args()

    tok, model = load_model(args.base_model, args.adapter)

    def run(text):
        raw = extract(tok, model, [text])[0]
        parsed = parse_json(raw)
        return json.dumps(parsed, indent=2, ensure_ascii=False) if parsed else f"Could not parse output:\n{raw}"

    gr.Interface(
        fn=run,
        inputs=gr.Textbox(lines=12, label="Job advert", value=EXAMPLE),
        outputs=gr.Code(language="json", label="Extracted data"),
        title="JobParse-Small",
        description="A 1.5B-parameter model fine-tuned with LoRA to extract structured data from messy job adverts.",
    ).launch()


if __name__ == "__main__":
    main()
