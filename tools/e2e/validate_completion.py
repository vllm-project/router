"""Validate a single non-streaming /v1/completions smoke response."""

import argparse
import json
from pathlib import Path


def validate_completion(data):
    if not isinstance(data, dict) or "error" in data:
        raise ValueError("expected a completion object without an error")
    choices = data.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("expected exactly one completion choice")
    choice = choices[0]
    if not isinstance(choice, dict):
        raise ValueError("expected a choice object")
    text = choice.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("completion text is empty or invalid")
    if choice.get("finish_reason") not in ("stop", "length"):
        raise ValueError("completion did not finish with stop or length")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("response", type=Path)
    args = parser.parse_args()
    try:
        validate_completion(json.loads(args.response.read_text(encoding="utf-8")))
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Invalid completion response: {exc}\n")
    print("Completion response is structurally valid and non-empty")


if __name__ == "__main__":
    main()
