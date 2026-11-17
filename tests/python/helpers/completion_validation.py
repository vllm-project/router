"""Validate one non-streaming smoke completion: exit 1 if invalid, 2 for CLI errors, 3 if unreadable."""

import argparse
import json
from pathlib import Path


def validate_single_completion(data):
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
    # The smoke requests 8 tokens: length is the limit, stop is normal EOS.
    # Abort/filter outcomes must fail even if the backend returns HTTP 200.
    if choice.get("finish_reason") not in ("stop", "length"):
        raise ValueError("completion did not finish with stop or length")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("response", type=Path)
    args = parser.parse_args()
    try:
        validate_single_completion(
            json.loads(args.response.read_text(encoding="utf-8"))
        )
    except OSError as exc:
        parser.exit(3, f"Cannot read completion response {args.response}: {exc}\n")
    except ValueError as exc:
        parser.exit(1, f"Invalid completion response: {exc}\n")
    print("Completion response is structurally valid and non-empty")


if __name__ == "__main__":
    main()
