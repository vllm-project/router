"""Reproduce the review's context-capped AgentX 12-session, 144-request slice."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

SOURCE_SHA256 = "e39cd2ff3eba21d4a3664be51da743ac3d2149a1933898cafc7bfeac8147eeef"
TRACE_FILE = "agentx-l4-12x12.jsonl"
TRACE_SHA256 = "1f37cb060e4609c5d52db9ca2d3674b23d32c6c42f3ed4a2cb4861e78b1f772b"
MAX_MODEL_LEN = 65536
SESSION_COUNT = 12
TURNS_PER_SESSION = 12


def _calls(entries):
    for entry in entries:
        if entry.get("type") == "subagent":
            yield from _calls(entry["requests"])
        else:
            yield entry


def extract(source: Path) -> bytes:
    """Select the review's three size cohorts without changing request fields."""
    with source.open("rb") as handle:
        if hashlib.file_digest(handle, "sha256").hexdigest() != SOURCE_SHA256:
            raise ValueError("upstream AgentX SHA-256 mismatch")
    eligible = []
    with source.open("rb") as handle:
        for line in handle:
            session = json.loads(line)
            turns = session["requests"][:TURNS_PER_SESSION]
            block = session["block_size"]
            if (
                len(turns) != TURNS_PER_SESSION
                or session["hash_id_scope"] != "local"
                or block <= 0
            ):
                continue
            requests = list(_calls(turns))
            if requests and all(
                req["in"] > 0
                and req["out"] > 1
                and req["in"] + req["out"] <= MAX_MODEL_LEN
                and len(req["hash_ids"]) == (req["in"] + block - 1) // block
                for req in requests
            ):
                eligible.append(
                    (
                        sum(req["in"] for req in requests),
                        session["id"],
                        {**session, "requests": turns},
                    )
                )
    eligible.sort(key=lambda item: (item[0], item[1]))
    if len(eligible) != 57:
        raise ValueError(f"expected 57 eligible source sessions, got {len(eligible)}")
    # One-based ranks 5-8, 27-30, and 50-53, matching the review's extractor.
    selected = [
        item[2] for start in (4, 26, 49) for item in eligible[start : start + 4]
    ]
    body = "".join(json.dumps(session) + "\n" for session in selected).encode("utf-8")
    if hashlib.sha256(body).hexdigest() != TRACE_SHA256:
        raise ValueError("extracted AgentX SHA-256 mismatch")
    return body


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    body = extract(args.source)
    destination = args.output or Path(__file__).resolve().parent / TRACE_FILE
    destination.write_bytes(body)
    print(f"Extracted 12 sessions and 144 requests: {TRACE_SHA256}")


if __name__ == "__main__":
    main()
