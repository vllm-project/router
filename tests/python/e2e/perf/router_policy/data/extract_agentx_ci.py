"""Rebuild the pinned AgentX slice, verifying source rows and output SHA256."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def extract(source: Path, manifest: dict) -> bytes:
    selected = {row["source_line"]: row for row in manifest["sessions"]}
    sessions = {}
    with source.open("rb") as handle:
        for line_number, raw in enumerate(handle, 1):
            if line_number in selected:
                expected = selected[line_number]
                if hashlib.sha256(raw).hexdigest() != expected["source_row_sha256"]:
                    raise ValueError(f"source row {line_number}: SHA256 mismatch")
                session = json.loads(raw)
                if session["id"] != expected["id"]:
                    raise ValueError(f"source row {line_number}: session ID mismatch")
                turns = manifest["turns_per_session"]
                if len(session["requests"]) < turns:
                    raise ValueError(f"source row {line_number}: not enough turns")
                sessions[session["id"]] = {
                    **session,
                    "requests": session["requests"][:turns],
                }
            if line_number == max(selected):
                break
    if len(sessions) != len(selected):
        raise ValueError("source does not contain every pinned session")
    body = "".join(
        json.dumps(sessions[row["id"]], sort_keys=True, separators=(",", ":")) + "\n"
        for row in manifest["sessions"]
    ).encode("utf-8")
    if hashlib.sha256(body).hexdigest() != manifest["trace_sha256"]:
        raise ValueError("extracted AgentX trace SHA256 mismatch")
    return body


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    data_dir = Path(__file__).resolve().parent
    manifest = json.loads(
        (data_dir / "agentx-ci-8x8.manifest.json").read_text(encoding="utf-8")
    )
    body = extract(args.source, manifest)
    destination = args.output or data_dir / manifest["trace_file"]
    destination.write_bytes(body)
    print(f"{destination}: sha256={manifest['trace_sha256']}")


if __name__ == "__main__":
    main()
