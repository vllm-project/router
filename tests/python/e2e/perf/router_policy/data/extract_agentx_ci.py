"""Reproduce the review's fixed heterogeneous AgentX 12-by-12 replay slice."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

# Fixed by the reference extractor at KaisennHu/router commit d35dec404b1132f.
# Four sessions from each of the small, medium, and large input-size cohorts.
SESSION_IDS = (
    "a89db4725a97da42ea24bd5128a093ffb215",
    "751fbb2089c7c4303f3ed241f3a7a27a3f2c",
    "b4c638cc16e89d17c18da83951a87fbc44dd",
    "3c1fbf5cf98484c7f9ef07f2d4b00e65d829",
    "e34c6bb77dc3a1cd393b29f573135d9d76a0",
    "6d5c6b957d5f634a7f820ddff7389e161def",
    "1007f04d2c64580308858bb42f292992929d",
    "578354185933aedcdc6c17fbfbfcfb48876b",
    "7754cd1191c469354df23c7d4e9de242f7cb",
    "1a271af3dedd40be2efd413ef35fe7f0f53b",
    "f204bbf87b0680d205fe8a14441fbde0ee71",
    "4ba38d0b213f0aeab695e16da9c91635a817",
)
SESSION_COUNT = len(SESSION_IDS)
TURNS_PER_SESSION = 12
MAX_MODEL_LEN = 163840
TRACE_FILE = "agentx-hetero12x12.jsonl"
TRACE_SHA256 = "e7e7fd7cd31f178483e2027281ce4592fca38bd35252c09b88e2a9dee1078638"


def extract(
    source: Path,
    *,
    session_ids: tuple[str, ...] = SESSION_IDS,
    turns_per_session: int = TURNS_PER_SESSION,
    expected_sha256: str = TRACE_SHA256,
) -> bytes:
    """Keep the fixed sessions in reference order, including all subagent calls."""
    if not session_ids or len(set(session_ids)) != len(session_ids):
        raise ValueError("expected nonempty, distinct session IDs")
    selected = set(session_ids)
    sessions = {}
    with source.open("rb") as handle:
        for raw in handle:
            session = json.loads(raw)
            session_id = session.get("id")
            if session_id not in selected:
                continue
            if session_id in sessions:
                raise ValueError(f"duplicate source session: {session_id}")
            requests = session["requests"][:turns_per_session]
            if len(requests) != turns_per_session:
                raise ValueError(f"{session_id}: fewer than {turns_per_session} turns")
            sessions[session_id] = {**session, "requests": requests}
            if len(sessions) == len(session_ids):
                break
    missing = selected - sessions.keys()
    if missing:
        raise ValueError(f"source is missing fixed sessions: {sorted(missing)}")
    # Match the reference serialization as well as its session selection.
    body = "".join(
        json.dumps(sessions[session_id]) + "\n" for session_id in session_ids
    ).encode("utf-8")
    if hashlib.sha256(body).hexdigest() != expected_sha256:
        raise ValueError("extracted AgentX trace SHA256 mismatch")
    return body


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    data_dir = Path(__file__).resolve().parent
    body = extract(args.source)
    destination = args.output or data_dir / TRACE_FILE
    destination.write_bytes(body)
    print(f"{destination}: sha256={TRACE_SHA256}")


if __name__ == "__main__":
    main()
