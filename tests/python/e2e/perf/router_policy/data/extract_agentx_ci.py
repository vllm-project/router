"""Extract the first 12 eligible AgentX sessions, retaining 12 turns each."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

SESSION_COUNT = 12
TURNS_PER_SESSION = 12
MAX_MODEL_LEN = 65536
TRACE_FILE = "agentx-ci-12x12.jsonl"
TRACE_SHA256 = "faf79824c90ddb72774a6f5f018f67821f26c48743cd5171cfbcfe9e00855537"


def _eligible(session: dict, turns: int) -> bool:
    requests = session.get("requests", [])[:turns]
    block = session.get("block_size")
    if (
        session.get("hash_id_scope") != "local"
        or type(block) is not int
        or block <= 0
        or len(requests) != turns
    ):
        return False
    for request in requests:
        input_tokens, output_tokens = request.get("in"), request.get("out")
        if (
            request.get("type") == "subagent"
            or type(input_tokens) is not int
            or type(output_tokens) is not int
            or input_tokens <= 0
            or output_tokens <= 1
            or input_tokens + output_tokens > MAX_MODEL_LEN
            or not isinstance(request.get("hash_ids"), list)
            or len(request["hash_ids"]) != (input_tokens + block - 1) // block
        ):
            return False
    return True


def extract(
    source: Path,
    *,
    session_count: int = SESSION_COUNT,
    turns_per_session: int = TURNS_PER_SESSION,
    expected_sha256: str = TRACE_SHA256,
) -> bytes:
    sessions = []
    seen = set()
    with source.open("rb") as handle:
        for raw in handle:
            session = json.loads(raw)
            session_id = session.get("id")
            if (
                not isinstance(session_id, str)
                or not session_id
                or session_id in seen
                or not _eligible(session, turns_per_session)
            ):
                continue
            seen.add(session_id)
            sessions.append(
                {**session, "requests": session["requests"][:turns_per_session]}
            )
            if len(sessions) == session_count:
                break
    if len(sessions) != session_count:
        raise ValueError(f"source does not contain {session_count} eligible sessions")
    body = "".join(
        json.dumps(session, sort_keys=True, separators=(",", ":")) + "\n"
        for session in sessions
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
