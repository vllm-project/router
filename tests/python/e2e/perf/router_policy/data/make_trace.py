"""Build AgentX replay trace: first N sessions, first T turns each."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def flat(entries):
    out = []
    for q in entries:
        if q.get("type") == "subagent":
            out.extend(q.get("requests") or [])
        elif "in" in q:
            out.append(q)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="SemiAnalysis AgentX traces.jsonl")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output JSONL (e.g. agentx-128-t4.jsonl)",
    )
    parser.add_argument("-n", "--sessions", type=int, default=128)
    parser.add_argument("-t", "--turns", type=int, default=4)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.source.read_text().splitlines()
        if line.strip()
    ]
    out_rows = []
    for row in rows[: args.sessions]:
        session = dict(row)
        session["requests"] = row["requests"][: args.turns]
        out_rows.append(session)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        for session in out_rows:
            handle.write(json.dumps(session) + "\n")

    req = sum(len(flat(s["requests"])) for s in out_rows)
    ins = [x["in"] for s in out_rows for x in flat(s["requests"])]
    print(
        "sessions=%d raw_total=%d flat_requests=%d sum_in=%d max_in=%d"
        % (len(out_rows), len(rows), req, sum(ins), max(ins) if ins else 0)
    )


if __name__ == "__main__":
    main()
