"""Extract the fixed TraceLab subsets used by router_policy cold replay.

Source: ``UW-SyFI/TraceLab`` revision ``7256bbf77e76d5cb83f4ddbd5ad31d8faee9dc7a``,
file ``data/v0.0.2/syfi_coding_trace.jsonl.gz``. The extractor keeps every
original ``provider == "codex"`` row for the eight merged session ids by default.
``--subset pd_disagg`` selects the review's six complete, context-capped sessions.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

SOURCE_SHA = "11ce51ec0a25e3d1d95b025bca2f7d1647e47571eb7cc968acd5fc64d4b4fb65"
TRACE_SHA = "d81e020824255602125cb0f5f94777601bc935fcd8739cd429665cf11331b849"
PD_TRACE_FILE = "tracelab-l4-6.jsonl"
PD_TRACE_SHA = "105c375c30634d8191a22cfbc56a6e2d63ce3e2ddc571ffe7519581b8086e2e0"
PD_MAX_MODEL_LEN = 65536
IDS = {
    "codex:20af69c9-826d-9ae3-51ce-185ac55fa8af",
    "codex:e2a0b3e9-6e0f-0860-76d6-10fa77b06519",
    "codex:99a6f681-fb99-26ab-4c57-1ae779940fb7",
    "codex:08918547-b261-28e8-6386-8cbbc819c7f9",
    "codex:76cfc159-d50c-9153-c30f-3a353fe6ccdf",
    "codex:fc3e5643-548d-4ba1-9f2b-f572e59439dd",
    "codex:6cde7e78-dc9d-8e50-5ba1-fc22e5beec32",
    "codex:ad4c8800-2882-f07b-8799-8b2d18caedb5",
}


def digest(path: Path) -> str:
    hashed = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hashed.update(chunk)
    return hashed.hexdigest()


def extract_pd_disagg(source: Path) -> bytes:
    """Keep complete sessions from the first sixteen keys that fit 64K context."""
    maxima: dict[tuple[str, str], int] = {}
    rows = []
    with gzip.open(source, "rb") as handle:
        for line in handle:
            row = json.loads(line)
            session = (row["provider"], row["session_id"])
            if len(maxima) < 16:
                maxima.setdefault(session, 0)
            if session in maxima:
                length = (row["input_tokens_total"] or 0) + (row["output_tokens"] or 0)
                maxima[session] = max(maxima[session], length)
                rows.append((session, row))
    keep = {session for session, length in maxima.items() if length <= PD_MAX_MODEL_LEN}
    selected = [row for session, row in rows if session in keep]
    if len(maxima) != 16 or len(keep) != 6 or len(selected) != 68:
        raise ValueError("expected six complete sessions and 68 requests")
    # Match the review's serialization as well as its session selection.
    body = "".join(json.dumps(row) + "\n" for row in selected).encode("utf-8")
    if hashlib.sha256(body).hexdigest() != PD_TRACE_SHA:
        raise ValueError("extracted PD TraceLab SHA-256 mismatch")
    return body


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--subset", choices=("merged", "pd_disagg"), default="merged")
    args = parser.parse_args()
    source = args.source
    data_dir = Path(__file__).resolve().parent
    if digest(source) != SOURCE_SHA:
        raise SystemExit("source SHA-256 mismatch")
    if args.subset == "pd_disagg":
        (data_dir / PD_TRACE_FILE).write_bytes(extract_pd_disagg(source))
        print(f"Extracted 6 complete sessions and 68 requests: {PD_TRACE_SHA}")
        return
    dest = data_dir / "tracelab-ci-uniform-28.jsonl"
    count = 0
    found: set[str] = set()
    with gzip.open(source, "rt", encoding="utf-8") as src, dest.open(
        "w", encoding="utf-8", newline="\n"
    ) as out:
        for line in src:
            row = json.loads(line)
            if row["provider"] == "codex" and row["session_id"] in IDS:
                out.write(line if line.endswith("\n") else line + "\n")
                count += 1
                found.add(row["session_id"])
    if count != 28 or found != IDS or digest(dest) != TRACE_SHA:
        raise SystemExit(
            f"extracted trace mismatch count={count} sessions={len(found)} "
            f"sha={digest(dest)}"
        )
    print(f"Extracted 8 complete sessions and 28 requests: {TRACE_SHA}")


if __name__ == "__main__":
    main()
