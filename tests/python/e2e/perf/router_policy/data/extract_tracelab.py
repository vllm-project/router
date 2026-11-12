"""Extract the eight TraceLab sessions used by the router_policy cold replay.

Source: ``UW-SyFI/TraceLab`` revision ``7256bbf77e76d5cb83f4ddbd5ad31d8faee9dc7a``,
file ``data/v0.0.2/syfi_coding_trace.jsonl.gz``. The extractor keeps every
original ``provider == "codex"`` row for the eight session ids, in source order.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
from pathlib import Path

SOURCE_SHA = "11ce51ec0a25e3d1d95b025bca2f7d1647e47571eb7cc968acd5fc64d4b4fb65"
TRACE_SHA = "d81e020824255602125cb0f5f94777601bc935fcd8739cd429665cf11331b849"
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


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {Path(sys.argv[0]).name} syfi_coding_trace.jsonl.gz")
    source = Path(sys.argv[1])
    dest = Path(__file__).resolve().parent / "tracelab-ci-uniform-28.jsonl"
    if digest(source) != SOURCE_SHA:
        raise SystemExit("source SHA-256 mismatch")
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
