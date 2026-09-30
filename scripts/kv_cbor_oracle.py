#!/usr/bin/env python3
"""Generate independent canonical-CBOR block-hash goldens using cbor2 5.6.5.

This exercises serialization, not a model vocabulary. Some synthetic IDs are
deliberately outside Qwen's vocabulary to cover all u32 CBOR width boundaries.
Install cbor2 only in an isolated Linux environment. No network or file writes
occur here; JSON is emitted on stdout, or compared with --check PATH.
"""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

import cbor2


CBOR2_VERSION = "5.6.5"
BLOCK_SIZE = 16
TOKEN_IDS = [
    0, 1, 22, 23, 24, 25, 254, 255, 256, 257, 65534, 65535, 65536, 65537,
    151644, 4294967295,
    151643, 151644, 151645, 151667, 151668, 198, 271, 872, 8948, 77091,
    0, 23, 24, 255, 256, 65536,
    # A trailing partial block must not be hashed.
    4294967295, 42, 151644,
]


def generate():
    version = importlib.metadata.version("cbor2")
    if version != CBOR2_VERSION:
        raise ValueError(f"expected cbor2 {CBOR2_VERSION}, found {version}")
    cases = []
    for seed in (0, 42):
        seed_cbor = cbor2.dumps(str(seed), canonical=True)
        parent = hashlib.sha256(seed_cbor).digest()
        case = {"seed": seed, "seed_cbor_hex": seed_cbor.hex(),
                "none_hash": parent.hex(), "block_cbor_hex": [], "block_hashes": []}
        for offset in range(0, len(TOKEN_IDS) - BLOCK_SIZE + 1, BLOCK_SIZE):
            tokens = tuple(TOKEN_IDS[offset:offset + BLOCK_SIZE])
            encoded = cbor2.dumps((parent, tokens, None), canonical=True)
            parent = hashlib.sha256(encoded).digest()
            case["block_cbor_hex"].append(encoded.hex())
            case["block_hashes"].append(parent.hex())
        cases.append(case)
    return {"schema_version": 1, "algorithm": "sha256_cbor", "cbor2_version": version,
            "block_size": BLOCK_SIZE, "token_ids": TOKEN_IDS,
            "trailing_partial_tokens": len(TOKEN_IDS) % BLOCK_SIZE, "cases": cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", type=Path)
    args = parser.parse_args()
    result = generate()
    if args.check:
        expected = json.loads(args.check.read_text(encoding="utf-8"))
        if result != expected:
            raise ValueError("generated canonical-CBOR oracle differs from fixture")
    else:
        json.dump(result, sys.stdout, indent=2)
        print()
    print("PASS: cbor2 5.6.5, block size 16, seeds 0/42, 2 complete blocks and 3 trailing IDs",
          file=sys.stderr)


if __name__ == "__main__":
    main()
