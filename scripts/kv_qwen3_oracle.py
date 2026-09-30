#!/usr/bin/env python3
"""Reproduce the pinned Qwen3 text oracle with public synthetic fixtures.

No network access, model execution, or repository writes. Install dependencies
only in an isolated Linux environment. Output JSON on stdout; diagnostics go
to stderr. Full mode refuses a tokenizer with any other hash. --template-only
checks only rendered strings and explicitly does not emit token-ID goldens.
"""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys


REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
TOKENIZER_SHA256 = "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"
CONFIG_SHA256 = "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101"
VOCAB_SHA256 = "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"
MERGES_SHA256 = "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5"


def verified(path, expected):
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != expected:
        raise ValueError(f"{path.name}: expected {expected}, got {digest}")
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer-directory", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, default=Path(__file__).resolve().parents[1]
                        / "tests/fixtures/kv_qwen3/prompts.json")
    parser.add_argument("--template-only", action="store_true")
    parser.add_argument("--check", type=Path, help="compare the full result with a saved oracle")
    args = parser.parse_args()
    directory = args.tokenizer_directory
    config = json.loads(verified(directory / "tokenizer_config.json", CONFIG_SHA256))
    fixtures = json.loads(args.fixtures.read_text(encoding="utf-8"))
    if fixtures["revision"] != REVISION:
        raise ValueError("fixture revision mismatch")

    from transformers import AutoTokenizer
    if args.template_only:
        # The real pinned Jinja template is evaluated by Transformers. The
        # reconstructed backend is used only to host apply_chat_template;
        # tokenize=False ensures this is not claimed as tokenizer equivalence.
        verified(directory / "vocab.json", VOCAB_SHA256)
        verified(directory / "merges.txt", MERGES_SHA256)
    else:
        verified(directory / "tokenizer.json", TOKENIZER_SHA256)
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
    if tokenizer.chat_template != config["chat_template"]:
        raise ValueError("loaded template differs from pinned configuration")

    output = {
        "schema_version": 1,
        "mode": "template-only" if args.template_only else "canonical-tokenizer",
        "model": "Qwen/Qwen3-0.6B",
        "revision": REVISION,
        "tokenizer_sha256": None if args.template_only else TOKENIZER_SHA256,
        "tokenizer_config_sha256": CONFIG_SHA256,
        "versions": {name: importlib.metadata.version(name)
                     for name in ("transformers", "tokenizers", "jinja2")},
        "chat": [],
        "completion": [],
    }
    for case in fixtures["chat"]:
        request = case["request"]
        kwargs = request.get("chat_template_kwargs", {})
        rendered = tokenizer.apply_chat_template(
            request["messages"], tokenize=False, add_generation_prompt=True, **kwargs)
        if rendered != case["rendered"]:
            raise ValueError(f"{case['name']}: rendered fixture differs from upstream template")
        result = {"name": case["name"], "request": request, "rendered": rendered}
        if not args.template_only:
            token_ids = tokenizer.apply_chat_template(
                request["messages"], tokenize=True, add_generation_prompt=True, **kwargs)
            if token_ids != tokenizer.encode(rendered, add_special_tokens=False):
                raise ValueError(f"{case['name']}: template and encoder disagree")
            result["token_ids"] = token_ids
        output["chat"].append(result)
    if not args.template_only:
        for case in fixtures["completion"]:
            request = case["request"]
            prompt = request["prompt"]
            ids = (tokenizer.encode(prompt, add_special_tokens=request.get("add_special_tokens", True))
                   if isinstance(prompt, str) else prompt)
            output["completion"].append({"name": case["name"], "request": request, "token_ids": ids})
    if args.check:
        expected = json.loads(args.check.read_text(encoding="utf-8"))
        if output != expected:
            raise ValueError("generated result differs from saved oracle")
    else:
        json.dump(output, sys.stdout, ensure_ascii=False, indent=2)
        print()
    print(f"PASS: {len(output['chat'])} chat and {len(output['completion'])} completion fixtures; "
          f"mode={output['mode']}", file=sys.stderr)


if __name__ == "__main__":
    main()
