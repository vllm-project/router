#!/usr/bin/env python3
"""Print chat token ids using Python vLLM's tokenizer.

Spawned by `src/backend/preprocess.rs` tests
(`vllm_chat_tokenize_matches_python_vllm_*`). Those tests skip unless
`VLLM_ROUTER_MODEL` is a model dir with `tokenizer.json` and this
process can `import vllm`. No model names or golden id lists in-tree.

  python tests/python_vllm_chat_ids.py /path/to/model \\
    '[{"role":"user","content":"hello"}]'

  VLLM_ROUTER_MODEL=/path/to/model cargo test --lib \\
    vllm_chat_tokenize_matches_python_vllm -- --nocapture
"""

from __future__ import annotations

import json
import sys


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: python_vllm_chat_ids.py MODEL_DIR MESSAGES_JSON")
    model, messages_raw = sys.argv[1], sys.argv[2]
    from vllm.tokenizers import get_tokenizer

    tok = get_tokenizer(model, trust_remote_code=True)
    messages = json.loads(messages_raw)
    enc = tok.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
    )
    # transformers returns BatchEncoding (a Mapping, not necessarily dict).
    ids = enc["input_ids"] if hasattr(enc, "__getitem__") and "input_ids" in enc else enc
    if ids and isinstance(ids[0], (list, tuple)):
        ids = ids[0]
    print(json.dumps([int(x) for x in ids]))


if __name__ == "__main__":
    main()
