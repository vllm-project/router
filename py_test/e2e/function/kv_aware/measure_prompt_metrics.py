#!/usr/bin/env python3
"""Report token and block metrics for kv_aware e2e prompts (aligns with issue #331 planning).

Uses the same strings as ``config.py`` / ``dataset.py``. Token counts follow the
chat template path (``apply_chat_template`` + generation prompt), matching Router
grpc chat when the same tokenizer/model is configured.

Examples (from repo root)::

  pip install transformers  # if not already available
  python -m py_test.e2e.function.kv_aware.measure_prompt_metrics \\
    --model-dir /path/to/Qwen3-0.6B

  # Optional live checks (router / worker already running):
  python -m py_test.e2e.function.kv_aware.measure_prompt_metrics \\
    --model-dir /path/to/model --router-url http://127.0.0.1:8000 \\
    --worker-http http://127.0.0.1:9000

Environment: ``MODEL_DIR`` or ``KV_AWARE_E2E_MODEL`` as default model path/id.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any

from py_test.e2e.function.kv_aware.config import (
    KV_AWARE_E2E_MODEL,
    kv_aware_index_settle_sec,
)
from py_test.e2e.function.kv_aware.dataset import (
    default_prefix_reuse_scenario,
    lane_prime_user,
    lane_reuse_user,
)


def _chat_input_token_ids(tokenizer: Any, user_content: str) -> list[int]:
    enc = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
    )
    ids = enc["input_ids"]
    if ids and isinstance(ids[0], (list, tuple)):
        ids = ids[0]
    return [int(x) for x in ids]


def _lcp_length(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def _device_full_blocks_estimate(
    input_tokens: int, shared_prefix_tokens: int, block_size: int
) -> dict[str, int]:
    """Conservative block counts for kv_aware credit planning (not exact Router hash)."""
    if block_size <= 0:
        return {"device_blocks_cap": 0, "shared_full_blocks": 0}
    cap = max(0, (input_tokens - 1) // block_size)
    shared_full = max(0, (min(shared_prefix_tokens, input_tokens) - 1) // block_size)
    shared_full = min(shared_full, cap)
    return {"device_blocks_cap": cap, "shared_full_blocks": shared_full}


def _measure_body(tokenizer: Any, label: str, text: str) -> dict[str, Any]:
    ids = _chat_input_token_ids(tokenizer, text)
    return {
        "label": label,
        "char_len": len(text),
        "input_tokens": len(ids),
        "token_ids_head": ids[:8],
    }


def _pair_metrics(
    tokenizer: Any,
    *,
    prime_label: str,
    prime_text: str,
    reuse_label: str,
    reuse_text: str,
    block_size: int,
) -> dict[str, Any]:
    prime = _measure_body(tokenizer, prime_label, prime_text)
    reuse = _measure_body(tokenizer, reuse_label, reuse_text)
    prime_ids = _chat_input_token_ids(tokenizer, prime_text)
    reuse_ids = _chat_input_token_ids(tokenizer, reuse_text)
    shared = _lcp_length(prime_ids, reuse_ids)
    blocks = _device_full_blocks_estimate(reuse["input_tokens"], shared, block_size)
    new_prefill_if_full_credit = max(0, reuse["input_tokens"] - shared)
    return {
        "prime": prime,
        "reuse": reuse,
        "shared_prefix_tokens": shared,
        "reuse_token_delta": reuse["input_tokens"] - prime["input_tokens"],
        "block_estimates": blocks,
        "new_prefill_tokens_upper_if_all_shared_credited": new_prefill_if_full_credit,
    }


def _http_json(url: str, timeout: float = 10.0) -> dict[str, Any] | None:
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return None


def _discover_block_size(
    worker_http: str | None, router_url: str | None
) -> tuple[int, str]:
    for base, tag in (
        (worker_http, "worker_http_get_server_info"),
        (router_url, "router_get_server_info"),
    ):
        if not base:
            continue
        data = _http_json(f"{base.rstrip('/')}/get_server_info")
        if not data:
            continue
        raw = data.get("kv_block_size")
        if isinstance(raw, int) and raw > 0:
            return raw, tag
    return 16, "default_assumed_16"


def _router_sample_n_prompt(
    router_url: str, model: str, user: str, timeout: float
) -> int | None:
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": user}],
            "max_tokens": 1,
            "stream": True,
            "temperature": 0.0,
        }
    ).encode()
    req = urllib.request.Request(
        f"{router_url.rstrip('/')}/v1/chat/completions",
        data=body,
        headers={"content-type": "application/json"},
        method="POST",
    )
    stages_raw: str | None = None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            stages_raw = resp.headers.get("x-router-stages")
            for line in resp:
                raw = line.decode("utf-8", errors="replace").strip()
                if raw.startswith(": router-stages "):
                    stages_raw = raw[len(": router-stages ") :].strip()
                    break
                if raw.strip() == "data: [DONE]":
                    break
    except (urllib.error.URLError, TimeoutError, OSError):
        return None
    if not stages_raw:
        return None
    try:
        stages = json.loads(stages_raw)
        n = stages.get("n_prompt_tokens")
        return int(n) if isinstance(n, int) else None
    except json.JSONDecodeError:
        return None


def _recommendations(report: dict[str, Any]) -> list[str]:
    out: list[str] = []
    block_size = int(report["block_size"])
    for key in ("single_lane",):
        pair = report["scenarios"].get(key)
        if not pair:
            continue
        shared_blocks = pair["block_estimates"]["shared_full_blocks"]
        if shared_blocks < 4:
            out.append(
                f"{key}: shared_full_blocks={shared_blocks} (<4); "
                "consider lengthening KV_AWARE_SHARED_PREFIX for stronger KV credit."
            )
    dual = report["scenarios"].get("dual_lane") or {}
    a_ids = dual.get("A", {}).get("prime", {}).get("token_ids_head")
    b_ids = dual.get("B", {}).get("prime", {}).get("token_ids_head")
    if a_ids and b_ids and a_ids == b_ids:
        out.append(
            "dual_lane: lane A/B prime share identical token prefix head; "
            "lanes may collide in KV index."
        )
    if block_size == 16 and report["block_size_source"].startswith("default"):
        out.append(
            "block_size not read from get_server_info; pass --worker-http or "
            "--router-url if discovery differs from 16."
        )
    return out


def build_report(
    *,
    model_dir: str,
    block_size: int | None,
    worker_http: str | None,
    router_url: str | None,
    router_model: str,
    router_probe_timeout: float,
) -> dict[str, Any]:
    if block_size is None or block_size <= 0:
        block_size, block_source = _discover_block_size(worker_http, router_url)
    else:
        block_source = "cli"

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    scenario = default_prefix_reuse_scenario()
    single = _pair_metrics(
        tokenizer,
        prime_label="single_prime",
        prime_text=scenario.prime_user,
        reuse_label="single_reuse",
        reuse_text=scenario.reuse_user,
        block_size=block_size,
    )
    dual: dict[str, Any] = {}
    for lane in ("A", "B"):
        dual[lane] = _pair_metrics(
            tokenizer,
            prime_label=f"lane_{lane}_prime",
            prime_text=lane_prime_user(lane),
            reuse_label=f"lane_{lane}_reuse",
            reuse_text=lane_reuse_user(lane, round_index=1),
            block_size=block_size,
        )

    a_prime_ids = _chat_input_token_ids(tokenizer, lane_prime_user("A"))
    b_prime_ids = _chat_input_token_ids(tokenizer, lane_prime_user("B"))
    dual["lanes_independent_prefix"] = _lcp_length(a_prime_ids, b_prime_ids) < min(
        16, len(a_prime_ids), len(b_prime_ids)
    )

    report: dict[str, Any] = {
        "model_dir": model_dir,
        "block_size": block_size,
        "block_size_source": block_source,
        "e2e_env": {
            "KV_AWARE_INDEX_SETTLE_SEC": kv_aware_index_settle_sec(),
            "KV_AWARE_E2E_MODEL": KV_AWARE_E2E_MODEL,
        },
        "scenarios": {
            "single_lane": single,
            "dual_lane": dual,
        },
        "live": {},
    }

    if router_url:
        n = _router_sample_n_prompt(
            router_url,
            router_model,
            scenario.prime_user,
            router_probe_timeout,
        )
        report["live"]["router_url"] = router_url
        report["live"]["router_n_prompt_tokens_prime_sample"] = n
        if n is not None and n != single["prime"]["input_tokens"]:
            report["live"]["tokenizer_router_mismatch"] = {
                "offline_input_tokens": single["prime"]["input_tokens"],
                "router_n_prompt_tokens": n,
            }

    if worker_http:
        report["live"]["worker_http"] = worker_http
        report["live"]["worker_get_server_info"] = _http_json(
            f"{worker_http.rstrip('/')}/get_server_info"
        )

    report["recommendations"] = _recommendations(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        default=os.environ.get("MODEL_DIR", "").strip() or KV_AWARE_E2E_MODEL,
        help="HF model id or local snapshot (tokenizer for offline counts)",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=0,
        help="Override kv block size (0 = fetch or default 16)",
    )
    parser.add_argument(
        "--router-url",
        default=os.environ.get("ROUTER_URL", "").strip(),
        help="Optional live router for get_server_info / n_prompt_tokens sample",
    )
    parser.add_argument(
        "--worker-http",
        default=os.environ.get("WORKER_HTTP", "").strip(),
        help="Optional worker OpenAI HTTP base for get_server_info",
    )
    parser.add_argument(
        "--router-model",
        default=os.environ.get("VLLM_ROUTER_MODEL", KV_AWARE_E2E_MODEL),
        help="Model field when probing router chat",
    )
    parser.add_argument(
        "--router-timeout",
        type=float,
        default=120.0,
        help="Timeout for optional router probe",
    )
    parser.add_argument(
        "--json-out",
        default="",
        help="Write report JSON to this path (default: stdout only)",
    )
    args = parser.parse_args()
    if not args.model_dir:
        parser.error("set --model-dir or MODEL_DIR or KV_AWARE_E2E_MODEL")

    block = args.block_size if args.block_size > 0 else None
    report = build_report(
        model_dir=args.model_dir,
        block_size=block,
        worker_http=args.worker_http or None,
        router_url=args.router_url or None,
        router_model=args.router_model,
        router_probe_timeout=args.router_timeout,
    )
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            f.write(text)
            f.write("\n")
    print(text)


if __name__ == "__main__":
    try:
        main()
    except ImportError as exc:
        if "transformers" in str(exc):
            sys.exit("missing dependency: pip install transformers\n" f"({exc})")
        raise
