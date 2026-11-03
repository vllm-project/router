"""L1 prefix encoding cache perf: Qwen3-0.6B, 16k, shape turns.

The worker and the router stay up through L1 on, including hit-rate changes.
Before L1 off, the worker and the router both cold-start.

    export VLLM_RS=/path/to/vllm-rs
    # optional local snapshot; default is Qwen/Qwen3-0.6B
    # export MODEL=/path/to/Qwen3-0.6B
    pytest -sv tests/python/e2e/perf/l1_encode/test_l1_encode.py::test_l1_encode \\
        -k "l1on and hit0.30 and c4"
"""

from __future__ import annotations

import os
import sys

from tests.python.e2e.perf.l1_encode.case_loader import (
    bench_max_tokens,
    bench_timeout,
    bench_tokens,
)
from tests.python.e2e.perf.l1_encode.scenario import (
    check_encode_reuse,
    check_l1_startup_log,
    format_scenario_result,
    l1_bypass_lines,
    wave_id,
    write_result,
)


def test_l1_encode(
    l1_stack,
    l1_case,
    l1_output_dir,
    bench_mod,
    l1_on: bool,
    hit_rate: float,
    concurrency: int,
    requests: int,
) -> None:
    l1_stack.ensure_router(l1_on, hit_rate)
    os.environ["ROUTER_URL"] = l1_stack.router_url
    os.environ["MODEL"] = l1_case.model
    settings = bench_mod.read_perf_settings(
        backend_label="l1on" if l1_on else "l1off",
        bench_tokens=bench_tokens(l1_case),
        bench_max_tokens=bench_max_tokens(l1_case),
        bench_timeout=bench_timeout(l1_case),
    )
    try:
        payload = bench_mod.run_prefix_kvhit_wave(
            settings,
            hit_rate=hit_rate,
            concurrency=concurrency,
            requests=requests,
            shape="turns",
            turns=l1_case.turns,
        )
    except SystemExit as exc:
        raise AssertionError(
            f"{exc}. If the worker rejected the prompt as longer than "
            "max-model-len or max-num-batched-tokens, raise both above "
            "prompt_tokens + max_tokens. Do not set them equal to --tokens."
        ) from None

    row_id = wave_id(l1_on, hit_rate, concurrency, requests)
    summary = (payload.get("wave") or {}).get("summary") or {}
    metrics = l1_stack.l1_metrics_text()
    artifact = write_result(
        l1_output_dir,
        row_id,
        {
            "id": row_id,
            "l1": "on" if l1_on else "off",
            "tokens": settings["tokens"],
            "turns": l1_case.turns,
            "metrics_l1": metrics,
            **payload,
        },
    )
    print(
        format_scenario_result(
            scenario_id=row_id,
            summary=summary,
            artifact=artifact,
        ),
        file=sys.stderr,
        flush=True,
    )

    bench_mod.assert_wave_summary(payload["wave"], requests)
    check_encode_reuse(summary, l1_on=l1_on, hit_rate=hit_rate)
    check_l1_startup_log(l1_stack.router_log_text(), l1_on=l1_on)
    if l1_on:
        bad = l1_bypass_lines(metrics)
        if bad:
            raise AssertionError(
                "L1 skipped the tree (vllm_tokenizer_l1_bypasses_total):\n"
                + "\n".join(bad)
            )
