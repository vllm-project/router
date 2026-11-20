"""L1 prefix encoding cache perf: Qwen3-0.6B, 16k, shape turns.

Each ``test_name`` in the case JSON is one pytest class. That class starts one
worker and one router and keeps them for every hit rate and wave.

    export VLLM_RS=/path/to/vllm-rs
    # optional local snapshot; default is Qwen/Qwen3-0.6B
    # export MODEL=/path/to/Qwen3-0.6B
    pytest -sv tests/python/e2e/perf/l1_encode/test_l1_encode.py::test_qwen3_16k_l1on \\
        -k "hit0.30 and c4"
"""

from __future__ import annotations

import os
import sys

from tests.python.e2e.perf.l1_encode.case_loader import (
    bench_max_tokens,
    bench_timeout,
    bench_tokens,
    load_cases,
)
from tests.python.e2e.perf.l1_encode.check import (
    check_encode_reuse,
    check_l1_startup_log,
    format_scenario_result,
    l1_bypass_lines,
    wave_id,
    write_result,
)


def _bind_case(case):
    class _L1Test:
        l1_case = case

        def test_l1_encode(
            self,
            l1_stack,
            l1_case,
            l1_output_dir,
            bench_mod,
            hit_rate: float,
            concurrency: int,
            requests: int,
        ) -> None:
            l1_on = l1_case.l1_on
            os.environ["ROUTER_URL"] = l1_stack.router_url
            os.environ["MODEL"] = l1_case.model
            settings = bench_mod.read_perf_settings(
                backend_label=case.test_name,
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

            row_id = wave_id(hit_rate, concurrency, requests)
            summary = (payload.get("wave") or {}).get("summary") or {}
            metrics = l1_stack.l1_metrics_text()
            artifact = write_result(l1_output_dir, row_id, payload)
            print(
                format_scenario_result(
                    scenario_id=f"{case.test_name}/{row_id}",
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

    _L1Test.__name__ = case.test_name
    _L1Test.__qualname__ = case.test_name
    return _L1Test


for _case in load_cases():
    globals()[_case.test_name] = _bind_case(_case)
