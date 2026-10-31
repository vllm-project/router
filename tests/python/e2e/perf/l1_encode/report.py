"""Pure helpers for one L1 encode perf scenario. No processes."""

from __future__ import annotations

from typing import Any


def hit_tag(hit_rate: float) -> str:
    """Node-id fragment. The number is the configured hit rate, decimal kept.

    ``hit0-`` selects rate 0. ``hit0`` also matches ``hit0.30`` and ``hit0.99999``.
    """
    if hit_rate <= 0.0:
        return "0"
    if hit_rate >= 0.99:
        return f"{hit_rate:.5f}".rstrip("0").rstrip(".")
    return f"{hit_rate:.2f}"


def wave_id(l1_on: bool, hit_rate: float, concurrency: int, requests: int) -> str:
    arm = "l1on" if l1_on else "l1off"
    return f"{arm}-hit{hit_tag(hit_rate)}-c{concurrency}-r{requests}"


def summary_p50(summary: dict[str, Any], key: str) -> float | None:
    block = summary.get(key)
    if isinstance(block, dict) and isinstance(block.get("p50"), (int, float)):
        return float(block["p50"])
    return None


def check_encode_reuse(
    summary: dict[str, Any], *, l1_on: bool, hit_rate: float
) -> None:
    """Structural L1 signal. Encode time is printed, not compared across scenarios."""
    if summary_p50(summary, "encode_ms") is None:
        raise AssertionError(
            "missing wave.summary.encode_ms; start the router with VLLM_ROUTER_STAGES=1"
        )
    reuse = summary_p50(summary, "encode_reuse")
    if reuse is None:
        raise AssertionError(
            "missing wave.summary.encode_reuse; start the router with VLLM_ROUTER_STAGES=1"
        )
    if not l1_on:
        if reuse != 0.0:
            raise AssertionError(f"L1 off encode_reuse p50={reuse}, expected 0")
        return
    if hit_rate <= 0.0:
        if reuse >= 0.02:
            raise AssertionError(
                f"hit 0 encode_reuse p50={reuse}, expected about 0 "
                "(one boundary token out of the prompt)"
            )
        return
    if hit_rate >= 0.99:
        if reuse <= 0.90:
            raise AssertionError(
                f"hit {hit_rate:g} encode_reuse p50={reuse}, expected above 0.90"
            )
        return
    low, high = hit_rate - 0.15, hit_rate + 0.15
    if not low <= reuse <= high:
        raise AssertionError(
            f"hit {hit_rate:g} encode_reuse p50={reuse}, expected near {hit_rate:g} "
            f"({low:.2f}..{high:.2f})"
        )


def check_l1_startup_log(text: str, *, l1_on: bool) -> None:
    """``L1 enabled`` is logged when the router first loads the tokenizer."""
    if l1_on:
        if "L1 disabled for this model" in text:
            raise AssertionError(
                "router logged 'L1 disabled for this model'; this tokenizer cannot use L1"
            )
        if "L1 enabled" not in text:
            raise AssertionError(
                "router did not log 'L1 enabled'. L1 did not attach for this model."
            )
        return
    if "L1 enabled" in text:
        raise AssertionError("L1 off router logged 'L1 enabled'")


def l1_bypass_lines(metrics_text: str) -> list[str]:
    """Prometheus lines where a prompt skipped the L1 tree."""
    bad: list[str] = []
    for line in metrics_text.splitlines():
        if line.startswith("#") or not line.startswith(
            "vllm_tokenizer_l1_bypasses_total"
        ):
            continue
        try:
            value = float(line.split()[-1])
        except ValueError:
            continue
        if value > 0:
            bad.append(line)
    return bad


def _fmt_summary_block(block: Any) -> str:
    if not isinstance(block, dict):
        return "n/a"
    parts: list[str] = []
    for stat in ("min", "p50", "avg", "max"):
        value = block.get(stat)
        if isinstance(value, (int, float)):
            parts.append(f"{stat}={float(value):.3f}")
    return " ".join(parts) if parts else "n/a"


def format_scenario_result(
    *,
    scenario_id: str,
    summary: dict[str, Any],
    artifact: str | None,
) -> str:
    """One scenario's wave stats and the JSON path. No warmup, no cross-scenario summary."""
    lines = [f"[perf] {scenario_id}"]
    lines.extend(
        f"[perf] {key} {_fmt_summary_block(summary.get(key))}"
        for key in ("ttft_ms", "e2e_ms", "encode_ms", "frontend_ms")
    )
    lines.append(f"[perf] result path: {artifact or '(not saved)'}")
    return "\n".join(lines)
