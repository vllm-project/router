"""Pure helpers for one L1 encode perf scenario. No processes."""

from __future__ import annotations

from typing import Any


def hit_tag(hit_rate: float) -> str:
    """Node-id fragment. ``hit0-`` is not a substring of ``hit030``."""
    if hit_rate <= 0.0:
        return "0"
    if abs(hit_rate - 0.3) < 1e-9:
        return "030"
    if abs(hit_rate - 0.99999) < 1e-8:
        return "99999"
    text = f"{hit_rate:.5f}".rstrip("0").rstrip(".")
    return text.replace(".", "")


def wave_id(l1_on: bool, hit_rate: float, concurrency: int, requests: int) -> str:
    arm = "l1on" if l1_on else "l1off"
    return f"{arm}-hit{hit_tag(hit_rate)}-c{concurrency}-r{requests}"


def summary_p50(summary: dict[str, Any], key: str) -> float | None:
    block = summary.get(key)
    if isinstance(block, dict) and isinstance(block.get("p50"), (int, float)):
        return float(block["p50"])
    return None


def first_prompt_tokens(payload: dict[str, Any]) -> int | None:
    wave = payload.get("wave") or {}
    details = wave.get("requests_detail") or []
    if details and isinstance(details[0].get("prompt_tokens"), int):
        return int(details[0]["prompt_tokens"])
    warmup = payload.get("warmup") or {}
    for name in ("hit", "miss"):
        row = warmup.get(name) or {}
        if isinstance(row.get("prompt_tokens"), int):
            return int(row["prompt_tokens"])
    return None


def format_stage_brief(fields: dict[str, Any]) -> str:
    encode = fields.get("encode_ms")
    reuse = fields.get("encode_reuse")
    if not isinstance(encode, (int, float)):
        return "n/a"
    reuse_s = f"{reuse:.4f}" if isinstance(reuse, (int, float)) else "n/a"
    return f"encode_ms={float(encode):.3f} encode_reuse={reuse_s}"


def check_encode_reuse(
    summary: dict[str, Any], *, l1_on: bool, hit_rate: float
) -> None:
    """Structural L1 signal. Encode-time ratios are printed, not asserted."""
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


def format_scenario_result(
    *,
    scenario_id: str,
    encode_ms_p50: float | None,
    encode_reuse_p50: float | None,
    prompt_tokens: int | None,
    artifact: str | None,
) -> str:
    """One scenario's bench numbers and the JSON path. No cross-scenario summary."""

    def num(value: float | None, digits: int) -> str:
        if isinstance(value, (int, float)):
            return f"{float(value):.{digits}f}"
        return "n/a"

    prompt = str(prompt_tokens) if prompt_tokens is not None else "n/a"
    path = artifact or "(not saved)"
    return (
        f"[perf] {scenario_id} encode_ms p50={num(encode_ms_p50, 3)} "
        f"encode_reuse p50={num(encode_reuse_p50, 4)} prompt_tokens={prompt}\n"
        f"[perf] result path: {path}"
    )
