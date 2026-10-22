#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Collect stable vLLM HTTP metadata and per-DP throughput observations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import httpx
from prometheus_client.parser import text_string_to_metric_families

from tools.calibration.common import write_json

THROUGHPUT_COUNTERS = (
    "vllm:prompt_tokens_total",
    "vllm:prompt_tokens_cached_total",
    "vllm:generation_tokens_total",
    "vllm:request_success_total",
)


async def collect_metadata(client: httpx.AsyncClient, model: str) -> tuple[dict[str, object], tuple[int, ...]]:
    """Build the hashable metadata available from /version, /v1/models, and /metrics."""
    version_response = await client.get("/version")
    version_response.raise_for_status()
    version_body = version_response.json()
    if not isinstance(version_body, dict) or not isinstance(version_body.get("version"), str):
        raise ValueError("/version did not return a string version")

    models_response = await client.get("/v1/models")
    models_response.raise_for_status()
    models_body = models_response.json()
    entries = models_body.get("data") if isinstance(models_body, dict) else None
    if not isinstance(entries, list):
        raise ValueError("/v1/models did not return a model list")
    selected = next(
        (entry for entry in entries if isinstance(entry, dict) and entry.get("id") == model),
        None,
    )
    if selected is None:
        raise ValueError(f"model {model!r} is not served by /v1/models")

    metrics_response = await client.get("/metrics")
    metrics_response.raise_for_status()
    samples = _parse_samples(metrics_response.text)
    engine_ids = _engine_ids(samples)
    cache_configs = _cache_configs(samples, engine_ids)
    metric_names = sorted(name for name in THROUGHPUT_COUNTERS if any(sample[0] == name for sample in samples))
    metadata: dict[str, object] = {
        "schema_version": 1,
        "backend": {"type": "vllm", "version": version_body["version"]},
        "model": {key: selected[key] for key in ("id", "root", "max_model_len") if key in selected},
        "deployment": {
            "engine_count": len(engine_ids),
            "cache_config": cache_configs,
            "throughput_metric_names": metric_names,
        },
    }
    return metadata, engine_ids


async def collect_throughput_counters(client: httpx.AsyncClient, engine: int) -> dict[str, float]:
    """Read selected cumulative counters for one EngineCore/DP rank."""
    response = await client.get("/metrics")
    response.raise_for_status()
    values = dict.fromkeys(THROUGHPUT_COUNTERS, 0.0)
    found_engine = False
    for name, labels, value in _parse_samples(response.text):
        if labels.get("engine") != str(engine):
            continue
        found_engine = True
        if name in values:
            values[name] += value
    if not found_engine:
        raise ValueError(f"/metrics does not expose engine={engine}")
    return values


def counter_delta(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    """Calculate nonnegative metric counter deltas for one measurement window."""
    return {name: max(after.get(name, 0) - value, 0) for name, value in before.items()}


def metadata_hash(metadata: dict[str, object]) -> str:
    """Return SHA-256 over canonical UTF-8 metadata JSON."""
    canonical = json.dumps(metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )
    return "sha256:" + hashlib.sha256(canonical).hexdigest()


def persist_metadata(root: Path, metadata: dict[str, object]) -> str:
    """Persist metadata once and reject mixing stages from different deployments."""
    path = root / "metadata.json"
    digest = metadata_hash(metadata)
    record = {"metadata": metadata, "metadata_hash": digest}
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous != record:
            raise ValueError("prefill and decode metadata do not match")
    else:
        write_json(path, record)
    return digest


def assemble_profile(root: Path) -> Path | None:
    """Create the three-field profile after both measurement stages complete."""
    metadata_path = root / "metadata.json"
    prefill_path = root / "prefill" / "result.json"
    decode_path = root / "decode" / "result.json"
    if not all(path.exists() for path in (metadata_path, prefill_path, decode_path)):
        return None
    metadata_record = json.loads(metadata_path.read_text(encoding="utf-8"))
    prefill = json.loads(prefill_path.read_text(encoding="utf-8"))
    decode = json.loads(decode_path.read_text(encoding="utf-8"))
    if prefill["metadata_hash"] != decode["metadata_hash"]:
        raise ValueError("prefill and decode metadata hashes do not match")
    if prefill["scope"] != decode["scope"]:
        raise ValueError("prefill and decode calibration scopes do not match")
    profile = {
        "metadata": metadata_record["metadata"],
        "metadata_hash": metadata_record["metadata_hash"],
        "calibration": {
            "scope": prefill["scope"],
            "prefill": prefill["calibration"],
            "decode": decode["calibration"],
        },
    }
    output = root / "calibration.json"
    write_json(output, profile)
    return output


def validate_engine(engine: int, engine_ids: tuple[int, ...]) -> None:
    """Require calibration traffic to target an exposed internal-DP rank."""
    if engine not in engine_ids:
        raise ValueError(f"engine {engine} is not present in /metrics: {engine_ids}")


def resolve_calibration_engine(requested_engine: int | None, engine_ids: tuple[int, ...]) -> tuple[int, int | None]:
    """Select the observed engine and the optional request-routing header value."""
    engine = engine_ids[0] if requested_engine is None else requested_engine
    validate_engine(engine, engine_ids)
    request_rank = engine if len(engine_ids) > 1 else None
    return engine, request_rank


def infer_kv_capacity_tokens(metadata: dict[str, object]) -> int:
    """Derive one EngineCore rank's logical HBM KV capacity from cache metrics."""
    deployment = metadata.get("deployment")
    cache_config = deployment.get("cache_config") if isinstance(deployment, dict) else None
    if not isinstance(cache_config, dict):
        raise ValueError("/metrics does not expose vllm:cache_config_info")
    try:
        block_size = int(str(cache_config["block_size"]))
        num_gpu_blocks = int(str(cache_config["num_gpu_blocks"]))
        engine_count = int(deployment["engine_count"])
    except (KeyError, ValueError) as exc:
        raise ValueError("cache_config_info and deployment must expose integer capacity fields") from exc
    if engine_count <= 0:
        raise ValueError("engine_count must be positive")
    # In an internally managed DP deployment, the aggregated /metrics endpoint
    # repeats the deployment-wide num_gpu_blocks value for every engine label.
    # Calibration pins one EngineCore, so its usable share is the aggregate
    # logical token capacity divided by the homogeneous engine count.
    capacity = block_size * num_gpu_blocks // engine_count
    if capacity <= 0:
        raise ValueError("derived KV capacity must be positive")
    return capacity


def _parse_samples(text: str) -> list[tuple[str, dict[str, str], float]]:
    """Parse Prometheus exposition into stable sample tuples."""
    return [
        (sample.name, dict(sample.labels), float(sample.value))
        for family in text_string_to_metric_families(text)
        for sample in family.samples
    ]


def _engine_ids(samples: list[tuple[str, dict[str, str], float]]) -> tuple[int, ...]:
    """Discover numeric EngineCore ranks from metric labels."""
    values = {int(labels["engine"]) for _, labels, _ in samples if labels.get("engine", "").isdigit()}
    if not values:
        raise ValueError("/metrics does not expose numeric engine labels")
    return tuple(sorted(values))


def _cache_configs(samples: list[tuple[str, dict[str, str], float]], engine_ids: tuple[int, ...]) -> dict[str, str]:
    """Return one common cache config and reject heterogeneous internal DP."""
    configs: dict[int, dict[str, str]] = {}
    for name, labels, _ in samples:
        if name != "vllm:cache_config_info" or "engine" not in labels:
            continue
        engine = int(labels["engine"])
        configs[engine] = {key: value for key, value in labels.items() if key != "engine"}
    if not configs:
        raise ValueError("/metrics does not expose vllm:cache_config_info")
    missing = set(engine_ids).difference(configs)
    if missing:
        raise ValueError(f"cache_config_info is missing engines: {sorted(missing)}")
    first = configs[engine_ids[0]]
    if any(configs[engine] != first for engine in engine_ids[1:]):
        raise ValueError("initial calibration supports only homogeneous internal DP")
    return dict(sorted(first.items()))
