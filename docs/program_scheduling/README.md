# Program Scheduling

Program scheduling keeps a logical Program identity for one agent task and schedules its turns against backend capacity instead of routing each request independently.

| Document | Covers |
| --- | --- |
| [Program scheduling internals](program-scheduling.md) | Native (non-PD) agent-aware scheduler: Program identity contract, admission math, progress TTL, offline calibration, engine KV-control boundary. |
| [Agent integration for workflow TTL prediction](agent-integration.md) | Agent-facing request contract: stable Program identity, lifecycle signaling, workflow taxonomy, request examples, and rollout validation. |
| [Tuning DeepSeek V4 Flash on Ascend A3](ascend-a3-deepseek-v4-flash.md) | Worked example using mean job completion time to choose between native routing, affinity only and full Program scheduling. |
| [PD Program Scheduling](pd-program-scheduling.md) | Program-scoped scheduling in HTTP PD mode: D-rank affinity, bounded prefill permits, warm remote KV reuse, automatic TTL, configuration reference, diagnostics, troubleshooting. |

The PD document requires `--vllm-pd-disaggregation` with `--kv-connector nixl`; every other deployment uses the native scheduler.
