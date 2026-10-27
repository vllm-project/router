# E2E performance (`tests/python/e2e/perf`)

GPU/NPU performance E2E suites for vLLM Router **agent-aware** (program scheduling) replay.

| Suite | Purpose | Runbook |
| ----- | ------- | ------- |
| [`agentaware/`](agentaware/README.md) | Cold vLLM + **Router** + trace **`replay`** (binding-only vs program scheduling) | [`agentaware/README.md`](agentaware/README.md) |

Shared vLLM lifecycle and case loading: [`../helpers/`](../helpers/).

Pytest marker: `e2e_perf` (run explicitly; not part of default CI unit/integration paths).

Example:

```bash
pytest -s -v tests/python/e2e/perf/agentaware/run_benchmark.py \
  --test-config-file tests/python/e2e/perf/agentaware/cases/binding_only/glm52_24x12_router_replay.json
```
