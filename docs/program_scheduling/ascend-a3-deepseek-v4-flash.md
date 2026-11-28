# Tuning DeepSeek V4 Flash on Ascend A3

A JiuwenSwarm agent runs 256 SWE-bench tasks against DeepSeek V4 Flash on Ascend A3, with up to 128 tasks active at once. The goal is to reduce **mean task duration after client admission**. The agent runner times each task after it acquires a concurrency slot. Across two runs of the same tasks, mean job completion time (JCT) fell from 4,325.22 s on the baseline to 3,665.82 s on the TTL run, **a 15.25% reduction**.

To choose a routing mode, run the live agent on the same ordered SWE-bench instances under native routing, affinity only and full scheduling. The agent generates fresh model requests and tool calls in each run.

## 1. Invoke the tuning skill

Give the agent the run directories, service endpoints and deployment targets. For example:

```text
$program-ttl-tuning
Choose native routing, affinity only, or full scheduling for DeepSeek V4 Flash (served as dsv4).
For the comparison use one Ascend A3, one worker, TP4 × internal DP4.
Router http://127.0.0.1:3001, worker http://127.0.0.1:8000, worker metrics at /metrics.
Baseline run directory: results/baseline/
TTL run directory: results/pt/
Calibrate the cost models first. For each routing mode, run JiuwenSwarm (code.normal) on the same 256 SWE-bench instances in the same order, at concurrency 128 with a 7,200 s task timeout. Use a fresh result directory for each run. Then read summary.json for metrics and requests.jsonl for request timings.
Minimize mean task duration after client admission (JCT) across all tasks, including failures and timeouts. Report its reduction relative to native routing, p95 JCT, and wall time for the whole run.
Use service targets for the minimum task completion rate and maximum client p95 time to first token (TTFT). For each unspecified limit, use its measured baseline on the same tasks with the same concurrency and timeout.
Return the eligible routing mode with the lowest mean JCT and its config diff.
```

This example has no specified service targets, so both comparisons use the measured baseline. Read existing logs, metrics and diagnostics on the live service.

## 2. Calibrate the Router and verify client opt-in before the comparison

### Calibrate the cost models from engine timings

Before comparing policies, check how long this A3 deployment takes to process uncached input (prefill) and to generate one output step for a batch (decode). If a calibration already exists for the same model, engine settings and parallel layout, check it against engine samples and reuse it if it fits. For a first calibration, measure on an idle worker. To capture costs at concurrency 128, use engine timings from native routing under that load. If those timings are missing, collect a native baseline first.

For example, cold prefills of 1k, 4k and 8k tokens taking 0.07, 0.28 and 0.84 s give `0.04 + 0.02x + 0.01x²` seconds, with `x` in thousands of tokens. These values illustrate the fit. The two run summaries contain only aggregate timings, so the A3 fit needs engine samples. Write the validated prefill and decode coefficients into the Router configuration and hold them fixed across the policy comparison.

<details>
<summary>Fit prefill and decode from native-baseline logs at normal concurrency</summary>

The AgentInfer active calibration commands need an idle worker. For a loaded fit, join baseline engine observations by request ID and rank and produce two JSON arrays:

| Input | Fields |
|---|---|
| `prefill-points.json` | Group uncached prefills by at least three prompt lengths. Use `prompt_tokens` and `median_ttft_seconds` from first scheduled prefill to first token, excluding admission wait. |
| `decode-points.json` | For steady decode-only windows record `batch_size`, `total_context_tokens` across the batch, and `median_iteration_seconds`. Vary batch and context independently. Normalize speculative steps by actual output-token advancement. |

Run this Python in the AgentInfer checkout. Check predictions on held-out windows and map the six selected coefficients to the Router. Set `prefill_cost_model.decode_throughput_alpha` separately. A starting per-rank token budget is `floor(0.95 × smallest rank startup capacity)`. [Calibration tools](https://github.com/openJiuwen-ai/agent-infer/blob/f764c9a7c387eef632defa96fb3ccae6f654c0f9/tools/calibration/README.md#supported-deployment-scope) · [Router field mapping](program-scheduling.md#offline-calibration-models)

```python
from tools.calibration.calibrate_prefill import fit_prefill
from tools.calibration.calibrate_decode import fit_decode
import json
import math
from pathlib import Path

def fit_file(path, fitter):
    return fitter(json.loads(Path(path).read_text()))["coefficients"]

prefill = fit_file("prefill-points.json", fit_prefill)
decode = fit_file("decode-points.json", fit_decode)
fixed = decode["decode_step_fixed_seconds"]
if not math.isfinite(fixed) or fixed <= 0:
    raise ValueError("Router requires fixed_step_seconds > 0. Check the decode samples and refit.")
print(prefill, decode, sep="\n")
```

</details>

### Launch the comparison modes

Use the native `--policy cache_aware` command as the baseline. Keep the worker endpoints and DP layout the same in all runs. For the two Program modes, add these arguments to the native command, using the calibration JSON from the preceding step:

```diff
  --policy cache_aware
+ --enable-program-scheduling
+ --program-scheduling-config-json "$(jq -c --argjson binding "$BINDING_ONLY" '.binding_only = $binding' program-scheduling.json)"
```

Set `BINDING_ONLY=true` for affinity only and `BINDING_ONLY=false` for full scheduling. The optional C0 control uses the full-scheduling command with a per-request TTL override, shown below.

### Verify two turns from the actual client

Run a short JiuwenSwarm task through the Router's `/v1/chat/completions`. Capture two successive outgoing requests from the same lead agent and take a diagnostics snapshot after each response (or in flight if the entry expires). The JiuwenSwarm adapter sets `program_id` to the current agent session ID. The example below shows the relevant fields with an illustrative ID. [JiuwenSwarm adapter](https://github.com/openJiuwen-ai/agent-infer/blob/f764c9a7c387eef632defa96fb3ccae6f654c0f9/agentinfer/agentbench/agents/jiuwenswarm/extension.py#L194-L212)

```json
{"vllm_xargs":{"agentic_context":"{\"session_id\":\"a3-example-42\",\"task_id\":\"a3-example-42\",\"program_id\":\"a3-example-42\",\"agent_id\":\"lead\",\"agent_role\":\"lead\",\"expected_resume\":true}"}}
```

```bash
ROUTER=http://127.0.0.1:3001
SESSION_ID=a3-example-42  # replace with the session ID observed on the wire
curl -fsS "$ROUTER/scheduling/diagnostics" |
  jq -c --arg sid "$SESSION_ID" '.ranks[] as $r | $r.programs[]
    | select(.session_id == $sid) | {session_id,step_id,program,generation,dp_rank:$r.dp_rank}'
```

Look for the captured session, the same redacted `program` and `generation`, and an advancing `step_id` across the two snapshots. For a new Program without client step IDs, the Router reports step 0 then 1 for two successive requests. A global `program_admit` count includes other clients. The historical runs lack these snapshots, so the results below retain their original labels.

<details>
<summary>Optional control: set zero TTL on every request under full scheduling (C0)</summary>

Use the same full-scheduling Router config for C0 and C. In an isolated AgentInfer checkout, insert the snippet immediately before `upstream_request = self.client.build_request(...)` in the **live benchmark request proxy**. Export `A3_TTL_ARM=C0` for zero TTL or `A3_TTL_ARM=C` for automatic TTL before launching the benchmark. Leave the variable unset for native and affinity-only runs. [Proxy forwarding code](https://github.com/openJiuwen-ai/agent-infer/blob/f764c9a7c387eef632defa96fb3ccae6f654c0f9/agentinfer/agentbench/request_proxy/server.py#L133-L150)

```python
import os

arm = os.environ.get("A3_TTL_ARM")
if arm is not None:
    if arm not in {"C0", "C"} or self.endpoint != "/v1/chat/completions":
        raise ValueError("Check A3_TTL_ARM and the benchmark endpoint")
    payload = json.loads(body)
    xargs = payload["vllm_xargs"]
    ctx = json.loads(xargs["agentic_context"])
    if "kv_retention_ttl" in ctx or "kv_retention_ttl_seconds" in ctx:
        raise ValueError("The live agent already overrides TTL")
    if arm == "C0":
        ctx["kv_retention_ttl_seconds"] = 0
        xargs["agentic_context"] = json.dumps(ctx)
        body = json.dumps(payload).encode()
    print("ttl_check", request_id, ctx["program_id"], ctx.get("kv_retention_ttl_seconds"), flush=True)
```

After each control run, match the `ttl_check` lines in `proxy.stdout.log` to the request IDs in that run's `requests.jsonl`, including retries. Verify there is one line per attempt. Every C0 value must be `0` and every C value must be `None`. For eligible C0 completions, verify that the Router's armed TTL is zero with source `request_hint`. Check that C actually armed a positive TTL before attributing a difference to retention.

</details>

### Run SWE-bench with JiuwenSwarm

On the isolated deployment, restart the Router and all workers after the probe and before each measured run. Start when services are ready, queues are empty and enabled ranks have zero request and continuity samples. Use the same prepared SWE-bench dataset and ordered instance list for each routing mode. If the original list is unavailable, the `jq` command below rebuilds it from the baseline task index. Set `RUN_DIR` to a new directory each time. [SWE-bench setup](https://github.com/openJiuwen-ai/agent-infer/blob/f764c9a7c387eef632defa96fb3ccae6f654c0f9/docs/en/how-to/run-benchmark.md#prepare-swe-bench-data)

```bash
jq -r '.tasks | sort_by(.task_position)[] | .instance_id' results/baseline/task_index.json > a3-tasks.txt
RUN_DIR=results/native-1
vllm bench serve --agentinfer run \
  --selection-path "$PWD/a3-tasks.txt" --task-num 256 --max-concurrency 128 --timeout 7200 \
  --agent-type jiuwenswarm --agent-executable jiuwenswarm --agent-profile code.normal \
  --endpoint /v1/chat/completions --model dsv4 --base-url http://127.0.0.1:3001 \
  --metrics-url http://127.0.0.1:8000/metrics --result-dir "$RUN_DIR"
```

## 3. Read the benchmark results

**Read task duration and batch wall time.** A task may wait for a free client slot before its timer starts. In `summary.json`, `.tasks.duration_seconds` gives the mean and p95 of the time **after** each task acquires its slot. The mean covers all 256 recorded tasks, including failures and timeouts. For elapsed time across the whole run, read `.run_wall_time_seconds`. [Live benchmark runner](https://github.com/openJiuwen-ai/agent-infer/blob/f764c9a7c387eef632defa96fb3ccae6f654c0f9/agentinfer/agentbench/benchkit/runner.py#L225-L240)

Check that each run's `task_index.json` lists the same 256 instance IDs in the same order. Live agents may take different paths, so also compare completed instance IDs and input/output token totals when interpreting timing differences. Calculate the relative reduction in mean task duration as `(baseline mean − candidate mean) / baseline mean`. Early failures can shorten the mean, so check the completion rate (agent runs completed divided by tasks planned) and client p95 TTFT. SWE-bench patch correctness is evaluated separately.

```bash
jq '{task_seconds_after_client_admission: (.tasks.duration_seconds | {mean, p95}),
     batch_wall_time_s: .run_wall_time_seconds,
     completed: .tasks.completed, ttft_p95_s: .requests.ttft_seconds.p95}' "$RUN_DIR/summary.json"
```

**Calculate token-weighted cache reuse.** The baseline had `215,777,280 / 268,142,686 = 80.47%` cached prompt tokens. The TTL run had `269,561,856 / 296,502,450 = 90.91%`. Evaluate the following at each arm's end, using its actual worker selector and observation window. Take one series per underlying counter, then sum worker increases before dividing.

```promql
100 * sum(increase(vllm:prompt_tokens_cached_total{job="a3-workers"}[30m]))
    / sum(increase(vllm:prompt_tokens_total{job="a3-workers"}[30m]))
```

The 80–88% baseline hit band helps find a workload under cache pressure. Compare routing modes at normal concurrency even outside that band. In a separate benchmark with a 63.74% baseline hit rate, both runs completed 24/24 tasks and wall time fell 24.9%. [Benchmark results](https://github.com/vllm-project/router/pull/286#user-content-test-result)

**Measure the return gap.** After the run, use the request log, `requests.jsonl`. For consecutive successful, non-overlapping requests from the same Program, subtract the previous `finished_at` from the next `started_at`. The first pair comes from the A3 example. The second pair has both session and actor identity, allowing a Program-level calculation. Times are UTC. [Pinned trace](https://github.com/openJiuwen-ai/agent-infer/blob/f764c9a7c387eef632defa96fb3ccae6f654c0f9/tests/agentbench/replay/claude_trace_8session_requests.jsonl)

| Pair | Previous request finished | Next request arrived | Gap |
|---|---|---|---:|
| A3, same session | `219e53d3…` 08:43:49.281150 | `bab99015…` 08:43:50.315012 | 1.033862 s |
| Pinned trace, same session and actor | `699fc599…` 04:37:53.610538 | `fade2058…` 04:37:56.059261 | 2.448723 s |

The A3 TTL run had 6,842 successful, non-overlapping adjacent session pairs and a session-gap p95 of 8.061 s. A session may contain multiple actors. For a Program-level comparison, join each proxy request ID to the outgoing `program_id` as in the C0/C check. Where only session identity is available, report session-level gaps.

**If the live runs meet the targets, use section 4 to choose a route.** If performance falls short, inspect cache reuse, queueing and same-Program return gaps to decide what to change next.

| Observation | Check and next experiment |
|---|---|
| Cache reuse rises and mean JCT gets worse | Inspect Router queue time, the blocking reason and the worker's waiting count. Test `admission_waiting_request_threshold` on its own if that gate is blocking requests and the worker has headroom. |
| The same Program often returns just after its TTL expires | Check the armed TTL histogram's `source` label. For `fitted`, check the rank continuity window. For `return_prediction`, check the Program's prediction support. For `request_hint`, inspect the request override. If an automatic TTL repeatedly reaches `max_acting_ttl_seconds`, test a slightly higher cap and check JCT and queue time again. |
| Predicted prefill or decode times are consistently off | Compare engine samples at matched input lengths or matched batch sizes and context lengths. If the error persists, refit the affected model using fresh engine samples and validate on held-out data. |

Change one factor per run. Use the same SWE-bench tasks and check mean JCT, completion rate and p95 TTFT before keeping the change.

## 4. Which routing mode should you deploy at normal concurrency?

Both runs used the same 256 tasks in the same order, a maximum concurrency of 128 and a 7,200 s timeout per task.

| Metric | Baseline | TTL |
|---|---:|---:|
| Mean task duration after client admission | 4,325.22 s | 3,665.82 s |
| p95 task duration after client admission | 7,217.03 s | 7,217.97 s |
| Whole-run wall time | 14,451.60 s | 12,185.16 s |
| Completed / planned tasks (rate) | 158/256 (61.72%) | 180/256 (70.31%) |
| Client p95 TTFT | 468.33 s | 394.47 s |

After client admission, tasks in the TTL run took 659.40 s less on average (about 11 minutes). The TTL run completed 22 more tasks, reduced client p95 TTFT by 73.86 s and cut whole-run wall time by 15.68%. More than 5% of tasks hit the 7,200 s timeout in each run, so p95 JCT stayed near two hours. **Enable Program scheduling if repeated live-agent runs on the same SWE-bench tasks reproduce the reduction in mean JCT and meet the completion rate and p95 TTFT targets.**

### Should other deployments enable Program scheduling?

This example tested a maximum concurrency of 128. On another service, run the same representative tasks with your agent under native routing and the candidate modes at both normal and peak concurrency. Use the behavior you observe to choose a mode.

| Observed workload or deployment | Recommendation |
|---|---|
| At low concurrency, native routing shows little queueing and already reuses context, or most requests are independent single turns | **Keep native routing.** These requests seldom need the scheduler to reserve capacity for a later turn. |
| Successive turns of the same agent often land on different DP ranks, causing a long prefix to be recomputed | **Try affinity only** (`binding_only: true`) so later turns return to the same rank. Keep it if it consistently lowers mean JCT and full scheduling adds no further benefit. |
| At high concurrency, agents compete for KV cache and return quickly from tools, yet often repeat long prefills even on the same rank | **Compare full scheduling with affinity only.** Enable full scheduling if it further reduces mean JCT while meeting the completion rate and p95 TTFT targets. If reserving capacity increases queueing and mean JCT, use the eligible affinity-only or native mode. |
| The model, accelerator, node count or TP/DP layout changes | **Establish a native baseline on the new deployment.** Check KV capacity per rank, calibrate prefill and decode costs, and compare the three modes at the new deployment's normal concurrency. |
