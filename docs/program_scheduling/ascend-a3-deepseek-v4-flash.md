# Tuning DeepSeek V4 Flash on Ascend A3

A jiuwenswarm operator wants to reduce **mean job completion time (JCT)** for 256 tasks served by DeepSeek V4 Flash on Ascend A3 at a maximum concurrency of 128. Across two runs of the same tasks, mean JCT fell from 4,325.22 s on the baseline to 3,665.82 s on the TTL run, **a 15.25% reduction**.

Replay the same requests to see whether affinity alone lowers mean JCT versus native routing and whether full scheduling saves additional time.

## 1. Invoke the tuning skill

Give the agent the run directories, service endpoints and deployment targets. For example:

```text
$program-ttl-tuning
Choose native routing, affinity only, or full scheduling for DeepSeek V4 Flash (served as dsv4).
For the replay use one Ascend A3, one worker, TP4 × internal DP4.
Router http://127.0.0.1:3001, worker http://127.0.0.1:8000, worker metrics at /metrics.
Baseline run directory: results/baseline/
TTL run directory: results/pt/
Calibrate the cost models before the comparison. Replay the same baseline requests.jsonl on the isolated deployment with 256 tasks at concurrency 128, then compare the run summaries.
Minimize mean JCT across all tasks. Report its reduction relative to native routing and report p95 JCT.
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
from pathlib import Path

def fit_file(path, fitter):
    return fitter(json.loads(Path(path).read_text()))["coefficients"]

print(fit_file("prefill-points.json", fit_prefill))
print(fit_file("decode-points.json", fit_decode))
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

Send two turns through the **actual client** to the Router's `/v1/chat/completions`. Save each outgoing body and take a diagnostics snapshot immediately after each response. If the entry expires first, capture it while the request is in flight. The first body's scheduling fields are shown below. Change `request_id` to `r2` and `step_id` to `2` for the second turn. Use a fresh session ID for each probe.

```json
{"vllm_xargs":{"agentic_context":"{\"session_id\":\"a3-example-42\",\"program_id\":\"a3-example-42:lead\",\"request_id\":\"r1\",\"step_id\":1,\"expected_resume\":true}"}}
```

| Client request | Outgoing `program_id` / `step_id` | Matching Router entry (illustrative) |
|---|---|---|
| Turn 1, `r1` | `a3-example-42:lead` / `1` | `step_id=1, program=9f31…b72a, generation=1, dp_rank=2` |
| Turn 2, `r2` | `a3-example-42:lead` / `2` | `step_id=2, program=9f31…b72a, generation=1, dp_rank=2` |

```bash
ROUTER=http://127.0.0.1:3001
SESSION_ID=a3-example-42  # use this probe's unique session
curl -fsS "$ROUTER/scheduling/diagnostics" |
  jq -c --arg sid "$SESSION_ID" '.ranks[] as $r | $r.programs[]
    | select(.session_id == $sid) | {session_id,step_id,program,generation,dp_rank:$r.dp_rank}'
```

The two snapshots must show the client's session, successive steps, and the same redacted `program` and `generation`. A global `program_admit` count includes other clients. The historical runs lack these snapshots, so the results below retain their original labels.

<details>
<summary>Optional control: set zero TTL on every request under full scheduling (C0)</summary>

C0 uses full scheduling with a zero TTL on every request. C uses the same configuration with automatic TTL. Insert this before `return body` in the pinned `ReplayTransport._body` chat branch and set `A3_REPLAY_ARM` for each run. Check the request IDs in the send log. Every C0 attempt should print `0` and every C attempt `None`. [Replay transport](https://github.com/openJiuwen-ai/agent-infer/blob/f764c9a7c387eef632defa96fb3ccae6f654c0f9/agentinfer/agentbench/replay/transport.py#L192-L209)

```python
ctx = json.loads(body["vllm_xargs"]["agentic_context"])
if "kv_retention_ttl" in ctx or "kv_retention_ttl_seconds" in ctx:
    raise ValueError("source request already overrides TTL")
if os.environ["A3_REPLAY_ARM"] == "C0":
    ctx["kv_retention_ttl_seconds"] = 0
    body["vllm_xargs"]["agentic_context"] = json.dumps(ctx)
print(ctx["request_id"], ctx.get("kv_retention_ttl_seconds"), flush=True)
```

</details>

Replay the same requests on an isolated deployment at the same concurrency and timeout. Before timing each arm, restart the Router and all workers. Start when the services are ready, queues are empty and rank sample counts are zero. Exclude the two identity probes from the measurements. [One-session replay command](https://github.com/vllm-project/router/pull/286#user-content-end-to-end-replay)

## 3. Use the replay logs to evaluate and diagnose the configuration

**Calculate mean JCT.** Set `RUN_DIR` to each run directory in turn. In `summary.json`, `.tasks.duration_seconds.mean` is the sum of the recorded durations for all 256 tasks divided by 256, including failures and timeouts. The relative reduction is `(baseline mean − candidate mean) / baseline mean`. Report p95 JCT to show the tail. Early failures can also shorten mean JCT, so check the completion rate (completed tasks divided by planned tasks) alongside client p95 TTFT.

```bash
jq '{jct_mean_s: .tasks.duration_seconds.mean, jct_p95_s: .tasks.duration_seconds.p95,
     completed: .tasks.completed, ttft_p95_s: .requests.ttft_seconds.p95}' "$RUN_DIR/summary.json"
```

**Calculate token-weighted cache reuse.** The baseline had `215,777,280 / 268,142,686 = 80.47%` cached prompt tokens. The TTL run had `269,561,856 / 296,502,450 = 90.91%`. Evaluate the following at each arm's end, using its actual worker selector and observation window. Take one series per underlying counter, then sum worker increases before dividing.

```promql
100 * sum(increase(vllm:prompt_tokens_cached_total{job="a3-workers"}[30m]))
    / sum(increase(vllm:prompt_tokens_total{job="a3-workers"}[30m]))
```

The 80–88% baseline hit band helps find a workload under cache pressure. It is still worth comparing routes outside that band at normal concurrency. In a separate replay with a 63.74% baseline hit rate, both arms completed 24/24 tasks and wall time fell 24.9%. For this A3 deployment, use mean per-task JCT to choose a route. [Benchmark results](https://github.com/vllm-project/router/pull/286#user-content-test-result)

**Measure the return gap.** Group `requests.jsonl` by Program and subtract the previous `finished_at` from the next `started_at`. The first pair comes from the A3 example. The second pair has both session and actor identity, allowing a Program-level calculation. Times are UTC. [Pinned trace](https://github.com/openJiuwen-ai/agent-infer/blob/f764c9a7c387eef632defa96fb3ccae6f654c0f9/tests/agentbench/replay/claude_trace_8session_requests.jsonl)

| Pair | Previous request finished | Next request arrived | Gap |
|---|---|---|---:|
| A3, same session | `219e53d3…` 08:43:49.281150 | `bab99015…` 08:43:50.315012 | 1.033862 s |
| Pinned trace, same session and actor | `699fc599…` 04:37:53.610538 | `fade2058…` 04:37:56.059261 | 2.448723 s |

The A3 TTL run had 6,842 successful, non-overlapping adjacent session pairs and a session-gap p95 of 8.061 s. A session may contain multiple actors. Group by the outgoing `program_id` collected in step 2 to compare these intervals with that Program's armed TTL.

**If the replay meets the targets, use section 4 to choose a route.** If performance falls short, inspect cache reuse, queueing and same-Program return gaps to decide what to change next.

| Observation | Check and next experiment |
|---|---|
| Cache reuse rises and mean JCT gets worse | Inspect Router queue time, the blocking reason and the worker's waiting count. Test `admission_waiting_request_threshold` on its own if that gate is blocking requests and the worker has headroom. |
| The same Program often returns just after its TTL expires | Confirm the statistics window is full. If the armed TTL repeatedly reaches `max_acting_ttl_seconds`, test a slightly higher cap and check JCT and queue time again. |
| Predicted prefill or decode times are consistently off | Compare engine samples at matched input lengths or matched batch sizes and context lengths. If the error persists, refit the affected model using fresh engine samples and validate on held-out data. |

Change one factor per replay. Use the same requests and check mean JCT, completion rate and p95 TTFT before keeping the change.

## 4. Which routing mode should you deploy at normal concurrency?

Both runs used the same 256 tasks in the same order, a maximum concurrency of 128 and a 7,200 s timeout per task.

| Example run (max concurrency 128) | Mean JCT | p95 JCT | Completed / planned (rate) | Client p95 TTFT |
|---|---:|---:|---:|---:|
| Baseline | 4,325.22 s | 7,217.03 s | 158/256 (61.72%) | 468.33 s |
| TTL | 3,665.82 s | 7,217.97 s | 180/256 (70.31%) | 394.47 s |

The TTL run saved 659.40 s per task on average (about 11 minutes), completed 22 more tasks and reduced client p95 TTFT by 73.86 s. More than 5% of tasks hit the 7,200 s timeout in each run, so p95 JCT stayed near two hours. **Enable Program scheduling if a fixed replay reproduces the reduction in mean JCT and meets the completion rate and p95 TTFT targets.**

### Should other deployments enable Program scheduling?

This example tested a maximum concurrency of 128. On another service, replay the same requests against native routing and the candidate modes at both normal and peak concurrency. Use the behavior you observe to choose a mode.

| Observed workload or deployment | Recommendation |
|---|---|
| At low concurrency, native routing shows little queueing and already reuses context, or most requests are independent single turns | **Keep native routing.** These requests seldom need the scheduler to reserve capacity for a later turn. |
| Successive turns of the same agent often land on different DP ranks, causing a long prefix to be recomputed | **Try affinity only** (`binding_only: true`) so later turns return to the same rank. Keep it if it consistently lowers mean JCT and full scheduling adds no further benefit. |
| At high concurrency, agents compete for KV cache and return quickly from tools, yet often repeat long prefills even on the same rank | **Compare full scheduling with affinity only.** Enable full scheduling if it further reduces mean JCT while meeting the completion rate and p95 TTFT targets. If reserving capacity increases queueing and mean JCT, use the eligible affinity-only or native mode. |
| The model, accelerator, node count or TP/DP layout changes | **Establish a native baseline on the new deployment.** Check KV capacity per rank, calibrate prefill and decode costs, and compare the three modes at the new deployment's normal concurrency. |
