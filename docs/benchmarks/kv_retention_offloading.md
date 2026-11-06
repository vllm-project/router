# KV retention on OffloadingConnector, hint on and off

A prototype of [RFC #427](https://github.com/vllm-project/router/issues/427)'s retain and release hints on vLLM's
native CPU offload, and the comparison the RFC asks for: which tier retention can protect, and what it does to GPU hits,
CPU restores, recomputation and eviction pressure. It changes nothing in the Router and does not define the Router-side
contract, which the RFC tracks separately. Its field names follow that proposal so that they can be renamed when it
lands.

## Summary

On one GPU, one model and one trace family, with three seeds per arm, retention protected a single prefix on either tier.
In the trace replay, no setting gained reliably beyond what two runs of the same arm differ by. A CPU lease raised CPU
restores every time it ran, and computed tokens and latency did not reliably follow. A larger CPU tier moved far more than
any hint.

**What the runs showed:**

- **One prefix, in a probe.** An 8,000-token prompt pushed out of the pool by other prompts came back in
  63 ms with a GPU hold, against 274 ms restored from the CPU tier without one. With a CPU tier the
  size of the pool, it came back in 277 ms with a lease, against 1,068 ms recomputed without one.
- **What a repeat moves.** The no-hint arms ran twice on each seed. At a CPU tier four times the pool, the two runs differ
  by up to 0.4 points of computed tokens, 2.0 of GPU hits and 12% in
  resume TTFT p90. At a tier the size of the pool, by up to 1.8, 2.2 and
  10%.
- **GPU holds in the replay: no gain.** Seed by seed, GPU hits moved +0.7, +0.9 and +0.1 points with a 10 s hold over 25% of the pool, and
  +0.7, −0.3 and −0.4 points with a 30 s hold over 50% of the pool. Computed tokens moved by at most 0.5 points. Resume TTFT
  p90 moved +1%, +8% and −7% with the first, and −4%, +35% and +19% with the second, beyond a repeat on two seeds. The holds
  kept 12.5% and 26.7% of the pool on average.
- **CPU leases at a tier the size of the pool: no reliable gain.** A 30 s lease over 90% of the tier raised CPU restores (+1.5, +5.3 and +3.8
  points), beyond a repeat. Computed tokens moved −0.7, +1.3 and −5.3 points and resume TTFT p90 +8%, +19% and −6%. A first
  round's −6.7, −5.0 and −6.2 points did not reproduce: see
  [The 30 s lease, run by run](#the-30-s-lease-run-by-run). A 120 s lease over 50% of the tier moved computed tokens −0.4, +6.4 and +3.6
  points, beyond a repeat on two seeds.
- **A hold on top of a lease: worse than no hint.** A 30 s hold over 50% of the pool added to the 120 s lease over 50% of the tier (`both-1x`) moved
  computed tokens +2.1, +8.9 and +5.0 points, each beyond a repeat, and resume TTFT p90 +5%, +54% and +14%.
- **The same 30 s lease at a tier four times the pool:** no clear change. Computed tokens moved +0.9, −0.9 and −0.7 points, CPU restores −0.5, +1.3 and 0.0 points and resume TTFT p90 +12%, +3% and +2%. All-turn TTFT p90 moved +48%, +9% and +1%, beyond a repeat on one seed. No leased chunk was evicted in any run.
- **Capacity.** With no hint, a 4x CPU tier in place of a 1x tier took computed tokens from 40.9% (40.8–62.9) to
  12.5% (10.2–13.6), and resume TTFT p90 from 6,471 (4,738–14,579) to 1,366 (718–3,840) ms. Neither pair of ranges overlaps.
- **Stores.** vLLM counted no failed store in any run, leases included.

**What that suggests, which these runs do not measure directly:**

- A GPU hold costs other sessions what it gives one. Used holds had been held for about a second at the median
  (0.3 to 1.2 s), when LRU would probably still have had those blocks, and the holds that could have mattered were
  refused for lack of budget or gave way to demand. At this load the pool looks close to zero-sum.
- A lease moves restores more reliably than it moves computation. A 30 s lease outlasts the 1x tier's turnover and not the 4x tier's:
  without a hint, the first evicted its whole capacity about every 14 s, and the second about every
  262 s. That would explain why restores rose at 1x and hardly at 4x, where the lease protects what LRU would
  have kept anyway.
- A lease's budget can be taken by a few long sessions: under the 120 s lease over 50% of the tier, 65% of ended leases covered
  nothing, against 46% under the 30 s lease over 90% of the tier. The two settings differ in TTL and in budget together, so
  these runs cannot say which of the two mattered.

## The hint and the answer

A hint rides a request, as `kv_transfer_params.kv_retention`. The prototype has no request-free control path, so a release
rides a request too.

```json
{
  "version": "0.1",
  "hint_id": "session-a:7",
  "op": "retain",
  "scope": {"tenant": "t", "cache_namespace": "default", "session": "session-a"},
  "generation": 7,
  "tiers": ["gpu", "cpu"],
  "ttl_seconds": 120.0,
  "priority": 0
}
```

Each tier answers for itself. A non-streaming response carries the answer in its `kv_transfer_params`, as in this one from a probe. The connector also
writes every answer, and how each hold or lease ended, to a JSONL file of receipts, which is what the replay reads.

```json
{
  "worker_generation": "8d819a24b6ec461096923f9578322750",
  "hint_id": "gpu-probe:1",
  "acks": [
    {
      "hint_id": "gpu-probe:1",
      "tier": "gpu",
      "state": "accepted",
      "reason": null,
      "worker_generation": "8d819a24b6ec461096923f9578322750",
      "ttl_seconds": 300.0,
      "tokens": 8000
    }
  ],
  "observed": {
    "gpu_hit_tokens": 7984,
    "cpu_hit_tokens": 16
  }
}
```

| RFC #427 | Here |
|---|---|
| Logical session TTL | Not here: the Router's or the client's own |
| Requested retention | The hint's `tiers` and `ttl_seconds` |
| Acknowledged retention | `acks`: `accepted` or `rejected` with a reason, and the tier's own TTL, which can be shorter than the one requested |
| Observed residency | `observed`: tokens the request found in the GPU pool and in the CPU tier, which `usage.prompt_tokens_details.cached_tokens` cannot tell apart. The receipts total the same prompt tokens as vLLM's counters in every run. Their split matches exactly in the one run with no preemption and drifts with preemptions, by up to 15.0% on GPU hits and 22.0% on CPU restores, as a receipt reports a request's last lookup and vLLM's counters its first. The tables take token shares from vLLM's counters |
| Eviction notification | A hold's receipt ends `used`, `expired`, `released` or `evicted`; a lease's receipts count its evicted chunks |
| Idempotency | The same `hint_id` renews a live hold and holds nothing more; a release with nothing to release succeeds |
| Generation | Per session, an older `generation` is refused. A hint may name the `worker_generation` it expects, and is refused if that is not this worker's |
| Tenant and namespace scope | Holds and leases are keyed by tenant, namespace and session, so another tenant's release finds nothing and gets the answer an unknown release gets. A hold's namespace must be the request's `cache_salt` |
| Budgets | A share of the pool or tier, per tenant and per engine. Over a budget a hold is refused and a lease covers only the prefix's head |

## How each tier retains

**GPU tier: a hold.** The connector answers `delay=True` from `request_finished_all_groups`, so vLLM keeps the finished
request's blocks referenced. The worker reports the request as finished sending when the hold ends, and vLLM frees the
blocks at that step. Releases reach the worker in the connector's metadata. A hold ends on the session's next request,
once that has taken its own references, or on a release, its TTL, `reset_cache`, or demand. A conversation's last turn asks
for nothing and still names its session, so it ends the previous turn's hold too. vLLM counts a held request as
work to do, so the engine keeps stepping and a hold expires on time on an idle engine too, in a loop that sleeps 1 ms
between steps, whose CPU cost was not measured.

A hold is hard: held blocks cannot be evicted, and they leave the pool smaller for everyone else. So it yields. A waiting
request that does not fit releases holds oldest first, and a running request that cannot grow does too, through
`has_pending_block_frees`, instead of preempting anything. vLLM's proposed eviction priority
([vLLM #59714](https://github.com/vllm-project/vllm/pull/59714)) would leave the blocks evictable, which this prototype
does not attempt.

**CPU tier: a lease.** A `CachePolicy` over vLLM's LRU, in which chunks without a live lease are evicted first and leased
chunks only when nothing else is evictable, so a store fails only where plain LRU would fail too. `check_lease_policy.py`
checks that against vLLM's LRU on random operations (6,922 evictions granted and 9,297 refused, each as LRU would, with 15,950 leased and 13,405 unleased chunks evicted, and never a higher rank before a lower one). A lease's clock restarts when its turn ends, and a
turn that outlasted the TTL opens the lease again, so the TTL counts the pause and not the turn.

## Setup

| | |
|---|---|
| Engine | vLLM 0.32.0, `Qwen/Qwen3-4B-Instruct-2507-FP8`, one RTX 5060 Ti 16 GB on a PCIe Gen3 x8 link. A 5 GiB pool (36,400 tokens), `max-num-batched-tokens` 512, `long-prefill-token-threshold` 256, priority scheduling, a write-through CPU tier ([`engine.yaml`](../../benches/kv_retention/engine.yaml)) |
| Load | AIPerf 0.13.0 `agentic_replay` of the SemiAnalysis Claude Code traces, block-scaled 32x to fit a 32k context: 16 session trees, think times capped at 120 s, the recorded output lengths, 10 minutes plus a 60 s grace, straight to the engine. A conversation is a session, and its turn index the generation |
| Host | Engine on cores 8-15 and AIPerf on cores 0-7 of the same machine, in one private network namespace |
| Runs | A fresh engine per run. Each of three seeds draws where the sessions enter the traces, so a seed pairs the arms, and the seeds differ from each other by more than any arm moves computed tokens. There are two rounds, on one host: see [Two rounds](#two-rounds). The second started 15 hours after the first's last run, and for each seed ran `off-1x`, `cpu30-1x`, `both-1x`, `off-4x`, `gpu10-4x`, `gpu30-4x` and `cpu30-4x` in that order. The first ran `off-1x-first` and `cpu-1x` round-robin with its other arms, so that each hinted arm started within 40 minutes of its baseline. `stock-4x` ran first, on one seed |
| Measures | Token shares are vLLM's `prompt_tokens_by_source` over the whole run, warm-up included. TTFT and recomputed tokens are AIPerf's profiling phase. A resume is a turn sent at least 1 s after its conversation's previous turn ended. Recomputed tokens are computed tokens beyond what a turn added to its previous prompt, an upper bound where a prompt is not an extension of the previous one |
| Validity | The GPU's compute processes are checked before, every 15 s during and after each run. Another process, or a check that cannot be read, invalidates a run, which is set aside and rerun. All 31 runs in the tables, and every other run of both rounds and the control, passed it at the first attempt. Two earlier smoke runs on seed 11 overlapped another process on the GPU, before this check existed: they are in no table, and informed only the choice of the second hold setting |

The only change between arms is the hint and how much each tier acknowledges. Every hint asks for 120 s, the think-time
cap, on the tiers it names, on every turn but a conversation's last. `off-4x-first` and `off-1x-first` are the same arms as
`off-4x` and `off-1x`, in the first round: the two on one seed are a repeat.

| Arm | Round | CPU tier | GPU hold: TTL, budget | CPU lease: TTL, budget |
|---|---:|---:|---|---|
| `stock-4x` | 1 | 20 GiB, 4x | vLLM's own connector and LRU: a control, on seed 11 | |
| `off-4x-first` | 1 | 20 GiB, 4x | none | none |
| `off-4x` | 2 | 20 GiB, 4x | none | none |
| `gpu10-4x` | 2 | 20 GiB, 4x | 10 s, 25% of the pool | none |
| `gpu30-4x` | 2 | 20 GiB, 4x | 30 s, 50% of the pool | none |
| `cpu30-4x` | 2 | 20 GiB, 4x | none | 30 s, 90% of the tier |
| `off-1x-first` | 1 | 5 GiB, 1x | none | none |
| `off-1x` | 2 | 5 GiB, 1x | none | none |
| `cpu-1x` | 1 | 5 GiB, 1x | none | 120 s, 50% of the tier |
| `cpu30-1x` | 2 | 5 GiB, 1x | none | 30 s, 90% of the tier |
| `both-1x` | 2 | 5 GiB, 1x | 30 s, 50% of the pool | 120 s, 50% of the tier |

The first values were a 10 s hold, Program scheduling's `max_acting_ttl_seconds` default, over a quarter of the pool, and
a lease as long as the think-time cap over half the tier. Each was changed once after its first results, and both first
values stay in the tables. The 10 s hold over 25% of the pool could reach few restores: in `off-4x`, 58% of restored tokens
belong to prompts over a quarter of the pool and 46% follow pauses over 10 s. The 120 s lease over 50% of the tier sat at or
near its budget for the whole of each run, at 98% of it on average.

## Live probes

Read from `vllm:prompt_tokens_by_source` around each request, on token-id prompts, one engine per probe, with the code the
second round ran.

| Probe | Hint off | Hint on |
|---|---|---|
| GPU tier: 8,000 tokens, then 54,600 tokens of other prompts, then the prompt again | 8,000 restored, 274 ms | 7,984 GPU hits and 16 restored, 63 ms |
| CPU tier of 1x: the same, with 109,200 tokens of other prompts | 8,000 computed, 1,068 ms | 8,000 restored, 277 ms |
| Negative control | 0 answers in the receipts | |
| Expiry: a hold of 3 s on an idle engine, then the filler | | `expired` after 3.0 s, and the prompt then comes back from the CPU tier (8,000 restored) |
| Overload: three holds of 10,000 tokens, then a 20,000-token prompt | | Served in 4.6 s, computing 20,000 tokens; the 3 holds `evicted`, oldest first |
| Overload: three new holds, then a running request that grows past the free pool | | 0 preemptions; 1 more hold `evicted` |
| Refusal: a hold over the budget, and a hint of an unknown version | | The hold is `rejected` with `over_tenant_budget`; the unknown version is ignored. Both requests are served |
| A conversation's last turn, after a held turn | | The hold ends `used` |
| A turn longer than its lease: a 30 s lease, a 48 s turn, then 21 s of other prompts | | 8,000 restored. On the code before the fix, the same probe computes all 8,000 |

## Replay results

Each cell is the median of the runs and their range. The no-hint arm matches its controls. On seed 11, `stock-4x`, vLLM's
own connector, had 54.5 / 35.2 / 10.3% of prompt tokens from the GPU, the CPU tier and computation, and `off-4x-first`, this
connector with no hint, had 54.4 / 35.4 / 10.2%, with all-turn TTFT p90 of 761 and 836 ms. [Flatshare v1.22's
benchmarks](https://github.com/StevenWang-CY/flatshare/blob/v1.22/docs/BENCHMARKS.md) publish the same point at
55.2 / 34.6 / 10.2% and 787 ms, in 686 requests against 687 here.

### Where prompt tokens came from

| Arm | Runs | GPU hit % | CPU restore % | Computed % | Recomputed tokens |
|---|---:|---:|---:|---:|---:|
| `stock-4x` | 1 | 54.5 | 35.2 | 10.3 | 88,413 |
| `off-4x-first` | 3 | 54.4 (45.6–59.2) | 35.4 (28.0–41.1) | 12.8 (10.2–13.2) | 252,356 (82,501–390,751) |
| `off-4x` | 3 | 54.2 (45.8–57.2) | 35.5 (30.3–40.6) | 12.5 (10.2–13.6) | 241,532 (84,660–415,428) |
| `gpu10-4x` | 3 | 54.9 (46.8–57.3) | 34.9 (30.1–40.1) | 12.6 (10.2–13.1) | 244,844 (84,978–388,268) |
| `gpu30-4x` | 3 | 55.0 (45.5–56.8) | 34.7 (30.4–41.1) | 12.7 (10.3–13.4) | 246,466 (87,288–401,458) |
| `cpu30-4x` | 3 | 53.8 (45.5–57.9) | 35.1 (30.3–41.8) | 11.8 (11.2–12.7) | 211,846 (119,750–355,499) |
| `off-1x-first` | 3 | 50.1 (22.9–52.1) | 7.3 (6.1–14.9) | 42.6 (41.8–62.2) | 1,130,790 (993,455–2,025,823) |
| `off-1x` | 3 | 52.3 (22.8–53.5) | 6.9 (5.6–14.3) | 40.9 (40.8–62.9) | 1,088,987 (980,720–2,077,278) |
| `cpu-1x` | 3 | 47.6 (18.2–52.7) | 6.1 (5.9–13.2) | 46.3 (41.3–68.6) | 1,239,738 (996,611–2,086,511) |
| `cpu30-1x` | 3 | 52.7 (16.3–53.8) | 10.7 (7.1–19.6) | 40.2 (35.6–64.1) | 976,087 (954,713–1,993,001) |
| `both-1x` | 3 | 47.6 (19.6–50.7) | 6.5 (6.3–8.6) | 45.9 (43.0–71.8) | 1,242,354 (1,046,277–2,021,409) |

### Time to first token (ms)

| Arm | Runs | Resume p50 | Resume p90 | All p50 | All p90 | All p99 | Requests |
|---|---:|---:|---:|---:|---:|---:|---:|
| `stock-4x` | 1 | 137 | 664 | 127 | 761 | 9,252 | 688 |
| `off-4x-first` | 3 | 177 (141–454) | 1,216 (711–3,896) | 139 (132–343) | 1,246 (836–3,908) | 10,397 (9,256–10,850) | 693 (687–911) |
| `off-4x` | 3 | 204 (140–444) | 1,366 (718–3,840) | 153 (132–352) | 1,365 (807–3,880) | 9,701 (8,818–11,284) | 692 (688–916) |
| `gpu10-4x` | 3 | 189 (142–418) | 1,277 (723–4,150) | 149 (131–337) | 1,279 (864–3,973) | 9,655 (7,793–10,525) | 693 (688–918) |
| `gpu30-4x` | 3 | 190 (143–472) | 1,620 (692–5,171) | 142 (130–389) | 1,576 (794–4,818) | 9,363 (9,344–12,052) | 692 (685–897) |
| `cpu30-4x` | 3 | 189 (145–484) | 1,389 (801–3,938) | 151 (135–423) | 1,377 (1,195–4,242) | 9,131 (8,918–9,587) | 693 (688–911) |
| `off-1x-first` | 3 | 321 (196–5,892) | 6,817 (4,957–16,199) | 224 (172–5,218) | 7,424 (6,051–15,336) | 20,720 (19,921–26,327) | 665 (652–669) |
| `off-1x` | 3 | 267 (181–5,710) | 6,471 (4,738–14,579) | 213 (181–5,294) | 6,704 (5,581–14,468) | 19,688 (19,500–26,061) | 665 (657–675) |
| `cpu-1x` | 3 | 516 (192–5,699) | 7,156 (5,218–16,211) | 274 (183–5,256) | 7,833 (6,148–16,081) | 20,528 (18,896–26,182) | 640 (635–665) |
| `cpu30-1x` | 3 | 252 (194–5,984) | 6,057 (5,100–17,415) | 195 (191–5,638) | 6,568 (5,486–16,235) | 18,657 (18,292–25,237) | 659 (646–666) |
| `both-1x` | 3 | 498 (222–7,141) | 7,357 (4,973–22,435) | 258 (217–6,790) | 8,281 (6,710–20,868) | 20,130 (18,882–32,150) | 650 (601–662) |

### Seed by seed, against the baseline

| Arm | Baseline | GPU hits, points | CPU restores, points | Computed, points | Resume p50 | Resume p90 | All p90 | Output tokens/s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `off-4x-first` | `stock-4x` | -0.1 | +0.3 | -0.2 | +2% | +7% | +10% | -0% |
| `off-4x` | `off-4x-first` | -0.2 / +0.2 / -2.0 | +0.1 / -0.6 / +2.3 | +0.1 / +0.4 / -0.3 | -0% / -2% / +15% | +1% / -1% / +12% | -3% / -1% / +10% | +0% / +1% / +1% |
| `gpu10-4x` | `off-4x` | +0.7 / +0.9 / +0.1 | -0.7 / -0.4 / -0.1 | +0.0 / -0.5 / +0.0 | +1% / -6% / -8% | +1% / +8% / -7% | +7% / +2% / -6% | +0% / +0% / -0% |
| `gpu30-4x` | `off-4x` | +0.7 / -0.3 / -0.4 | -0.8 / +0.5 / +0.2 | +0.1 / -0.2 / +0.2 | +2% / +7% / -7% | -4% / +35% / +19% | -2% / +24% / +15% | -0% / -2% / -1% |
| `cpu30-4x` | `off-4x` | -0.5 / -0.3 / +0.7 | -0.5 / +1.3 / +0.0 | +0.9 / -0.9 / -0.7 | +3% / +9% / -7% | +12% / +3% / +2% | +48% / +9% / +1% | +0% / +1% / -0% |
| `off-1x` | `off-1x-first` | +1.4 / -0.1 / +2.2 | -0.5 / -0.5 / -0.4 | -0.9 / +0.7 / -1.8 | -7% / -3% / -17% | -4% / -10% / -5% | -8% / -6% / -10% | -0% / +0% / +3% |
| `cpu-1x` | `off-1x-first` | +0.6 / -4.7 / -2.5 | -0.2 / -1.7 / -1.2 | -0.4 / +6.4 / +3.6 | -2% / -3% / +61% | +5% / +0% / +5% | +2% / +5% / +6% | -0% / -4% / -2% |
| `cpu30-1x` | `off-1x` | -0.8 / -6.5 / +1.5 | +1.5 / +5.3 / +3.8 | -0.7 / +1.3 / -5.3 | +7% / +5% / -6% | +8% / +19% / -6% | -2% / +12% / -2% | +1% / -3% / +1% |
| `both-1x` | `off-1x` | -2.8 / -3.2 / -4.7 | +0.7 / -5.7 / -0.3 | +2.1 / +8.9 / +5.0 | +22% / +25% / +87% | +5% / +54% / +14% | +20% / +44% / +24% | +1% / -11% / -4% |

### GPU holds

| Arm | Runs | Accepted | Refused | Used | Expired | Evicted | Used, % of tokens held | Held when used, p50 (s) | Pool held, mean % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `gpu10-4x` | 3 | 380 (356–586) | 313 (300–319) | 250 (249–279) | 32 (19–40) | 69 (66–318) | 66 (38–70) | 1.0 (0.5–1.2) | 12.5 (8.6–13.1) |
| `gpu30-4x` | 3 | 493 (454–686) | 194 (186–218) | 308 (283–352) | 10 (5–30) | 134 (116–392) | 58 (38–67) | 1.2 (0.3–1.2) | 26.7 (20.2–36.5) |
| `both-1x` | 3 | 516 (412–522) | 121 (70–238) | 188 (45–306) | 3 (1–20) | 209 (203–474) | 41 (8–49) | 1.1 (1.0–1.6) | 18.7 (10.6–31.7) |

### CPU leases

| Arm | Runs | Accepted | Leased chunks, mean | Chunks over budget | Leased chunks evicted | Ended leases covering nothing, % |
|---|---:|---:|---:|---:|---:|---:|
| `cpu30-4x` | 3 | 680 (675–893) | 4,350 (3,936–5,042) | 4,165 (902–6,822) | 0 (0–0) | 1 (0–3) |
| `cpu-1x` | 3 | 628 (626–652) | 1,119 (1,090–1,119) | 242,578 (218,106–246,830) | 46 (0–463) | 65 (59–76) |
| `cpu30-1x` | 3 | 646 (637–653) | 1,924 (1,855–1,943) | 113,946 (92,883–147,330) | 2,882 (2,116–8,094) | 46 (26–54) |
| `both-1x` | 3 | 637 (592–650) | 1,119 (1,106–1,120) | 247,059 (225,122–249,892) | 440 (0–513) | 65 (63–75) |

### Eviction pressure and cost

| Arm | Runs | GPU blocks evicted/s | Idle before evict, p50 bucket (s) | CPU chunks evicted/s | CPU store failures | Restored GiB | Stored GiB | KV in use, mean % | Waiting, mean | Queue, mean (s) | Preemptions | Output tokens/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `stock-4x` | 1 | 162 | 10.0 | — | 0 | 165 | 50 | 16 | 0.4 | 0.50 | 1 | 41 |
| `off-4x-first` | 3 | 172 (163–323) | 10.0 (5.0–10.0) | 35 (22–63) | 0 (0–0) | 167 (155–321) | 68 (50–103) | 17 (16–26) | 0.5 (0.5–1.1) | 0.69 (0.50–1.15) | 1 (0–8) | 48 (41–55) |
| `off-4x` | 3 | 181 (164–324) | 10.0 (5.0–10.0) | 35 (23–64) | 0 (0–0) | 167 (167–318) | 67 (50–104) | 25 (15–39) | 0.7 (0.5–1.7) | 0.69 (0.52–1.23) | 1 (1–8) | 48 (41–55) |
| `gpu10-4x` | 3 | 181 (162–319) | 10.0 (5.0–10.0) | 35 (23–64) | 0 (0–0) | 167 (164–315) | 66 (50–104) | 36 (27–46) | 0.7 (0.4–1.6) | 0.68 (0.48–1.22) | 3 (1–9) | 48 (41–55) |
| `gpu30-4x` | 3 | 180 (160–323) | 10.0 (5.0–10.0) | 37 (23–63) | 0 (0–0) | 169 (162–317) | 70 (50–103) | 49 (47–58) | 0.9 (0.5–1.9) | 0.83 (0.49–1.42) | 3 (1–10) | 48 (41–54) |
| `cpu30-4x` | 3 | 179 (165–324) | 10.0 (5.0–10.0) | 34 (24–61) | 0 (0–0) | 169 (164–322) | 64 (51–100) | 25 (17–37) | 0.6 (0.6–1.5) | 0.62 (0.57–1.19) | 1 (1–2) | 48 (41–55) |
| `off-1x-first` | 3 | 189 (166–328) | 10.0 (5.0–10.0) | 164 (141–257) | 0 (0–0) | 42 (34–92) | 224 (192–349) | 41 (27–62) | 1.6 (1.3–5.0) | 1.62 (1.37–5.01) | 17 (12–32) | 39 (38–44) |
| `off-1x` | 3 | 182 (161–330) | 10.0 (5.0–10.0) | 158 (139–260) | 0 (0–0) | 39 (30–92) | 216 (188–356) | 41 (27–61) | 1.5 (1.4–5.0) | 1.48 (1.39–5.00) | 17 (11–40) | 39 (38–45) |
| `cpu-1x` | 3 | 196 (162–319) | 10.0 (5.0–10.0) | 216 (161–258) | 0 (0–0) | 34 (29–71) | 292 (218–351) | 44 (27–62) | 1.7 (1.4–5.1) | 1.80 (1.45–5.44) | 17 (13–36) | 39 (36–43) |
| `cpu30-1x` | 3 | 177 (164–333) | 10.0 (5.0–10.0) | 162 (146–214) | 0 (0–0) | 54 (32–103) | 221 (197–294) | 39 (27–64) | 1.5 (1.3–5.4) | 1.57 (1.33–5.57) | 9 (7–38) | 40 (37–45) |
| `both-1x` | 3 | 196 (168–296) | 10.0 (10.0–10.0) | 204 (168–257) | 0 (0–0) | 35 (28–45) | 276 (225–349) | 59 (54–67) | 1.6 (1.4–6.2) | 1.68 (1.47–7.03) | 14 (6–25) | 39 (34–43) |

### Restored tokens by the pause before the turn (thousands)

| Arm | Runs | Up to 1 s | 1-5 s | 5-10 s | 10-30 s | 30-120 s | Over 120 s |
|---|---:|---:|---:|---:|---:|---:|---:|
| `off-4x-first` | 3 | 42 (40–767) | 399 (298–905) | 240 (163–245) | 197 (158–231) | 217 (132–310) | 112 (46–136) |
| `off-4x` | 3 | 51 (46–713) | 417 (299–913) | 243 (168–297) | 191 (157–199) | 245 (156–309) | 112 (5–146) |
| `gpu10-4x` | 3 | 41 (38–743) | 414 (306–839) | 228 (175–274) | 189 (158–205) | 235 (176–309) | 112 (5–146) |
| `gpu30-4x` | 3 | 43 (24–782) | 507 (278–903) | 250 (171–264) | 177 (119–211) | 268 (153–287) | 107 (10–132) |
| `cpu30-4x` | 3 | 48 (47–716) | 431 (304–949) | 187 (174–263) | 201 (144–214) | 241 (200–357) | 82 (0–165) |
| `off-1x-first` | 3 | 45 (19–330) | 227 (160–299) | 19 (15–24) | 3 (0–15) | 0 (0–15) | 4 (2–18) |
| `off-1x` | 3 | 51 (22–272) | 177 (132–330) | 33 (14–48) | 8 (2–9) | 4 (3–18) | 3 (2–10) |
| `cpu-1x` | 3 | 35 (19–196) | 150 (56–254) | 12 (11–51) | 23 (17–26) | 11 (9–20) | 16 (6–22) |
| `cpu30-1x` | 3 | 38 (25–343) | 281 (115–360) | 14 (10–57) | 31 (17–61) | 1 (0–9) | 0 (0–0) |
| `both-1x` | 3 | 27 (14–129) | 142 (37–168) | 4 (4–21) | 21 (20–40) | 16 (6–93) | 13 (6–15) |

### Restored tokens by the prompt's share of the pool (thousands)

| Arm | Runs | Up to 25% | 25-50% | Over 50% |
|---|---:|---:|---:|---:|
| `off-4x-first` | 3 | 501 (318–1,173) | 462 (255–654) | 333 (48–893) |
| `off-4x` | 3 | 510 (358–1,217) | 477 (226–650) | 349 (48–832) |
| `gpu10-4x` | 3 | 465 (304–1,151) | 480 (249–696) | 381 (24–842) |
| `gpu30-4x` | 3 | 472 (303–1,138) | 521 (282–661) | 397 (42–869) |
| `cpu30-4x` | 3 | 527 (355–1,249) | 477 (231–623) | 371 (41–847) |
| `off-1x-first` | 3 | 163 (152–580) | 67 (11–129) | 22 (0–72) |
| `off-1x` | 3 | 176 (131–567) | 37 (28–100) | 36 (0–61) |
| `cpu-1x` | 3 | 162 (127–467) | 69 (12–75) | 6 (0–16) |
| `cpu30-1x` | 3 | 156 (127–673) | 107 (20–208) | 30 (0–43) |
| `both-1x` | 3 | 116 (87–293) | 120 (8–134) | 5 (0–13) |


## The 30 s lease, run by run

In the first round, with a CPU tier the size of the pool, the 30 s lease over 90% of the tier lowered computed tokens on all three seeds and
looked like a gain. It ran as a block, hours after its baselines. Each row here is that arm against a no-hint run, seed by seed, on seeds 11, 12
and 13 in that order. The control is the first round's code, run again after the second round.

| Run | Code | No-hint run it is against | Computed, points | CPU restores, points | GPU hits, points | Resume p90 |
|---|---|---|---:|---:|---:|---:|
| First round | before the fixes | first round, up to 3.5 h earlier | −6.7 / −5.0 / −6.2 | +3.7 / +6.5 / +4.1 | +3.0 / −1.5 / +2.1 | −32% / −16% / −11% |
| Second round | fixed | second round, 13 min earlier | −0.7 / +1.3 / −5.3 | +1.5 / +5.3 / +3.8 | −0.8 / −6.5 / +1.5 | +8% / +19% / −6% |
| Control, seeds 11 and 12 | before the fixes | second round, up to 4.7 h earlier | −1.5 / −1.1 | +2.4 / +4.7 | −0.9 / −3.6 | −2% / +2% |

- **The gain did not reproduce.** Started right after its baseline, on the fixed code, the arm moved computed tokens
  −0.7, +1.3 and −5.3 points, against −6.7, −5.0 and −6.2 in the first round. Seed 13 kept it, and seeds 11 and 12 did not.
  Resume TTFT p90 went from −32%, −16% and −11% to +8%, +19% and −6%.
- **The fixes did not lose it.** The control moved computed tokens −1.5 and −1.1 points. Its computed tokens came out
  4.3 and 4.5 points above the first round's, on the same code and the same seed, and 0.9 and 2.4 points below the
  fixed code's, so the fixes moved the arm by less than the same code moved between rounds. A no-hint repeat differs by at
  most 1.8, so on these two pairs this arm's runs spread further than the no-hint arm's. These runs do
  not say why.
- **The restores did reproduce.** CPU restores rose in all 8 runs, by +1.5 to +6.5 points, where a no-hint
  repeat moved them by at most 0.5. In the second round, restored tokens after pauses of 10-30 s were
  61, 17 and 31 thousand, against 9, 2 and 8 thousand without a hint: the range a 30 s lease should reach and LRU on this
  tier mostly does not. The shares sum to 100, so restores that do not replace computation replace GPU hits: those moved
  −0.8, −6.5 and +1.5 points in that round.
- **Computed tokens were below their baseline in 7 of the 8 runs**, and moved
  −6.7 to +1.3 points across them. A gain smaller than the spread is not excluded. A gain of the first round's size on every seed is not what the
  repeats show.

## Two rounds

Review of the first round found two defects in the prototype, and both counted against retention:

- A conversation's last turn carried no hint, so the connector could not tie it to its session, and the previous turn's hold
  stayed until its TTL or demand.
- A turn longer than a lease's TTL lost its lease before it ended, so the pause after it had no coverage.

Both are fixed in the code here, and the live probes above cover both. The second round ran on the fixed code, with repeats
of the no-hint arms. `stock-4x`, `off-4x-first`, `off-1x-first` and `cpu-1x` stay from the first round, as neither fix
touches them: the first three carry no hint, `cpu-1x` holds nothing, and none of its turns outlasted its 120 s lease. The
first round's other runs are in no table here, except for the 30 s lease's row above.

## What this does not show

- One GPU, one small model and one trace family, on a 5 GiB pool. Another card moves every ratio here.
- A hard hold, not an eviction priority, which would cost the pool less and protect less.
- A replay's next prompt carries the trace's text, not what the model generated, so output blocks never hit. A hold's
  output blocks and the write-through tier's are wasted here, so both the hold's cost and the tier's pressure are overstated
  against a real agent.
- Every session asks for retention on every turn but its last, with a fixed TTL: no mix of hinted and unhinted traffic, no
  prediction of who returns when. The lease budget has no per-session cap. A later hint that changes a session's priority
  re-ranks only the lease's new chunks.
- A CPU tier the size of the pool is a stress point, not a recommended deployment.
- `both-1x` adds the hold to the first lease, and ran in the other round from the lease alone, so these runs do not separate
  what the hold added. The hold with the 30 s lease over 90% of the tier was not run.
- Spread is thinly measured: one repeat per seed of each no-hint arm, and two of the 30 s lease. A gain smaller than that
  spread would not show here.
- No Router in the path: hints went straight to the engine, and one tenant sent them all, so the tenant budget was the
  engine budget. The prototype trusts the hint's tenant and namespace as sent. Whoever can reach the engine can send a hint,
  so a Router in front would have to set them.
- Worker restart, several workers and TP above 1. Releases travel in the connector's metadata so that they would reach
  another process, and a stale `worker_generation` is refused in the unit tests, but one worker ran, with no restart.

## Reproduce

The engine needs vLLM 0.32.0 and the model in its own environment. The load generator needs AIPerf 0.13.0, the corpus and
its `weka_trace_scaled` loader from [Flatshare v1.22](https://github.com/StevenWang-CY/flatshare/tree/v1.22/python/flatshare_bench),
and the endpoint plugin in this directory, which adds the hint.

```sh
git clone --branch v1.22 https://github.com/StevenWang-CY/flatshare
uv venv --python 3.12 bench
VIRTUAL_ENV=$PWD/bench uv pip install flatshare/python/flatshare_stats "flatshare/python/flatshare_bench[aiperf]" \
  -e benches/kv_retention/aiperf
# The corpus: semianalysisai/cc-traces-weka-062126 on Hugging Face, block-scaled 32x to fit a 32k context.
bench/bin/flatshare-bench scale-traces traces.jsonl weka-x32 --factor 32

# On a shared host, run these in a network namespace of their own: the engine binds 127.0.0.1.
RUN="bench/bin/python benches/kv_retention/run_eval.py --vllm /path/to/vllm"
$RUN probe --out runs/probes
REPLAY="replay --out runs/replay --corpus weka-x32 --aiperf bench/bin/aiperf"
$RUN $REPLAY --arms stock-4x --seeds 11
$RUN $REPLAY --arms off-4x-first off-1x-first cpu-1x --seeds 11 12 13
$RUN $REPLAY --seeds 11 12 13  # off-1x cpu30-1x both-1x off-4x gpu10-4x gpu30-4x cpu30-4x, in that order per seed
bench/bin/python benches/kv_retention/report.py runs/replay
```

`run_eval.py` starts one engine per probe and per run, with `benches/kv_retention` on its `PYTHONPATH`, and pins the
engine to cores 8-15 and AIPerf to cores 0-7 (`--engine-cores`, `--load-cores`). `report.py` prints the replay tables. The
first round's other runs and the control are the same `replay` on the prototype's modules at `30f6d75`, in this PR's
history. With vLLM's Python, `check_lease_policy.py` runs the policy check, on no GPU. The lifecycle's and the report's
tests need no vLLM:

```sh
pytest tests/python/unit/test_kv_retention.py
```
