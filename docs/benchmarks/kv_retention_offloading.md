# KV retention on OffloadingConnector, hint on and off

A prototype of [RFC #427](https://github.com/vllm-project/router/issues/427)'s retain and release hints on vLLM's
native CPU offload, and the comparison the RFC asks for: which tier retention can protect, and what it does to GPU hits,
CPU restores, recomputation and eviction pressure. It changes nothing in the Router and does not define the Router-side
contract, which the RFC tracks separately. Its field names follow that proposal so that they can be renamed when it
lands.

## Summary

On one GPU, one model and one trace family, with three seeds per arm, retention protected a single prefix on either tier.
In the trace replay it did not help on the GPU tier. On the CPU tier one setting of two went the right way, only where the
tier was small, by an amount three seeds do not settle. A larger CPU tier moved far more than any hint.

**What the runs showed:**

- **One prefix, in a probe.** An 8,000-token prompt pushed out of the pool by other prompts came back in
  63 ms with a GPU hold, against 273 ms restored from the CPU tier without one. With a CPU tier the
  size of the pool, it came back in 277 ms with a lease, against 1,075 ms recomputed without one.
- **GPU holds in the replay: no gain.** Seed by seed, GPU hits moved +1.1, 0.0 and −4.3 points with a 10 s hold over 25% of the pool, and
  −1.2, −0.7 and −5.4 points with a 30 s hold over 50% of the pool. All-turn TTFT p90 moved −7%, +20% and +18% with the first and +6%, +9% and +64%
  with the second, where two no-hint runs of one seed differ by 10%. The holds kept 12.6% and 28.6% of the pool on average.
- **CPU leases at a tier the size of the pool.** A 120 s lease over 50% of the tier did not help: computed tokens moved
  −0.4, +6.4 and +3.6 points. A 30 s lease over 90% of the tier lowered them in all three seeds (−6.7, −5.0 and −6.2 points, from a
  median of 42.6%), raised CPU restores (+3.7, +6.5 and +4.1 points) and lowered resume TTFT p90
  (−32%, −16% and −11%). Without pairing the seeds that is inside their spread: computed tokens 36.5% (35.1–57.2) against
  42.6% (41.8–62.2). A 30 s hold over 50% of the pool added to the first lease (`both-1x`) did not help either: −1.1, +9.1 and −0.3 points.
- **The same lease at a tier four times the pool:** no clear change. Computed tokens moved +0.4, −0.4 and −1.5 points, CPU restores −0.4, +2.4 and +2.4 points and resume TTFT p90 −4%, +1% and +2%. No leased chunk was evicted in any run.
- **Capacity.** With no hint, a 4x CPU tier in place of a 1x tier took computed tokens from 42.6% (41.8–62.2) to
  12.8% (10.2–13.2), and resume TTFT p90 from 6,817 (4,957–16,199) to 1,216 (711–3,896) ms. Neither pair of ranges overlaps.
- **Stores.** vLLM counted no failed store in any run, leases included.

**What that suggests, which these runs do not measure directly:**

- A GPU hold costs other sessions what it gives one. Used holds had been held for about a second at the median
  (0.6 to 1.3 s), when LRU would probably still have had those blocks, and the holds that could have mattered were
  refused for lack of budget or gave way to demand. At this load the pool looks close to zero-sum.
- A lease helps where the tier turns over faster than sessions return. Without a hint, the 1x tier evicted its whole
  capacity about every 14 s, and the 4x tier about every 258 s. A 30 s lease outlasts the first and not the second, which would explain why it moved tokens at 1x and not at 4x, where it protects what LRU would have kept anyway.
- A lease's budget can be taken by a few long sessions: under the 120 s lease over 50% of the tier, 65% of ended leases covered
  nothing, against 48% under the 30 s lease over 90% of the tier. The two settings differ in TTL and in budget together, so
  these runs cannot say which of the two mattered.
- Whether the 30 s lease over 90% of the tier gains anything beyond run-to-run spread is open: see
  [Reading the lease's gain](#reading-the-leases-gain).

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
  "worker_generation": "39e6030bcd964a4586cff1229cb13687",
  "hint_id": "gpu-probe:1",
  "acks": [
    {
      "hint_id": "gpu-probe:1",
      "tier": "gpu",
      "state": "accepted",
      "reason": null,
      "worker_generation": "39e6030bcd964a4586cff1229cb13687",
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
once that has taken its own references, or on a release, its TTL, `reset_cache`, or demand. vLLM counts a held request as
work to do, so the engine keeps stepping and a hold expires on time on an idle engine too, in a loop that sleeps 1 ms
between steps, whose CPU cost was not measured.

A hold is hard: held blocks cannot be evicted, and they leave the pool smaller for everyone else. So it yields. A waiting
request that does not fit releases holds oldest first, and a running request that cannot grow does too, through
`has_pending_block_frees`, instead of preempting anything. vLLM's proposed eviction priority
([vLLM #59714](https://github.com/vllm-project/vllm/pull/59714)) would leave the blocks evictable, which this prototype
does not attempt.

**CPU tier: a lease.** A `CachePolicy` over vLLM's LRU, in which chunks without a live lease are evicted first and leased
chunks only when nothing else is evictable, so a store fails only where plain LRU would fail too. `check_lease_policy.py`
checks that against vLLM's LRU on random operations (6,922 evictions granted and 9,297 refused, each as LRU would, with 15,950 leased and 13,405 unleased chunks evicted, and never a higher rank before a lower one).

## Setup

| | |
|---|---|
| Engine | vLLM 0.32.0, `Qwen/Qwen3-4B-Instruct-2507-FP8`, one RTX 5060 Ti 16 GB on a PCIe Gen3 x8 link. A 5 GiB pool (36,400 tokens), `max-num-batched-tokens` 512, `long-prefill-token-threshold` 256, priority scheduling, a write-through CPU tier ([`engine.yaml`](../../benches/kv_retention/engine.yaml)) |
| Load | AIPerf 0.13.0 `agentic_replay` of the SemiAnalysis Claude Code traces, block-scaled 32x to fit a 32k context: 16 session trees, think times capped at 120 s, the recorded output lengths, 10 minutes plus a 60 s grace, straight to the engine. A conversation is a session, and its turn index the generation |
| Host | Engine on cores 8-15 and AIPerf on cores 0-7 of the same machine, in one private network namespace |
| Runs | A fresh engine per run. Each of three seeds draws where the sessions enter the traces, so a seed pairs the arms, and the seeds differ from each other by more than most arms do. For each seed, `off-4x`, `gpu10-4x`, `gpu30-4x`, `off-1x`, `cpu-1x` and `both-1x` ran round-robin. `cpu30-1x` and then `cpu30-4x`, added after the first results, ran afterwards as blocks, not interleaved. `stock-4x` ran first, on one seed |
| Measures | Token shares are vLLM's `prompt_tokens_by_source` over the whole run, warm-up included. TTFT and recomputed tokens are AIPerf's profiling phase. A resume is a turn sent at least 1 s after its conversation's previous turn ended. Recomputed tokens are computed tokens beyond what a turn added to its previous prompt, an upper bound where a prompt is not an extension of the previous one |
| Validity | The GPU's compute processes are checked before, every 15 s during and after each run. Another process, or a check that cannot be read, invalidates a run, which is set aside and rerun. All 25 runs in the tables passed it at the first attempt. Two earlier smoke runs on seed 11 overlapped another process on the GPU, before this check existed: they are in no table, and informed only the choice of the second hold setting |

The only change between arms is the hint and how much each tier acknowledges. Every hint asks for 120 s, the think-time
cap, on the tiers it names, on every turn but a conversation's last.

| Arm | CPU tier | GPU hold: TTL, budget | CPU lease: TTL, budget |
|---|---:|---|---|
| `stock-4x` | 20 GiB, 4x | vLLM's own connector and LRU: a control, on seed 11 | |
| `off-4x` | 20 GiB, 4x | none | none |
| `gpu10-4x` | 20 GiB, 4x | 10 s, 25% of the pool | none |
| `gpu30-4x` | 20 GiB, 4x | 30 s, 50% of the pool | none |
| `cpu30-4x` | 20 GiB, 4x | none | 30 s, 90% of the tier |
| `off-1x` | 5 GiB, 1x | none | none |
| `cpu-1x` | 5 GiB, 1x | none | 120 s, 50% of the tier |
| `cpu30-1x` | 5 GiB, 1x | none | 30 s, 90% of the tier |
| `both-1x` | 5 GiB, 1x | 30 s, 50% of the pool | 120 s, 50% of the tier |

The first values were a 10 s hold, Program scheduling's `max_acting_ttl_seconds` default, over a quarter of the pool, and
a lease as long as the think-time cap over half the tier. Each was changed once after its first results, and both first
values stay in the tables. The 10 s hold over 25% of the pool could reach few restores: in `off-4x`, 58% of restored tokens
belong to prompts over a quarter of the pool and 46% follow pauses over 10 s. The 120 s lease over 50% of the tier sat at or
near its budget for the whole of each run, at 98% of it on average.

## Live probes

Read from `vllm:prompt_tokens_by_source` around each request, on token-id prompts, one engine per probe, with the code the
replay ran.

| Probe | Hint off | Hint on |
|---|---|---|
| GPU tier: 8,000 tokens, then 54,600 tokens of other prompts, then the prompt again | 8,000 restored, 273 ms | 7,984 GPU hits and 16 restored, 63 ms |
| CPU tier of 1x: the same, with 109,200 tokens of other prompts | 8,000 computed, 1,075 ms | 8,000 restored, 277 ms |
| Negative control | 0 answers in the receipts | |
| Expiry: a hold of 3 s on an idle engine, then the filler | | `expired` after 3.0 s, and the prompt then comes back from the CPU tier (8,000 restored) |
| Overload: three holds of 10,000 tokens, then a 20,000-token prompt | | Served in 4.6 s, computing 20,000 tokens; the 3 holds `evicted`, oldest first |
| Overload: three new holds, then a running request that grows past the free pool | | 0 preemptions; 1 more hold `evicted` |
| Refusal: a hold over the budget, and a hint of an unknown version | | The hold is `rejected` with `over_tenant_budget`; the unknown version is ignored. Both requests are served |

## Replay results

Each cell is the median of the runs and their range. `off-4x` matches its controls. On seed 11, `stock-4x`, vLLM's own
connector, had 54.5 / 35.2 / 10.3% of prompt tokens from the GPU, the CPU tier and computation, and `off-4x`, this connector
with no hint, had 54.4 / 35.4 / 10.2%, with all-turn TTFT p90 of 761 and 836 ms. [Flatshare v1.22's
benchmarks](https://github.com/StevenWang-CY/flatshare/blob/v1.22/docs/BENCHMARKS.md) publish the same point at
55.2 / 34.6 / 10.2% and 787 ms, in 686 requests against 687 here.

### Where prompt tokens came from

| Arm | Runs | GPU hit % | CPU restore % | Computed % | Recomputed tokens |
|---|---:|---:|---:|---:|---:|
| `stock-4x` | 1 | 54.5 | 35.2 | 10.3 | 88,413 |
| `off-4x` | 3 | 54.4 (45.6–59.2) | 35.4 (28.0–41.1) | 12.8 (10.2–13.2) | 252,356 (82,501–390,751) |
| `gpu10-4x` | 3 | 54.9 (45.6–55.5) | 34.2 (32.5–41.0) | 12.6 (10.3–13.4) | 244,652 (87,861–403,598) |
| `gpu30-4x` | 3 | 53.1 (44.9–53.7) | 36.6 (34.7–42.1) | 11.5 (10.2–13.0) | 193,275 (84,400–381,831) |
| `cpu30-4x` | 3 | 54.3 (43.6–58.2) | 35.1 (30.4–43.6) | 11.4 (10.6–12.9) | 194,818 (100,296–359,356) |
| `off-1x` | 3 | 50.1 (22.9–52.1) | 7.3 (6.1–14.9) | 42.6 (41.8–62.2) | 1,130,790 (993,455–2,025,823) |
| `cpu-1x` | 3 | 47.6 (18.2–52.7) | 6.1 (5.9–13.2) | 46.3 (41.3–68.6) | 1,239,738 (996,611–2,086,511) |
| `cpu30-1x` | 3 | 52.2 (21.4–55.2) | 11.3 (9.8–21.4) | 36.5 (35.1–57.2) | 977,175 (820,692–1,864,070) |
| `both-1x` | 3 | 50.8 (16.2–53.6) | 6.8 (5.7–12.5) | 42.4 (40.7–71.3) | 1,098,515 (979,095–2,137,311) |

### Time to first token (ms)

| Arm | Runs | Resume p50 | Resume p90 | All p50 | All p90 | All p99 | Requests |
|---|---:|---:|---:|---:|---:|---:|---:|
| `stock-4x` | 1 | 137 | 664 | 127 | 761 | 9,252 | 688 |
| `off-4x` | 3 | 177 (141–454) | 1,216 (711–3,896) | 139 (132–343) | 1,246 (836–3,908) | 10,397 (9,256–10,850) | 693 (687–911) |
| `gpu10-4x` | 3 | 192 (144–429) | 1,518 (707–4,593) | 145 (131–367) | 1,466 (775–4,679) | 9,612 (9,267–10,707) | 692 (686–913) |
| `gpu30-4x` | 3 | 208 (142–491) | 1,931 (731–3,942) | 148 (131–440) | 2,049 (889–4,278) | 10,177 (9,321–15,405) | 688 (686–911) |
| `cpu30-4x` | 3 | 176 (142–486) | 1,235 (685–3,937) | 144 (134–410) | 1,301 (999–4,120) | 9,239 (8,928–9,988) | 694 (685–908) |
| `off-1x` | 3 | 321 (196–5,892) | 6,817 (4,957–16,199) | 224 (172–5,218) | 7,424 (6,051–15,336) | 20,720 (19,921–26,327) | 665 (652–669) |
| `cpu-1x` | 3 | 516 (192–5,699) | 7,156 (5,218–16,211) | 274 (183–5,256) | 7,833 (6,148–16,081) | 20,528 (18,896–26,182) | 640 (635–665) |
| `cpu30-1x` | 3 | 267 (181–5,146) | 6,101 (3,390–13,656) | 204 (167–4,704) | 6,101 (4,271–13,188) | 17,642 (15,639–24,774) | 670 (661–687) |
| `both-1x` | 3 | 301 (172–7,056) | 6,850 (5,917–15,629) | 229 (157–6,572) | 7,375 (7,137–15,270) | 28,015 (20,503–37,212) | 643 (622–665) |

### Seed by seed, against the baseline

| Arm | Baseline | GPU hits, points | CPU restores, points | Computed, points | Resume p50 | Resume p90 | All p90 | Output tokens/s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `off-4x` | `stock-4x` | -0.1 | +0.3 | -0.2 | +2% | +7% | +10% | -0% |
| `gpu10-4x` | `off-4x` | +1.1 / -0.0 / -4.3 | -1.3 / -0.1 / +4.5 | +0.2 / +0.2 / -0.2 | +2% / -6% / +9% | -1% / +18% / +25% | -7% / +20% / +18% | +0% / +0% / +1% |
| `gpu30-4x` | `off-4x` | -1.2 / -0.7 / -5.4 | +1.2 / +0.9 / +6.8 | +0.1 / -0.2 / -1.3 | +1% / +8% / +18% | +3% / +1% / +59% | +6% / +9% / +64% | +0% / +1% / -0% |
| `cpu30-4x` | `off-4x` | -0.1 / -2.1 / -1.0 | -0.4 / +2.4 / +2.4 | +0.4 / -0.4 / -1.5 | +1% / +7% / -0% | -4% / +1% / +2% | +19% / +5% / +4% | +0% / -0% / +0% |
| `cpu-1x` | `off-1x` | +0.6 / -4.7 / -2.5 | -0.2 / -1.7 / -1.2 | -0.4 / +6.4 / +3.6 | -2% / -3% / +61% | +5% / +0% / +5% | +2% / +5% / +6% | -0% / -4% / -2% |
| `cpu30-1x` | `off-1x` | +3.0 / -1.5 / +2.1 | +3.7 / +6.5 / +4.1 | -6.7 / -5.0 / -6.2 | -7% / -13% / -17% | -32% / -16% / -11% | -29% / -14% / -18% | +3% / +1% / +4% |
| `both-1x` | `off-1x` | +1.5 / -6.7 / +0.7 | -0.4 / -2.4 / -0.4 | -1.1 / +9.1 / -0.3 | -12% / +20% / -6% | +19% / -4% / +0% | +18% / -0% / -1% | -0% / -5% / -1% |

### GPU holds

| Arm | Runs | Accepted | Refused | Used | Expired | Evicted | Used, % of tokens held | Held when used, p50 (s) | Pool held, mean % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `gpu10-4x` | 3 | 358 (352–573) | 322 (315–327) | 244 (227–247) | 38 (20–40) | 71 (69–325) | 66 (36–68) | 0.9 (0.6–1.2) | 12.6 (8.9–13.5) |
| `gpu30-4x` | 3 | 459 (445–675) | 218 (214–230) | 293 (246–306) | 12 (6–24) | 147 (128–416) | 57 (34–60) | 1.1 (0.8–1.3) | 28.6 (20.3–36.2) |
| `both-1x` | 3 | 489 (487–545) | 143 (68–165) | 288 (33–313) | 8 (1–20) | 179 (168–511) | 51 (5–56) | 1.2 (0.3–1.3) | 22.6 (7.3–31.4) |

### CPU leases

| Arm | Runs | Accepted | Leased chunks, mean | Chunks over budget | Leased chunks evicted | Ended leases covering nothing, % |
|---|---:|---:|---:|---:|---:|---:|
| `cpu30-4x` | 3 | 681 (672–890) | 4,258 (3,823–5,073) | 4,651 (961–5,733) | 0 (0–0) | 1 (0–1) |
| `cpu-1x` | 3 | 628 (626–652) | 1,119 (1,090–1,119) | 242,578 (218,106–246,830) | 46 (0–463) | 65 (59–76) |
| `cpu30-1x` | 3 | 657 (648–678) | 1,930 (1,825–1,963) | 111,308 (85,872–146,550) | 5,722 (2,580–6,255) | 48 (33–48) |
| `both-1x` | 3 | 632 (613–652) | 1,119 (1,110–1,120) | 237,000 (234,297–255,350) | 121 (0–279) | 69 (65–83) |

### Eviction pressure and cost

| Arm | Runs | GPU blocks evicted/s | Idle before evict, p50 bucket (s) | CPU chunks evicted/s | CPU store failures | Restored GiB | Stored GiB | KV in use, mean % | Waiting, mean | Queue, mean (s) | Preemptions | Output tokens/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `stock-4x` | 1 | 162 | 10.0 | — | 0 | 165 | 50 | 16 | 0.4 | 0.50 | 1 | 41 |
| `off-4x` | 3 | 172 (163–323) | 10.0 (5.0–10.0) | 35 (22–63) | 0 (0–0) | 167 (155–321) | 68 (50–103) | 17 (16–26) | 0.5 (0.5–1.1) | 0.69 (0.50–1.15) | 1 (0–8) | 48 (41–55) |
| `gpu10-4x` | 3 | 190 (159–322) | 10.0 (5.0–10.0) | 36 (23–63) | 0 (0–0) | 179 (160–318) | 67 (50–104) | 37 (25–47) | 0.7 (0.4–1.7) | 0.70 (0.47–1.24) | 2 (1–7) | 48 (41–55) |
| `gpu30-4x` | 3 | 190 (167–329) | 5.0 (5.0–10.0) | 31 (23–63) | 0 (0–0) | 189 (171–325) | 61 (50–103) | 51 (46–55) | 1.0 (0.5–1.7) | 1.00 (0.51–1.33) | 1 (1–7) | 48 (41–55) |
| `cpu30-4x` | 3 | 175 (162–334) | 10.0 (5.0–10.0) | 32 (22–62) | 0 (0–0) | 170 (164–335) | 63 (49–102) | 23 (16–40) | 0.7 (0.5–1.6) | 0.67 (0.52–1.21) | 2 (1–4) | 48 (41–55) |
| `off-1x` | 3 | 189 (166–328) | 10.0 (5.0–10.0) | 164 (141–257) | 0 (0–0) | 42 (34–92) | 224 (192–349) | 41 (27–62) | 1.6 (1.3–5.0) | 1.62 (1.37–5.01) | 17 (12–32) | 39 (38–44) |
| `cpu-1x` | 3 | 196 (162–319) | 10.0 (5.0–10.0) | 216 (161–258) | 0 (0–0) | 34 (29–71) | 292 (218–351) | 44 (27–62) | 1.7 (1.4–5.1) | 1.80 (1.45–5.44) | 17 (13–36) | 39 (36–43) |
| `cpu30-1x` | 3 | 184 (156–331) | 10.0 (5.0–10.0) | 164 (138–220) | 0 (0–0) | 59 (44–119) | 223 (184–302) | 39 (23–62) | 1.4 (0.9–5.1) | 1.41 (0.96–5.02) | 16 (6–24) | 40 (38–45) |
| `both-1x` | 3 | 177 (158–318) | 10.0 (10.0–10.0) | 187 (167–263) | 0 (0–0) | 35 (28–66) | 256 (226–358) | 59 (52–66) | 2.4 (1.4–5.6) | 2.46 (1.46–6.08) | 10 (8–25) | 39 (36–43) |

### Restored tokens by the pause before the turn (thousands)

| Arm | Runs | Up to 1 s | 1-5 s | 5-10 s | 10-30 s | 30-120 s | Over 120 s |
|---|---:|---:|---:|---:|---:|---:|---:|
| `off-4x` | 3 | 42 (40–767) | 399 (298–905) | 240 (163–245) | 197 (158–231) | 217 (132–310) | 112 (46–136) |
| `gpu10-4x` | 3 | 38 (38–821) | 523 (285–861) | 220 (177–250) | 203 (156–210) | 236 (156–307) | 107 (5–144) |
| `gpu30-4x` | 3 | 45 (42–752) | 527 (317–957) | 258 (161–259) | 198 (149–212) | 286 (171–304) | 111 (5–175) |
| `cpu30-4x` | 3 | 52 (47–853) | 389 (297–931) | 167 (163–256) | 196 (156–212) | 280 (168–319) | 146 (31–165) |
| `off-1x` | 3 | 45 (19–330) | 227 (160–299) | 19 (15–24) | 3 (0–15) | 0 (0–15) | 4 (2–18) |
| `cpu-1x` | 3 | 35 (19–196) | 150 (56–254) | 12 (11–51) | 23 (17–26) | 11 (9–20) | 16 (6–22) |
| `cpu30-1x` | 3 | 27 (17–341) | 332 (80–436) | 40 (31–131) | 41 (41–61) | 0 (0–26) | 0 (0–4) |
| `both-1x` | 3 | 7 (6–214) | 186 (38–210) | 9 (3–15) | 23 (18–51) | 9 (5–79) | 18 (12–21) |

### Restored tokens by the prompt's share of the pool (thousands)

| Arm | Runs | Up to 25% | 25-50% | Over 50% |
|---|---:|---:|---:|---:|
| `off-4x` | 3 | 501 (318–1,173) | 462 (255–654) | 333 (48–893) |
| `gpu10-4x` | 3 | 469 (294–1,174) | 609 (256–653) | 370 (37–873) |
| `gpu30-4x` | 3 | 515 (333–1,203) | 582 (257–682) | 427 (48–884) |
| `cpu30-4x` | 3 | 542 (312–1,256) | 500 (249–618) | 392 (28–930) |
| `off-1x` | 3 | 163 (152–580) | 67 (11–129) | 22 (0–72) |
| `cpu-1x` | 3 | 162 (127–467) | 69 (12–75) | 6 (0–16) |
| `cpu30-1x` | 3 | 146 (127–788) | 196 (19–253) | 31 (0–46) |
| `both-1x` | 3 | 85 (59–424) | 137 (26–150) | 19 (0–25) |


## Reading the lease's gain

This section is interpretation. Three seeds cannot establish an effect: the same sign in all three happens one time in
four with no effect at all.

In favour of the 30 s lease over 90% of the tier at the 1x tier:

- CPU restores, what a lease acts on directly, rose by +3.7, +6.5 and +4.1 points, where the other two 1x arms moved
  −2.4 to −0.2.
- Resume TTFT p90 moved −32%, −16% and −11%, where the other two moved −4% to +19%.
- Restored tokens after pauses of 10-30 s, seed by seed, were 61, 41 and 41 thousand against 3, 0 and 15 thousand
  without a hint. That is the pause range a 30 s lease should reach and LRU on this tier mostly does not.
- At the 4x tier, where a 30 s lease should do little, the same lease moved computed tokens +0.4, −0.4 and −1.5 points and
  resume TTFT p90 −4%, +1% and +2%.

Against reading it as settled:

- Computed tokens separate it only in part. On seeds 11 and 13 the other two 1x arms moved them −1.1 to +3.6
  points against its −6.7 and −6.2. On seed 12 they moved +6.4 and +9.1 points against its
  −5.0, at least as far the other way.
- The spread at 1x is not measured. No 1x run was repeated on the same seed. The only such pair is `stock-4x` and `off-4x`,
  at 4x on seed 11, which differ by −0.2 points of computed tokens and +7% in resume TTFT p90.
  The same lease at 4x, with no leased chunk evicted, still moved CPU restores by up to +2.4 points and GPU
  hits by up to −2.1. Unless its reordering of evictions did that, one pair at 4x can differ by that much. The
  1x tier queues longer, at
  1.62 s against 0.69 s on average without a hint, and its pairs may spread further.
- Its runs came after the rest, up to 3.5 hours after their baselines on the same seed.
- It is one setting on one trace, chosen after the first setting failed, and it changed the TTL and the budget together.

So these runs support a lead, not a result to act on. Interleaved repeats of `off-1x` and `cpu30-1x` on the same seeds would
measure the spread they lack.

## Defects in these runs

Review found two defects after these runs. Both are fixed in the code here, which has not yet run on a GPU. Both count
against retention, so neither inflates a gain, though either could hide a small one.

- **A conversation's last turn did not end the previous turn's hold.** It carried no hint, so the connector could not tie
  it to its session. Those holds, 17 to 26 per run of 352 to 675, stayed until their TTL or demand. They
  count as expired or evicted, not used, and kept 1.2% to 13.2% of the run's held token-seconds past that last turn's arrival.
- **A turn longer than a lease's TTL lost its lease before it ended**, so the pause after it had no coverage. That reached
  1, 13 and 4 turns of 682 to 708 in `cpu30-1x` (at most 3.3% of a run's prompt tokens) and
  0, 2 and 0 in `cpu30-4x`. No turn outlasted a 120 s lease.

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
- `both-1x` adds the hold to the first lease. The hold with the 30 s lease over 90% of the tier was not run.
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
$RUN $REPLAY --seeds 11 12 13  # off-4x gpu10-4x gpu30-4x off-1x cpu-1x both-1x, round-robin
$RUN $REPLAY --arms cpu30-1x --seeds 11 12 13
$RUN $REPLAY --arms cpu30-4x --seeds 11 12 13
bench/bin/python benches/kv_retention/report.py runs/replay
```

`run_eval.py` starts one engine per probe and per run, with `benches/kv_retention` on its `PYTHONPATH`, and pins the
engine to cores 8-15 and AIPerf to cores 0-7 (`--engine-cores`, `--load-cores`). `report.py` prints the tables above. With
vLLM's Python, `check_lease_policy.py` runs the policy check, on no GPU. The lifecycle's and the report's tests need no
vLLM:

```sh
pytest tests/python/unit/test_kv_retention.py
```
