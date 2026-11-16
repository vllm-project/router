# KV transfer planning for kv_aware

## What it does

`kv_aware` picks a worker for each request. Sometimes the worker it picks is idle but has no cache, while a busy peer holds the request's prefix. The transfer planner asks one question after the pick: would it be cheaper for the picked worker to fetch the prefix from that peer than to recompute it, or than to wait on the busy peer?

- **`off`** (default): no planner, no extra work.
- **`shadow`**: the planner runs and its answer is counted. Routing and the request are unchanged.
- **`active`**: as shadow, and a planned request also carries `kv_transfer_params` so the destination engine fetches the prefix from the source.

The planner never changes which worker is picked, in any mode.

## One planning attempt

```mermaid
sequenceDiagram
    participant C as Client
    participant R as Router (kv_aware)
    participant P as Planner (plan_transfer)
    participant D as Destination worker
    participant S as Source worker

    C->>R: request
    R->>R: 1. look up the prompt's blocks in the KV index
    R->>R: 2. score workers and pick the destination
    R->>P: 3. picked worker, peers, loads, destination's index hits
    P->>P: 4. rank peers that share the destination's cache identity
    P->>P: 5. run the gates on each peer in rank order
    P-->>R: 6. plan, or the skip reason
    R->>R: 7. count vllm_router_kv_transfer_plans_total{outcome}
    alt shadow mode, or no plan
        R->>D: 8a. request, unchanged
    else active mode and a plan
        R->>R: 8b. look up the source's peer address, take an in-flight slot
        R->>D: 9. request + kv_transfer_params (source address)
        D->>S: 10. peer lookup and KV read (engine to engine)
        R->>R: 11. count vllm_router_kv_transfer_requests_total{outcome}
    end
    D-->>C: response
    R->>R: 12. free the in-flight slot when the request ends
```

Steps 1 to 7 are the same in shadow and active mode. Steps 8b to 12 exist only in active mode.

## Decision flow

Checks run in this order. The first one that fails gives the outcome label.

**Before the planner (call site, `kv_aware.rs`):**

| Check | Outcome label |
|---|---|
| The pick came from a session binding | `session_binding` |
| The request has no token ids to match | `missing_tokens` |

**Planner, request level (`plan_transfer`, `kv_transfer.rs`):**

| Check | Outcome label |
|---|---|
| Planner disabled or its constants invalid | `disabled` |
| Picked position is outside the peer list (a router bug) | `misaligned_input` |
| Destination's cache key unknown | `unverified_identity` |
| No lookup of the destination's own prompt (no tokens, several prompts, no credit) | `unknown_destination_input` |
| No other worker shares the destination's cache identity | `no_peer` |

**Peer ranking.** Peers that share the destination's cache identity are ranked by: longest prefix of the destination's prompt first, then fewer active requests, then position. A peer's prefix is measured from the destination's own prompt lookup, so it always describes tokens the destination's prompt contains. Only `Device` blocks from block 0 count, continued by `HostPinned` segments; `Disk` and `External` are ignored.

**Planner, per peer, in rank order (`plan_from_source`):**

| # | Check | Outcome label |
|---|---|---|
| 1 | Peer was excluded for this request (retry) | `source_excluded` |
| 2 | `require_capability` is on and source cannot send to destination | `incompatible` |
| 3 | Peer's prefix is not longer than the destination's | `picked_holds_longest_prefix` |
| 4 | Peer holds fewer than `min_prefix_blocks` extra blocks | `prefix_too_short` |
| 5 | Peer has fewer than `source_load_threshold` active requests | `source_not_busy` |
| 6 | Nothing left to credit (the last prompt token is never credited) | `prefix_too_short` |
| 7 | Cost rule fails (below) | `cost_rule` |

The first peer that passes all seven is planned (`planned`).

**Which reason is reported when no peer passes.** One request produces one outcome. The planner reports the reason of the highest-ranked peer, but prefers a peer that failed an ordinary check (3 to 7) over one that failed 1 or 2. A retry exclusion or a layout mismatch on the longest holder says less about lost transfers than why the next usable holder was rejected. So the counter answers "why did the best *usable* peer not qualify", not "why did the longest holder not qualify".

## Cost rule

A plan is made only if both hold:

```
fetch_ms    < wait_ms
transfer_ms < saved_ms * margin
```

| Term | Formula | Inputs come from |
|---|---|---|
| `credited` | `min(missing_blocks * block_size, input_tokens - 1 - dest_prefix_blocks * block_size)` | KV index hits for the destination's prompt |
| `transfer_ms` | `transfer_fixed_ms + credited * transfer_us_per_token / 1000` | `--kv-transfer-fixed-ms`, `--kv-transfer-us-per-token` |
| `dest_queue_ms` | `dest.active_prefill_tokens * prefill_us_per_token / 1000` | router's in-flight accounting, `--kv-transfer-prefill-us-per-token` |
| `source_wait_ms` | `source.active_prefill_tokens * prefill_us_per_token / 1000` | same |
| `fetch_ms` | `dest_queue_ms + transfer_ms + prefill(remaining on destination after the fetch)` | remaining tokens use the routing estimate (tier credits `--kv-penalty-*`) |
| `wait_ms` | `source_wait_ms + prefill(remaining on source)` | same routing estimate |
| `saved_ms` | `prefill(destination remaining before - after the fetch)` | same |

`prefill(n) = n * prefill_us_per_token / 1000`.

**Worked example** (defaults: block size 16, 50 us/token prefill, 5 ms + 5 us/token transfer, margin 0.8). Prompt is 2,000 tokens. Destination is idle with no cache. Source holds 100 device blocks, has 4 active requests and 10,000 queued prefill tokens.

- `credited = min(100 * 16, 1999 - 0) = 1600` tokens
- `transfer_ms = 5 + 1600 * 0.005 = 13`
- `saved_ms = 1600 * 0.05 = 80`
- `dest_queue_ms = 0`, `source_wait_ms = 10000 * 0.05 = 500`
- `fetch_ms = 0 + 13 + prefill(400) = 33`
- `wait_ms = 500 + prefill(400) = 520`
- Rule: `33 < 520` and `13 < 80 * 0.8 = 64`. Planned.

## Known bias in the estimates

The planned rate from shadow mode is an upper estimate. Read it with these three facts in mind.

1. **The wait path is priced as serial time.** `source_wait_ms` treats the source's queued tokens as if they run one after another. Routing itself scores workers as `(new prefill + queued tokens) * (active requests + 1)`, so the two numbers are not on the same scale, and a batching engine drains its queue faster than the serial estimate. This overstates `wait_ms`.
2. **The source's export work is free.** `fetch_ms` has no term for the busy source serving the read. This understates `fetch_ms`.
3. **The counter counts selections, not requests.** `vllm_router_kv_transfer_plans_total` is incremented once per selection. A retried request is counted again, and there is no per-request total to divide by.

Points 1 and 2 both favour `planned`. Calibrate the constants against a measured baseline before acting on the rate.

## Modes and settings

| Mode | `require_capability` | What a plan needs | What happens to a planned request |
|---|---|---|---|
| `off` | any | (no planner) | nothing |
| `shadow` | `false` | shared cache identity only | counted, request unchanged |
| `shadow` | `true` | plus: source can export, destination can import, same connector, version and layout fingerprint | counted, request unchanged |
| `active` | forced `true` | same as above | counted; then see below |

In `active` mode a planned request goes through these checks, in order. Each failure is counted on `vllm_router_kv_transfer_requests_total{outcome}` and the request is sent unchanged.

| Check | Outcome |
|---|---|
| Destination is a gRPC worker | `grpc_destination` |
| Source has no engine-reported peer address | `no_peer_endpoint` |
| Source already has `max_inflight_per_source` fetches in flight | `inflight_cap` |
| Client sent its own `kv_transfer_params` | `client_params_present` |
| Body is not a JSON object | `invalid_body` |
| Otherwise | `sent` |

**Where capabilities come from.**
- *Engine-reported:* a worker's `peer_lookup` record on `/kv_event_sources` gives both a capability and a peer address (host, port, engine id, TP/DCP/PP sizes).
- *Static:* `kv_transfer.capabilities` in the router config gives a capability only. It has no address, so a statically declared source can be planned in shadow mode but always ends as `no_peer_endpoint` in active mode. A static entry wins over the engine's capability for the same worker; the engine's address is still stored.

**Kill switch.** `KvAwarePolicy::set_peer_fetch_enabled(false)` makes an active-mode router behave as shadow for requests selected from then on. In-flight fetches finish normally. There is no HTTP endpoint for it yet.

**Not supported.**
- Prefill/decode (PD) routing: any mode other than `off` is rejected at startup.
- gRPC workers with `active`: rejected at startup. A gRPC worker discovered later is never asked to fetch.

## Metrics

| Metric | Labels | Meaning |
|---|---|---|
| `vllm_router_kv_transfer_plans_total` | `outcome` | One per selection: `planned` or a skip label from the tables above |
| `vllm_router_kv_transfer_requests_total` | `outcome` | Active mode only, one per planned request: `sent`, `grpc_destination`, `no_peer_endpoint`, `inflight_cap`, `client_params_present`, `invalid_body` |
| `vllm_router_kv_transfer_estimated_saved_ms` | none | `saved_ms` of each planned transfer |
| `vllm_router_kv_transfer_estimated_cost_ms` | `path` = `fetch` or `wait` | Both estimates, whenever the cost rule was evaluated |
