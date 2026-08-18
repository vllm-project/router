//! Discrete-event replay of the codex-swebenchpro agent trace (610
//! sessions / 20230 turns, real per-turn prompt/completion token counts
//! in tests/data/codex_trace_tokens.csv) against the real policy
//! implementations, modeling a P/D-disaggregated fleet:
//!
//! - 4 prefill workers, each a capacity-1 FIFO server: a turn routed to
//!   a P that has seen the session before prefills only the token
//!   increment; routed anywhere else it pays a full prefill. Queue wait
//!   is the TTFT cost of imbalance.
//! - 4 decode workers with per-token service time; concurrency spread
//!   is the ITL cost of imbalance.
//! - Closed loop: 256 sessions active, zero think time, next session
//!   admitted when one finishes — the vllm-bench-tui replay shape.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, VecDeque};
use std::sync::Arc;

use vllm_router_rs::core::{BasicWorker, Worker, WorkerType};
use vllm_router_rs::policies::{
    ConsistentHashPolicy, LoadBalancingPolicy, RendezvousHashPolicy, RequestHeaders,
    RoundRobinPolicy, StickyTablePolicy,
};

const P_WORKERS: usize = 4;
const D_WORKERS: usize = 4;
const MAX_ACTIVE: usize = 256;
const PREFILL_TOKS_PER_SEC: f64 = 10_000.0;
const TPOT_SECS: f64 = 0.03;
/// Even a fully-cached turn re-processes something (new user message).
const MIN_PREFILL_TOKENS: u64 = 64;

struct Turn {
    isl: u64,
    osl: u64,
}

fn load_trace() -> Vec<Vec<Turn>> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/codex_trace_tokens.csv");
    let text = std::fs::read_to_string(path).expect("run tests/data extraction first");
    let mut order: Vec<String> = Vec::new();
    let mut by_session: HashMap<String, Vec<Turn>> = HashMap::new();
    for line in text.lines().skip(1) {
        let mut parts = line.split(',');
        let sid = parts.next().unwrap().to_string();
        let isl: u64 = parts.next().unwrap().parse().unwrap();
        let osl: u64 = parts.next().unwrap().parse().unwrap();
        if !by_session.contains_key(&sid) {
            order.push(sid.clone());
        }
        by_session
            .entry(sid)
            .or_default()
            .push(Turn { isl, osl: osl.max(1) });
    }
    order.into_iter().map(|sid| by_session.remove(&sid).unwrap()).collect()
}

#[derive(Default)]
struct Stats {
    makespan_secs: f64,
    prefill_mtok: f64,
    p_busy_secs: Vec<f64>,
    d_peak_inflight: Vec<usize>,
    waits_secs: Vec<f64>,
    sticky_turns: usize,
    turns_with_prev: usize,
}

impl Stats {
    fn p_busy_spread(&self) -> f64 {
        let max = self.p_busy_secs.iter().cloned().fold(0.0, f64::max);
        let mean = self.p_busy_secs.iter().sum::<f64>() / self.p_busy_secs.len() as f64;
        max / mean
    }
    fn wait_pct(&mut self, q: f64) -> f64 {
        self.waits_secs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        self.waits_secs[((self.waits_secs.len() - 1) as f64 * q) as usize]
    }
    fn sticky_pct(&self) -> f64 {
        self.sticky_turns as f64 / self.turns_with_prev.max(1) as f64
    }
}

struct PrefillServer {
    queue: VecDeque<(usize, u64, u64)>, // (session, new_tokens, enqueue_us)
    running: Option<(usize, u64, u64)>, // (session, new_tokens, start_us)
}

fn run(
    policy_p: &dyn LoadBalancingPolicy,
    policy_d: &dyn LoadBalancingPolicy,
    sessions: &[Vec<Turn>],
) -> Stats {
    let mk = |prefix: &str, n: usize| -> Vec<Arc<dyn Worker>> {
        (0..n)
            .map(|i| {
                Arc::new(BasicWorker::new(
                    format!("http://{}{}:8000", prefix, i),
                    WorkerType::Regular,
                )) as Arc<dyn Worker>
            })
            .collect()
    };
    let p_workers = mk("p", P_WORKERS);
    let d_workers = mk("d", D_WORKERS);

    let mut servers: Vec<PrefillServer> = (0..P_WORKERS)
        .map(|_| PrefillServer {
            queue: VecDeque::new(),
            running: None,
        })
        .collect();

    // Event kinds: 0 = turn arrives (a=session), 1 = prefill done
    // (a=p_idx, b=session), 2 = decode done (a=d_idx, b=session).
    let mut heap: BinaryHeap<Reverse<(u64, u8, usize, usize)>> = BinaryHeap::new();
    let mut next_unstarted = MAX_ACTIVE.min(sessions.len());
    for sid in 0..next_unstarted {
        heap.push(Reverse((sid as u64 * 50_000, 0, sid, 0)));
    }

    let mut cur_turn = vec![0usize; sessions.len()];
    let mut prefix_cache: Vec<HashMap<usize, u64>> = vec![HashMap::new(); sessions.len()];
    let mut last_p: Vec<Option<usize>> = vec![None; sessions.len()];
    let mut chosen_d = vec![0usize; sessions.len()];
    let mut d_inflight = vec![0usize; D_WORKERS];

    let mut stats = Stats {
        p_busy_secs: vec![0.0; P_WORKERS],
        d_peak_inflight: vec![0; D_WORKERS],
        ..Default::default()
    };
    let mut end_us = 0u64;
    // Stand-in request body sized like the real serialized messages
    // (~4 chars/token) so body-size-gated behavior sees realistic costs.
    let blob = "x".repeat(1 << 20);

    let start_prefill = |server: &mut PrefillServer,
                         heap: &mut BinaryHeap<Reverse<(u64, u8, usize, usize)>>,
                         stats: &mut Stats,
                         p_idx: usize,
                         t: u64| {
        if server.running.is_none() {
            if let Some((sid, new_tok, enq_us)) = server.queue.pop_front() {
                stats.waits_secs.push((t - enq_us) as f64 / 1e6);
                server.running = Some((sid, new_tok, t));
                let dur = (new_tok as f64 / PREFILL_TOKS_PER_SEC * 1e6) as u64;
                heap.push(Reverse((t + dur, 1, p_idx, sid)));
            }
        }
    };

    while let Some(Reverse((t, kind, a, b))) = heap.pop() {
        end_us = end_us.max(t);
        match kind {
            0 => {
                let sid = a;
                let turn = &sessions[sid][cur_turn[sid]];
                let mut headers = RequestHeaders::new();
                headers.insert("x-session-id".to_string(), format!("sess-{}", sid));
                let body = &blob[..(turn.isl as usize * 4).min(blob.len())];
                let p = policy_p
                    .select_worker_with_headers(&p_workers, Some(body), Some(&headers))
                    .unwrap();
                let d = policy_d
                    .select_worker_with_headers(&d_workers, Some(body), Some(&headers))
                    .unwrap();
                chosen_d[sid] = d;

                if let Some(prev) = last_p[sid] {
                    stats.turns_with_prev += 1;
                    if prev == p {
                        stats.sticky_turns += 1;
                    }
                }
                last_p[sid] = Some(p);

                let cached = prefix_cache[sid].get(&p).copied().unwrap_or(0);
                let new_tok = if turn.isl >= cached {
                    (turn.isl - cached).max(MIN_PREFILL_TOKENS)
                } else {
                    turn.isl // context compaction: prefix no longer matches
                };
                prefix_cache[sid].insert(p, turn.isl);
                stats.prefill_mtok += new_tok as f64 / 1e6;

                p_workers[p].increment_load();
                servers[p].queue.push_back((sid, new_tok, t));
                start_prefill(&mut servers[p], &mut heap, &mut stats, p, t);
            }
            1 => {
                let (p, sid) = (a, b);
                let (_, new_tok, start_us) = servers[p].running.take().unwrap();
                stats.p_busy_secs[p] += (t - start_us) as f64 / 1e6;
                let _ = new_tok;
                p_workers[p].decrement_load();
                start_prefill(&mut servers[p], &mut heap, &mut stats, p, t);

                let d = chosen_d[sid];
                d_workers[d].increment_load();
                d_inflight[d] += 1;
                stats.d_peak_inflight[d] = stats.d_peak_inflight[d].max(d_inflight[d]);
                let dur = (sessions[sid][cur_turn[sid]].osl as f64 * TPOT_SECS * 1e6) as u64;
                heap.push(Reverse((t + dur, 2, d, sid)));
            }
            _ => {
                let (d, sid) = (a, b);
                d_workers[d].decrement_load();
                d_inflight[d] -= 1;
                cur_turn[sid] += 1;
                if cur_turn[sid] < sessions[sid].len() {
                    heap.push(Reverse((t, 0, sid, 0)));
                } else if next_unstarted < sessions.len() {
                    heap.push(Reverse((t, 0, next_unstarted, 0)));
                    next_unstarted += 1;
                }
            }
        }
    }

    stats.makespan_secs = end_us as f64 / 1e6;
    stats
}

#[test]
fn codex_trace_replay_compares_policies() {
    let sessions = load_trace();
    let turns: usize = sessions.iter().map(Vec::len).sum();
    let total_isl: u64 = sessions.iter().flatten().map(|t| t.isl).sum();
    let max_turns = sessions.iter().map(Vec::len).max().unwrap();
    println!(
        "trace: {} sessions, {} turns (max {}/session), {:.1} Mtok total prompt",
        sessions.len(),
        turns,
        max_turns,
        total_isl as f64 / 1e6
    );

    let mk_pair = |name: &str| -> (Box<dyn LoadBalancingPolicy>, Box<dyn LoadBalancingPolicy>) {
        match name {
            "sticky_table" => (
                Box::new(StickyTablePolicy::new()),
                Box::new(StickyTablePolicy::new()),
            ),
            "consistent_hash" => (
                Box::new(ConsistentHashPolicy::new()),
                Box::new(ConsistentHashPolicy::new()),
            ),
            "rendezvous_hash" => (
                Box::new(RendezvousHashPolicy::new()),
                Box::new(RendezvousHashPolicy::new()),
            ),
            _ => (
                Box::new(RoundRobinPolicy::new()),
                Box::new(RoundRobinPolicy::new()),
            ),
        }
    };

    println!(
        "{:<16} {:>10} {:>12} {:>11} {:>9} {:>9} {:>8}  {}",
        "policy", "makespan", "prefillMtok", "P max/avg", "wait p50", "wait p99", "sticky%", "D peak inflight"
    );
    let mut results: Vec<(&str, Stats)> = Vec::new();
    for name in ["sticky_table", "consistent_hash", "rendezvous_hash", "round_robin"] {
        let (pp, pd) = mk_pair(name);
        let mut s = run(pp.as_ref(), pd.as_ref(), &sessions);
        let (w50, w99) = (s.wait_pct(0.50), s.wait_pct(0.99));
        println!(
            "{:<16} {:>9.0}s {:>12.1} {:>10.3}x {:>8.2}s {:>8.1}s {:>7.1}%  {:?}",
            name,
            s.makespan_secs,
            s.prefill_mtok,
            s.p_busy_spread(),
            w50,
            w99,
            100.0 * s.sticky_pct(),
            s.d_peak_inflight,
        );
        results.push((name, s));
    }

    let sticky = &results[0].1;
    let chash = &results[1].1;
    let rr = &results[3].1;

    // Stickiness holds; migrations are rare.
    assert!(sticky.sticky_pct() >= 0.95, "sticky% {}", sticky.sticky_pct());
    // Both sticky policies pay near-minimal prefill; round_robin pays the
    // full-refetch tax on nearly every turn.
    assert!(sticky.prefill_mtok < 1.15 * chash.prefill_mtok);
    assert!(rr.prefill_mtok > 2.0 * sticky.prefill_mtok);
    // Load-aware placement beats the load-blind hash on prefill balance
    // and on the queueing it causes.
    assert!(sticky.p_busy_spread() < chash.p_busy_spread());
    let (mut s2, mut c2) = (results[0].1.waits_secs.clone(), results[1].1.waits_secs.clone());
    s2.sort_by(|x, y| x.partial_cmp(y).unwrap());
    c2.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let p99 = |v: &Vec<f64>| v[(v.len() - 1) * 99 / 100];
    assert!(p99(&s2) <= p99(&c2), "p99 wait {:.1} !<= {:.1}", p99(&s2), p99(&c2));
    assert!(sticky.makespan_secs <= 1.05 * chash.makespan_secs);
    assert!(sticky.makespan_secs < rr.makespan_secs);
}
