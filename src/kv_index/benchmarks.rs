//! Reproducible CPU microbenchmark for the baseline and unified KV index.
//!
//! Copy this file and the `mod benchmarks` test declaration to the #493
//! baseline. Run the same profile and command for both builds:
//!
//! ```text
//! KV_BENCH_ITERS=10 RUSTFLAGS='--check-cfg=cfg(kv_bench_baseline)' \
//!   cargo test --features cpu,ready kv_index_cpu_workloads -- --ignored --nocapture --test-threads=1
//! KV_BENCH_ITERS=10 RUSTFLAGS='--check-cfg=cfg(kv_bench_baseline) --cfg kv_bench_baseline' \
//!   cargo test --features cpu,ready kv_index_cpu_workloads -- --ignored --nocapture --test-threads=1
//! ```
//!
//! Use the second command only for the baseline. `--release` must either be
//! present in both commands or absent in both. Output is one JSON object per
//! workload. These tests measure Router CPU work, not GPU inference or network.
//! Dense timings start after local hashes have been built; hybrid timings include
//! request sequence hashing. A writer's apply time includes lock waiting,
//! validation and update work, so it is an upper bound on lock wait, not an
//! isolated lock-wait measurement. The writer is closed-loop with no publisher
//! queue; queue lag is not measured.

#![cfg(all(test, feature = "cpu", feature = "ready"))]

use std::hint::black_box;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};
use std::time::{Duration, Instant};

use crate::kv_hash::BlockKeyGenerator;
use crate::kv_index::hybrid::{HybridEventGroup, HybridEventIndex, HybridGroupKind};
use crate::kv_index::wire::ExternalBlockHash;
use crate::kv_index::{
    BlockRemoved, BlockStored, KVEvent, KVEventBatch, KvBlockIndexer, MatchQuery, ResidencyOwner,
    RoutableTarget, StorageTier, TierMatch,
};

const BLOCK: usize = 16;

#[cfg(kv_bench_baseline)]
fn attach(index: HybridEventIndex, _: Arc<KvBlockIndexer>) -> HybridEventIndex {
    index
}

#[cfg(not(kv_bench_baseline))]
fn attach(index: HybridEventIndex, shared: Arc<KvBlockIndexer>) -> HybridEventIndex {
    index.with_shared_index(shared)
}

fn iterations() -> usize {
    std::env::var("KV_BENCH_ITERS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(10)
        .clamp(1, 100)
}

fn percentile_us(samples: &mut [u64], percentile: usize) -> f64 {
    samples.sort_unstable();
    let pos = ((samples.len() - 1) * percentile).div_ceil(100);
    samples[pos] as f64 / 1_000.0
}

fn emit(
    name: &str,
    workers: usize,
    groups: usize,
    tokens: usize,
    mut samples: Vec<u64>,
    elapsed_ns: u128,
) {
    let p50 = percentile_us(&mut samples, 50);
    let p95 = percentile_us(&mut samples, 95);
    let p99 = percentile_us(&mut samples, 99);
    println!(
        "{}",
        serde_json::json!({
            "workload": name, "workers": workers, "groups": groups,
            "tokens": tokens, "block_tokens": BLOCK, "samples": samples.len(),
            "p50_us": p50, "p95_us": p95, "p99_us": p99,
            "throughput_unit": if name.starts_with("writer_") { "events_per_second" } else { "all_target_queries_per_second" },
            "qps": samples.len() as f64 * 1e9 / elapsed_ns.max(1) as f64,
        })
    );
}

fn worker(i: usize) -> ResidencyOwner {
    ResidencyOwner::Worker {
        source: Arc::from(format!("worker-{i}")),
        dp_rank: 0,
        incarnation: 0,
    }
}

struct DenseFixture {
    index: Arc<KvBlockIndexer>,
    query: Arc<MatchQuery>,
    blocks: Vec<(Arc<str>, Arc<str>)>,
}

fn dense_fixture(workers: usize, tokens: usize) -> DenseFixture {
    let index = Arc::new(KvBlockIndexer::new());
    let blocks: Vec<(Arc<str>, Arc<str>)> = (0..tokens / BLOCK)
        .map(|i| {
            (
                Arc::from(format!("b:{i:064x}")),
                Arc::from(format!("local-{i}")),
            )
        })
        .collect();
    for id in 0..workers {
        index.store(0, worker(id), StorageTier::Device, None, &blocks);
    }
    let query = Arc::new(MatchQuery {
        group_idx: 0,
        local_hashes: blocks.iter().map(|(_, local)| local.clone()).collect(),
        tiers_of_interest: vec![StorageTier::Device],
    });
    let hits = index.find_matches(&query);
    assert_eq!(hits.len(), workers);
    assert!(hits
        .iter()
        .all(|hit| hit.segments == [(0, (tokens / BLOCK) as u32)]));
    DenseFixture {
        index,
        query,
        blocks,
    }
}

fn dense(workers: usize, tokens: usize, iters: usize) {
    let DenseFixture { index, query, .. } = dense_fixture(workers, tokens);
    for _ in 0..2 {
        black_box(index.find_matches(&query));
    }
    let started = Instant::now();
    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let start = Instant::now();
        black_box(index.find_matches(&query));
        samples.push(start.elapsed().as_nanos() as u64);
    }
    emit(
        "dense_query",
        workers,
        1,
        tokens,
        samples,
        started.elapsed().as_nanos(),
    );
}

fn dense_concurrent(iters: usize) {
    let workers = 8;
    let groups = 1;
    let tokens = 8192;
    let DenseFixture {
        index,
        query,
        blocks,
    } = dense_fixture(workers, tokens);
    let barrier = Arc::new(Barrier::new(6));
    let active = Arc::new(AtomicUsize::new(4));
    let writer_ready = Arc::new(AtomicBool::new(false));
    let mut readers = Vec::new();
    for _ in 0..4 {
        let index = index.clone();
        let query = query.clone();
        let barrier = barrier.clone();
        let active = active.clone();
        let writer_ready = writer_ready.clone();
        readers.push(std::thread::spawn(move || {
            barrier.wait();
            while !writer_ready.load(Ordering::Acquire) {
                std::thread::yield_now();
            }
            let mut samples = Vec::with_capacity(iters);
            for _ in 0..iters {
                let start = Instant::now();
                black_box(index.find_matches(&query));
                samples.push(start.elapsed().as_nanos() as u64);
            }
            active.fetch_sub(1, Ordering::Release);
            samples
        }));
    }
    let writer_index = index.clone();
    let writer_barrier = barrier.clone();
    let writer_active = active.clone();
    let writer_start = writer_ready.clone();
    let owner = worker(0);
    let last_block = [blocks.last().unwrap().clone()];
    let last_hash = [last_block[0].0.clone()];
    let parent = blocks[blocks.len() - 2].0.clone();
    let writer = std::thread::spawn(move || {
        writer_barrier.wait();
        let started = Instant::now();
        writer_start.store(true, Ordering::Release);
        let mut samples = Vec::with_capacity(10_000);
        let mut count = 0usize;
        while writer_active.load(Ordering::Acquire) != 0
            && started.elapsed() < Duration::from_secs(60)
        {
            let start = Instant::now();
            if count.is_multiple_of(2) {
                writer_index.remove(0, &owner, StorageTier::Device, &last_hash);
            } else {
                writer_index.store(
                    0,
                    owner.clone(),
                    StorageTier::Device,
                    Some(parent.clone()),
                    &last_block,
                );
            }
            let latency = start.elapsed().as_nanos() as u64;
            if samples.len() < 10_000 {
                samples.push(latency);
            } else {
                samples[count % 10_000] = latency;
            }
            count += 1;
        }
        (
            samples,
            count,
            started.elapsed().as_nanos(),
            writer_active.load(Ordering::Acquire) != 0,
        )
    });
    let started = Instant::now();
    barrier.wait();
    let mut reader_samples = Vec::new();
    for reader in readers {
        reader_samples.extend(reader.join().unwrap());
    }
    let elapsed = started.elapsed().as_nanos();
    let (mut writer_samples, writer_events, writer_elapsed, stopped_early) = writer.join().unwrap();
    let hits = index.find_matches(&query);
    assert_eq!(hits.len(), workers);
    for hit in hits {
        let missing_last = hit.target == worker(0).target() && !writer_events.is_multiple_of(2);
        assert_eq!(
            hit.segments,
            [(0, (tokens / BLOCK - usize::from(missing_last)) as u32)]
        );
    }
    emit(
        "dense_4readers_1writer",
        workers,
        groups,
        tokens,
        reader_samples,
        elapsed,
    );
    if !writer_samples.is_empty() {
        let p50 = percentile_us(&mut writer_samples, 50);
        let p95 = percentile_us(&mut writer_samples, 95);
        let p99 = percentile_us(&mut writer_samples, 99);
        println!(
            "{}",
            serde_json::json!({
                "workload":"dense_writer_apply_including_lock_wait", "workers":workers,
                "groups":groups, "tokens":tokens, "events":writer_events,
                "samples":writer_samples.len(), "sampling":"last 10000 events at most",
                "p50_us":p50, "p95_us":p95, "p99_us":p99,
                "events_per_second":writer_events as f64 * 1e9 / writer_elapsed.max(1) as f64,
                "stopped_early":stopped_early
            })
        );
    }
    println!(
        "{}",
        serde_json::json!({"workload":"dense_writer_queue_lag", "queue_lag_us":null,
        "reason":"closed-loop writer; no producer queue"})
    );
}

fn event(group: u32, tokens: &[u32], hashes: &[[u8; 32]], checkpoint: bool) -> KVEvent {
    KVEvent::BlockStored(BlockStored {
        extra_keys: Some(vec![rmpv::Value::Nil; hashes.len()]),
        block_hashes: hashes
            .iter()
            .map(|h| ExternalBlockHash::Bytes(h.to_vec()))
            .collect(),
        parent_block_hash: None,
        token_ids: tokens.to_vec(),
        block_size: BLOCK as u32,
        lora_id: None,
        medium: Some("GPU".to_owned()),
        lora_name: None,
        group_idx: Some(group),
        kv_cache_spec_kind: Some(
            if checkpoint {
                "mamba"
            } else {
                "full_attention"
            }
            .to_owned(),
        ),
        kv_cache_spec_sliding_window: None,
        locality: None,
        ownership: None,
        session_id: None,
    })
}

fn hybrid_fixture(
    workers: usize,
    groups: usize,
    tokens: usize,
) -> (Vec<Arc<HybridEventIndex>>, Arc<Vec<u32>>, Vec<KVEventBatch>) {
    let prompt = Arc::new(
        (0..=tokens)
            .map(|i| (i % 50_000) as u32)
            .collect::<Vec<_>>(),
    );
    let hashes = BlockKeyGenerator::new(BLOCK, 0).generate_block_keys(&prompt[..tokens]);
    let layout: Vec<_> = (0..groups)
        .map(|i| HybridEventGroup {
            group_idx: i as u32,
            block_size: BLOCK as u32,
            kind: if i == 0 {
                HybridGroupKind::FullAttention
            } else {
                HybridGroupKind::Checkpoint
            },
        })
        .collect();
    let initial = KVEventBatch {
        ts: 1.0,
        data_parallel_rank: Some(0),
        events: (0..groups)
            .map(|i| event(i as u32, &prompt[..tokens], &hashes, i != 0))
            .collect(),
    };
    let shared = Arc::new(KvBlockIndexer::new());
    let mut indices = Vec::with_capacity(workers);
    for id in 0..workers {
        let target = RoutableTarget {
            instance_id: Arc::from(format!("worker-{id}")),
            dp_rank: 0,
        };
        let index = HybridEventIndex::new(
            target,
            layout.clone(),
            BLOCK as u32,
            0,
            groups * (tokens / BLOCK) + 1,
        )
        .unwrap();
        let index = Arc::new(attach(index, shared.clone()));
        index.begin_empty(0);
        index.apply_batch(0, &initial).unwrap();
        assert_eq!(index.reusable_tokens(&prompt, usize::MAX), tokens as u64);
        indices.push(index);
    }
    let last = *hashes.last().unwrap();
    let remove = KVEventBatch {
        ts: 1.0,
        data_parallel_rank: Some(0),
        events: vec![KVEvent::BlockRemoved(BlockRemoved {
            block_hashes: vec![ExternalBlockHash::Bytes(last.to_vec())],
            medium: Some("GPU".into()),
            group_idx: Some(0),
            locality: None,
            ownership: None,
        })],
    };
    let store = KVEventBatch {
        ts: 1.0,
        data_parallel_rank: Some(0),
        events: vec![event(0, &prompt[tokens - BLOCK..tokens], &[last], false)],
    };
    (indices, prompt, vec![remove, store])
}

fn hybrid_query(indices: &[Arc<HybridEventIndex>], prompt: &[u32]) {
    for index in indices {
        black_box(index.reusable_tokens(prompt, usize::MAX));
    }
}

fn hybrid(workers: usize, groups: usize, tokens: usize, iters: usize) {
    let (indices, prompt, _) = hybrid_fixture(workers, groups, tokens);
    for _ in 0..2 {
        hybrid_query(&indices, &prompt);
    }
    let started = Instant::now();
    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let start = Instant::now();
        hybrid_query(&indices, &prompt);
        samples.push(start.elapsed().as_nanos() as u64);
    }
    emit(
        "hybrid_all_targets",
        workers,
        groups,
        tokens,
        samples,
        started.elapsed().as_nanos(),
    );
}

fn concurrent(iters: usize) {
    let workers = 8;
    let groups = 4;
    let tokens = 8192;
    let (indices, prompt, events) = hybrid_fixture(workers, groups, tokens);
    let barrier = Arc::new(Barrier::new(6));
    let active = Arc::new(AtomicUsize::new(4));
    let writer_ready = Arc::new(AtomicBool::new(false));
    let mut readers = Vec::new();
    for _ in 0..4 {
        let indices = indices.clone();
        let prompt = prompt.clone();
        let barrier = barrier.clone();
        let active = active.clone();
        let writer_ready = writer_ready.clone();
        readers.push(std::thread::spawn(move || {
            barrier.wait();
            while !writer_ready.load(Ordering::Acquire) {
                std::thread::yield_now();
            }
            let mut samples = Vec::with_capacity(iters);
            for _ in 0..iters {
                let start = Instant::now();
                hybrid_query(&indices, &prompt);
                samples.push(start.elapsed().as_nanos() as u64);
            }
            active.fetch_sub(1, Ordering::Release);
            samples
        }));
    }
    let writer = indices[0].clone();
    let writer_barrier = barrier.clone();
    let writer_active = active.clone();
    let writer_start = writer_ready.clone();
    let writer = std::thread::spawn(move || {
        writer_barrier.wait();
        let started = Instant::now();
        writer_start.store(true, Ordering::Release);
        let mut samples = Vec::with_capacity(10_000);
        let mut count = 0usize;
        while writer_active.load(Ordering::Acquire) != 0
            && started.elapsed() < Duration::from_secs(60)
        {
            let start = Instant::now();
            writer
                .apply_batch(count as u64 + 1, &events[count % 2])
                .unwrap();
            let latency = start.elapsed().as_nanos() as u64;
            if samples.len() < 10_000 {
                samples.push(latency);
            } else {
                samples[count % 10_000] = latency;
            }
            count += 1;
        }
        (
            samples,
            count,
            started.elapsed().as_nanos(),
            writer_active.load(Ordering::Acquire) != 0,
        )
    });
    let started = Instant::now();
    barrier.wait();
    let mut reader_samples = Vec::new();
    for reader in readers {
        reader_samples.extend(reader.join().unwrap());
    }
    let elapsed = started.elapsed().as_nanos();
    let (mut writer_samples, writer_events, writer_elapsed, stopped_early) = writer.join().unwrap();
    emit(
        "hybrid_4readers_1writer",
        workers,
        groups,
        tokens,
        reader_samples,
        elapsed,
    );
    if !writer_samples.is_empty() {
        let p50 = percentile_us(&mut writer_samples, 50);
        let p95 = percentile_us(&mut writer_samples, 95);
        let p99 = percentile_us(&mut writer_samples, 99);
        println!(
            "{}",
            serde_json::json!({
                "workload":"writer_apply_including_lock_wait", "workers":workers,
                "groups":groups, "tokens":tokens, "events":writer_events,
                "samples":writer_samples.len(), "sampling":"last 10000 events at most",
                "p50_us":p50, "p95_us":p95, "p99_us":p99,
                "events_per_second":writer_events as f64 * 1e9 / writer_elapsed.max(1) as f64,
                "stopped_early":stopped_early
            })
        );
    }
    println!(
        "{}",
        serde_json::json!({"workload":"writer_queue_lag", "queue_lag_us":null,
        "reason":"closed-loop writer; no producer queue"})
    );
}

fn interleaved_checkpoints(partial: bool, iters: usize) {
    use crate::kv_index::hybrid::{common_reusable_tokens, HybridGroupView};
    let target = RoutableTarget {
        instance_id: Arc::from("interleaved"),
        dp_rank: 0,
    };
    let points = 2048_u32;
    let segments = |parity| {
        (0..points)
            .filter(|p| p % 2 == parity)
            .map(|p| (p, p + 1))
            .collect()
    };
    let hits: Vec<Vec<TierMatch>> = (0..2)
        .map(|g| {
            vec![TierMatch {
                target: target.clone(),
                tier: StorageTier::Device,
                locality: crate::kv_index::Locality::Local,
                segments: if partial { Vec::new() } else { segments(g) },
            }]
        })
        .collect();
    let partials: Vec<Vec<u64>> = (0..2)
        .map(|g| {
            if partial {
                (1..=points)
                    .filter(|p| p % 2 == g)
                    .map(|p| u64::from(p) * 16)
                    .collect()
            } else {
                Vec::new()
            }
        })
        .collect();
    let views: Vec<_> = (0..2)
        .map(|g| HybridGroupView {
            group_idx: g as u32,
            block_size: if partial { 65536 } else { 16 },
            kind: HybridGroupKind::Checkpoint,
            matches: &hits[g],
            page_evidence: true,
            partial_ends: &partials[g],
        })
        .collect();
    let started = Instant::now();
    let mut samples = Vec::new();
    for _ in 0..iters {
        let start = Instant::now();
        assert_eq!(black_box(common_reusable_tokens(&target, &views, 32768)), 0);
        samples.push(start.elapsed().as_nanos() as u64);
    }
    emit(
        if partial {
            "interleaved_partial_checkpoints"
        } else {
            "interleaved_checkpoints"
        },
        1,
        2,
        32768,
        samples,
        started.elapsed().as_nanos(),
    );
}

#[test]
#[ignore = "CPU microbenchmark; run explicitly with --ignored --nocapture"]
fn kv_index_cpu_workloads() {
    let n = iterations();
    println!(
        "{}",
        serde_json::json!({"workload":"metadata", "iters":n,
        "baseline":cfg!(kv_bench_baseline), "profile":if cfg!(debug_assertions) {"debug"} else {"release"}})
    );
    for tokens in [8192, 32768] {
        for workers in [1, 8, 32] {
            dense(workers, tokens, n);
        }
        for groups in [4, 16] {
            for workers in [1, 8] {
                hybrid(workers, groups, tokens, n);
            }
        }
    }
    interleaved_checkpoints(false, n);
    interleaved_checkpoints(true, n);
    concurrent(n);
    dense_concurrent(n);
}

/// Replay real wire batches at identical event watermarks in both versions.
#[test]
#[ignore = "requires KV_REPLAY_FIXTURE recorded from a real engine"]
fn replay_recorded_engine_events() {
    #[derive(serde::Deserialize)]
    struct Frame {
        seq: u64,
        payload: Vec<u8>,
    }
    #[derive(serde::Deserialize)]
    struct Fixture {
        capability: crate::kv_index::hybrid::HybridCapabilities,
        frames: Vec<Frame>,
        prompts: Vec<Vec<u32>>,
    }
    let path = std::env::var("KV_REPLAY_FIXTURE").expect("KV_REPLAY_FIXTURE is required");
    let fixture: Fixture = serde_json::from_reader(std::fs::File::open(path).unwrap()).unwrap();
    let rank = &fixture.capability.ranks[0];
    let source = crate::kv_index::subscriber::EventSource {
        source: Arc::from("recorded-engine"),
        dp_rank: rank.rank,
        pub_endpoint: "tcp://127.0.0.1:28851".into(),
        replay_endpoint: Some("tcp://127.0.0.1:28852".into()),
        topic: rank
            .publisher_topic
            .clone()
            .expect("recorded publisher topic"),
        hwm: None,
    };
    let index = fixture
        .capability
        .index_for_rank(
            rank.rank,
            &source,
            RoutableTarget {
                instance_id: source.source.clone(),
                dp_rank: rank.rank,
            },
            1_000_000,
        )
        .unwrap();
    let index = attach(index, Arc::new(KvBlockIndexer::new()));
    assert_eq!(fixture.frames.first().unwrap().seq, 0);
    index.begin_empty(0);
    for frame in fixture.frames {
        index.apply_payload(frame.seq, &frame.payload).unwrap();
        for (prompt, tokens) in fixture.prompts.iter().enumerate() {
            for cap in [usize::MAX, 1056, 1040, 544, 0] {
                let matched = index.reusable_tokens(tokens, cap);
                println!(
                    "{}",
                    serde_json::json!({"seq":frame.seq,"prompt":prompt,
                    "cap":cap,"matched":matched})
                );
            }
        }
    }
}
