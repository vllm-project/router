//! Per-request routing-input cost on the CPU fast path.
//!
//! Establishes the "before" numbers for the tokenizer/L0 request-path
//! integration (roadmap #244, Issue 1): what the router spends today to
//! derive the routing key from a typed request and to select a worker,
//! before any tokenizer runs in the request path.
//!
//! Groups:
//! - `routing_key/extract_text_for_routing/{completion,chat}/{size}`:
//!   `GenerationRequest::extract_text_for_routing` as called once per request
//!   in `Router::route_typed_request`, including dropping the returned
//!   `String` as the router does. (`iter_with_large_drop` would keep a whole
//!   sample's worth of 16 KiB strings alive and needs tens of GB.)
//! - `policy/{cache_aware,rendezvous_hash}/{size}`: `select_worker` over four
//!   healthy workers with the routing text of a hot-set prompt.
//! - `cache_aware_key_format/{raw_text,one_char_per_token,digit_tagged}/{tokens}`:
//!   `Tree::insert` (64 keys into a fresh tree) and
//!   `Tree::prefix_match_with_counts` (one probe sharing the first half of
//!   the warmed keys) for three candidate key encodings at equal token
//!   counts; raw text uses about four bytes per token. This is input for the discussion on
//!   token-id routing keys (PR #237); it does not exercise any router code
//!   path beyond the tree.
//!
//! Inputs come from `tests/common/bench_corpus.rs` (seeded, offline), so the
//! benchmark needs no network and no tokenizer file.
//!
//! Run with: `cargo bench --bench routing_input`

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::sync::Arc;
use std::time::{Duration, Instant};
use vllm_router_rs::core::{BasicWorker, Worker, WorkerType};
use vllm_router_rs::policies::{
    CacheAwareConfig, CacheAwarePolicy, LoadBalancingPolicy, RendezvousHashPolicy,
};
use vllm_router_rs::protocols::spec::GenerationRequest;
use vllm_router_rs::tree::Tree;

#[path = "../tests/common/mod.rs"]
mod common;
use common::bench_corpus::{
    chat, completion_text, seeded_text, to_chat_request, to_completion_request, Corpus, CorpusKind,
    SEED, SIZES,
};

/// A routing-key builder for one candidate encoding.
type KeyFn = Box<dyn Fn(u64) -> String>;

const WORKER_COUNT: usize = 4;
const TOKEN_COUNTS: [usize; 4] = [128, 512, 2048, 8192];
const VOCAB: u32 = 32_000;

fn size_label(size: usize) -> String {
    if size >= 1024 {
        format!("{}KiB", size / 1024)
    } else {
        format!("{size}B")
    }
}

fn workers() -> Vec<Arc<dyn Worker>> {
    (0..WORKER_COUNT)
        .map(|i| {
            Arc::new(BasicWorker::new(
                format!("http://127.0.0.1:{}", 30_000 + i),
                WorkerType::Regular,
            )) as Arc<dyn Worker>
        })
        .collect()
}

fn bench_extract_text(c: &mut Criterion) {
    let mut group = c.benchmark_group("routing_key/extract_text_for_routing");
    for size in SIZES {
        let corpus = Corpus::new(CorpusKind::Hot64, size);
        let prompt = corpus.prompt(0);

        // The completion routing text is the prompt itself, so its cost
        // scales with the prompt: report bytes per second.
        group.throughput(Throughput::Bytes(size as u64));
        let completion = to_completion_request(&completion_text(&prompt));
        group.bench_with_input(
            BenchmarkId::new("completion", size_label(size)),
            &completion,
            |b, req| {
                b.iter(|| black_box(req.extract_text_for_routing()));
            },
        );

        // The chat routing text is `session_params.session_id` or empty, so
        // its cost does not depend on the prompt: report calls per second.
        group.throughput(Throughput::Elements(1));
        let chat_req = to_chat_request(&chat(None, &prompt));
        group.bench_with_input(
            BenchmarkId::new("chat", size_label(size)),
            &chat_req,
            |b, req| {
                b.iter(|| black_box(req.extract_text_for_routing()));
            },
        );
    }
    group.finish();
}

fn bench_policy_select(c: &mut Criterion) {
    let mut group = c.benchmark_group("policy");
    let workers = workers();
    for size in SIZES {
        let corpus = Corpus::new(CorpusKind::Hot64, size);
        let prompts: Vec<String> = (0..64).map(|i| corpus.prompt(i)).collect();

        // cache_aware: warm the tree with the hot set once, then measure the
        // steady state (every request is a tree hit that also re-inserts).
        let cache_aware = CacheAwarePolicy::with_config(CacheAwareConfig {
            eviction_interval_secs: 0,
            ..CacheAwareConfig::default()
        });
        for w in &workers {
            cache_aware.add_worker(w.as_ref());
        }
        for p in &prompts {
            cache_aware.select_worker(&workers, Some(p));
        }
        let mut i = 0usize;
        group.bench_with_input(
            BenchmarkId::new("cache_aware", size_label(size)),
            &prompts,
            |b, prompts| {
                b.iter(|| {
                    let p = &prompts[i % prompts.len()];
                    i += 1;
                    black_box(cache_aware.select_worker(&workers, Some(p)))
                });
            },
        );

        let rendezvous = RendezvousHashPolicy::new();
        let mut j = 0usize;
        group.bench_with_input(
            BenchmarkId::new("rendezvous_hash", size_label(size)),
            &prompts,
            |b, prompts| {
                b.iter(|| {
                    let p = &prompts[j % prompts.len()];
                    j += 1;
                    black_box(rendezvous.select_worker(&workers, Some(p)))
                });
            },
        );
    }
    group.finish();
}

/// Token ids drawn uniformly from the vocabulary; the first `shared` ids are
/// identical across sequences so the tree has a prefix to match.
fn synthetic_ids(seq: u64, n: usize, shared: usize) -> Vec<u32> {
    let mut shared_rng = StdRng::seed_from_u64(SEED);
    let mut own_rng = StdRng::seed_from_u64(SEED ^ (seq + 1));
    (0..n)
        .map(|k| {
            if k < shared {
                shared_rng.random_range(0..VOCAB)
            } else {
                own_rng.random_range(0..VOCAB)
            }
        })
        .collect()
}

/// One unassigned Unicode scalar per token id (planes 4-7), so the character
/// radix tree counts tokens, not digits.
fn one_char_per_token(ids: &[u32]) -> String {
    ids.iter()
        .map(|&id| char::from_u32(0x40000 + id).expect("id fits in planes 4-7"))
        .collect()
}

/// The tagged decimal encoding proposed in PR #237: a record-separator tag,
/// then every id terminated by a unit separator.
fn digit_tagged(ids: &[u32]) -> String {
    let mut s = String::with_capacity(2 + ids.len() * 7);
    s.push('\u{1e}');
    for id in ids {
        s.push_str(&id.to_string());
        s.push('\u{1f}');
    }
    s
}

/// Plain text of roughly the same token count (about four bytes per token):
/// the first `shared` tokens' worth of bytes is identical across sequences
/// and the rest is seeded per sequence, mirroring `synthetic_ids`.
fn raw_text(seq: u64, n_tokens: usize, shared: usize) -> String {
    let mut s = seeded_text(SEED, shared * 4);
    s.push_str(&seeded_text(SEED ^ (seq + 1), (n_tokens - shared) * 4));
    s
}

fn bench_key_format(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_aware_key_format");
    let tenant = "http://127.0.0.1:30000";
    for n in TOKEN_COUNTS {
        let shared = n / 2;
        let variants: [(&str, KeyFn); 3] = [
            ("raw_text", Box::new(move |seq| raw_text(seq, n, shared))),
            (
                "one_char_per_token",
                Box::new(move |seq| one_char_per_token(&synthetic_ids(seq, n, shared))),
            ),
            (
                "digit_tagged",
                Box::new(move |seq| digit_tagged(&synthetic_ids(seq, n, shared))),
            ),
        ];
        for (name, make_key) in variants.iter() {
            let keys: Vec<String> = (0..64).map(make_key).collect();
            let key_bytes: usize = keys.iter().map(String::len).sum::<usize>() / keys.len();
            let key_chars: usize =
                keys.iter().map(|k| k.chars().count()).sum::<usize>() / keys.len();
            eprintln!("{name}/{n} tokens: key ~{key_bytes} bytes, ~{key_chars} chars");

            // insert: 64 distinct keys into an empty tree per iteration, so
            // throughput counts every inserted token. The tree is emptied with
            // `remove_tenant` outside the timed region rather than dropped:
            // `Node` holds a strong reference to its parent, so dropping a
            // populated `Tree` never frees its nodes, and a fresh tree per
            // iteration grows memory until the process is killed.
            group.throughput(Throughput::Elements((keys.len() * n) as u64));
            group.bench_with_input(
                BenchmarkId::new(format!("{name}/insert"), n),
                &keys,
                |b, keys| {
                    let tree = Tree::new();
                    b.iter_custom(|iters| {
                        let mut total = Duration::ZERO;
                        for _ in 0..iters {
                            let t0 = Instant::now();
                            for k in keys {
                                tree.insert(k, tenant);
                            }
                            total += t0.elapsed();
                            tree.remove_tenant(tenant);
                        }
                        total
                    });
                },
            );

            // prefix_match: a tree warmed with the 64 keys, matching a key that
            // shares the first half with them and diverges after.
            let tree = Tree::new();
            for k in &keys {
                tree.insert(k, tenant);
            }
            let probe = make_key(1_000);
            group.throughput(Throughput::Elements(n as u64));
            group.bench_with_input(
                BenchmarkId::new(format!("{name}/prefix_match"), n),
                &probe,
                |b, probe| {
                    b.iter(|| black_box(tree.prefix_match_with_counts(probe)));
                },
            );
        }
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_extract_text,
    bench_policy_select,
    bench_key_format
);
criterion_main!(benches);
