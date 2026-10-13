//! CPU cost of routing inputs, before tokenizer integration (#244).
//!
//! Groups:
//! - `routing_key`: extract routing text from completion and chat requests.
//! - `policy`: select a worker from a warmed hot set.
//! - `cache_aware_key_format`: compare tree key encodings at equal token counts.
//! - `edge/cache_aware`: long keys, match thresholds, load balance and UTF-8.
//! - `edge/long_input`: parse, extract and serialize long request bodies.
//! - `edge/rendezvous`: long prompts with and without session headers.
//! - `edge/token_ids`: parse and extract pre-tokenized prompts.
//! - `edge/workers`: selection across 1 to 64 workers.
//!
//! Inputs are deterministic and offline. Edge fixtures are shared with CI
//! tests; cache-aware cases also assert their expected branch before timing.
//! Returned routing strings are dropped inside the timed call. Keeping a
//! whole sample's strings alive would require tens of GB.
//!
//! Run: `cargo bench --bench routing_input` (append `-- edge/` for edge cases).
//! See `docs/benchmarks/router_overhead.md` for the method and group details.

use axum::Json;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::{Duration, Instant};
use vllm_router_rs::policies::{
    CacheAwareConfig, CacheAwarePolicy, LoadBalancingPolicy, RendezvousHashPolicy,
};
use vllm_router_rs::protocols::spec::{
    ChatCompletionRequest, CompletionRequest, GenerationRequest,
};
use vllm_router_rs::tree::Tree;

#[path = "../tests/common/mod.rs"]
mod common;
use common::bench_corpus::{
    chat, completion_ids, completion_text, seeded_ids, seeded_text, seeded_utf8_text,
    to_chat_request, to_completion_request, Corpus, CorpusKind, LONG_SIZES, PROMPT_ID_COUNTS, SEED,
    SIZES,
};
use common::routing_edge::{
    cache_aware_cases, json_like_prompt, rendezvous_pair, session_headers, size_label, workers,
    CacheAwareFixture, WORKER_COUNTS,
};

/// A routing-key builder for one candidate encoding.
type KeyFn = Box<dyn Fn(u64) -> String>;

const WORKER_COUNT: usize = 4;
const TOKEN_COUNTS: [usize; 4] = [128, 512, 2048, 8192];
const VOCAB: u32 = 32_000;
/// Inputs at or above this size get criterion's minimum sample count.
const LARGE_INPUT_BYTES: usize = 512 * 1024;

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
    let workers = workers(WORKER_COUNT);
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

            // Time all 64 inserts. Prune outside the timer to break the
            // parent/child Arc cycles that dropping a populated tree would retain.
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

/// Use fewer samples for large inputs.
fn edge_timing<M: criterion::measurement::Measurement>(
    group: &mut criterion::BenchmarkGroup<'_, M>,
    input_bytes: usize,
) {
    group
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(5))
        .sample_size(if input_bytes >= LARGE_INPUT_BYTES {
            10
        } else {
            50
        });
}

fn bench_edge_cache_aware(c: &mut Criterion) {
    let mut group = c.benchmark_group("edge/cache_aware");
    for case in cache_aware_cases().into_iter().filter(|case| case.bench) {
        let built = case.build();
        assert_eq!(
            built.fixture.select(&built.probe),
            Some(case.expect),
            "{} must take its intended branch",
            case.name
        );
        edge_timing(&mut group, case.key_bytes);
        group.throughput(Throughput::Elements(1));
        group.bench_function(BenchmarkId::from_parameter(&case.name), |b| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    built.fixture.reset();
                    let t0 = Instant::now();
                    black_box(built.fixture.select(&built.probe));
                    total += t0.elapsed();
                }
                total
            });
        });
    }
    group.finish();
}

fn bench_edge_long_input(c: &mut Criterion) {
    let mut group = c.benchmark_group("edge/long_input");
    for size in LONG_SIZES {
        let label = size_label(size);
        edge_timing(&mut group, size);
        let prompt = seeded_text(SEED ^ size as u64, size);
        let completion_body = serde_json::to_vec(&completion_text(&prompt)).expect("json");
        let chat_body = serde_json::to_vec(&chat(None, &prompt)).expect("json");
        eprintln!(
            "long_input/{label}: prompt {size} bytes, completion body {} bytes, chat body {} bytes",
            completion_body.len(),
            chat_body.len()
        );
        let completion: CompletionRequest =
            serde_json::from_slice(&completion_body).expect("completion");
        let chat_req: ChatCompletionRequest = serde_json::from_slice(&chat_body).expect("chat");

        group.throughput(Throughput::Bytes(completion_body.len() as u64));
        group.bench_function(format!("deserialize/completion/{label}"), |b| {
            b.iter(|| {
                black_box(Json::<CompletionRequest>::from_bytes(&completion_body).expect("parse"))
            });
        });
        group.bench_function(format!("serialize/completion/{label}"), |b| {
            b.iter(|| black_box(serde_json::to_vec(&completion).expect("json")));
        });
        group.throughput(Throughput::Bytes(chat_body.len() as u64));
        group.bench_function(format!("deserialize/chat/{label}"), |b| {
            b.iter(|| {
                black_box(Json::<ChatCompletionRequest>::from_bytes(&chat_body).expect("parse"))
            });
        });
        group.bench_function(format!("serialize/chat/{label}"), |b| {
            b.iter(|| black_box(serde_json::to_vec(&chat_req).expect("json")));
        });

        // Include dropping the returned text, as in the request path.
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_function(format!("extract/completion/{label}"), |b| {
            b.iter(|| black_box(completion.extract_text_for_routing()));
        });
        group.throughput(Throughput::Elements(1));
        group.bench_function(format!("extract/chat/{label}"), |b| {
            b.iter(|| black_box(chat_req.extract_text_for_routing()));
        });

        // Equal byte sizes of CJK text expose character-dependent parsing cost.
        let utf8_prompt = seeded_utf8_text(SEED ^ size as u64, size);
        let utf8_completion = serde_json::to_vec(&completion_text(&utf8_prompt)).expect("json");
        let utf8_chat = serde_json::to_vec(&chat(None, &utf8_prompt)).expect("json");
        group.throughput(Throughput::Bytes(utf8_completion.len() as u64));
        group.bench_function(format!("deserialize/completion_utf8/{label}"), |b| {
            b.iter(|| {
                black_box(Json::<CompletionRequest>::from_bytes(&utf8_completion).expect("parse"))
            });
        });
        group.throughput(Throughput::Bytes(utf8_chat.len() as u64));
        group.bench_function(format!("deserialize/chat_utf8/{label}"), |b| {
            b.iter(|| {
                black_box(Json::<ChatCompletionRequest>::from_bytes(&utf8_chat).expect("parse"))
            });
        });
    }
    group.finish();
}

fn bench_edge_rendezvous(c: &mut Criterion) {
    let mut group = c.benchmark_group("edge/rendezvous");
    let policy = RendezvousHashPolicy::new();
    let workers = workers(WORKER_COUNT);
    let session = session_headers("edge-session");
    group.throughput(Throughput::Elements(1));
    for size in LONG_SIZES {
        let label = size_label(size);
        edge_timing(&mut group, size);
        // Reuse the pair checked by the session-header test.
        let (prompt, _) = rendezvous_pair(size);
        group.bench_function(format!("body/{label}"), |b| {
            b.iter(|| black_box(policy.select_worker_with_headers(&workers, Some(&prompt), None)));
        });
        group.bench_function(format!("header/{label}"), |b| {
            b.iter(|| {
                black_box(policy.select_worker_with_headers(
                    &workers,
                    Some(&prompt),
                    Some(&session),
                ))
            });
        });
    }
    for size in [16 * 1024, 1024 * 1024] {
        edge_timing(&mut group, size);
        let prompt = json_like_prompt(size);
        group.bench_function(format!("json_like_prompt/{}", size_label(size)), |b| {
            b.iter(|| black_box(policy.select_worker(&workers, Some(&prompt))));
        });
    }
    group.finish();
}

fn bench_edge_token_ids(c: &mut Criterion) {
    let mut group = c.benchmark_group("edge/token_ids");
    for n in PROMPT_ID_COUNTS {
        let body = serde_json::to_vec(&completion_ids(&seeded_ids(SEED, n))).expect("json");
        eprintln!("token_ids/ids{n}: body {} bytes", body.len());
        edge_timing(&mut group, body.len());
        group.throughput(Throughput::Bytes(body.len() as u64));
        group.bench_function(format!("deserialize/ids{n}"), |b| {
            b.iter(|| black_box(Json::<CompletionRequest>::from_bytes(&body).expect("parse")));
        });
        let req: CompletionRequest = serde_json::from_slice(&body).expect("parse");
        group.throughput(Throughput::Elements(1));
        group.bench_function(format!("extract/ids{n}"), |b| {
            b.iter(|| black_box(req.extract_text_for_routing()));
        });
    }
    group.finish();
}

fn bench_edge_workers(c: &mut Criterion) {
    let mut group = c.benchmark_group("edge/workers");
    let corpus = Corpus::new(CorpusKind::Hot64, 2048);
    let prompts: Vec<String> = (0..64).map(|i| corpus.prompt(i)).collect();
    edge_timing(&mut group, 2048);
    group.throughput(Throughput::Elements(1));
    let policy = RendezvousHashPolicy::new();
    for n in WORKER_COUNTS {
        let workers = workers(n);
        let mut i = 0usize;
        group.bench_function(format!("rendezvous/{n}"), |b| {
            b.iter(|| {
                let p = &prompts[i % prompts.len()];
                i += 1;
                black_box(policy.select_worker(&workers, Some(p)))
            });
        });

        // Repeated hits leave the tree unchanged, so no reset is needed.
        let fixture = CacheAwareFixture::round_robin(&prompts, n);
        for (k, p) in prompts.iter().enumerate() {
            assert_eq!(fixture.select(p), Some(k % n), "{n} workers, prompt {k}");
        }
        let mut j = 0usize;
        group.bench_function(format!("cache_aware/{n}"), |b| {
            b.iter(|| {
                let p = &prompts[j % prompts.len()];
                j += 1;
                black_box(fixture.select(p))
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_extract_text,
    bench_policy_select,
    bench_key_format,
    bench_edge_cache_aware,
    bench_edge_long_input,
    bench_edge_rendezvous,
    bench_edge_token_ids,
    bench_edge_workers
);
criterion_main!(benches);
