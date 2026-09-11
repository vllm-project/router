//! Benchmarks for the exact-match tokenizer cache (`CachedTokenizer`).
//!
//! Compares cached `encode()` against the uncached HuggingFace tokenizer for
//! pure hits, pure misses, mixed hit ratios and concurrent hits, and prints
//! the estimated cache memory per prompt size.
//!
//! Run with: `cargo bench --bench tokenizer_cache_benchmark`
//!
//! The tokenizer is TinyLlama/TinyLlama-1.1B-Chat-v1.0 `tokenizer.json`,
//! fetched into `.tokenizer_cache/` by `common::ensure_tokenizer_cached` on
//! first use. Reported numbers were taken with the file whose SHA-256 is
//! `bcd04f0eadf90287bd26e1a183ac487d8a141b09b06aecb7725bbdd343640f2e`.
//!
//! Timing boundaries: every `encode` group drops the returned `Encoding`
//! outside the timed region (`iter_with_large_drop` / `iter_batched_ref`),
//! and the miss case also builds and drops its fresh cache outside it, so
//! the numbers are per-call costs, not cache lifecycle costs.

use criterion::{
    black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput,
};
use std::path::PathBuf;
use std::sync::{Arc, OnceLock};
use std::thread;
use std::time::Instant;
use vllm_router_rs::tokenizer::{
    cache::{estimate_entry_bytes, CachedTokenizer, TokenizerCacheConfig},
    huggingface::HuggingFaceTokenizer,
    traits::*,
};

#[path = "../tests/common/mod.rs"]
mod common;
use common::ensure_tokenizer_cached;

static TOKENIZER_PATH: OnceLock<PathBuf> = OnceLock::new();

fn tokenizer_path() -> &'static PathBuf {
    TOKENIZER_PATH.get_or_init(ensure_tokenizer_cached)
}

fn load_tokenizer() -> Arc<dyn Tokenizer> {
    Arc::new(
        HuggingFaceTokenizer::from_file(tokenizer_path().to_str().unwrap())
            .expect("Failed to load tokenizer"),
    )
}

const SHORT_PROMPT: &str = "What is the capital of France?";
const MEDIUM_PROMPT: &str = "Write a detailed explanation of quantum computing, including its principles, current applications, and future potential. Be sure to cover both the theoretical foundations and practical implementations.";
const LONG_PROMPT: &str = "You are an expert software engineer. Review the following code and provide detailed feedback on performance optimizations, potential bugs, and architectural improvements. Consider scalability, maintainability, and best practices. The code implements a distributed caching system with the following requirements: 1) High availability across multiple regions, 2) Sub-millisecond latency for cache hits, 3) Automatic failover and recovery, 4) Support for both LRU and LFU eviction policies, 5) Real-time monitoring and alerting. Please analyze each component thoroughly and suggest concrete improvements with code examples where appropriate.";

fn generate_system_prompt(size: usize) -> String {
    let domains = [
        "mathematics",
        "physics",
        "chemistry",
        "biology",
        "computer science",
        "engineering",
        "medicine",
        "law",
        "economics",
        "philosophy",
    ];
    let mut prompt = String::from("You are a helpful AI assistant with expertise in ");
    let mut i = 0;
    while prompt.len() < size {
        prompt.push_str(domains[i % domains.len()]);
        prompt.push_str(", ");
        i += 1;
    }
    prompt
}

fn prompts() -> Vec<(&'static str, String)> {
    vec![
        ("short_30B", SHORT_PROMPT.to_string()),
        ("medium_230B", MEDIUM_PROMPT.to_string()),
        ("long_670B", LONG_PROMPT.to_string()),
        ("system_4KB", generate_system_prompt(4000)),
        ("system_16KB", generate_system_prompt(16000)),
    ]
}

fn roomy_config() -> TokenizerCacheConfig {
    TokenizerCacheConfig {
        max_entries: 4096,
        max_bytes: 1 << 30,
        max_entry_bytes: 1 << 30,
    }
}

/// Print the estimated retained bytes for each prompt size once.
fn report_memory(inner: &Arc<dyn Tokenizer>) {
    println!("\nEstimated cache memory per entry (TinyLlama tokenizer):");
    println!(
        "{:<14} {:>10} {:>8} {:>14} {:>12}",
        "prompt", "input_B", "tokens", "estimate_B", "B/token"
    );
    for (name, prompt) in prompts() {
        let encoding = inner.encode(&prompt).unwrap();
        let tokens = encoding.token_ids().len();
        let estimate = estimate_entry_bytes(&prompt, &encoding);
        println!(
            "{:<14} {:>10} {:>8} {:>14} {:>12.1}",
            name,
            prompt.len(),
            tokens,
            estimate,
            estimate as f64 / tokens.max(1) as f64
        );
    }
    println!();
}

/// Uncached encode, cache hit, the `Encoding` clone a hit pays for, and a
/// cache miss into an empty cache, per prompt size. Returned encodings and
/// the per-miss caches are dropped outside the timed region.
fn bench_encode_paths(c: &mut Criterion) {
    let inner = load_tokenizer();
    report_memory(&inner);
    let cached = CachedTokenizer::new(inner.clone(), roomy_config()).unwrap();

    let mut group = c.benchmark_group("tokenizer_cache/encode");
    for (name, prompt) in prompts() {
        let encoding = inner.encode(&prompt).unwrap();
        cached.encode(&prompt).unwrap();
        group.throughput(Throughput::Bytes(prompt.len() as u64));

        group.bench_with_input(BenchmarkId::new("uncached", name), &prompt, |b, p| {
            b.iter_with_large_drop(|| inner.encode(black_box(p)).unwrap())
        });
        group.bench_with_input(BenchmarkId::new("hit", name), &prompt, |b, p| {
            b.iter_with_large_drop(|| cached.encode(black_box(p)).unwrap())
        });
        group.bench_with_input(BenchmarkId::new("clone_only", name), &encoding, |b, e| {
            b.iter_with_large_drop(|| black_box(e).clone())
        });
        group.bench_with_input(BenchmarkId::new("miss", name), &prompt, |b, p| {
            b.iter_batched_ref(
                || {
                    CachedTokenizer::new(
                        inner.clone(),
                        TokenizerCacheConfig {
                            max_entries: 16,
                            ..roomy_config()
                        },
                    )
                    .unwrap()
                },
                |cache| cache.encode(black_box(p)).unwrap(),
                BatchSize::SmallInput,
            )
        });
    }
    group.finish();
}

/// A hot set of 64 medium prompts served at a fixed hit ratio, with cold
/// prompts unique per call, against the same sequence uncached.
fn bench_mixed_workload(c: &mut Criterion) {
    let inner = load_tokenizer();
    let hot: Vec<String> = (0..64)
        .map(|i| format!("{MEDIUM_PROMPT} (variant {i})"))
        .collect();

    let mut group = c.benchmark_group("tokenizer_cache/mixed");
    group.throughput(Throughput::Elements(1));
    for hit_pct in [50u64, 90, 99] {
        let label = format!("hit{hit_pct}pct");

        group.bench_with_input(
            BenchmarkId::new("cached", &label),
            &hit_pct,
            |b, &hit_pct| {
                b.iter_custom(|iters| {
                    let cache = CachedTokenizer::new(
                        inner.clone(),
                        TokenizerCacheConfig {
                            max_entries: 1024,
                            max_bytes: 64 << 20,
                            max_entry_bytes: 1 << 20,
                        },
                    )
                    .unwrap();
                    for prompt in &hot {
                        cache.encode(prompt).unwrap();
                    }
                    let mut cold = 0u64;
                    let start = Instant::now();
                    for i in 0..iters {
                        if i % 100 < hit_pct {
                            black_box(cache.encode(&hot[i as usize % hot.len()]).unwrap());
                        } else {
                            cold += 1;
                            let prompt = format!("{MEDIUM_PROMPT} (cold {cold})");
                            black_box(cache.encode(&prompt).unwrap());
                        }
                    }
                    start.elapsed()
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("uncached", &label),
            &hit_pct,
            |b, &hit_pct| {
                b.iter_custom(|iters| {
                    let mut cold = 0u64;
                    let start = Instant::now();
                    for i in 0..iters {
                        if i % 100 < hit_pct {
                            black_box(inner.encode(&hot[i as usize % hot.len()]).unwrap());
                        } else {
                            cold += 1;
                            let prompt = format!("{MEDIUM_PROMPT} (cold {cold})");
                            black_box(inner.encode(&prompt).unwrap());
                        }
                    }
                    start.elapsed()
                })
            },
        );
    }
    group.finish();
}

/// All-hit encodes spread across N threads sharing one cache. Reported time
/// is wall clock per encode across all threads, so flat or rising numbers
/// with more threads indicate lock contention.
fn bench_concurrent_hits(c: &mut Criterion) {
    let inner = load_tokenizer();
    let cache = Arc::new(CachedTokenizer::new(inner.clone(), roomy_config()).unwrap());
    let hot: Arc<Vec<String>> = Arc::new(
        (0..16)
            .map(|i| format!("{SHORT_PROMPT} (variant {i})"))
            .collect(),
    );
    for prompt in hot.iter() {
        cache.encode(prompt).unwrap();
    }

    let mut group = c.benchmark_group("tokenizer_cache/concurrent_short");
    group.throughput(Throughput::Elements(1));
    for threads in [1usize, 2, 4, 8] {
        group.bench_with_input(BenchmarkId::new("hit", threads), &threads, |b, &threads| {
            b.iter_custom(|iters| {
                let per_thread = iters.div_ceil(threads as u64);
                let start = Instant::now();
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let cache = cache.clone();
                        let hot = hot.clone();
                        thread::spawn(move || {
                            for i in 0..per_thread as usize {
                                black_box(cache.encode(&hot[(i + t) % hot.len()]).unwrap());
                            }
                        })
                    })
                    .collect();
                for handle in handles {
                    handle.join().unwrap();
                }
                start.elapsed()
            })
        });
        group.bench_with_input(
            BenchmarkId::new("uncached", threads),
            &threads,
            |b, &threads| {
                b.iter_custom(|iters| {
                    let per_thread = iters.div_ceil(threads as u64);
                    let start = Instant::now();
                    let handles: Vec<_> = (0..threads)
                        .map(|t| {
                            let inner = inner.clone();
                            let hot = hot.clone();
                            thread::spawn(move || {
                                for i in 0..per_thread as usize {
                                    black_box(inner.encode(&hot[(i + t) % hot.len()]).unwrap());
                                }
                            })
                        })
                        .collect();
                    for handle in handles {
                        handle.join().unwrap();
                    }
                    start.elapsed()
                })
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_encode_paths,
    bench_mixed_workload,
    bench_concurrent_hits
);
criterion_main!(benches);
