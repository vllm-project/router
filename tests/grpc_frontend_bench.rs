//! Offline gRPC frontend benchmark.
//!
//! ```text
//! cargo test --release --test grpc_frontend_bench grpc_frontend_l0_overhead -- --ignored --nocapture
//! cargo test --release --test grpc_frontend_bench grpc_frontend_l1_overhead -- --ignored --nocapture
//! ```
#![cfg(unix)]

#[path = "common/grpc_frontend_fixture.rs"]
mod fixture;
#[path = "common/mock_vllm_rs.rs"]
mod mock;

use std::sync::Arc;
use std::time::{Duration, Instant};

use fixture::{chat_request, model_fixture, qwen_model_fixture};
use mock::{MockVllmRs, MockVllmRsServer};
use serde_json::{json, Value};
use tokio::sync::Barrier;
use vllm_router_rs::backend::{EngineFrontend, L1CacheConfig};

fn settings(name: &str, default: &str) -> Vec<usize> {
    std::env::var(name)
        .unwrap_or_else(|_| default.into())
        .split(',')
        .map(|s| {
            s.trim()
                .parse()
                .expect("positive integer benchmark setting")
        })
        .inspect(|n| assert!(*n > 0))
        .collect()
}

fn prompt(size: usize, index: usize) -> String {
    let prefix = format!("request {index:016x}: ");
    let mut text = prefix;
    while text.len() < size {
        text.push_str("the quick brown fox jumps over the lazy dog. ");
    }
    text.truncate(size);
    text
}

fn cpu_seconds() -> f64 {
    // SAFETY: usage is a valid writable rusage pointer.
    let usage = unsafe {
        let mut usage: libc::rusage = std::mem::zeroed();
        assert_eq!(libc::getrusage(libc::RUSAGE_SELF, &mut usage), 0);
        usage
    };
    usage.ru_utime.tv_sec as f64
        + usage.ru_utime.tv_usec as f64 / 1e6
        + usage.ru_stime.tv_sec as f64
        + usage.ru_stime.tv_usec as f64 / 1e6
}

fn percentile(sorted: &[f64], fraction: f64) -> f64 {
    sorted[((sorted.len() - 1) as f64 * fraction).round() as usize]
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
#[ignore = "offline performance measurement; run in release mode"]
async fn grpc_frontend_l0_overhead() {
    let sizes = settings("VLLM_ROUTER_BENCH_SIZES", "200,16384,131072");
    let concurrencies = settings("VLLM_ROUTER_BENCH_CONCURRENCY", "1,8");
    let seconds = settings("VLLM_ROUTER_BENCH_MEASURE_SECS", "2")[0];
    let model = Arc::new(model_fixture());
    let worker = MockVllmRsServer::spawn_with_state(MockVllmRs {
        capture_limit: 0,
        ..Default::default()
    })
    .await;
    for size in sizes {
        assert!(size >= 32, "prompt must fit its unique request marker");
        for &concurrency in &concurrencies {
            for corpus in ["hot64", "cold"] {
                for enabled in [false, true] {
                    let frontend = Arc::new(if enabled {
                        EngineFrontend::with_tokenizer_cache(
                            Duration::from_secs(30),
                            Default::default(),
                        )
                        .unwrap()
                    } else {
                        EngineFrontend::new()
                    });
                    // Warm up outside measurement; cold requests use different IDs.
                    for i in 0..64 {
                        let request = chat_request(&model, &prompt(size, i), false);
                        let prepared = frontend.prepare(request).await.unwrap();
                        let response = frontend.dispatch(&worker.grpc_url, prepared).await;
                        assert!(response.status().is_success());
                        axum::body::to_bytes(response.into_body(), usize::MAX)
                            .await
                            .unwrap();
                    }
                    let before = frontend.tokenizer_cache_stats().unwrap_or_default();
                    let barrier = Arc::new(Barrier::new(concurrency + 1));
                    let mut tasks = tokio::task::JoinSet::new();
                    for t in 0..concurrency {
                        let (frontend, model, barrier, url) = (
                            frontend.clone(),
                            model.clone(),
                            barrier.clone(),
                            worker.grpc_url.clone(),
                        );
                        tasks.spawn(async move {
                            let mut latency = Vec::new();
                            let mut encoding = Vec::new();
                            barrier.wait().await;
                            let start = Instant::now();
                            let mut i = t;
                            while start.elapsed() < Duration::from_secs(seconds as u64) {
                                let index = if corpus == "hot64" { i % 64 } else { 64 + i };
                                let request = chat_request(&model, &prompt(size, index), false);
                                let t0 = Instant::now();
                                let prepared = frontend.prepare(request).await.unwrap();
                                encoding.push(prepared.tokenized.encode_ms * 1000.0);
                                let response = frontend.dispatch(&url, prepared).await;
                                assert!(response.status().is_success());
                                axum::body::to_bytes(response.into_body(), usize::MAX)
                                    .await
                                    .unwrap();
                                latency.push(t0.elapsed().as_secs_f64() * 1e6);
                                i += concurrency;
                            }
                            (latency, encoding)
                        });
                    }
                    let cpu_start = cpu_seconds();
                    let wall_start = Instant::now();
                    barrier.wait().await;
                    let (mut latency, mut encoding) = (Vec::new(), Vec::new());
                    while let Some(result) = tasks.join_next().await {
                        let (l, e) = result.unwrap();
                        latency.extend(l);
                        encoding.extend(e);
                    }
                    let wall = wall_start.elapsed().as_secs_f64();
                    let cpu = cpu_seconds() - cpu_start;
                    latency.sort_by(f64::total_cmp);
                    encoding.sort_by(f64::total_cmp);
                    let stats = frontend.tokenizer_cache_stats().unwrap_or_default();
                    let requests = latency.len();
                    assert!(requests > 0);
                    if enabled {
                        assert_eq!(
                            stats.hits + stats.misses - before.hits - before.misses,
                            requests as u64
                        );
                        if corpus == "hot64" {
                            assert_eq!(stats.misses, before.misses);
                        } else {
                            assert_eq!(stats.hits, before.hits);
                        }
                    }
                    println!(
                        "{}",
                        json!({
                            "benchmark": "grpc_frontend_l0", "corpus": corpus, "input_bytes": size,
                            "concurrency": concurrency, "l0": enabled, "requests": requests,
                            "wall_s": wall, "requests_per_s": requests as f64 / wall,
                            "latency_p50_us": percentile(&latency, 0.5), "latency_p99_us": percentile(&latency, 0.99),
                            "encode_p50_us": percentile(&encoding, 0.5), "encode_p99_us": percentile(&encoding, 0.99),
                            "cpu_s": cpu, "cpu_us_per_request": cpu * 1e6 / requests as f64,
                            "cache_bytes": stats.bytes, "cache_entries": stats.entries,
                            "hits": stats.hits - before.hits, "misses": stats.misses - before.misses,
                        })
                    );
                }
            }
        }
    }
}

fn reuse_targets(default: &str) -> Vec<f64> {
    std::env::var("VLLM_ROUTER_BENCH_REUSE")
        .unwrap_or_else(|_| default.into())
        .split(',')
        .map(|s| {
            let pct: f64 = s.trim().parse().expect("reuse percent");
            assert!(
                (0.0..=100.0).contains(&pct),
                "reuse percent must be 0..=100"
            );
            pct
        })
        .collect()
}

fn chat_messages(
    model: &tempfile::TempDir,
    messages: &[Value],
) -> vllm_router_rs::protocols::spec::ChatCompletionRequest {
    serde_json::from_value(json!({
        "model": model.path().to_str().unwrap(),
        "messages": messages,
        "max_tokens": 8,
        "stream": false,
    }))
    .unwrap()
}

async fn prompt_tokens(
    frontend: &EngineFrontend,
    model: &tempfile::TempDir,
    messages: &[Value],
) -> usize {
    frontend
        .prepare(chat_messages(model, messages))
        .await
        .unwrap()
        .tokenized
        .token_ids
        .len()
}

const TURN: &str = "The function reads the file and returns the matching line. ";

/// Shared chat history plus one unique user turn.
///
/// This is the CPU-only shape of the 128k run: every request renders the same
/// earlier turns, and only the last user message changes. `notes` is how many
/// times `" note"` was appended so the rendered prompt lands on `target`
/// tokens. A 0% target uses a unique system message instead of `shared`.
struct L1Prompt {
    shared: Vec<Value>,
    notes: usize,
    unique_system: bool,
    /// User text before the `" note"` padding. Short when the tail budget is
    /// smaller than the full instruction sentence.
    stem: String,
}

impl L1Prompt {
    fn messages(&self, index: usize) -> Vec<Value> {
        let mut messages = if self.unique_system {
            vec![json!({
                "role": "system",
                "content": format!("session {index:08x} agent"),
            })]
        } else {
            self.shared.clone()
        };
        let mut content = self.stem.replace("{index}", &format!("{index:08x}"));
        for _ in 0..self.notes {
            content.push_str(" note");
        }
        messages.push(json!({"role": "user", "content": content}));
        messages
    }
}

async fn build_l1_prompt(
    sizer: &EngineFrontend,
    model: &tempfile::TempDir,
    target: usize,
    reuse_pct: f64,
) -> L1Prompt {
    assert!(target >= 32, "prompt must be at least 32 tokens");
    // Reserve the shortest unique user turn so a 99.99% target does not
    // build a full-length prefix and then append another turn on top.
    let marker_tail = prompt_tokens(
        sizer,
        model,
        &[
            json!({"role": "system", "content": "You are a repository agent. Answer from the files below."}),
            json!({"role": "user", "content": "q00000000"}),
        ],
    )
    .await;
    let marker_base = prompt_tokens(
        sizer,
        model,
        &[json!({"role": "system", "content": "You are a repository agent. Answer from the files below."})],
    )
    .await;
    let min_tail = marker_tail.saturating_sub(marker_base).max(1);
    let shared_goal = ((target as f64) * (reuse_pct / 100.0)).round() as usize;
    let shared_goal = shared_goal.min(target.saturating_sub(min_tail));
    if shared_goal == 0 {
        let prompt = L1Prompt {
            shared: Vec::new(),
            notes: 0,
            unique_system: true,
            stem: "s{index} measured: name the function.".to_string(),
        };
        let notes = pad_notes(sizer, model, &prompt, target).await;
        return L1Prompt { notes, ..prompt };
    }

    let mut shared = vec![json!({
        "role": "system",
        "content": "You are a repository agent. Answer from the files below.",
    })];
    let mut turn = 0usize;
    while prompt_tokens(sizer, model, &shared).await < shared_goal {
        shared.push(json!({"role": "user", "content": format!("question {turn}: {TURN}")}));
        shared.push(json!({"role": "assistant", "content": format!("answer {turn}: {TURN}")}));
        turn += 1;
        assert!(
            turn < 100_000,
            "shared history did not reach {shared_goal} tokens"
        );
    }
    shrink_last_message(sizer, model, &mut shared, shared_goal).await;

    // The GPU run's tail is a short instruction padded with " note". When that
    // sentence is longer than the uncached budget, keep the same turn shape
    // and use a fixed-width marker so 99.99% can still be a short tail.
    let sentence = "s{index} measured: name the function.";
    let mut prompt = L1Prompt {
        shared,
        notes: 0,
        unique_system: false,
        stem: sentence.to_string(),
    };
    let with_sentence = prompt_tokens(sizer, model, &prompt.messages(0)).await;
    if with_sentence > target {
        prompt.stem = "q{index}".to_string();
    }
    let notes = pad_notes(sizer, model, &prompt, target).await;
    L1Prompt { notes, ..prompt }
}

/// Drop characters off the last message until the shared chat is at or under `goal`.
async fn shrink_last_message(
    sizer: &EngineFrontend,
    model: &tempfile::TempDir,
    messages: &mut [Value],
    goal: usize,
) {
    if prompt_tokens(sizer, model, messages).await <= goal {
        return;
    }
    let last = messages.last().unwrap()["content"]
        .as_str()
        .unwrap()
        .to_string();
    let mut lo = 1usize;
    let mut hi = last.len();
    let mut best = 1usize;
    while lo <= hi {
        let mid = (lo + hi) / 2;
        messages.last_mut().unwrap()["content"] = json!(&last[..mid]);
        if prompt_tokens(sizer, model, messages).await <= goal {
            best = mid;
            lo = mid + 1;
        } else if mid == 0 {
            break;
        } else {
            hi = mid - 1;
        }
    }
    messages.last_mut().unwrap()["content"] = json!(&last[..best]);
}

/// Smallest `" note"` repeat count whose rendered prompt has at least `target` tokens.
async fn pad_notes(
    sizer: &EngineFrontend,
    model: &tempfile::TempDir,
    prompt: &L1Prompt,
    target: usize,
) -> usize {
    if prompt_tokens(sizer, model, &prompt.messages(0)).await >= target {
        return 0;
    }
    let mut lo = 0usize;
    let mut hi = target;
    while lo < hi {
        let mid = (lo + hi) / 2;
        let trial = L1Prompt {
            notes: mid,
            ..prompt.clone_parts()
        };
        if prompt_tokens(sizer, model, &trial.messages(0)).await < target {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

impl L1Prompt {
    fn clone_parts(&self) -> Self {
        Self {
            shared: self.shared.clone(),
            notes: self.notes,
            unique_system: self.unique_system,
            stem: self.stem.clone(),
        }
    }
}

fn l1_config() -> L1CacheConfig {
    L1CacheConfig {
        max_insert_bytes: 256 * 1024 * 1024,
        max_insert_segments: 1_000_000,
        ..Default::default()
    }
}

#[tokio::test]
async fn l1_bench_prompts_hit_the_requested_reuse() {
    let model = qwen_model_fixture();
    let sizer = EngineFrontend::new();
    // The new user turn's template is a few dozen tokens, so a literal
    // 0.01% tail needs a very long prompt. At 8192 the shortest turn is
    // still above 99.5%.
    for (reuse, tokens, low, high) in [
        (0.0, 512, 0.0, 2.0),
        (70.0, 512, 60.0, 80.0),
        (99.99, 8192, 99.5, 100.0),
    ] {
        let prompt = build_l1_prompt(&sizer, &model, tokens, reuse).await;
        let l1 =
            EngineFrontend::with_encoding_caches(Duration::from_secs(30), None, Some(l1_config()))
                .unwrap();
        let warm = l1
            .prepare(chat_messages(&model, &prompt.messages(0)))
            .await
            .unwrap();
        assert!(!warm.tokenized.token_ids.is_empty());
        let measured = l1
            .prepare(chat_messages(&model, &prompt.messages(1)))
            .await
            .unwrap();
        let total = measured.tokenized.token_ids.len();
        let pct = 100.0 * measured.tokenized.cached_tokens as f64 / total as f64;
        assert!(
            total + 8 >= tokens,
            "target {tokens} rendered {total} tokens at reuse {reuse}"
        );
        assert!(
            (low..high).contains(&pct),
            "reuse target {reuse} measured {pct:.2} ({}/{})",
            measured.tokenized.cached_tokens,
            total
        );
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
#[ignore = "offline performance measurement; run in release mode"]
async fn grpc_frontend_l1_overhead() {
    let targets = settings("VLLM_ROUTER_BENCH_TOKENS", "2048,8192");
    let concurrencies = settings("VLLM_ROUTER_BENCH_CONCURRENCY", "1,4,8,16,32,64");
    let seconds = settings("VLLM_ROUTER_BENCH_MEASURE_SECS", "2")[0];
    let reuses = reuse_targets("0,30,70,99.99");
    let model = Arc::new(qwen_model_fixture());
    let sizer = EngineFrontend::new();
    let worker = MockVllmRsServer::spawn_with_state(MockVllmRs {
        capture_limit: 0,
        ..Default::default()
    })
    .await;
    for target in targets {
        for &reuse in &reuses {
            let prompt = Arc::new(build_l1_prompt(&sizer, &model, target, reuse).await);
            let sample = sizer
                .prepare(chat_messages(&model, &prompt.messages(0)))
                .await
                .unwrap();
            let prompt_tokens = sample.tokenized.token_ids.len();
            for &concurrency in &concurrencies {
                for enabled in [false, true] {
                    let frontend = Arc::new(if enabled {
                        EngineFrontend::with_encoding_caches(
                            Duration::from_secs(30),
                            None,
                            Some(l1_config()),
                        )
                        .unwrap()
                    } else {
                        EngineFrontend::new()
                    });
                    // One warmup stores the shared prefix. Measured tails differ.
                    let request = chat_messages(&model, &prompt.messages(0));
                    let prepared = frontend.prepare(request).await.unwrap();
                    let response = frontend.dispatch(&worker.grpc_url, prepared).await;
                    assert!(response.status().is_success());
                    axum::body::to_bytes(response.into_body(), usize::MAX)
                        .await
                        .unwrap();
                    let before = frontend.l1_cache_stats();
                    let barrier = Arc::new(Barrier::new(concurrency + 1));
                    let mut tasks = tokio::task::JoinSet::new();
                    for t in 0..concurrency {
                        let (frontend, model, prompt, barrier, url) = (
                            frontend.clone(),
                            model.clone(),
                            prompt.clone(),
                            barrier.clone(),
                            worker.grpc_url.clone(),
                        );
                        tasks.spawn(async move {
                            let mut latency = Vec::new();
                            let mut encoding = Vec::new();
                            let mut reuse = Vec::new();
                            barrier.wait().await;
                            let start = Instant::now();
                            let mut i = t + concurrency;
                            while start.elapsed() < Duration::from_secs(seconds as u64) {
                                let request = chat_messages(&model, &prompt.messages(i));
                                let t0 = Instant::now();
                                let prepared = frontend.prepare(request).await.unwrap();
                                let tokens = prepared.tokenized.token_ids.len().max(1);
                                encoding.push(prepared.tokenized.encode_ms * 1000.0);
                                reuse.push(
                                    100.0 * prepared.tokenized.cached_tokens as f64 / tokens as f64,
                                );
                                let response = frontend.dispatch(&url, prepared).await;
                                assert!(response.status().is_success());
                                axum::body::to_bytes(response.into_body(), usize::MAX)
                                    .await
                                    .unwrap();
                                latency.push(t0.elapsed().as_secs_f64() * 1e6);
                                i += concurrency;
                            }
                            (latency, encoding, reuse)
                        });
                    }
                    let cpu_start = cpu_seconds();
                    let wall_start = Instant::now();
                    barrier.wait().await;
                    let (mut latency, mut encoding, mut reuse_samples) =
                        (Vec::new(), Vec::new(), Vec::new());
                    while let Some(result) = tasks.join_next().await {
                        let (l, e, r) = result.unwrap();
                        latency.extend(l);
                        encoding.extend(e);
                        reuse_samples.extend(r);
                    }
                    let wall = wall_start.elapsed().as_secs_f64();
                    let cpu = cpu_seconds() - cpu_start;
                    latency.sort_by(f64::total_cmp);
                    encoding.sort_by(f64::total_cmp);
                    let requests = latency.len();
                    assert!(requests > 0);
                    let reuse_pct = reuse_samples.iter().sum::<f64>() / requests as f64;
                    let stats = frontend.l1_cache_stats();
                    let (hits, lookups, cache_bytes, cache_nodes) = match (enabled, stats, before) {
                        (true, Some(stats), Some(before)) => (
                            stats.hits - before.hits,
                            stats.lookups - before.lookups,
                            stats.bytes,
                            stats.nodes,
                        ),
                        _ => (0, 0, 0, 0),
                    };
                    println!(
                        "{}",
                        json!({
                            "benchmark": "grpc_frontend_l1",
                            "reuse_target_pct": reuse,
                            "reuse_pct": reuse_pct,
                            "prompt_tokens": prompt_tokens,
                            "concurrency": concurrency,
                            "l1": enabled,
                            "requests": requests,
                            "wall_s": wall,
                            "requests_per_s": requests as f64 / wall,
                            "latency_p50_us": percentile(&latency, 0.5),
                            "latency_p99_us": percentile(&latency, 0.99),
                            "encode_p50_us": percentile(&encoding, 0.5),
                            "encode_p99_us": percentile(&encoding, 0.99),
                            "cpu_s": cpu,
                            "cpu_us_per_request": cpu * 1e6 / requests as f64,
                            "cache_bytes": cache_bytes,
                            "cache_nodes": cache_nodes,
                            "hits": hits,
                            "lookups": lookups,
                        })
                    );
                }
            }
        }
    }
}
