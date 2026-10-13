//! Offline gRPC frontend benchmark.
//! cargo test --release --test grpc_frontend_bench -- --ignored --nocapture
#![cfg(unix)]

#[path = "common/grpc_frontend_fixture.rs"]
mod fixture;
#[path = "common/mock_vllm_rs.rs"]
mod mock;

use std::sync::Arc;
use std::time::{Duration, Instant};

use fixture::{chat_request, model_fixture};
use mock::{MockVllmRs, MockVllmRsServer};
use serde_json::json;
use tokio::sync::Barrier;
use vllm_router_rs::backend::EngineFrontend;

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
