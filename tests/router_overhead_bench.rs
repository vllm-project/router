//! Router overhead harness: what the router adds on top of a worker that
//! does nothing, measured end to end on one machine.
//!
//! One `vllm-router` process (the real binary, release profile recommended)
//! is spawned per scenario cell in front of in-process mock workers that
//! answer instantly. A client drives the router, and the harness reports
//! client-observed latency, throughput, how the requests spread across the
//! workers, and the router process's own CPU time and peak RSS, taken per
//! process with `wait4(2)` after the process is stopped. The `direct_mock`
//! scenario drives a mock worker without a router and is the floor every
//! other row is read against.
//!
//! Numbers from this harness describe router cost only. Nothing here can
//! show a routing *benefit* (KV-cache hits, time to first token); that needs
//! real workers.
//!
//! The client and the mock workers share one 8-thread runtime in this
//! process; the router runs as a separate process with its default thread
//! count. On a machine with fewer spare cores than that, router rows include
//! CPU contention the `direct_mock` row does not have; the report records
//! the thread counts and the CPU this process used per cell so it is visible.
//!
//! Ignored by default. Run with:
//!
//! ```text
//! cargo test --release --test router_overhead_bench -- --ignored --nocapture
//! ```
//!
//! Knobs (environment variables, all optional):
//! - `VLLM_ROUTER_BENCH_SCENARIOS`: comma list of `direct_mock`,
//!   `completions_off`, `completions_rendezvous`, `chat_off` (default: all)
//! - `VLLM_ROUTER_BENCH_SIZES`: prompt bytes, default `200,2048,16384`
//! - `VLLM_ROUTER_BENCH_CORPORA`: `hot64,cold,mixed90,short_shared_prefix,
//!   long_shared_prefix`, default `hot64,cold`
//! - `VLLM_ROUTER_BENCH_CONCURRENCY`: default `1,64`
//! - `VLLM_ROUTER_BENCH_RATE`: target requests per second across all
//!   client tasks; `0` (default) is closed loop, each task sends the next
//!   request as soon as the previous response is read
//! - `VLLM_ROUTER_BENCH_REPEATS`: how many times to run the whole matrix,
//!   rotating the scenario order each time; default `1`
//! - `VLLM_ROUTER_BENCH_WARMUP_SECS` / `VLLM_ROUTER_BENCH_MEASURE_SECS`:
//!   default `5` / `20`
//! - `VLLM_ROUTER_BENCH_WORKERS`: mock workers behind the router, default `4`
//! - `VLLM_ROUTER_BENCH_ROUTER_CPUS`: Linux only; run the router under
//!   `taskset -c <list>` so only the router is pinned
//! - `VLLM_ROUTER_BENCH_OUT_DIR`: default `target/router_overhead`
//!
//! See `docs/benchmarks/router_overhead.md` for the method and the report
//! template.

#![cfg(unix)]

mod common;

use common::bench_corpus::{chat, completion_text, Corpus, CorpusKind, SIZES};
use common::bench_mock::BenchMockWorker;
use reqwest::header::CONTENT_TYPE;
use serde::Serialize;
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Worker threads of the runtime that hosts the client and the mock workers.
const HARNESS_WORKER_THREADS: usize = 8;
/// `--eviction-interval` passed to every router. Longer than any cell, so
/// the cache-aware tree is never evicted while a cell runs: rows for
/// non-repeating corpora include the tree's growth, deterministically.
const EVICTION_INTERVAL_SECS: u64 = 3600;
/// `--max-tree-size` passed to every router (the CLI default, made explicit).
const MAX_TREE_SIZE: usize = 67_108_864;
/// Attempts to start a router when its port was taken between pick and bind.
const ROUTER_START_ATTEMPTS: usize = 3;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Route {
    Completions,
    Chat,
}

impl Route {
    fn path(self) -> &'static str {
        match self {
            Route::Completions => "/v1/completions",
            Route::Chat => "/v1/chat/completions",
        }
    }

    fn body(self, prompt: &str) -> String {
        match self {
            Route::Completions => completion_text(prompt).to_string(),
            Route::Chat => chat(None, prompt).to_string(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Scenario {
    name: &'static str,
    /// `None` drives a mock worker directly (no router).
    policy: Option<&'static str>,
    route: Route,
}

const SCENARIOS: [Scenario; 4] = [
    Scenario {
        name: "direct_mock",
        policy: None,
        route: Route::Completions,
    },
    Scenario {
        name: "completions_off",
        policy: Some("cache_aware"),
        route: Route::Completions,
    },
    Scenario {
        name: "completions_rendezvous",
        policy: Some("rendezvous_hash"),
        route: Route::Completions,
    },
    Scenario {
        name: "chat_off",
        policy: Some("cache_aware"),
        route: Route::Chat,
    },
];

struct Config {
    scenarios: Vec<Scenario>,
    sizes: Vec<usize>,
    corpora: Vec<CorpusKind>,
    concurrency: Vec<usize>,
    rate: f64,
    repeats: usize,
    warmup: Duration,
    measure: Duration,
    workers: usize,
    router_cpus: Option<String>,
    out_dir: PathBuf,
}

fn env_list(name: &str, default: &str) -> Vec<String> {
    std::env::var(name)
        .unwrap_or_else(|_| default.to_string())
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

fn env_num<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

impl Config {
    fn from_env() -> Config {
        let scenarios = env_list("VLLM_ROUTER_BENCH_SCENARIOS", "")
            .into_iter()
            .map(|n| {
                *SCENARIOS
                    .iter()
                    .find(|s| s.name == n)
                    .unwrap_or_else(|| panic!("unknown scenario {n}"))
            })
            .collect::<Vec<_>>();
        let scenarios = if scenarios.is_empty() {
            SCENARIOS.to_vec()
        } else {
            scenarios
        };
        let default_sizes = SIZES
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let router_cpus = std::env::var("VLLM_ROUTER_BENCH_ROUTER_CPUS")
            .ok()
            .filter(|s| !s.trim().is_empty());
        assert!(
            router_cpus.is_none() || cfg!(target_os = "linux"),
            "VLLM_ROUTER_BENCH_ROUTER_CPUS needs Linux `taskset`"
        );
        Config {
            scenarios,
            sizes: env_list("VLLM_ROUTER_BENCH_SIZES", &default_sizes)
                .iter()
                .map(|s| s.parse().expect("size"))
                .collect(),
            corpora: env_list("VLLM_ROUTER_BENCH_CORPORA", "hot64,cold")
                .iter()
                .map(|c| CorpusKind::parse(c).unwrap_or_else(|| panic!("unknown corpus {c}")))
                .collect(),
            concurrency: env_list("VLLM_ROUTER_BENCH_CONCURRENCY", "1,64")
                .iter()
                .map(|c| c.parse().expect("concurrency"))
                .collect(),
            rate: env_num("VLLM_ROUTER_BENCH_RATE", 0.0),
            repeats: env_num("VLLM_ROUTER_BENCH_REPEATS", 1usize).max(1),
            warmup: Duration::from_secs(env_num("VLLM_ROUTER_BENCH_WARMUP_SECS", 5u64)),
            measure: Duration::from_secs(env_num("VLLM_ROUTER_BENCH_MEASURE_SECS", 20u64)),
            workers: env_num("VLLM_ROUTER_BENCH_WORKERS", 4usize).max(1),
            router_cpus,
            out_dir: PathBuf::from(
                std::env::var("VLLM_ROUTER_BENCH_OUT_DIR")
                    .unwrap_or_else(|_| "target/router_overhead".to_string()),
            ),
        }
    }
}

/// Resource usage of one child process, read with `wait4(2)` after it exited.
#[derive(Clone, Copy, Debug, Default, Serialize)]
struct ProcessUsage {
    user_cpu_s: f64,
    sys_cpu_s: f64,
    max_rss_bytes: u64,
}

/// A running router. Dropping it without [`stop`](Self::stop) (for example
/// when an assertion fails mid-run) still kills and reaps the process.
struct RouterProcess {
    child: Option<Child>,
    port: u16,
    log_path: PathBuf,
}

impl RouterProcess {
    fn spawn(
        worker_urls: &[String],
        policy: &str,
        log_path: &Path,
        router_cpus: Option<&str>,
    ) -> RouterProcess {
        let port = portpicker::pick_unused_port().expect("router port");
        let prometheus_port = loop {
            let p = portpicker::pick_unused_port().expect("prometheus port");
            if p != port {
                break p;
            }
        };
        let log = fs::File::create(log_path).expect("router log file");
        let bin = env!("CARGO_BIN_EXE_vllm-router");
        // `taskset` execs the router in place, so the pid we wait on is the
        // router's own.
        let mut cmd = match router_cpus {
            Some(cpus) => {
                let mut c = Command::new("taskset");
                c.arg("-c").arg(cpus).arg(bin);
                c
            }
            None => Command::new(bin),
        };
        cmd.arg("--host")
            .arg("127.0.0.1")
            .arg("--port")
            .arg(port.to_string())
            .arg("--prometheus-port")
            .arg(prometheus_port.to_string())
            .arg("--policy")
            .arg(policy)
            .arg("--eviction-interval")
            .arg(EVICTION_INTERVAL_SECS.to_string())
            .arg("--max-tree-size")
            .arg(MAX_TREE_SIZE.to_string())
            .arg("--log-level")
            .arg("warn")
            .arg("--worker-startup-check-interval")
            .arg("1")
            .arg("--worker-urls");
        for url in worker_urls {
            cmd.arg(url);
        }
        let child = cmd
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(log)
            .spawn()
            .expect("spawn vllm-router");
        RouterProcess {
            child: Some(child),
            port,
            log_path: log_path.to_path_buf(),
        }
    }

    /// Spawn a router and wait until it serves `/health`. A router that
    /// exits because its port was taken after it was picked is restarted on
    /// fresh ports; any other early exit fails the run with its log.
    async fn start(
        worker_urls: &[String],
        policy: &str,
        log_path: &Path,
        router_cpus: Option<&str>,
        client: &reqwest::Client,
    ) -> RouterProcess {
        for attempt in 1..=ROUTER_START_ATTEMPTS {
            let mut router = RouterProcess::spawn(worker_urls, policy, log_path, router_cpus);
            match router.wait_ready(client).await {
                Ok(()) => return router,
                Err(exit) => {
                    let log = fs::read_to_string(&router.log_path).unwrap_or_default();
                    if log.contains("Address already in use") && attempt < ROUTER_START_ATTEMPTS {
                        eprintln!("   router port was taken, retrying ({attempt})");
                        continue;
                    }
                    panic!("vllm-router exited before becoming ready ({exit}):\n{log}");
                }
            }
        }
        unreachable!("the last attempt either returns or panics")
    }

    fn url(&self) -> String {
        format!("http://127.0.0.1:{}", self.port)
    }

    async fn wait_ready(&mut self, client: &reqwest::Client) -> Result<(), String> {
        let deadline = Instant::now() + Duration::from_secs(60);
        let health = format!("{}/health", self.url());
        loop {
            let child = self.child.as_mut().expect("router is running");
            if let Ok(Some(status)) = child.try_wait() {
                return Err(status.to_string());
            }
            if let Ok(resp) = client.get(&health).send().await {
                if resp.status().is_success() {
                    return Ok(());
                }
            }
            assert!(
                Instant::now() < deadline,
                "vllm-router did not become ready; see {}",
                self.log_path.display()
            );
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }

    /// Stop the router and return its own resource usage. `SIGKILL` is used
    /// on purpose: the load has already stopped, and CPU time and peak RSS
    /// are accounted regardless of how the process ends.
    // The child is reaped by `wait4` below, which std's `Child` cannot see.
    #[allow(clippy::zombie_processes)]
    fn stop(mut self) -> ProcessUsage {
        let child = self.child.take().expect("router is running");
        let pid = child.id() as libc::pid_t;
        // SAFETY: plain libc calls on a pid we spawned and have not reaped;
        // `rusage` is a POD struct that wait4 fills in.
        unsafe {
            libc::kill(pid, libc::SIGKILL);
            let mut status: libc::c_int = 0;
            let mut usage: libc::rusage = std::mem::zeroed();
            let waited = libc::wait4(pid, &mut status, 0, &mut usage);
            assert_eq!(waited, pid, "wait4 failed for the router process");
            ProcessUsage {
                user_cpu_s: timeval_secs(usage.ru_utime),
                sys_cpu_s: timeval_secs(usage.ru_stime),
                max_rss_bytes: max_rss_bytes(usage.ru_maxrss),
            }
        }
    }
}

impl Drop for RouterProcess {
    fn drop(&mut self) {
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

fn timeval_secs(tv: libc::timeval) -> f64 {
    tv.tv_sec as f64 + tv.tv_usec as f64 / 1e6
}

/// `ru_maxrss` is bytes on macOS and kibibytes on Linux and the BSDs.
fn max_rss_bytes(ru_maxrss: libc::c_long) -> u64 {
    if cfg!(target_os = "macos") {
        ru_maxrss as u64
    } else {
        ru_maxrss as u64 * 1024
    }
}

/// CPU seconds (user + system) this process has used so far: the client
/// and the in-process mock workers.
fn self_cpu_s() -> f64 {
    // SAFETY: getrusage fills a POD struct for the calling process.
    unsafe {
        let mut usage: libc::rusage = std::mem::zeroed();
        libc::getrusage(libc::RUSAGE_SELF, &mut usage);
        timeval_secs(usage.ru_utime) + timeval_secs(usage.ru_stime)
    }
}

#[derive(Clone, Debug, Default)]
struct LoadResult {
    /// Successful requests started inside the measurement window.
    requests: u64,
    /// Failed requests started inside the measurement window.
    errors: u64,
    /// Successful requests over warmup and measurement together: every
    /// request the router served in this cell.
    completed_ok: u64,
    measure_secs: f64,
    latencies_us: Vec<u64>,
}

/// One load cell: what to send, to where, how hard and for how long.
#[derive(Clone)]
struct LoadSpec {
    base_url: String,
    route: Route,
    corpus: Arc<Corpus>,
    concurrency: usize,
    rate: f64,
    warmup: Duration,
    measure: Duration,
}

/// Drive `base_url` with `concurrency` tasks. Closed loop by default: each
/// task sends the next request as soon as the previous response body has
/// been read. With `rate > 0` every task paces itself to `rate /
/// concurrency` requests per second (a rate cap, still at most one request
/// in flight per task), so arms can be compared at the same offered load.
/// Request `i` of the run uses `corpus.prompt(i)`; tasks interleave indices
/// so every corpus kind sees the sequence it was designed for. Every body is
/// built the same way for every corpus, outside the timed region. Latency
/// and error counts cover requests *started* inside the measurement window.
async fn run_load(client: &reqwest::Client, spec: LoadSpec) -> LoadResult {
    let LoadSpec {
        base_url,
        route,
        corpus,
        concurrency,
        rate,
        warmup,
        measure,
    } = spec;
    let url = format!("{base_url}{}", route.path());
    let start = Instant::now();
    let warm_end = start + warmup;
    let end = warm_end + measure;
    let pace = if rate > 0.0 {
        Some(Duration::from_secs_f64(concurrency as f64 / rate))
    } else {
        None
    };

    let mut tasks = Vec::with_capacity(concurrency);
    for task in 0..concurrency {
        let client = client.clone();
        let url = url.clone();
        let corpus = Arc::clone(&corpus);
        tasks.push(tokio::spawn(async move {
            let mut latencies = Vec::with_capacity(8192);
            let mut errors = 0u64;
            let mut completed_ok = 0u64;
            let mut i = task;
            let mut next_at = start;
            loop {
                if let Some(pace) = pace {
                    tokio::time::sleep_until(tokio::time::Instant::from_std(next_at)).await;
                    next_at += pace;
                }
                if Instant::now() >= end {
                    break;
                }
                let body = route.body(&corpus.prompt(i));
                let t0 = Instant::now();
                let ok = match client
                    .post(&url)
                    .header(CONTENT_TYPE, "application/json")
                    .body(body)
                    .send()
                    .await
                {
                    Ok(resp) => {
                        let ok = resp.status().is_success();
                        let _ = resp.bytes().await;
                        ok
                    }
                    Err(_) => false,
                };
                if ok {
                    completed_ok += 1;
                }
                if t0 >= warm_end {
                    if ok {
                        latencies.push(t0.elapsed().as_micros() as u64);
                    } else {
                        errors += 1;
                    }
                }
                i += concurrency;
            }
            (latencies, errors, completed_ok)
        }));
    }

    let mut result = LoadResult {
        measure_secs: measure.as_secs_f64(),
        ..Default::default()
    };
    for task in tasks {
        let (latencies, errors, completed_ok) = task.await.expect("load task");
        result.requests += latencies.len() as u64;
        result.errors += errors;
        result.completed_ok += completed_ok;
        result.latencies_us.extend(latencies);
    }
    result.latencies_us.sort_unstable();
    result
}

fn percentile(sorted: &[u64], p: f64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let rank = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[rank.min(sorted.len() - 1)]
}

fn median(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = values.len();
    if n % 2 == 1 {
        values[n / 2]
    } else {
        (values[n / 2 - 1] + values[n / 2]) / 2.0
    }
}

#[derive(Clone, Debug, Serialize)]
struct Row {
    repeat: usize,
    scenario: String,
    policy: Option<String>,
    route: String,
    size_bytes: usize,
    corpus: String,
    concurrency: usize,
    rate: f64,
    workers: usize,
    warmup_secs: f64,
    measure_secs: f64,
    requests: u64,
    errors: u64,
    /// Successful requests over warmup and measurement; the denominator of
    /// `router_cpu_ms_per_1k_requests`.
    served_requests: u64,
    rps: f64,
    p50_us: u64,
    p90_us: u64,
    p99_us: u64,
    mean_us: f64,
    /// `mean_us` minus the `direct_mock` mean for the same size/corpus/
    /// concurrency/repeat, when that row was measured in this run. Means
    /// subtract; percentiles do not, so no such column exists for them.
    mean_minus_direct_us: Option<f64>,
    /// Fraction of the generation requests each mock worker served
    /// (router scenarios only).
    worker_shares: Option<Vec<f64>>,
    /// Largest worker share divided by the even share `1 / workers`:
    /// `1.0` is perfectly even, `workers` is everything on one worker.
    hotspot: Option<f64>,
    router: Option<ProcessUsage>,
    /// Router CPU over its whole life (startup, warmup, measurement) per
    /// thousand requests it served over warmup and measurement. Startup is
    /// a small fixed cost included in the numerator.
    router_cpu_ms_per_1k_requests: Option<f64>,
    /// CPU seconds the client and the mock workers used during the cell.
    harness_cpu_s: f64,
}

type CellKey = (String, usize, String, usize);

fn cell_key(r: &Row) -> CellKey {
    (
        r.scenario.clone(),
        r.size_bytes,
        r.corpus.clone(),
        r.concurrency,
    )
}

fn command_output(program: &str, args: &[&str]) -> Option<String> {
    Command::new(program)
        .args(args)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
}

#[derive(Serialize)]
struct Report {
    commit: String,
    /// Tracked files differ from `commit`.
    dirty: bool,
    rustc: String,
    os: &'static str,
    arch: &'static str,
    cpus: usize,
    profile: &'static str,
    harness_worker_threads: usize,
    /// `TOKIO_WORKER_THREADS` as seen by the router, or `default`.
    router_worker_threads: String,
    /// `taskset -c` list for the router, or `none`.
    router_cpus: String,
    eviction_interval_secs: u64,
    max_tree_size: usize,
    repeats: usize,
    rate: f64,
    rows: Vec<Row>,
}

fn fmt_opt(v: Option<f64>, digits: usize) -> String {
    v.map(|v| format!("{v:.digits$}"))
        .unwrap_or_else(|| "-".to_string())
}

fn render_rows(rows: &[Row]) -> String {
    let mut out = String::new();
    out.push_str("| rep | scenario | size | corpus | c | requests | errors | rps | p50 us | p90 us | p99 us | mean us | mean-direct us | hotspot | router cpu ms/1k req | router max rss MiB | harness cpu s |\n");
    out.push_str(
        "|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n",
    );
    for r in rows {
        let rss = r.router.map(|u| u.max_rss_bytes as f64 / (1024.0 * 1024.0));
        out.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} | {:.0} | {} | {} | {} | {:.1} | {} | {} | {} | {} | {:.1} |\n",
            r.repeat,
            r.scenario,
            r.size_bytes,
            r.corpus,
            r.concurrency,
            r.requests,
            r.errors,
            r.rps,
            r.p50_us,
            r.p90_us,
            r.p99_us,
            r.mean_us,
            fmt_opt(r.mean_minus_direct_us, 1),
            fmt_opt(r.hotspot, 2),
            fmt_opt(r.router_cpu_ms_per_1k_requests, 1),
            fmt_opt(rss, 1),
            r.harness_cpu_s,
        ));
    }
    out
}

/// Median over repeats per cell, with the spread of the mean latency
/// (`(max - min) / median`) so run-to-run noise is visible next to any
/// difference between cells.
fn render_summary(rows: &[Row]) -> String {
    let mut cells: BTreeMap<CellKey, Vec<&Row>> = BTreeMap::new();
    for r in rows {
        cells.entry(cell_key(r)).or_default().push(r);
    }
    let mut out = String::new();
    out.push_str("| scenario | size | corpus | c | runs | rps (median) | p50 us | p99 us | mean us | mean spread | mean-direct us | hotspot | router cpu ms/1k req | router max rss MiB |\n");
    out.push_str("|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n");
    for ((scenario, size, corpus, c), rs) in cells {
        let med = |f: &dyn Fn(&Row) -> Option<f64>| -> Option<f64> {
            let mut v: Vec<f64> = rs.iter().filter_map(|r| f(r)).collect();
            if v.is_empty() {
                None
            } else {
                Some(median(&mut v))
            }
        };
        let means: Vec<f64> = rs.iter().map(|r| r.mean_us).collect();
        let mean_med = med(&|r| Some(r.mean_us)).unwrap_or(0.0);
        let spread = if means.len() > 1 && mean_med > 0.0 {
            let max = means.iter().cloned().fold(f64::MIN, f64::max);
            let min = means.iter().cloned().fold(f64::MAX, f64::min);
            Some((max - min) / mean_med * 100.0)
        } else {
            None
        };
        out.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {:.1} | {} | {} | {} | {} | {} |\n",
            scenario,
            size,
            corpus,
            c,
            rs.len(),
            fmt_opt(med(&|r| Some(r.rps)), 0),
            fmt_opt(med(&|r| Some(r.p50_us as f64)), 0),
            fmt_opt(med(&|r| Some(r.p99_us as f64)), 0),
            mean_med,
            spread
                .map(|s| format!("{s:.1}%"))
                .unwrap_or_else(|| "-".to_string()),
            fmt_opt(med(&|r| r.mean_minus_direct_us), 1),
            fmt_opt(med(&|r| r.hotspot), 2),
            fmt_opt(med(&|r| r.router_cpu_ms_per_1k_requests), 1),
            fmt_opt(
                med(&|r| r.router.map(|u| u.max_rss_bytes as f64 / (1024.0 * 1024.0))),
                1
            ),
        ));
    }
    out
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
#[ignore = "load test: run explicitly with --ignored --nocapture"]
async fn router_overhead() {
    let cfg = Config::from_env();
    fs::create_dir_all(&cfg.out_dir).expect("out dir");
    let client = reqwest::Client::builder()
        .pool_max_idle_per_host(256)
        .timeout(Duration::from_secs(30))
        .build()
        .expect("client");

    let mut workers = Vec::with_capacity(cfg.workers);
    for _ in 0..cfg.workers {
        workers.push(BenchMockWorker::start().await);
    }
    let worker_urls: Vec<String> = workers.iter().map(|w| w.url().to_string()).collect();

    let profile = if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    };
    let commit = command_output("git", &["rev-parse", "--short", "HEAD"])
        .unwrap_or_else(|| "unknown".into());
    let dirty = command_output("git", &["status", "--porcelain", "--untracked-files=no"])
        .is_some_and(|s| !s.is_empty());
    let rustc = command_output("rustc", &["-V"]).unwrap_or_else(|| "unknown".into());
    let cpus = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(0);
    let router_worker_threads =
        std::env::var("TOKIO_WORKER_THREADS").unwrap_or_else(|_| "default".to_string());
    let router_cpus = cfg
        .router_cpus
        .clone()
        .unwrap_or_else(|| "none".to_string());
    eprintln!(
        "router overhead harness: commit={commit} dirty={dirty} rustc=\"{rustc}\" os={} arch={} cpus={cpus} profile={profile} workers={} repeats={} rate={} warmup={:?} measure={:?} router_worker_threads={router_worker_threads} router_cpus={router_cpus}",
        std::env::consts::OS,
        std::env::consts::ARCH,
        cfg.workers,
        cfg.repeats,
        cfg.rate,
        cfg.warmup,
        cfg.measure
    );
    if profile == "debug" {
        eprintln!("warning: debug profile; use `cargo test --release ...` for reportable numbers");
    }

    let mut rows: Vec<Row> = Vec::new();
    for repeat in 0..cfg.repeats {
        // Rotate the scenario order every repeat so a slow drift of the
        // machine does not always land on the same scenario.
        let mut order = cfg.scenarios.clone();
        let shift = repeat % order.len();
        order.rotate_left(shift);
        for scenario in &order {
            for &size in &cfg.sizes {
                for &kind in &cfg.corpora {
                    for &concurrency in &cfg.concurrency {
                        let corpus = Arc::new(Corpus::new(kind, size));
                        let label = format!(
                            "r{}-{}-{}-{}-c{}",
                            repeat,
                            scenario.name,
                            size,
                            kind.name(),
                            concurrency
                        );
                        eprintln!("== {label}");
                        for w in &workers {
                            w.reset_served();
                        }
                        let spec = LoadSpec {
                            base_url: String::new(),
                            route: scenario.route,
                            corpus,
                            concurrency,
                            rate: cfg.rate,
                            warmup: cfg.warmup,
                            measure: cfg.measure,
                        };
                        let cpu_before = self_cpu_s();
                        let (load, usage) = match scenario.policy {
                            None => (
                                run_load(
                                    &client,
                                    LoadSpec {
                                        base_url: worker_urls[0].clone(),
                                        ..spec.clone()
                                    },
                                )
                                .await,
                                None,
                            ),
                            Some(policy) => {
                                let log_path = cfg.out_dir.join(format!("{label}.router.log"));
                                let router = RouterProcess::start(
                                    &worker_urls,
                                    policy,
                                    &log_path,
                                    cfg.router_cpus.as_deref(),
                                    &client,
                                )
                                .await;
                                let load = run_load(
                                    &client,
                                    LoadSpec {
                                        base_url: router.url(),
                                        ..spec.clone()
                                    },
                                )
                                .await;
                                (load, Some(router.stop()))
                            }
                        };
                        let harness_cpu_s = self_cpu_s() - cpu_before;
                        let served: Vec<u64> = workers.iter().map(|w| w.served()).collect();
                        let total_served: u64 = served.iter().sum();
                        let (worker_shares, hotspot) = if usage.is_some() && total_served > 0 {
                            let shares: Vec<f64> = served
                                .iter()
                                .map(|&n| n as f64 / total_served as f64)
                                .collect();
                            let max = shares.iter().cloned().fold(0.0, f64::max);
                            (Some(shares), Some(max * cfg.workers as f64))
                        } else {
                            (None, None)
                        };
                        let mean_us = if load.latencies_us.is_empty() {
                            0.0
                        } else {
                            load.latencies_us.iter().sum::<u64>() as f64
                                / load.latencies_us.len() as f64
                        };
                        let row = Row {
                            repeat,
                            scenario: scenario.name.to_string(),
                            policy: scenario.policy.map(str::to_string),
                            route: scenario.route.path().to_string(),
                            size_bytes: size,
                            corpus: kind.name().to_string(),
                            concurrency,
                            rate: cfg.rate,
                            workers: cfg.workers,
                            warmup_secs: cfg.warmup.as_secs_f64(),
                            measure_secs: load.measure_secs,
                            requests: load.requests,
                            errors: load.errors,
                            served_requests: load.completed_ok,
                            rps: load.requests as f64 / load.measure_secs,
                            p50_us: percentile(&load.latencies_us, 0.50),
                            p90_us: percentile(&load.latencies_us, 0.90),
                            p99_us: percentile(&load.latencies_us, 0.99),
                            mean_us,
                            mean_minus_direct_us: None,
                            worker_shares,
                            hotspot,
                            router: usage,
                            router_cpu_ms_per_1k_requests: usage.map(|u| {
                                if load.completed_ok == 0 {
                                    0.0
                                } else {
                                    (u.user_cpu_s + u.sys_cpu_s) * 1000.0
                                        / (load.completed_ok as f64 / 1000.0)
                                }
                            }),
                            harness_cpu_s,
                        };
                        eprintln!(
                            "   requests={} errors={} rps={:.0} p50={}us p99={}us mean={:.1}us hotspot={} router_cpu_ms_per_1k={}",
                            row.requests,
                            row.errors,
                            row.rps,
                            row.p50_us,
                            row.p99_us,
                            row.mean_us,
                            fmt_opt(row.hotspot, 2),
                            fmt_opt(row.router_cpu_ms_per_1k_requests, 1)
                        );
                        rows.push(row);
                    }
                }
            }
        }
    }

    for w in &mut workers {
        w.stop().await;
    }

    // Fill in the mean delta against direct_mock measured in the same repeat.
    let direct: BTreeMap<(usize, usize, String, usize), f64> = rows
        .iter()
        .filter(|r| r.policy.is_none())
        .map(|r| {
            (
                (r.repeat, r.size_bytes, r.corpus.clone(), r.concurrency),
                r.mean_us,
            )
        })
        .collect();
    for r in rows.iter_mut() {
        if r.policy.is_some() {
            r.mean_minus_direct_us = direct
                .get(&(r.repeat, r.size_bytes, r.corpus.clone(), r.concurrency))
                .map(|d| r.mean_us - d);
        }
    }

    let report = Report {
        commit,
        dirty,
        rustc,
        os: std::env::consts::OS,
        arch: std::env::consts::ARCH,
        cpus,
        profile,
        harness_worker_threads: HARNESS_WORKER_THREADS,
        router_worker_threads,
        router_cpus,
        eviction_interval_secs: EVICTION_INTERVAL_SECS,
        max_tree_size: MAX_TREE_SIZE,
        repeats: cfg.repeats,
        rate: cfg.rate,
        rows: rows.clone(),
    };
    let json_path = cfg.out_dir.join("summary.json");
    fs::write(
        &json_path,
        serde_json::to_string_pretty(&report).expect("json"),
    )
    .expect("write");
    let mut md = String::new();
    md.push_str("## Per run\n\n");
    md.push_str(&render_rows(&rows));
    md.push_str("\n## Median over runs\n\n");
    md.push_str(&render_summary(&rows));
    let md_path = cfg.out_dir.join("summary.md");
    fs::write(&md_path, &md).expect("write");
    eprintln!(
        "\n{md}\nwritten: {} and {}",
        json_path.display(),
        md_path.display()
    );

    let total_errors: u64 = rows.iter().map(|r| r.errors).sum();
    assert_eq!(total_errors, 0, "requests failed during the benchmark");
}
