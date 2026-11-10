#![cfg(all(feature = "cpu", feature = "ready"))]

//! S1: the shadow KV transfer planner, end to end, with mock workers that
//! publish real KV events.
//!
//! Each test starts a real `kv_aware` router (`RouterFactory::create_router`,
//! the same path `main` takes) with `kv_transfer: {mode: shadow}` and only
//! [`KvEventMock`] backends. The mocks answer discovery over HTTP and publish
//! `BlockStored` / `BlockRemoved` / `AllBlocksCleared` over ZMQ, so the router
//! builds its index from the wire exactly as it does against vLLM. Requests carry
//! token ids (`/v1/completions` with an integer prompt) and each test asserts on
//! the delta of `vllm_router_kv_transfer_plans_total{outcome}`.
//!
//! # How the counter is read
//!
//! The metric is recorded through the global `metrics` recorder, so the test
//! binary installs a `PrometheusBuilder` recorder once (no HTTP listener) and
//! scrapes `PrometheusHandle::render()`. The recorder is process-global, so tests
//! take a lock and run one at a time; assertions compare the complete outcome
//! delta around the request under test, never absolute values.
//!
//! # How a worker is made busy
//!
//! The router's `active_requests` / `active_prefill_tokens` come from its own
//! slot counters. A request carrying `"priority": HOLD_PRIORITY` is held open by the
//! mock, so the counters stay raised. Hold prompts reuse the worker's cached
//! prefix plus an uncached tail, which lets them route to that worker (cache
//! affinity) while charging it `tail` prefill tokens each. Each test comment
//! works the cost arithmetic out: router cost is
//! `(new_prefill + active_prefill_tokens) * (active_requests + 1)`, and the
//! planner prices 50 us per prefilled token, 5 ms + 5 us per transferred token.
//!
//! # Convergence
//!
//! Index convergence is polled through `KvAwarePolicy::decision_for` (a side
//! effect free selection over the same registered workers and index): a test
//! proceeds only when the router itself reports the expected cache credit.

mod common;

use axum::body::Body;
use axum::http::Request;
use axum::Router;
use common::kv_event_mock::{KvEventMock, KvEventMockConfig, Residency, HOLD_PRIORITY};
use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};
use serde_json::json;
use std::collections::BTreeMap;
use std::sync::{Arc, OnceLock};
use std::time::Duration;
use tokio::task::JoinHandle;
use tower::ServiceExt;
use vllm_router_rs::config::types::{KvTransferConfig, KvTransferMode};
use vllm_router_rs::config::{PolicyConfig, RouterConfig, RoutingMode};
use vllm_router_rs::policies::KvAwarePolicy;
use vllm_router_rs::routers::RouterFactory;
use vllm_router_rs::server::AppContext;

const BLOCK_SIZE: u32 = 16;
const COUNTER: &str = "vllm_router_kv_transfer_plans_total";
const SAVED_MS: &str = "vllm_router_kv_transfer_estimated_saved_ms";
const COST_MS: &str = "vllm_router_kv_transfer_estimated_cost_ms";

// ---------------------------------------------------------------------------
// Metrics and serialization
// ---------------------------------------------------------------------------

fn metrics() -> &'static PrometheusHandle {
    static HANDLE: OnceLock<PrometheusHandle> = OnceLock::new();
    HANDLE.get_or_init(|| {
        PrometheusBuilder::new()
            .install_recorder()
            .expect("install the test metrics recorder")
    })
}

/// Every `outcome` label of the transfer-plan counter, with its current value.
fn plan_counts() -> BTreeMap<String, u64> {
    let mut counts = BTreeMap::new();
    for line in metrics().render().lines() {
        let Some(rest) = line.strip_prefix(COUNTER).and_then(|r| r.strip_prefix('{')) else {
            continue;
        };
        let outcome = rest
            .split("outcome=\"")
            .nth(1)
            .and_then(|tail| tail.split('"').next())
            .expect("outcome label");
        let value: f64 = line
            .rsplit(' ')
            .next()
            .and_then(|value| value.parse().ok())
            .expect("counter value");
        counts.insert(outcome.to_owned(), value as u64);
    }
    counts
}

/// Observation count and sum of every histogram series this PR adds, keyed by
/// `saved`, `cost:fetch` and `cost:wait`. The test recorder has no buckets
/// configured, so the exporter renders these as summaries; `_count` and `_sum`
/// are present either way.
fn cost_histograms() -> BTreeMap<String, (u64, f64)> {
    let mut series: BTreeMap<String, (u64, f64)> = BTreeMap::new();
    for line in metrics().render().lines() {
        for (name, base) in [(SAVED_MS, "saved"), (COST_MS, "cost")] {
            for (suffix, is_count) in [("_count", true), ("_sum", false)] {
                let Some(rest) = line.strip_prefix(&format!("{name}{suffix}")) else {
                    continue;
                };
                let key = match rest.strip_prefix('{') {
                    Some(labels) => {
                        let path = labels
                            .split("path=\"")
                            .nth(1)
                            .and_then(|tail| tail.split('"').next())
                            .expect("path label");
                        format!("{base}:{path}")
                    }
                    None => base.to_owned(),
                };
                let value: f64 = line
                    .rsplit(' ')
                    .next()
                    .and_then(|value| value.parse().ok())
                    .expect("histogram value");
                let entry = series.entry(key).or_default();
                if is_count {
                    entry.0 = value as u64;
                } else {
                    entry.1 = value;
                }
            }
        }
    }
    series
}

/// Per-series change in `(count, sum)` between two scrapes (only series whose
/// count changed).
fn cost_delta(
    before: &BTreeMap<String, (u64, f64)>,
    after: &BTreeMap<String, (u64, f64)>,
) -> BTreeMap<String, (u64, f64)> {
    after
        .iter()
        .filter_map(|(key, &(count, sum))| {
            let (old_count, old_sum) = before.get(key).copied().unwrap_or((0, 0.0));
            (count > old_count).then(|| (key.clone(), (count - old_count, sum - old_sum)))
        })
        .collect()
}

/// Assert the cost histogram delta has exactly the expected series, each with
/// the expected observation count and a sum within 0.01 ms of `sum`.
fn assert_cost_observations(actual: &BTreeMap<String, (u64, f64)>, expected: &[(&str, u64, f64)]) {
    let keys: Vec<_> = actual.keys().map(String::as_str).collect();
    let want: Vec<_> = expected.iter().map(|(key, _, _)| *key).collect();
    assert_eq!(keys, want, "series observed: {actual:?}");
    for &(key, count, sum) in expected {
        let (got_count, got_sum) = actual[key];
        assert_eq!(got_count, count, "{key} count");
        assert!((got_sum - sum).abs() < 0.01, "{key} sum {got_sum} != {sum}");
    }
}

/// What changed between two scrapes (only non-zero entries).
fn delta(before: &BTreeMap<String, u64>, after: &BTreeMap<String, u64>) -> BTreeMap<String, u64> {
    after
        .iter()
        .filter_map(|(outcome, value)| {
            let change = value - before.get(outcome).copied().unwrap_or(0);
            (change > 0).then(|| (outcome.clone(), change))
        })
        .collect()
}

fn outcomes(expected: &[(&str, u64)]) -> BTreeMap<String, u64> {
    expected
        .iter()
        .map(|(outcome, count)| ((*outcome).to_owned(), *count))
        .collect()
}

// ---------------------------------------------------------------------------
// Token helpers
// ---------------------------------------------------------------------------

/// `len` distinct token ids; different `seed`s never collide.
fn seq(seed: u32, len: usize) -> Vec<u32> {
    (0..len as u32).map(|i| seed * 1_000_000 + i).collect()
}

/// `prefix` followed by `tail` tokens that no worker has cached.
fn with_tail(prefix: &[u32], tail_seed: u32, tail: usize) -> Vec<u32> {
    let mut tokens = prefix.to_vec();
    tokens.extend(seq(90 + tail_seed, tail));
    tokens
}

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------

/// One mock per `(instance_id, model)`. Two mocks share a cache identity when
/// their models are equal (block size is fixed).
struct Options {
    workers: Vec<(&'static str, &'static str)>,
    router_temperature: f64,
    session_ttl_secs: Option<u64>,
    transfer: KvTransferConfig,
}

impl Options {
    fn new(workers: &[(&'static str, &'static str)]) -> Self {
        Self {
            workers: workers.to_vec(),
            router_temperature: 0.0,
            session_ttl_secs: None,
            transfer: KvTransferConfig {
                mode: KvTransferMode::Shadow,
                require_capability: false,
                ..KvTransferConfig::default()
            },
        }
    }
}

struct Fixture {
    app: Router,
    ctx: Arc<AppContext>,
    mocks: Vec<KvEventMock>,
    held: Vec<JoinHandle<u16>>,
    _serial: tokio::sync::MutexGuard<'static, ()>,
}

static SERIAL: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

impl Fixture {
    async fn start(options: Options) -> Self {
        let serial = SERIAL.lock().await;
        metrics();

        let mut mocks = Vec::new();
        for (instance, model) in &options.workers {
            mocks.push(
                KvEventMock::start(KvEventMockConfig::new(instance, model, BLOCK_SIZE)).await,
            );
        }
        let config = RouterConfig {
            mode: RoutingMode::Regular {
                worker_urls: mocks.iter().map(|mock| mock.url.clone()).collect(),
            },
            policy: PolicyConfig::KvAware {
                router_temperature: options.router_temperature,
                kv_penalty_host: 0.8,
                kv_penalty_external: 0.6,
                kv_penalty_disk: 0.3,
                static_kv_event_sources: Default::default(),
                kv_transfer: Some(Box::new(options.transfer.clone())),
            },
            kv_aware_session_affinity_ttl_secs: options.session_ttl_secs,
            port: 0,
            request_timeout_secs: 10,
            worker_startup_timeout_secs: 5,
            worker_startup_check_interval_secs: 1,
            max_concurrent_requests: 64,
            queue_size: 0,
            ..RouterConfig::default()
        };
        let ctx = common::create_test_context(config.clone());
        let router = RouterFactory::create_router(&ctx)
            .await
            .expect("router starts against the mock workers");
        let app = common::test_app::create_test_app_with_context(
            Arc::from(router),
            Arc::clone(&ctx),
            &config,
        );
        // The router subscribes during construction; the SUB connect itself is
        // asynchronous, so wait for each publisher to see it before any event.
        for mock in &mocks {
            mock.wait_subscribed().await;
        }
        Self {
            app,
            ctx,
            mocks,
            held: Vec::new(),
            _serial: serial,
        }
    }

    fn mock(&self, instance: &str) -> &KvEventMock {
        self.mocks
            .iter()
            .find(|mock| mock.config().instance_id == instance)
            .unwrap_or_else(|| panic!("no mock {instance}"))
    }

    /// What the router's own policy would pick for `tokens` right now, and the
    /// work it would charge there (new prefill plus queued prefill tokens). Pure.
    fn decision(&self, tokens: &[u32]) -> (String, u64) {
        let policy = self.ctx.policy_registry.get_default_policy();
        let kv = policy
            .as_any()
            .downcast_ref::<KvAwarePolicy>()
            .expect("kv_aware policy");
        let workers: Vec<_> = self
            .ctx
            .worker_registry
            .get_all()
            .into_iter()
            .filter(|worker| worker.is_available())
            .collect();
        let decision = kv.decision_for(&workers, Some(tokens)).expect("a decision");
        (
            decision.target.instance_id.to_string(),
            decision.components.ptoken,
        )
    }

    /// Wait until the router's index makes `tokens` route to `instance` with
    /// exactly `ptoken` tokens of remaining work. Polls; never a fixed sleep.
    async fn converge(&self, tokens: &[u32], instance: &str, ptoken: u64) {
        let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
        loop {
            let seen = self.decision(tokens);
            if seen == (instance.to_owned(), ptoken) {
                return;
            }
            assert!(
                tokio::time::Instant::now() < deadline,
                "index never converged: wanted ({instance}, {ptoken}), last saw {seen:?}"
            );
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }

    /// Wait until `tokens` route with exactly `ptoken` remaining work, whichever
    /// worker is picked (used to see credit disappear).
    async fn converge_work(&self, tokens: &[u32], ptoken: u64) {
        let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
        loop {
            let seen = self.decision(tokens);
            if seen.1 == ptoken {
                return;
            }
            assert!(
                tokio::time::Instant::now() < deadline,
                "index never converged: wanted work {ptoken}, last saw {seen:?}"
            );
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }

    /// POST `body` to `route` through the router and return the status.
    async fn post(&self, route: &str, body: serde_json::Value, session: Option<&str>) -> u16 {
        post(&self.app, route, body, session).await
    }

    /// A completion carrying token ids.
    async fn complete(&self, tokens: &[u32], session: Option<&str>) -> u16 {
        self.post(
            "/v1/completions",
            json!({"prompt": tokens, "max_tokens": 1}),
            session,
        )
        .await
    }

    /// Open `count` held requests, each `prefix` plus a distinct `tail`-token
    /// suffix, one at a time, and check each lands on `instance`.
    async fn make_busy(&mut self, instance: &str, prefix: &[u32], tail: usize, count: usize) {
        for i in 0..count {
            let app = self.app.clone();
            let tokens = with_tail(prefix, i as u32, tail);
            self.held.push(tokio::spawn(async move {
                post(
                    &app,
                    "/v1/completions",
                    json!({"prompt": tokens, "max_tokens": 1, "priority": HOLD_PRIORITY}),
                    None,
                )
                .await
            }));
            // Fails with a clear message if the request routed elsewhere.
            self.mock(instance).wait_held(i + 1).await;
        }
    }

    /// Release held requests and check they all completed.
    async fn finish(mut self) {
        for mock in &self.mocks {
            mock.release_held();
        }
        for task in self.held.drain(..) {
            assert_eq!(task.await.expect("held request task"), 200);
        }
    }
}

async fn post(app: &Router, route: &str, body: serde_json::Value, session: Option<&str>) -> u16 {
    let mut request = Request::builder()
        .method("POST")
        .uri(route)
        .header("content-type", "application/json");
    if let Some(session) = session {
        request = request.header("x-session-id", session);
    }
    let response = app
        .clone()
        .oneshot(request.body(Body::from(body.to_string())).unwrap())
        .await
        .expect("router response");
    let status = response.status().as_u16();
    // Drain the body: the router releases its load accounting when it ends.
    let _ = axum::body::to_bytes(response.into_body(), usize::MAX).await;
    status
}

/// Run `request`, returning the outcome delta it produced.
async fn outcomes_of<F: std::future::Future>(request: F) -> (F::Output, BTreeMap<String, u64>) {
    let before = plan_counts();
    let result = request.await;
    (result, delta(&before, &plan_counts()))
}

/// Like [`outcomes_of`], also returning the cost histogram delta.
async fn observed_of<F: std::future::Future>(
    request: F,
) -> (
    F::Output,
    BTreeMap<String, u64>,
    BTreeMap<String, (u64, f64)>,
) {
    let (before, hist_before) = (plan_counts(), cost_histograms());
    let result = request.await;
    (
        result,
        delta(&before, &plan_counts()),
        cost_delta(&hist_before, &cost_histograms()),
    )
}

// ---------------------------------------------------------------------------
// Scenarios (design doc section 3.4). Block size 16, min_prefix_blocks 64.
//
// Planner prices, for a destination with nothing cached that would fetch `c`
// tokens from a source: transfer = 5 ms + c * 5 us; fetch = transfer + remaining
// prefill * 50 us; wait = source queued tokens * 50 us + source remaining
// prefill * 50 us. A plan needs fetch < wait and transfer < 0.8 * saved.
// ---------------------------------------------------------------------------

/// 1: A holds a 100-block prefix and is busy; B is idle and cold -> `planned`.
///
/// Prompt P = 1700 tokens, A caches 100 blocks (1600 tokens).
/// Two held requests (1600 cached + 300 tail) each charge A 300 tokens and route
/// to A (300 < 1900, then (300+300)*2 = 1200 < 1900). Then for P:
/// A = (100 + 600) * 3 = 2100 > B = 1700, so B is picked.
/// Plan: transfer 13 ms, fetch 13 + 5 = 18 ms, wait 600 * 0.05 + 5 = 35 ms.
#[tokio::test]
async fn s1_busy_source_with_long_prefix_is_planned() {
    let mut fx = Fixture::start(Options::new(&[("a", "m"), ("b", "m")])).await;
    let prompt = seq(1, 1700);
    fx.mock("a")
        .store_prefix(&prompt, 100, Residency::Device)
        .await;
    fx.converge(&prompt, "a", 100).await;
    fx.make_busy("a", &prompt[..1600], 300, 2).await;

    let (b_before, a_before) = (fx.mock("b").completions(), fx.mock("a").completions());
    let (status, delta, costs) = observed_of(fx.complete(&prompt, None)).await;

    assert_eq!(status, 200);
    assert_eq!(delta, outcomes(&[("planned", 1)]));
    // Saved prefill: 1600 credited tokens * 0.05 ms = 80 ms.
    assert_cost_observations(
        &costs,
        &[
            ("cost:fetch", 1, 18.0),
            ("cost:wait", 1, 35.0),
            ("saved", 1, 80.0),
        ],
    );
    assert_eq!(fx.mock("b").completions() - b_before, 1, "B was the pick");
    assert_eq!(
        fx.mock("a").completions(),
        a_before,
        "A was only the source"
    );
    fx.finish().await;
}

/// 2: same prefix, A idle -> `source_not_busy`.
///
/// Under the default greedy picker an idle holder of a 100-block prefix always
/// beats an idle cold worker (cost 100 vs 1700), so the pick would be A itself
/// (`picked_holds_longest_prefix`) and the planner never sees an idle source.
/// The router only picks B here when the picker is non-greedy: with
/// `router_temperature` 2000 it samples with weight exp(-(cost - min) / T), which
/// gives B about 31% of requests. The test sends requests (each completes and
/// leaves both workers idle) until the planner has seen B picked, bounded at 300
/// (all-A probability 0.69^300, i.e. never). Every request must end in one of the
/// two outcomes this setup can produce, and none may plan.
#[tokio::test]
async fn s2_idle_source_is_not_busy() {
    let mut options = Options::new(&[("a", "m"), ("b", "m")]);
    options.router_temperature = 2000.0;
    let fx = Fixture::start(options).await;
    let prompt = seq(2, 1700);
    fx.mock("a")
        .store_prefix(&prompt, 100, Residency::Device)
        .await;
    fx.converge(&prompt, "a", 100).await;

    let before = plan_counts();
    let mut seen = BTreeMap::new();
    for _ in 0..300 {
        assert_eq!(fx.complete(&prompt, None).await, 200);
        seen = delta(&before, &plan_counts());
        if seen.contains_key("source_not_busy") {
            break;
        }
    }
    assert!(
        seen.get("source_not_busy").copied().unwrap_or(0) >= 1,
        "B was never picked in 300 requests: {seen:?}"
    );
    assert!(
        seen.keys()
            .all(|k| k == "source_not_busy" || k == "picked_holds_longest_prefix"),
        "unexpected outcomes with an idle source: {seen:?}"
    );
    fx.finish().await;
}

/// 2b: the deterministic form of 2. Greedy routing, but the source has one
/// request in flight and `source_load_threshold` is 2, so the source is loaded
/// enough to lose the pick yet not busy by the planner's rule.
///
/// One held request (1600 cached + 2400 tail) charges A 2400 tokens, so for the
/// 1700 token P: A = (100 + 2400) * 2 = 5000 > B = 1700 -> B is picked and A has
/// 1 active request < 2.
#[tokio::test]
async fn s2b_source_below_load_threshold_is_not_busy() {
    let mut options = Options::new(&[("a", "m"), ("b", "m")]);
    options.transfer.source_load_threshold = 2;
    let mut fx = Fixture::start(options).await;
    let prompt = seq(12, 1700);
    fx.mock("a")
        .store_prefix(&prompt, 100, Residency::Device)
        .await;
    fx.converge(&prompt, "a", 100).await;
    fx.make_busy("a", &prompt[..1600], 2400, 1).await;

    let b_before = fx.mock("b").completions();
    let (status, delta) = outcomes_of(fx.complete(&prompt, None)).await;

    assert_eq!(status, 200);
    assert_eq!(delta, outcomes(&[("source_not_busy", 1)]));
    assert_eq!(fx.mock("b").completions() - b_before, 1, "B was the pick");
    fx.finish().await;
}

/// 3: A holds only 40 blocks and is busy -> `prefix_too_short` (40 < 64).
///
/// Prompt P = 700 tokens, A caches 40 blocks (640). Two held requests
/// (640 cached + 100 tail) charge A 100 tokens each and route to A (100 < 740,
/// then (100+100)*2 = 400 < 740). For P: A = (60 + 200) * 3 = 780 > B = 700.
#[tokio::test]
async fn s3_short_prefix_is_not_worth_fetching() {
    let mut fx = Fixture::start(Options::new(&[("a", "m"), ("b", "m")])).await;
    let prompt = seq(3, 700);
    fx.mock("a")
        .store_prefix(&prompt, 40, Residency::Device)
        .await;
    fx.converge(&prompt, "a", 60).await;
    fx.make_busy("a", &prompt[..640], 100, 2).await;

    let b_before = fx.mock("b").completions();
    let (status, delta, costs) = observed_of(fx.complete(&prompt, None)).await;

    assert_eq!(status, 200);
    assert_eq!(delta, outcomes(&[("prefix_too_short", 1)]));
    // Skipped before the cost rule: nothing is observed.
    assert_cost_observations(&costs, &[]);
    assert_eq!(fx.mock("b").completions() - b_before, 1, "B was the pick");
    fx.finish().await;
}

/// 4: the longest holder (A, 100 blocks) is excluded by a retry; a busy
/// 80-block peer (C) is still eligible -> `planned` from C.
///
/// Setup (P = 1700 tokens):
/// * C stores 80 blocks, then three held requests (1280 cached + 120 tail) make
///   it busy. Each routes to C (120, then (120+120)*2 = 480, then
///   (120+240)*3 = 1080, all below the cold 1400), leaving 3 active and 360
///   queued tokens. For P, C = (420 + 360) * 4 = 3120 > B = 1700.
/// * A stores 100 blocks and takes one small held request (1600 + 50 tail).
///   For P, A = (100 + 50) * 2 = 300, the cheapest, so attempt 1 picks A.
/// * A answers the request with HTTP 500, the router retries with A excluded and
///   attempt 2 picks B (1700 < C's 3120).
///
/// Attempt 1 plans for destination A, which already holds the longest prefix
/// (`picked_holds_longest_prefix`). Attempt 2 ranks A (excluded) then C (busy,
/// 80 blocks, fetch 11.4 + 21 = 32.4 ms < wait 18 + 21 = 39 ms) -> `planned`.
#[tokio::test]
async fn s4_excluded_longest_peer_falls_back_to_eligible_busy_peer() {
    let mut fx = Fixture::start(Options::new(&[("a", "m"), ("b", "m"), ("c", "m")])).await;
    let prompt = seq(4, 1700);

    fx.mock("c")
        .store_prefix(&prompt, 80, Residency::Device)
        .await;
    fx.converge(&prompt, "c", 420).await;
    fx.make_busy("c", &prompt[..1280], 120, 3).await;

    fx.mock("a")
        .store_prefix(&prompt, 100, Residency::Device)
        .await;
    fx.converge(&prompt, "a", 100).await;
    fx.make_busy("a", &prompt[..1600], 50, 1).await;

    let (a_before, b_before, c_before) = (
        fx.mock("a").completions(),
        fx.mock("b").completions(),
        fx.mock("c").completions(),
    );
    fx.mock("a").fail_next(1);
    let (status, delta) = outcomes_of(fx.complete(&prompt, None)).await;

    assert_eq!(status, 200);
    assert_eq!(
        delta,
        outcomes(&[("picked_holds_longest_prefix", 1), ("planned", 1)])
    );
    assert_eq!(fx.mock("a").completions() - a_before, 1, "A failed once");
    assert_eq!(
        fx.mock("b").completions() - b_before,
        1,
        "B served the retry"
    );
    assert_eq!(
        fx.mock("c").completions(),
        c_before,
        "C was only the source"
    );
    fx.finish().await;
}

/// 5: the source holds 100 device blocks plus 25 External blocks, 280 queued
/// tokens and 4 active requests; the destination is idle -> `cost_rule`.
///
/// P = 2000 tokens (125 blocks). External is the owner domain, not a medium: the
/// mock publishes the second copy with `ownership: "pool"`, which the router
/// indexes as `StorageTier::External`. The router credits it at 0.6, so A's
/// work for P is 2000 - (100 + 25 * 0.6) * 16 = 160 tokens (the test waits for
/// exactly that, which proves the External tier was indexed).
/// Four held requests (1600 cached + 70 tail) charge A 70 tokens each (70 j^2 <
/// 1670), leaving 4 active and 280 queued. For P: A = (160 + 280) * 5 = 2200 >
/// B = 2000, so B is picked.
/// Only the 100 device blocks are transferable: transfer 5 + 1600 * 0.005 = 13
/// ms, fetch 13 + 400 * 0.05 = 33 ms, wait 280 * 0.05 + 160 * 0.05 = 22 ms.
/// Fetching loses, so `cost_rule` (design: fetch 33 ms against wait 22 ms).
#[tokio::test]
async fn s5_external_tier_makes_waiting_cheaper_than_fetching() {
    let mut fx = Fixture::start(Options::new(&[("a", "m"), ("b", "m")])).await;
    let prompt = seq(5, 2000);
    fx.mock("a")
        .store_prefix(&prompt, 100, Residency::Device)
        .await;
    fx.mock("a")
        .store_prefix(&prompt, 125, Residency::External)
        .await;
    fx.converge(&prompt, "a", 160).await;
    fx.make_busy("a", &prompt[..1600], 70, 4).await;

    let b_before = fx.mock("b").completions();
    let (status, delta, costs) = observed_of(fx.complete(&prompt, None)).await;

    assert_eq!(status, 200);
    assert_eq!(delta, outcomes(&[("cost_rule", 1)]));
    // The estimates survive the failed rule; nothing is saved because no plan
    // was made.
    assert_cost_observations(&costs, &[("cost:fetch", 1, 33.0), ("cost:wait", 1, 22.0)]);
    assert_eq!(fx.mock("b").completions() - b_before, 1, "B was the pick");
    fx.finish().await;
}

/// 6: requests without token ids -> `missing_tokens`.
///
/// A text prompt and a chat request both need the worker to render token ids;
/// the mock serves neither `/render` nor `/tokenize`, so the router has no ids
/// and never consults the index. A busy long-prefix source is set up so that the
/// missing ids are the only thing between this request and a plan.
#[tokio::test]
async fn s6_request_without_token_ids_is_missing_tokens() {
    let mut fx = Fixture::start(Options::new(&[("a", "m"), ("b", "m")])).await;
    let prompt = seq(6, 1700);
    fx.mock("a")
        .store_prefix(&prompt, 100, Residency::Device)
        .await;
    fx.converge(&prompt, "a", 100).await;
    fx.make_busy("a", &prompt[..1600], 300, 2).await;

    let (status, text_delta) = outcomes_of(fx.post(
        "/v1/completions",
        json!({"prompt": "the quick brown fox", "max_tokens": 1}),
        None,
    ))
    .await;
    assert_eq!(status, 200);
    assert_eq!(text_delta, outcomes(&[("missing_tokens", 1)]));

    let (status, chat_delta) = outcomes_of(fx.post(
        "/v1/chat/completions",
        json!({"messages": [{"role": "user", "content": "hello"}], "max_tokens": 1}),
        None,
    ))
    .await;
    assert_eq!(status, 200);
    assert_eq!(chat_delta, outcomes(&[("missing_tokens", 1)]));
    fx.finish().await;
}

/// 7: a request routed by a session binding -> `session_binding`.
///
/// Same setup as scenario 1 with `kv_aware_session_affinity_ttl_secs` set. The
/// first request of session `s` is a normal selection (B, planned) and binds the
/// session to B. The second request of `s` is served by the binding, and the
/// planner reports `session_binding` without planning, even though A is still
/// busy and a fetch would still look attractive.
#[tokio::test]
async fn s7_session_binding_skips_the_planner() {
    let mut options = Options::new(&[("a", "m"), ("b", "m")]);
    options.session_ttl_secs = Some(1800);
    let mut fx = Fixture::start(options).await;
    let prompt = seq(7, 1700);
    fx.mock("a")
        .store_prefix(&prompt, 100, Residency::Device)
        .await;
    fx.converge(&prompt, "a", 100).await;
    fx.make_busy("a", &prompt[..1600], 300, 2).await;

    let (status, first) = outcomes_of(fx.complete(&prompt, Some("s"))).await;
    assert_eq!(status, 200);
    assert_eq!(first, outcomes(&[("planned", 1)]));
    assert_eq!(fx.mock("b").completions(), 1, "first turn went to B");

    let (status, second) = outcomes_of(fx.complete(&prompt, Some("s"))).await;
    assert_eq!(status, 200);
    assert_eq!(second, outcomes(&[("session_binding", 1)]));
    assert_eq!(fx.mock("b").completions(), 2, "second turn stayed on B");
    fx.finish().await;
}

/// 8: two workers with different cache identities -> `no_peer`.
///
/// Same load setup as scenario 1, but A serves model `model-x` and B serves
/// `model-y`, so the router builds two cache identities. B is picked on cost and
/// has no peer in its identity group, whatever A holds.
#[tokio::test]
async fn s8_different_cache_identities_have_no_peer() {
    let mut fx = Fixture::start(Options::new(&[("a", "model-x"), ("b", "model-y")])).await;
    let prompt = seq(8, 1700);
    fx.mock("a")
        .store_prefix(&prompt, 100, Residency::Device)
        .await;
    fx.converge(&prompt, "a", 100).await;
    fx.make_busy("a", &prompt[..1600], 300, 2).await;

    let b_before = fx.mock("b").completions();
    let (status, delta) = outcomes_of(fx.complete(&prompt, None)).await;

    assert_eq!(status, 200);
    assert_eq!(delta, outcomes(&[("no_peer", 1)]));
    assert_eq!(fx.mock("b").completions() - b_before, 1, "B was the pick");
    fx.finish().await;
}

/// 9 (stretch): a lost event batch, then recovery.
///
/// The mock consumes one sequence number without publishing it and then
/// publishes an unrelated store, so the router's subscriber sees a gap with no
/// replay endpoint. It must drop everything it believed about A (credit is never
/// kept across an unrecoverable gap). Republishing the prefix after the gap is a
/// fresh observation and restores scenario 1's outcome. There is no replay
/// endpoint in the mock, so this exercises the invalidate-and-relearn path, not
/// replay itself.
#[tokio::test]
async fn s9_gap_invalidates_then_republish_recovers() {
    let mut fx = Fixture::start(Options::new(&[("a", "m"), ("b", "m")])).await;
    let prompt = seq(9, 1700);
    fx.mock("a")
        .store_prefix(&prompt, 100, Residency::Device)
        .await;
    fx.converge(&prompt, "a", 100).await;

    // Lose a batch: the next one arrives with seq + 2.
    fx.mock("a").skip_seq().await;
    let unrelated = seq(19, 16);
    fx.mock("a")
        .store_prefix(&unrelated, 1, Residency::Device)
        .await;
    // The prefix credit is gone (both workers cold: full 1700 tokens of work).
    fx.converge_work(&prompt, 1700).await;

    fx.mock("a")
        .store_prefix(&prompt, 100, Residency::Device)
        .await;
    fx.converge(&prompt, "a", 100).await;
    fx.make_busy("a", &prompt[..1600], 300, 2).await;

    let (status, delta) = outcomes_of(fx.complete(&prompt, None)).await;
    assert_eq!(status, 200);
    assert_eq!(delta, outcomes(&[("planned", 1)]));
    fx.finish().await;
}

/// `BlockRemoved` and `AllBlocksCleared` take credit away again, and a later
/// store brings it back, all through the same wiring. (Not a design-table row.)
/// Checked while both workers are idle, where the router's own pick shows the
/// index state directly; then a plan is made from the recovered index.
#[tokio::test]
async fn removal_events_drop_credit_and_restore_recovers_it() {
    let mut fx = Fixture::start(Options::new(&[("a", "m"), ("b", "m")])).await;
    let prompt = seq(10, 1700);
    let stored = fx
        .mock("a")
        .store_prefix(&prompt, 100, Residency::Device)
        .await;
    fx.converge(&prompt, "a", 100).await;

    fx.mock("a").evict(&stored).await;
    fx.converge_work(&prompt, 1700).await;

    fx.mock("a")
        .store_prefix(&prompt, 100, Residency::Device)
        .await;
    fx.converge(&prompt, "a", 100).await;

    fx.mock("a").clear_all().await;
    fx.converge_work(&prompt, 1700).await;

    fx.mock("a")
        .store_prefix(&prompt, 100, Residency::Device)
        .await;
    fx.converge(&prompt, "a", 100).await;
    fx.make_busy("a", &prompt[..1600], 300, 2).await;

    let (status, delta) = outcomes_of(fx.complete(&prompt, None)).await;
    assert_eq!(status, 200);
    assert_eq!(delta, outcomes(&[("planned", 1)]));
    fx.finish().await;
}
