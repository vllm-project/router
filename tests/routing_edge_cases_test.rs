//! CI checks for the fixtures timed by `benches/routing_input.rs`, plus
//! stale-tenant cleanup and the HTTP request-body limit.

mod common;

use axum::body::Body;
use axum::http::header::{CONTENT_LENGTH, CONTENT_TYPE};
use axum::http::{Request, StatusCode};
use common::bench_corpus::{completion_text, seeded_text, Corpus, CorpusKind, LONG_SIZES, SEED};
use common::bench_mock::BenchMockWorker;
use common::routing_edge::{
    ascii_fork, branch, cache_aware_cases, rendezvous_pair, session_headers, workers,
    CacheAwareFixture, BASE_LOADS, FIRST_HEALTHY, MIN_LOAD, TENANT, WORKERS, WORKER_COUNTS,
};
use std::sync::Arc;
use tower::ServiceExt;
use vllm_router_rs::config::{RouterConfig, RoutingMode};
use vllm_router_rs::policies::{LoadBalancingPolicy, RendezvousHashPolicy, RequestHeaders};
use vllm_router_rs::routers::{RouterFactory, RouterTrait};
use vllm_router_rs::tree::Tree;

/// Check character counts and branch selection, both before and after reset.
/// Collect failures so one run identifies every broken case.
#[test]
fn cache_aware_cases_take_the_intended_branch() {
    let mut failures = Vec::new();
    for case in cache_aware_cases() {
        let built = case.build();

        let warm: Vec<&str> = built.fixture.warm_keys().collect();
        assert_eq!(warm.len(), 1, "{}: one warmed key per case", case.name);
        let tree = Tree::new();
        tree.insert(warm[0], "tenant");
        let result = tree.prefix_match_with_counts(&built.probe);
        tree.remove_tenant("tenant");
        let matched = (result.matched_char_count, result.input_char_count);
        if matched != built.matched_chars {
            failures.push(format!(
                "{}: matched (chars, of) {:?}, expected {:?}",
                case.name, matched, built.matched_chars
            ));
        }

        for pass in ["fresh", "after reset"] {
            let got = built.fixture.select(&built.probe);
            if got != Some(case.expect) {
                failures.push(format!(
                    "{} ({pass}): selected {:?} = {}, expected {}",
                    case.name,
                    got,
                    got.map_or("none", branch),
                    branch(case.expect)
                ));
            }
            built.fixture.reset();
        }
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

/// An unhealthy tenant falls back to W0 and is removed from the tree.
/// After it recovers, the old key must miss (W3), not hit W1 or fall back again.
#[test]
fn stale_tenant_falls_back_and_is_forgotten() {
    let (warm, _) = ascii_fork(SEED, 2048, 2048);
    let fixture = CacheAwareFixture::new(vec![(TENANT, warm.clone())], &BASE_LOADS, &[TENANT]);
    assert_eq!(
        fixture.select(&warm),
        Some(FIRST_HEALTHY),
        "W1 down: {}",
        branch(FIRST_HEALTHY)
    );
    fixture.set_healthy(TENANT, true);
    assert_eq!(
        fixture.select(&warm),
        Some(MIN_LOAD),
        "W1 back: {}",
        branch(MIN_LOAD)
    );
}

/// The worker-scaling benchmark must hit each prompt's assigned tenant.
#[test]
fn round_robin_tenants_hit_for_every_worker_count() {
    let corpus = Corpus::new(CorpusKind::Hot64, 2048);
    let prompts: Vec<String> = (0..64).map(|i| corpus.prompt(i)).collect();
    for n in WORKER_COUNTS {
        let fixture = CacheAwareFixture::round_robin(&prompts, n);
        for (i, p) in prompts.iter().enumerate() {
            assert_eq!(fixture.select(p), Some(i % n), "{n} workers, prompt {i}");
        }
    }
}

/// A nonempty session header overrides both prompts; an empty one is ignored.
#[test]
fn session_header_overrides_long_prompts() {
    let policy = RendezvousHashPolicy::new();
    let workers = workers(WORKERS);
    let pick = |prompt: Option<&str>, headers: Option<&RequestHeaders>| {
        policy
            .select_worker_with_headers(&workers, prompt, headers)
            .expect("healthy workers")
    };
    let session = session_headers("edge-session");
    let empty = session_headers("");
    for size in LONG_SIZES {
        let (a, b) = rendezvous_pair(size);
        assert_ne!(
            pick(Some(&a), None),
            pick(Some(&b), None),
            "{size} B, no header"
        );
        let by_header = pick(None, Some(&session));
        assert_eq!(
            pick(Some(&a), Some(&session)),
            by_header,
            "{size} B, prompt a"
        );
        assert_eq!(
            pick(Some(&b), Some(&session)),
            by_header,
            "{size} B, prompt b"
        );
        assert_eq!(
            pick(Some(&a), Some(&empty)),
            pick(Some(&a), None),
            "{size} B, empty header"
        );
        assert_eq!(
            pick(Some(&b), Some(&empty)),
            pick(Some(&b), None),
            "{size} B, empty header"
        );
    }
}

/// A `/v1/completions` body of exactly `len` bytes.
fn completion_body_of_len(len: usize) -> String {
    let envelope = completion_text("").to_string().len();
    let body = completion_text(&seeded_text(SEED, len - envelope)).to_string();
    assert_eq!(body.len(), len);
    body
}

/// `--max-payload-size` is inclusive: a body of exactly the limit is routed
/// to the worker, one byte more is rejected before routing.
#[tokio::test]
async fn body_at_the_payload_limit_is_routed_and_one_byte_more_is_rejected() {
    const LIMIT: usize = 1024 * 1024;
    let mut worker = BenchMockWorker::start().await;
    let config = RouterConfig {
        mode: RoutingMode::Regular {
            worker_urls: vec![worker.url().to_string()],
        },
        max_payload_size: LIMIT,
        worker_startup_timeout_secs: 5,
        worker_startup_check_interval_secs: 1,
        ..Default::default()
    };
    let context = common::create_test_context(config.clone());
    let router: Arc<dyn RouterTrait> = Arc::from(
        RouterFactory::create_router(&context)
            .await
            .expect("router"),
    );
    let app = common::test_app::create_test_app(router, reqwest::Client::new(), &config);

    for (len, expected) in [
        (LIMIT, StatusCode::OK),
        (LIMIT + 1, StatusCode::PAYLOAD_TOO_LARGE),
    ] {
        let request = Request::builder()
            .method("POST")
            .uri("/v1/completions")
            .header(CONTENT_TYPE, "application/json")
            .header(CONTENT_LENGTH, len)
            .body(Body::from(completion_body_of_len(len)))
            .expect("request");
        let response = app.clone().oneshot(request).await.expect("response");
        assert_eq!(response.status(), expected, "body of {len} bytes");
    }
    assert_eq!(
        worker.served(),
        1,
        "only the body at the limit reaches the worker"
    );
    worker.stop().await;
}
