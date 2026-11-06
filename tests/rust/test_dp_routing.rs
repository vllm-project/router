#![cfg(all(feature = "cpu", feature = "ready"))]

//! Integration tests for intra-node data-parallel (DP) routing.
//!
//! These tests verify that when `intra_node_data_parallel_size > 1`, the router:
//!   1. Creates DPAwareWorker instances (not BasicWorker) so endpoint_url() strips @rank
//!   2. Successfully routes requests to the original host:port (no URL corruption)
//!   3. Sends X-data-parallel-rank headers to backend workers
//!   4. Correctly expands N workers × M ranks into N*M DP-aware workers
//!
//! Background: Commit d13949d introduced a regression where hostname:port@rank URLs
//! were parsed as HTTP userinfo (user:pass@host), because PD and regular routers created
//! BasicWorker instances instead of DPAwareWorker when intra_node_data_parallel_size > 1.
//! The @rank suffix in BasicWorker::endpoint_url() caused reqwest to interpret
//! http://node1:8087@1/v1/completions as username=node1, password=8087, host=1.

mod common;

#[cfg(all(test, feature = "cpu", feature = "ready"))]
mod dp_routing_tests {
    use vllm_router_rs::core::{BasicWorker, DPAwareWorker, Worker, WorkerType};
    use vllm_router_rs::routers::http::dp_utils;

    // =====================================================================
    // Test 1: DPAwareWorker vs BasicWorker endpoint URL behavior
    // =====================================================================
    // These tests document the core invariant: DPAwareWorker.endpoint_url()
    // must produce URLs without @rank, while BasicWorker would include it.

    #[test]
    fn test_basic_worker_includes_at_rank_in_endpoint_url() {
        // This test documents the problematic behavior that caused the bug:
        // BasicWorker stores the URL as-is, including @rank suffix.
        let worker = BasicWorker::new("http://node1:8087@1".to_string(), WorkerType::Regular);

        // BasicWorker.url() and base_url() both return the raw URL with @rank
        assert_eq!(worker.url(), "http://node1:8087@1");
        assert_eq!(worker.base_url(), "http://node1:8087@1");

        // endpoint_url() would produce a URL that reqwest parses as userinfo:
        // http://node1:8087@1/v1/completions → user=node1, pass=8087, host=1
        assert_eq!(
            worker.endpoint_url("/v1/completions"),
            "http://node1:8087@1/v1/completions",
            "BasicWorker includes @rank in endpoint URL (this is the bug)"
        );

        // BasicWorker is NOT DP-aware
        assert!(!worker.is_dp_aware());
        assert_eq!(worker.dp_rank(), None);
    }

    #[test]
    fn test_dp_aware_worker_strips_at_rank_from_endpoint_url() {
        // DPAwareWorker correctly separates base_url from dp_rank
        let worker = DPAwareWorker::new("http://node1:8087".to_string(), 1, 4, WorkerType::Regular);

        // url() includes @rank for identification/registry lookup
        assert_eq!(worker.url(), "http://node1:8087@1");

        // base_url() is clean — no @rank
        assert_eq!(worker.base_url(), "http://node1:8087");

        // endpoint_url() uses base_url, producing a valid URL
        assert_eq!(
            worker.endpoint_url("/v1/completions"),
            "http://node1:8087/v1/completions",
            "DPAwareWorker must strip @rank from endpoint URL"
        );

        // DP metadata is accessible separately
        assert!(worker.is_dp_aware());
        assert_eq!(worker.dp_rank(), Some(1));
        assert_eq!(worker.dp_size(), Some(4));
    }

    // =====================================================================
    // Test 2: parse_worker_url round-trip with DPAwareWorker
    // =====================================================================
    // Verifies the pattern used in PD and regular router initialization:
    // expand URL → parse back → create DPAwareWorker

    #[test]
    fn test_dp_expansion_and_worker_creation_round_trip() {
        // Simulate what get_dp_aware_workers() produces
        let expanded_urls = [
            "http://node1:8087@0".to_string(),
            "http://node1:8087@1".to_string(),
            "http://node1:8087@2".to_string(),
            "http://node1:8087@3".to_string(),
        ];
        let dp_size = 4;

        for (expected_rank, url) in expanded_urls.iter().enumerate() {
            let (base_url, dp_rank) = dp_utils::parse_worker_url(url);
            assert_eq!(base_url, "http://node1:8087");
            assert_eq!(dp_rank, Some(expected_rank));

            // Create DPAwareWorker (the correct path)
            let worker = DPAwareWorker::new(
                base_url.clone(),
                dp_rank.unwrap_or(0),
                dp_size,
                WorkerType::Regular,
            );

            // Verify the worker's endpoint URL is clean
            assert_eq!(
                worker.endpoint_url("/v1/chat/completions"),
                "http://node1:8087/v1/chat/completions",
                "DPAwareWorker at rank {} must produce clean endpoint URL",
                expected_rank
            );

            // Verify the worker's identification URL has the @rank
            assert_eq!(worker.url(), format!("http://node1:8087@{}", expected_rank));
        }
    }

    #[tokio::test]
    async fn test_get_dp_aware_workers_expansion() {
        let urls = vec![
            "http://node1:8087".to_string(),
            "http://node2:8087".to_string(),
        ];
        let dp_size = 4;

        let expanded = dp_utils::get_dp_aware_workers(&urls, &None, dp_size)
            .await
            .unwrap();

        // 2 workers × 4 ranks = 8 expanded URLs
        assert_eq!(expanded.len(), 8);

        // Verify each expanded URL can be parsed and creates a correct DPAwareWorker
        for url in &expanded {
            let (base_url, dp_rank) = dp_utils::parse_worker_url(url);
            assert!(
                dp_rank.is_some(),
                "Expanded URL should have dp_rank: {}",
                url
            );

            let worker = DPAwareWorker::new(
                base_url.clone(),
                dp_rank.unwrap(),
                dp_size,
                WorkerType::Regular,
            );

            // The critical check: endpoint_url must NOT contain @
            let endpoint = worker.endpoint_url("/v1/completions");
            assert!(
                !endpoint.contains('@'),
                "endpoint_url must not contain @ (got: {})",
                endpoint
            );

            // The endpoint must resolve to the original host
            assert!(
                endpoint.starts_with(&base_url),
                "endpoint_url must start with base URL: {} (got: {})",
                base_url,
                endpoint
            );
        }
    }

    #[tokio::test]
    async fn test_get_dp_aware_workers_ipv6_expansion() {
        let urls = vec!["https://[2a03:83e4:5006:0090:5f5a:f8c5:0400:0000]:20009".to_string()];
        let dp_size = 2;

        let expanded = dp_utils::get_dp_aware_workers(&urls, &None, dp_size)
            .await
            .unwrap();

        assert_eq!(expanded.len(), 2);

        for url in &expanded {
            let (base_url, dp_rank) = dp_utils::parse_worker_url(url);
            let worker =
                DPAwareWorker::new(base_url, dp_rank.unwrap(), dp_size, WorkerType::Regular);

            let endpoint = worker.endpoint_url("/v1/completions");
            assert!(
                !endpoint.contains('@'),
                "IPv6 endpoint_url must not contain @ (got: {})",
                endpoint
            );
            assert!(
                endpoint.starts_with("https://[2a03:83e4:5006:0090:5f5a:f8c5:0400:0000]:20009"),
                "IPv6 endpoint must preserve bracketed address (got: {})",
                endpoint
            );
        }
    }

    // =====================================================================
    // Test 3: Verify dp_size=1 creates BasicWorker (no expansion)
    // =====================================================================

    #[test]
    fn test_no_dp_expansion_when_dp_size_is_one() {
        let url = "http://node1:8087";
        let (base_url, dp_rank) = dp_utils::parse_worker_url(url);

        // No @rank in the URL, so parse_worker_url returns None
        assert_eq!(base_url, url);
        assert_eq!(dp_rank, None);

        // With dp_size=1, a BasicWorker is appropriate
        let worker = BasicWorker::new(url.to_string(), WorkerType::Regular);
        assert_eq!(
            worker.endpoint_url("/v1/completions"),
            "http://node1:8087/v1/completions"
        );
        assert!(!worker.is_dp_aware());
    }
}

// =====================================================================
// End-to-end integration tests with mock workers
// =====================================================================
// These tests start real HTTP servers, create a router with
// intra_node_data_parallel_size > 1, and verify requests actually
// reach the correct backend. If @rank corrupted the URL, reqwest
// would connect to the wrong host and the request would fail.

#[cfg(all(test, feature = "cpu", feature = "ready"))]
mod dp_e2e_tests {
    use super::common;
    use axum::body::Body;
    use axum::extract::Request;
    use common::mock_worker::{
        clear_captured_requests, get_captured_requests, HealthStatus, MockWorker, MockWorkerConfig,
        WorkerType,
    };
    use reqwest::Client;
    use serde_json::json;
    use tower::ServiceExt;
    use vllm_router_rs::config::{
        CircuitBreakerConfig, ConnectionMode, PolicyConfig, RetryConfig, RouterConfig, RoutingMode,
    };
    use vllm_router_rs::routers::RouterFactory;

    use std::sync::Arc;

    /// Helper to create a RouterConfig with DP settings for Regular mode
    fn make_regular_config(worker_urls: Vec<String>, dp_size: usize) -> RouterConfig {
        RouterConfig {
            mode: RoutingMode::Regular { worker_urls },
            pd_kv_event_sources: Default::default(),
            policy: PolicyConfig::RoundRobin,
            host: "127.0.0.1".to_string(),
            port: 0,
            max_payload_size: 256 * 1024 * 1024,
            request_timeout_secs: 10,
            worker_startup_timeout_secs: 5,
            worker_startup_check_interval_secs: 1,
            intra_node_data_parallel_size: dp_size,
            api_key: None,
            api_key_validation_urls: vec![],
            discovery: None,
            metrics: None,
            log_dir: None,
            log_level: None,
            request_id_headers: None,
            max_concurrent_requests: 64,
            queue_size: 0,
            queue_timeout_secs: 60,
            rate_limit_tokens_per_second: None,
            cors_allowed_origins: vec![],
            retry: RetryConfig::default(),
            circuit_breaker: CircuitBreakerConfig::default(),
            disable_retries: false,
            disable_circuit_breaker: false,
            kv_aware_session_affinity_ttl_secs: None,
            health_check: vllm_router_rs::config::HealthCheckConfig::default(),
            enable_igw: false,
            connection_mode: ConnectionMode::Http,
            history_backend: vllm_router_rs::config::HistoryBackend::Memory,
            enable_profiling: false,
            profile_timeout_secs: 30,
            epd: None,
            kv_connector: vllm_router_rs::config::KvConnector::Nixl,
            program_scheduling: None,
        }
    }

    /// Helper to create a RouterConfig for PD mode with DP settings
    fn make_pd_config(
        prefill_urls: Vec<(String, Option<u16>)>,
        decode_urls: Vec<String>,
        dp_size: usize,
    ) -> RouterConfig {
        RouterConfig {
            mode: RoutingMode::VllmPrefillDecode {
                prefill_urls,
                decode_urls,
                prefill_policy: None,
                decode_policy: None,
                discovery_address: None,
            },
            pd_kv_event_sources: Default::default(),
            policy: PolicyConfig::RoundRobin,
            host: "127.0.0.1".to_string(),
            port: 0,
            max_payload_size: 256 * 1024 * 1024,
            request_timeout_secs: 10,
            worker_startup_timeout_secs: 5,
            worker_startup_check_interval_secs: 1,
            intra_node_data_parallel_size: dp_size,
            api_key: None,
            api_key_validation_urls: vec![],
            discovery: None,
            metrics: None,
            log_dir: None,
            log_level: None,
            request_id_headers: None,
            max_concurrent_requests: 64,
            queue_size: 0,
            queue_timeout_secs: 60,
            rate_limit_tokens_per_second: None,
            cors_allowed_origins: vec![],
            retry: RetryConfig::default(),
            circuit_breaker: CircuitBreakerConfig::default(),
            disable_retries: false,
            disable_circuit_breaker: false,
            kv_aware_session_affinity_ttl_secs: None,
            health_check: vllm_router_rs::config::HealthCheckConfig::default(),
            enable_igw: false,
            connection_mode: ConnectionMode::Http,
            history_backend: vllm_router_rs::config::HistoryBackend::Memory,
            enable_profiling: false,
            profile_timeout_secs: 30,
            epd: None,
            kv_connector: vllm_router_rs::config::KvConnector::Nixl,
            program_scheduling: None,
        }
    }

    // -----------------------------------------------------------------
    // Regular Router + DP > 1: transparent proxy
    // -----------------------------------------------------------------

    type EpdRequests =
        Arc<std::sync::Mutex<Vec<(String, axum::http::HeaderMap, serde_json::Value)>>>;

    async fn epd_server() -> (String, EpdRequests, tokio::task::JoinHandle<()>) {
        use axum::{
            extract::State,
            http::{HeaderMap, Uri},
            routing::{get, post},
            Json, Router,
        };
        use serde_json::Value;
        use std::sync::Mutex;
        async fn backend(
            State(seen): State<EpdRequests>,
            uri: Uri,
            headers: HeaderMap,
            Json(body): Json<Value>,
        ) -> Json<Value> {
            seen.lock()
                .unwrap()
                .push((uri.path().into(), headers, body.clone()));
            if uri.path().starts_with("/e/") {
                let uuid = body["messages"][0]["content"][0]["uuid"].as_str().unwrap();
                Json(json!({"ec_transfer_params": {(uuid): {
                    "metadata": {"image_grid_thw": [[1, 2, 3]]},
                    "destination": body["ec_transfer_params"]["consumer_zmq"],
                }}}))
            } else if ["p", "p0", "p1"].contains(&uri.path().split('/').nth(1).unwrap()) {
                Json(json!({"kv_transfer_params": {
                    "do_remote_prefill": true, "remote_engine_id": "P"
                }}))
            } else {
                Json(json!({"choices": [{"message": {"content": "A"}}]}))
            }
        }

        let seen: EpdRequests = Arc::new(Mutex::new(Vec::new()));
        let mut app = Router::new();
        for (role, dp_size) in [
            ("e", 2),
            ("p", 2),
            ("d", 2),
            ("pd", 2),
            ("p0", 2),
            ("p1", 4),
        ] {
            let metrics: String = (0..dp_size)
                .map(|rank| format!("vllm:num_requests_running{{engine=\"{rank}\"}} 0\n"))
                .collect();
            app = app
                .route(&format!("/{role}/health"), get(|| async { "ok" }))
                .route(&format!("/{role}/v1/chat/completions"), post(backend))
                .route(
                    &format!("/{role}/metrics"),
                    get(move || {
                        let metrics = metrics.clone();
                        async move { metrics }
                    }),
                );
        }
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let app = app.with_state(seen.clone());
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        (base, seen, server)
    }

    #[tokio::test]
    async fn test_epd_dp2_keeps_encoder_destination_and_consumer_rank_aligned() {
        use axum::http::{HeaderMap, Method, StatusCode};
        use vllm_router_rs::config::{ConfigValidator, EpdConfig};

        for separate_pd in [false, true] {
            for push in [false, true] {
                let (base, seen, server) = epd_server().await;
                let consumer = format!("{base}/{}", if separate_pd { "p" } else { "pd" });
                let addresses: std::collections::HashMap<_, _> = if push {
                    (0..2)
                        .map(|rank| {
                            (
                                format!("{consumer}@{rank}"),
                                format!("tcp://127.0.0.1:{}", 19000 + rank),
                            )
                        })
                        .collect()
                } else {
                    Default::default()
                };
                let mut config = if separate_pd {
                    make_pd_config(vec![(consumer.clone(), None)], vec![format!("{base}/d")], 2)
                } else {
                    make_regular_config(vec![consumer.clone()], 2)
                };
                config.port = 8000;
                config.epd = Some(EpdConfig {
                    encoder_urls: vec![format!("{base}/e")],
                    consumer_zmq_addrs: addresses.clone(),
                });
                ConfigValidator::validate(&config).unwrap();
                if push {
                    let mut incomplete = config.clone();
                    incomplete
                        .epd
                        .as_mut()
                        .unwrap()
                        .consumer_zmq_addrs
                        .remove(&format!("{consumer}@1"));
                    // Map coverage is checked after discovery, not by syntax validation.
                    ConfigValidator::validate(&incomplete).unwrap();
                    let context = common::create_test_context(incomplete);
                    let error = RouterFactory::create_router(&context).await.err().unwrap();
                    assert!(error.contains("consumer_zmq_addrs"), "{error}");
                    assert!(error.contains(&format!("{consumer}@1")), "{error}");
                }
                let context = common::create_test_context(config);
                let router = RouterFactory::create_router(&context).await.unwrap();
                let mut headers = HeaderMap::new();
                headers.insert("x-data-parallel-rank", "99".parse().unwrap());
                for _ in 0..4 {
                    let response = router.route_transparent(Some(&headers), "/v1/chat/completions", &Method::POST, json!({
                        "model": "mock-model",
                        "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}]}]
                    })).await;
                    assert_eq!(response.status(), StatusCode::OK);
                    axum::body::to_bytes(response.into_body(), 4096)
                        .await
                        .unwrap();
                }
                let seen = seen.lock().unwrap();
                let mut ranks = Vec::new();
                for (path, headers, body) in seen.iter() {
                    if path.starts_with("/e/") {
                        assert!(!headers.contains_key("x-data-parallel-rank"));
                        continue;
                    }
                    assert_eq!(headers.get_all("x-data-parallel-rank").iter().count(), 1);
                    let rank = headers["x-data-parallel-rank"]
                        .to_str()
                        .unwrap()
                        .parse::<usize>()
                        .unwrap();
                    assert!(rank < 2);
                    if path.starts_with("/d/") {
                        assert!(body.get("ec_transfer_params").is_none());
                        continue;
                    }
                    ranks.push(rank);
                    let uuid = body["messages"][0]["content"][0]["uuid"].as_str().unwrap();
                    assert_eq!(body["messages"][0]["content"][0]["type"], "image_embeds");
                    if push {
                        assert_eq!(
                            body["ec_transfer_params"][uuid]["destination"],
                            addresses[&format!("{consumer}@{rank}")]
                        );
                    }
                }
                ranks.sort();
                assert_eq!(ranks, [0, 0, 1, 1]);
                assert_eq!(seen.len(), if separate_pd { 12 } else { 8 });
                server.abort();
            }
        }
    }

    #[tokio::test]
    async fn test_epd_regular_preserves_a_pinned_consumer() {
        use axum::http::{Method, StatusCode};
        use vllm_router_rs::config::{ConfigValidator, EpdConfig};

        let (base, seen, server) = epd_server().await;
        let consumer = format!("{base}/pd@1");
        let mut config = make_regular_config(vec![consumer.clone()], 2);
        config.port = 8000;
        config.epd = Some(EpdConfig {
            encoder_urls: vec![format!("{base}/e")],
            consumer_zmq_addrs: [(consumer, "tcp://127.0.0.1:19001".into())].into(),
        });
        ConfigValidator::validate(&config).unwrap();
        let context = common::create_test_context(config);
        let router = RouterFactory::create_router(&context).await.unwrap();
        assert_eq!(context.worker_registry.get_all().len(), 1);
        let response = router.route_transparent(None, "/v1/chat/completions", &Method::POST, json!({
            "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}]}]
        })).await;
        assert_eq!(response.status(), StatusCode::OK);
        axum::body::to_bytes(response.into_body(), 4096)
            .await
            .unwrap();
        let seen = seen.lock().unwrap();
        assert_eq!(seen.len(), 2);
        assert_eq!(seen[1].0, "/pd/v1/chat/completions");
        assert_eq!(seen[1].1["x-data-parallel-rank"], "1");
        server.abort();
    }

    #[tokio::test]
    async fn test_epd_regular_rejects_an_out_of_range_consumer() {
        use vllm_router_rs::config::EpdConfig;

        let (base, seen, server) = epd_server().await;
        for rank in [2, 5] {
            let consumer = format!("{base}/pd@{rank}");
            let mut config = make_regular_config(vec![consumer.clone()], 2);
            config.port = 8000;
            config.epd = Some(EpdConfig {
                encoder_urls: vec![format!("{base}/e")],
                consumer_zmq_addrs: [(consumer.clone(), "tcp://127.0.0.1:19001".into())].into(),
            });
            let context = common::create_test_context(config);
            let error = RouterFactory::create_router(&context).await.err().unwrap();
            assert!(error.contains(&consumer), "{error}");
            assert!(error.contains("DP size 2"), "{error}");
        }
        assert!(seen.lock().unwrap().is_empty());
        server.abort();
    }

    #[tokio::test]
    async fn test_epd_pd_rejects_out_of_range_pinned_ranks() {
        use vllm_router_rs::config::EpdConfig;

        let (base, seen, server) = epd_server().await;
        for role in ["p", "d"] {
            for rank in [2, 5] {
                let pinned = format!("{base}/{role}@{rank}");
                let prefill = if role == "p" {
                    pinned.clone()
                } else {
                    format!("{base}/p")
                };
                let decode = if role == "d" {
                    pinned.clone()
                } else {
                    format!("{base}/d")
                };
                // Both endpoints report DP=2, overriding the configured fallback.
                let mut config = make_pd_config(vec![(prefill, None)], vec![decode], 4);
                config.port = 8000;
                config.epd = Some(EpdConfig {
                    encoder_urls: vec![format!("{base}/e")],
                    consumer_zmq_addrs: Default::default(),
                });
                let context = common::create_test_context(config);
                let Err(error) = RouterFactory::create_router(&context).await else {
                    panic!("P/D must reject a pinned rank outside the discovered DP size");
                };
                assert!(error.contains(&pinned), "{error}");
                assert!(error.contains("DP size 2"), "{error}");
            }
        }
        assert!(seen.lock().unwrap().is_empty());
        server.abort();
    }

    #[tokio::test]
    async fn test_epd_address_map_matches_discovered_prefill_ranks() {
        use vllm_router_rs::config::{ConfigValidator, EpdConfig};

        let (base, _, server) = epd_server().await;
        let prefill = vec![(format!("{base}/p0"), None), (format!("{base}/p1"), None)];
        let mut config = make_pd_config(prefill, vec![format!("{base}/d")], 2);
        config.port = 8000;
        config.epd = Some(EpdConfig {
            encoder_urls: vec![format!("{base}/e")],
            consumer_zmq_addrs: [("p0", 2), ("p1", 4)]
                .into_iter()
                .flat_map(|(role, size)| {
                    let base = &base;
                    (0..size).map(move |rank| {
                        (
                            format!("{base}/{role}@{rank}"),
                            format!("tcp://127.0.0.1:{}", 19000 + rank),
                        )
                    })
                })
                .collect(),
        });
        ConfigValidator::validate(&config).unwrap();
        let context = common::create_test_context(config.clone());
        let _router = RouterFactory::create_router(&context).await.unwrap();
        assert_eq!(context.worker_registry.get_prefill_workers().len(), 6);

        for rank in [2, 3] {
            config
                .epd
                .as_mut()
                .unwrap()
                .consumer_zmq_addrs
                .remove(&format!("{base}/p1@{rank}"));
        }
        ConfigValidator::validate(&config).unwrap();
        let context = common::create_test_context(config);
        let error = RouterFactory::create_router(&context).await.err().unwrap();
        assert!(error.contains("consumer_zmq_addrs"), "{error}");
        assert!(error.contains(&format!("{base}/p1@2")), "{error}");
        server.abort();
    }

    #[tokio::test]
    async fn test_epd_rejects_rank_bound_push_handles_before_dispatch() {
        use axum::http::{HeaderMap, Method, StatusCode};
        use vllm_router_rs::config::{ConfigValidator, EpdConfig};

        for (separate_pd, dp_size, push) in [
            (false, 2, true),
            (true, 2, true),
            (false, 1, true),
            (false, 2, false),
        ] {
            let (base, seen, server) = epd_server().await;
            let consumer = format!("{base}/{}", if separate_pd { "p" } else { "pd" });
            let mut config = if separate_pd {
                make_pd_config(
                    vec![(consumer.clone(), None)],
                    vec![format!("{base}/d")],
                    dp_size,
                )
            } else {
                make_regular_config(vec![consumer.clone()], dp_size)
            };
            config.port = 8000;
            config.epd = Some(EpdConfig {
                encoder_urls: vec![format!("{base}/e")],
                consumer_zmq_addrs: if push {
                    (0..dp_size)
                        .map(|rank| {
                            let url = if dp_size == 1 {
                                consumer.clone()
                            } else {
                                format!("{consumer}@{rank}")
                            };
                            (url, format!("tcp://127.0.0.1:{}", 19000 + rank))
                        })
                        .collect()
                } else {
                    Default::default()
                },
            });
            ConfigValidator::validate(&config).unwrap();
            let context = common::create_test_context(config);
            let router = RouterFactory::create_router(&context).await.unwrap();
            let mut headers = HeaderMap::new();
            headers.insert("x-data-parallel-rank", "1".parse().unwrap());
            for mixed in [false, true] {
                let mut content = vec![
                    json!({"type": "image_embeds", "image_embeds": {"image_grid_thw": [1, 2, 3]}, "uuid": "cached"}),
                ];
                if mixed {
                    content.push(json!({"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}));
                }
                let response = router.route_transparent(if dp_size > 1 { Some(&headers) } else { None }, "/v1/chat/completions", &Method::POST, json!({
                    "messages": [{"role": "user", "content": content}],
                    "ec_transfer_params": {"cached": {"transfer_id": "old"}, "ec_items": [{"mm_hash": "cached", "transfer_id": "old"}]}
                })).await;
                let rejected = push && dp_size > 1;
                assert_eq!(
                    response.status(),
                    if rejected {
                        StatusCode::BAD_REQUEST
                    } else {
                        StatusCode::OK
                    }
                );
                axum::body::to_bytes(response.into_body(), 4096)
                    .await
                    .unwrap();
                if rejected {
                    assert!(seen.lock().unwrap().is_empty());
                }
            }
            server.abort();
        }
    }

    #[tokio::test]
    async fn test_regular_router_dp2_transparent_proxy_reaches_backend() {
        // Start a mock worker
        let mut worker = MockWorker::new(MockWorkerConfig {
            port: 0,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 0.0,
        });
        let worker_url = worker.start().await.unwrap();
        let port: u16 = worker_url.split(':').next_back().unwrap().parse().unwrap();
        clear_captured_requests(port);

        // Create router with dp_size=2
        let config = make_regular_config(vec![worker_url.clone()], 2);
        let app_context = common::create_test_context(config.clone());
        let router = RouterFactory::create_router(&app_context).await.unwrap();
        let router = Arc::from(router);

        // Wait for health checks
        tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

        // Create test app with transparent proxy enabled
        let app = common::test_app::create_test_app(Arc::clone(&router), Client::new(), &config);

        // Send a chat completion request through the transparent proxy
        let body = json!({
            "model": "mock-model",
            "messages": [{"role": "user", "content": "hello"}]
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_string(&body).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();

        // If the URL was corrupted by @rank, reqwest would fail to connect
        // (e.g., trying to reach host "0" or "1" instead of 127.0.0.1)
        assert_eq!(
            resp.status().as_u16(),
            200,
            "Request should succeed when DP > 1 (got {}). URL corruption by @rank would cause connection failure.",
            resp.status()
        );

        // Verify the mock worker received the request with X-data-parallel-rank header
        let captured = get_captured_requests(port);
        assert!(
            !captured.is_empty(),
            "Mock worker should have received at least one request"
        );

        // At least one request should have X-data-parallel-rank header
        let has_dp_rank_header = captured
            .iter()
            .any(|r| r.headers.contains_key("x-data-parallel-rank"));
        assert!(
            has_dp_rank_header,
            "Request should include X-data-parallel-rank header when DP > 1. Headers: {:?}",
            captured[0].headers
        );

        worker.stop().await;
    }

    #[tokio::test]
    async fn test_regular_router_dp1_no_dp_header() {
        // Baseline: with dp_size=1, no X-data-parallel-rank header should be sent
        let mut worker = MockWorker::new(MockWorkerConfig::default());
        let worker_url = worker.start().await.unwrap();
        let port: u16 = worker_url.split(':').next_back().unwrap().parse().unwrap();
        clear_captured_requests(port);

        let config = make_regular_config(vec![worker_url.clone()], 1);
        let app_context = common::create_test_context(config.clone());
        let router = RouterFactory::create_router(&app_context).await.unwrap();
        let router = Arc::from(router);

        tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

        let app = common::test_app::create_test_app(Arc::clone(&router), Client::new(), &config);

        let body = json!({
            "model": "mock-model",
            "messages": [{"role": "user", "content": "hello"}]
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_string(&body).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status().as_u16(), 200);

        let captured = get_captured_requests(port);
        assert!(!captured.is_empty());

        // With dp_size=1, no X-data-parallel-rank header
        let has_dp_rank_header = captured
            .iter()
            .any(|r| r.headers.contains_key("x-data-parallel-rank"));
        assert!(
            !has_dp_rank_header,
            "Should NOT include X-data-parallel-rank when DP = 1"
        );

        worker.stop().await;
    }

    #[tokio::test]
    async fn test_regular_router_dp2_get_v1_models() {
        let mut worker = MockWorker::new(MockWorkerConfig::default());
        let worker_url = worker.start().await.unwrap();
        let port: u16 = worker_url.split(':').next_back().unwrap().parse().unwrap();
        clear_captured_requests(port);

        let config = make_regular_config(vec![worker_url.clone()], 2);
        let app_context = common::create_test_context(config.clone());
        let router = RouterFactory::create_router(&app_context).await.unwrap();
        let router = Arc::from(router);

        tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

        let app = common::test_app::create_test_app(Arc::clone(&router), Client::new(), &config);

        let req = Request::builder()
            .uri("/v1/models")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(
            resp.status().as_u16(),
            200,
            "GET /v1/models must succeed when DP > 1 (got {}). @rank in the worker URL was misinterpreted as userinfo.",
            resp.status()
        );

        // The 200 alone could be misleading if a fallback were added later;
        // assert the body is the mock worker's model list.
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(
            body["data"][0]["id"].as_str(),
            Some("mock-model"),
            "GET /v1/models should return the worker's model list. Body: {:?}",
            body
        );

        // Verify the forwarded X-data-parallel-rank matches the selected
        // worker's rank (0 or 1 here; both DP ranks share this mock worker,
        // and worker selection order is not deterministic).
        let captured = get_captured_requests(port);
        let models_req = captured
            .iter()
            .find(|r| r.path == "/v1/models")
            .expect("Mock worker should have received the GET /v1/models request");
        let rank_values = models_req
            .headers
            .get("x-data-parallel-rank")
            .expect("GET proxy must forward X-data-parallel-rank when DP > 1");
        assert_eq!(
            rank_values.len(),
            1,
            "exactly one X-data-parallel-rank value expected, got {:?}",
            rank_values
        );
        let rank: usize = rank_values[0]
            .parse()
            .expect("X-data-parallel-rank must be numeric");
        assert!(
            rank < 2,
            "forwarded rank must be one of the router's DP ranks (0 or 1), got {}",
            rank
        );

        worker.stop().await;
    }

    #[tokio::test]
    async fn test_regular_router_dp2_get_v1_models_overrides_client_dp_rank() {
        // A client-supplied X-data-parallel-rank must not leak through or
        // duplicate the router-selected rank: the worker should receive
        // exactly one value, matching the router's selection.
        let mut worker = MockWorker::new(MockWorkerConfig::default());
        let worker_url = worker.start().await.unwrap();
        let port: u16 = worker_url.split(':').next_back().unwrap().parse().unwrap();
        clear_captured_requests(port);

        let config = make_regular_config(vec![worker_url.clone()], 2);
        let app_context = common::create_test_context(config.clone());
        let router = RouterFactory::create_router(&app_context).await.unwrap();
        let router = Arc::from(router);

        tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

        let app = common::test_app::create_test_app(Arc::clone(&router), Client::new(), &config);

        let req = Request::builder()
            .uri("/v1/models")
            .header("x-data-parallel-rank", "99")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(
            resp.status().as_u16(),
            200,
            "GET /v1/models must succeed even when the client sends a conflicting rank (got {})",
            resp.status()
        );

        let captured = get_captured_requests(port);
        let models_req = captured
            .iter()
            .find(|r| r.path == "/v1/models")
            .expect("Mock worker should have received the GET /v1/models request");
        let rank_values = models_req
            .headers
            .get("x-data-parallel-rank")
            .expect("router-selected X-data-parallel-rank must be forwarded");
        assert_eq!(
            rank_values.len(),
            1,
            "client-supplied rank must be filtered out; worker received {:?}",
            rank_values
        );
        assert_ne!(
            rank_values[0], "99",
            "worker must receive the router's selection, not the client's value"
        );
        let rank: usize = rank_values[0]
            .parse()
            .expect("X-data-parallel-rank must be numeric");
        assert!(
            rank < 2,
            "forwarded rank must be one of the router's DP ranks (0 or 1), got {}",
            rank
        );

        worker.stop().await;
    }

    // -----------------------------------------------------------------
    // Regular Router + DP > 1: worker registry verification
    // -----------------------------------------------------------------

    #[tokio::test]
    async fn test_regular_router_dp2_creates_correct_worker_count() {
        let mut worker = MockWorker::new(MockWorkerConfig::default());
        let worker_url = worker.start().await.unwrap();

        // dp_size=3 should expand 1 worker URL into 3 DP-aware workers
        let config = make_regular_config(vec![worker_url.clone()], 3);
        let app_context = common::create_test_context(config.clone());
        let _router = RouterFactory::create_router(&app_context).await.unwrap();

        tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

        let all_workers = app_context.worker_registry.get_all();
        assert_eq!(
            all_workers.len(),
            3,
            "1 worker URL × dp_size=3 should produce 3 DP-aware workers, got {}",
            all_workers.len()
        );

        // Verify each worker is DP-aware with correct properties
        for w in &all_workers {
            assert!(
                w.is_dp_aware(),
                "Worker {} should be DP-aware when dp_size > 1",
                w.url()
            );
            assert!(
                w.dp_rank().is_some(),
                "DP-aware worker should have a dp_rank"
            );
            assert_eq!(
                w.dp_size(),
                Some(3),
                "DP-aware worker should have dp_size=3"
            );

            // Critical: endpoint_url must not contain @
            let endpoint = w.endpoint_url("/v1/completions");
            assert!(
                !endpoint.contains('@'),
                "endpoint_url must not contain @rank (got: {})",
                endpoint
            );
        }

        // Verify all 3 ranks (0, 1, 2) are present
        let mut ranks: Vec<usize> = all_workers.iter().filter_map(|w| w.dp_rank()).collect();
        ranks.sort();
        assert_eq!(ranks, vec![0, 1, 2], "Should have ranks 0, 1, 2");

        worker.stop().await;
    }

    #[tokio::test]
    async fn test_regular_router_dp1_creates_basic_workers() {
        let mut worker = MockWorker::new(MockWorkerConfig::default());
        let worker_url = worker.start().await.unwrap();

        let config = make_regular_config(vec![worker_url.clone()], 1);
        let app_context = common::create_test_context(config.clone());
        let _router = RouterFactory::create_router(&app_context).await.unwrap();

        tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

        let all_workers = app_context.worker_registry.get_all();
        assert_eq!(all_workers.len(), 1, "dp_size=1 should produce 1 worker");

        let w = &all_workers[0];
        assert!(
            !w.is_dp_aware(),
            "Worker should NOT be DP-aware when dp_size=1"
        );
        assert_eq!(w.dp_rank(), None);
        assert!(!w.url().contains('@'), "URL should not contain @rank");

        worker.stop().await;
    }

    // -----------------------------------------------------------------
    // Multiple workers + DP > 1
    // -----------------------------------------------------------------

    #[tokio::test]
    async fn test_multiple_workers_dp2_expansion() {
        let mut worker1 = MockWorker::new(MockWorkerConfig::default());
        let mut worker2 = MockWorker::new(MockWorkerConfig::default());
        let url1 = worker1.start().await.unwrap();
        let url2 = worker2.start().await.unwrap();

        let config = make_regular_config(vec![url1.clone(), url2.clone()], 2);
        let app_context = common::create_test_context(config.clone());
        let _router = RouterFactory::create_router(&app_context).await.unwrap();

        tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

        let all_workers = app_context.worker_registry.get_all();
        assert_eq!(
            all_workers.len(),
            4,
            "2 worker URLs × dp_size=2 should produce 4 DP-aware workers"
        );

        // All should be DP-aware
        for w in &all_workers {
            assert!(w.is_dp_aware());
            let endpoint = w.endpoint_url("/test");
            assert!(!endpoint.contains('@'));
        }

        worker1.stop().await;
        worker2.stop().await;
    }

    // -----------------------------------------------------------------
    // vLLM PD Router + DP > 1: worker registry verification
    // -----------------------------------------------------------------

    #[tokio::test]
    async fn test_vllm_pd_router_dp2_creates_dp_aware_workers() {
        let mut prefill_worker = MockWorker::new(MockWorkerConfig {
            port: 0,
            worker_type: WorkerType::Prefill,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 0.0,
        });
        let mut decode_worker = MockWorker::new(MockWorkerConfig {
            port: 0,
            worker_type: WorkerType::Decode,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 0.0,
        });
        let prefill_url = prefill_worker.start().await.unwrap();
        let decode_url = decode_worker.start().await.unwrap();

        let config = make_pd_config(
            vec![(prefill_url.clone(), None)],
            vec![decode_url.clone()],
            2,
        );
        let app_context = common::create_test_context(config.clone());
        let _router = RouterFactory::create_router(&app_context).await.unwrap();

        tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

        let all_workers = app_context.worker_registry.get_all();
        // 1 prefill × dp_size=2 + 1 decode × dp_size=2 = 4 workers
        assert_eq!(
            all_workers.len(),
            4,
            "PD mode: 1 prefill + 1 decode × dp_size=2 should produce 4 workers, got {}",
            all_workers.len()
        );

        // All should be DP-aware
        for w in &all_workers {
            assert!(
                w.is_dp_aware(),
                "PD worker {} should be DP-aware when dp_size > 1",
                w.url()
            );

            let endpoint = w.endpoint_url("/v1/completions");
            assert!(
                !endpoint.contains('@'),
                "PD worker endpoint_url must not contain @rank (got: {})",
                endpoint
            );
        }

        // Verify prefill and decode workers separately
        let prefill_workers = app_context.worker_registry.get_prefill_workers();
        let decode_workers = app_context.worker_registry.get_decode_workers();
        assert_eq!(prefill_workers.len(), 2, "Should have 2 prefill DP workers");
        assert_eq!(decode_workers.len(), 2, "Should have 2 decode DP workers");

        prefill_worker.stop().await;
        decode_worker.stop().await;
    }

    // -----------------------------------------------------------------
    // vLLM PD Router + DP > 1: add_prefill_server / add_decode_server runtime path
    // -----------------------------------------------------------------
    // These tests verify the fix for D100422851: when dp_size > 1,
    // add_prefill_server/add_decode_server must create DPAwareWorker
    // (not BasicWorker) to prevent IPv6+DP URL corruption.

    #[tokio::test]
    async fn test_vllm_pd_router_add_prefill_server_dp2_creates_dp_aware_worker() {
        // Start initial PD workers for router creation
        let mut initial_prefill = MockWorker::new(MockWorkerConfig {
            port: 0,
            worker_type: WorkerType::Prefill,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 0.0,
        });
        let mut initial_decode = MockWorker::new(MockWorkerConfig {
            port: 0,
            worker_type: WorkerType::Decode,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 0.0,
        });
        let prefill_url = initial_prefill.start().await.unwrap();
        let decode_url = initial_decode.start().await.unwrap();

        // Start a NEW prefill worker to add at runtime
        let mut new_prefill = MockWorker::new(MockWorkerConfig {
            port: 0,
            worker_type: WorkerType::Prefill,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 0.0,
        });
        let new_prefill_url = new_prefill.start().await.unwrap();

        // Create vLLM PD router with dp_size=2
        let config = make_pd_config(
            vec![(prefill_url.clone(), None)],
            vec![decode_url.clone()],
            2,
        );
        let app_context = common::create_test_context(config.clone());
        let router = RouterFactory::create_router(&app_context).await.unwrap();

        tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

        // Initial workers: 1 prefill × 2 + 1 decode × 2 = 4
        assert_eq!(app_context.worker_registry.get_all().len(), 4);

        // Downcast to the user-facing vLLM PD router and add a new prefill server at runtime.
        use vllm_router_rs::routers::http::vllm_pd_router::VllmPDRouter;
        let pd_router = router.as_any().downcast_ref::<VllmPDRouter>().unwrap();

        let invalid = format!("{new_prefill_url}@2");
        let error = pd_router
            .add_prefill_server(invalid.clone(), None)
            .await
            .unwrap_err();
        assert!(error.to_string().contains("DP size 2"), "{error}");
        assert!(app_context.worker_registry.get_by_url(&invalid).is_none());
        // Add new prefill server (plain URL, no @rank — mimics service discovery)
        let result = pd_router
            .add_prefill_server(new_prefill_url.clone(), None)
            .await;
        assert!(
            result.is_ok(),
            "add_prefill_server should succeed: {:?}",
            result.err()
        );

        // The new worker should be registered as DPAwareWorker
        // With dp_size=2 and no @rank in URL, the router expands the bare URL
        // into one DPAwareWorker per rank.
        let new_worker_url = format!("{}@0", new_prefill_url);
        let worker = app_context.worker_registry.get_by_url(&new_worker_url);
        assert!(
            worker.is_some(),
            "DPAwareWorker should be registered with @0 suffix. Registry URLs: {:?}",
            app_context
                .worker_registry
                .get_all()
                .iter()
                .map(|w| w.url().to_string())
                .collect::<Vec<_>>()
        );

        let w = worker.unwrap();
        assert!(
            w.is_dp_aware(),
            "Runtime-added worker should be DP-aware when dp_size > 1"
        );
        assert_eq!(w.dp_rank(), Some(0));
        assert_eq!(w.dp_size(), Some(2));

        // Critical: endpoint_url must NOT contain @rank
        let endpoint = w.endpoint_url("/v1/completions");
        assert!(
            !endpoint.contains('@'),
            "Runtime-added worker endpoint_url must not contain @rank (got: {})",
            endpoint
        );

        initial_prefill.stop().await;
        initial_decode.stop().await;
        new_prefill.stop().await;
    }

    #[tokio::test]
    async fn test_vllm_pd_router_add_decode_server_dp2_creates_dp_aware_worker() {
        let mut initial_prefill = MockWorker::new(MockWorkerConfig {
            port: 0,
            worker_type: WorkerType::Prefill,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 0.0,
        });
        let mut initial_decode = MockWorker::new(MockWorkerConfig {
            port: 0,
            worker_type: WorkerType::Decode,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 0.0,
        });
        let prefill_url = initial_prefill.start().await.unwrap();
        let decode_url = initial_decode.start().await.unwrap();

        // New decode worker to add at runtime
        let mut new_decode = MockWorker::new(MockWorkerConfig {
            port: 0,
            worker_type: WorkerType::Decode,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 0.0,
        });
        let new_decode_url = new_decode.start().await.unwrap();

        let config = make_pd_config(
            vec![(prefill_url.clone(), None)],
            vec![decode_url.clone()],
            2,
        );
        let app_context = common::create_test_context(config.clone());
        let router = RouterFactory::create_router(&app_context).await.unwrap();

        tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

        use vllm_router_rs::routers::http::vllm_pd_router::VllmPDRouter;
        let pd_router = router.as_any().downcast_ref::<VllmPDRouter>().unwrap();

        let invalid = format!("{new_decode_url}@2");
        let error = pd_router
            .add_decode_server(invalid.clone())
            .await
            .unwrap_err();
        assert!(error.to_string().contains("DP size 2"), "{error}");
        assert!(app_context.worker_registry.get_by_url(&invalid).is_none());
        // Add new decode server at runtime
        let result = pd_router.add_decode_server(new_decode_url.clone()).await;
        assert!(
            result.is_ok(),
            "add_decode_server should succeed: {:?}",
            result.err()
        );

        // Verify the new worker is DPAwareWorker
        let new_worker_url = format!("{}@0", new_decode_url);
        let worker = app_context.worker_registry.get_by_url(&new_worker_url);
        assert!(
            worker.is_some(),
            "DPAwareWorker should be registered with @0 suffix"
        );

        let w = worker.unwrap();
        assert!(w.is_dp_aware());
        assert_eq!(w.dp_rank(), Some(0));

        let endpoint = w.endpoint_url("/v1/completions");
        assert!(
            !endpoint.contains('@'),
            "Decode worker endpoint_url must not contain @rank (got: {})",
            endpoint
        );

        initial_prefill.stop().await;
        initial_decode.stop().await;
        new_decode.stop().await;
    }

    #[tokio::test]
    async fn test_vllm_pd_router_add_prefill_server_dp1_creates_basic_worker() {
        // With dp_size=1, add_prefill_server should create BasicWorker
        let mut initial_prefill = MockWorker::new(MockWorkerConfig {
            port: 0,
            worker_type: WorkerType::Prefill,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 0.0,
        });
        let mut initial_decode = MockWorker::new(MockWorkerConfig {
            port: 0,
            worker_type: WorkerType::Decode,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 0.0,
        });
        let prefill_url = initial_prefill.start().await.unwrap();
        let decode_url = initial_decode.start().await.unwrap();

        let mut new_prefill = MockWorker::new(MockWorkerConfig {
            port: 0,
            worker_type: WorkerType::Prefill,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 0.0,
        });
        let new_prefill_url = new_prefill.start().await.unwrap();

        let config = make_pd_config(
            vec![(prefill_url.clone(), None)],
            vec![decode_url.clone()],
            1, // dp_size=1
        );
        let app_context = common::create_test_context(config.clone());
        let router = RouterFactory::create_router(&app_context).await.unwrap();

        tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

        use vllm_router_rs::routers::http::vllm_pd_router::VllmPDRouter;
        let pd_router = router.as_any().downcast_ref::<VllmPDRouter>().unwrap();

        let result = pd_router
            .add_prefill_server(new_prefill_url.clone(), None)
            .await;
        assert!(result.is_ok());

        // With dp_size=1, worker is registered with the original URL (no @rank)
        let worker = app_context.worker_registry.get_by_url(&new_prefill_url);
        assert!(
            worker.is_some(),
            "BasicWorker should be registered with original URL"
        );

        let w = worker.unwrap();
        assert!(
            !w.is_dp_aware(),
            "Worker should NOT be DP-aware when dp_size=1"
        );
        assert!(!w.url().contains('@'));

        initial_prefill.stop().await;
        initial_decode.stop().await;
        new_prefill.stop().await;
    }

    #[tokio::test]
    async fn test_vllm_pd_router_dp1_creates_basic_workers() {
        let mut prefill_worker = MockWorker::new(MockWorkerConfig {
            port: 0,
            worker_type: WorkerType::Prefill,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 0.0,
        });
        let mut decode_worker = MockWorker::new(MockWorkerConfig {
            port: 0,
            worker_type: WorkerType::Decode,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 0.0,
        });
        let prefill_url = prefill_worker.start().await.unwrap();
        let decode_url = decode_worker.start().await.unwrap();

        let config = make_pd_config(
            vec![(prefill_url.clone(), None)],
            vec![decode_url.clone()],
            1,
        );
        let app_context = common::create_test_context(config.clone());
        let _router = RouterFactory::create_router(&app_context).await.unwrap();

        tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

        let all_workers = app_context.worker_registry.get_all();
        assert_eq!(
            all_workers.len(),
            2,
            "PD mode dp_size=1: 1 prefill + 1 decode"
        );

        for w in &all_workers {
            assert!(
                !w.is_dp_aware(),
                "Worker should NOT be DP-aware when dp_size=1"
            );
            assert!(!w.url().contains('@'));
        }

        prefill_worker.stop().await;
        decode_worker.stop().await;
    }
}
