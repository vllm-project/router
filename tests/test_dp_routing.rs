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

#[cfg(test)]
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

#[cfg(test)]
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

    fn pd_mock(worker_type: WorkerType) -> MockWorker {
        MockWorker::new(MockWorkerConfig {
            port: 0,
            worker_type,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 0.0,
        })
    }

    fn worker_port(url: &str) -> u16 {
        url.rsplit(':').next().unwrap().parse().unwrap()
    }

    async fn pd_chat_response(app: axum::Router) -> axum::response::Response {
        let request = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                json!({
                    "model": "mock-model",
                    "messages": [{"role": "user", "content": "hello"}],
                    "max_tokens": 7,
                })
                .to_string(),
            ))
            .unwrap();
        app.oneshot(request).await.unwrap()
    }

    async fn send_pd_chat(app: axum::Router) {
        let response = pd_chat_response(app).await;
        let status = response.status();
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        assert_eq!(status.as_u16(), 200, "{}", String::from_utf8_lossy(&body));
    }

    fn captured_ranks(port: u16) -> std::collections::BTreeSet<usize> {
        get_captured_requests(port)
            .iter()
            .filter_map(|request| request.headers.get("x-data-parallel-rank"))
            .map(|values| {
                assert_eq!(values.len(), 1);
                values[0].parse().unwrap()
            })
            .collect()
    }

    #[tokio::test]
    async fn test_asymmetric_pd_static_requests_and_metrics() {
        for (legacy, prefill, decode, metrics, expected) in [
            (4, None, None, None, (4, 4)),
            (1, Some(4), Some(2), None, (4, 2)),
            (4, None, Some(2), None, (4, 2)),
            (2, Some(4), None, None, (4, 2)),
            (1, None, None, Some((4, 2)), (4, 2)),
            (1, Some(4), Some(2), Some((2, 4)), (4, 2)),
            (1, Some(4), Some(1), None, (4, 1)),
            (1, Some(1), Some(2), None, (1, 2)),
        ] {
            let mut p = pd_mock(WorkerType::Prefill);
            let mut d = pd_mock(WorkerType::Decode);
            if let Some((p_size, d_size)) = metrics {
                p.set_dp_metrics(p_size).await;
                d.set_dp_metrics(d_size).await;
            }
            let p_url = p.start().await.unwrap();
            let d_url = d.start().await.unwrap();
            clear_captured_requests(worker_port(&p_url));
            clear_captured_requests(worker_port(&d_url));
            let mut config =
                make_pd_config(vec![(p_url.clone(), None)], vec![d_url.clone()], legacy);
            config.prefill_data_parallel_size = prefill;
            config.decode_data_parallel_size = decode;
            let ctx = common::create_test_context(config.clone());
            let router = Arc::from(RouterFactory::create_router(&ctx).await.unwrap());
            assert_eq!(ctx.worker_registry.get_prefill_workers().len(), expected.0);
            assert_eq!(ctx.worker_registry.get_decode_workers().len(), expected.1);
            let app = common::test_app::create_test_app(router, Client::new(), &config);
            for _ in 0..8 {
                send_pd_chat(app.clone()).await;
            }
            for (url, size) in [(&p_url, expected.0), (&d_url, expected.1)] {
                let ranks = captured_ranks(worker_port(url));
                let expected_ranks = if size > 1 {
                    (0..size).collect()
                } else {
                    Default::default()
                };
                assert_eq!(
                    ranks, expected_ranks,
                    "{url}, legacy={legacy}, P={prefill:?}, D={decode:?}"
                );
            }
            let p_requests = get_captured_requests(worker_port(&p_url));
            let d_requests = get_captured_requests(worker_port(&d_url));
            assert_eq!(p_requests.len(), 8);
            assert_eq!(d_requests.len(), 8);
            for request in p_requests {
                assert_eq!(request.body.as_ref().unwrap()["max_tokens"], 1);
                assert!(request.body.unwrap()["kv_transfer_params"]
                    .get("remote_dp_size")
                    .is_none());
            }
            for request in d_requests {
                assert_eq!(request.body.unwrap()["max_tokens"], 7);
            }
            p.stop().await;
            d.stop().await;
        }
    }

    #[tokio::test]
    async fn test_asymmetric_pd_dynamic_add_remove_and_pinned_url() {
        use vllm_router_rs::routers::http::vllm_pd_router::VllmPDRouter;
        let mut p = pd_mock(WorkerType::Prefill);
        let mut d = pd_mock(WorkerType::Decode);
        let p_url = p.start().await.unwrap();
        let d_url = d.start().await.unwrap();
        let mut config = make_pd_config(vec![(format!("{p_url}@3"), None)], vec![d_url.clone()], 1);
        config.prefill_data_parallel_size = Some(4);
        config.decode_data_parallel_size = Some(2);
        let ctx = common::create_test_context(config.clone());
        let router = RouterFactory::create_router(&ctx).await.unwrap();
        let pd = router.as_any().downcast_ref::<VllmPDRouter>().unwrap();
        assert_eq!(ctx.worker_registry.get_prefill_workers().len(), 1);
        assert_eq!(
            ctx.worker_registry.get_prefill_workers()[0].dp_rank(),
            Some(3)
        );
        let mut added_p = pd_mock(WorkerType::Prefill);
        let mut added_d = pd_mock(WorkerType::Decode);
        let added_p_url = added_p.start().await.unwrap();
        let added_d_url = added_d.start().await.unwrap();
        pd.add_prefill_server(added_p_url.clone(), None)
            .await
            .unwrap();
        pd.add_decode_server(added_d_url.clone()).await.unwrap();
        assert_eq!(ctx.worker_registry.get_prefill_workers().len(), 5);
        assert_eq!(ctx.worker_registry.get_decode_workers().len(), 4);
        pd.remove_prefill_server(&added_p_url).await.unwrap();
        pd.remove_decode_server(&added_d_url).await.unwrap();
        assert_eq!(ctx.worker_registry.get_prefill_workers().len(), 1);
        assert_eq!(ctx.worker_registry.get_decode_workers().len(), 2);
        for worker in [&mut p, &mut d, &mut added_p, &mut added_d] {
            worker.stop().await;
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_asymmetric_pd_discovery_requests_and_peer_metadata() {
        use vllm_router_rs::config::KvConnector;
        for (connector, mode, prefill_size, reported_size, pinned_decode, role_overrides) in [
            (KvConnector::Nixl, None, 4, None, None, true),
            (KvConnector::MoriIO, Some("WRITE"), 4, Some(4), None, true),
            (KvConnector::MoriIO, Some("READ"), 4, Some(4), None, true),
            (KvConnector::MoriIO, Some("WRITE"), 1, Some(1), None, true),
            (KvConnector::MoriIO, Some("READ"), 8, Some(16), None, false),
            (KvConnector::MoriIO, Some("WRITE"), 8, Some(16), None, false),
            (KvConnector::MoriIO, Some("WRITE"), 4, None, None, true),
            (KvConnector::MoriIO, Some("WRITE"), 4, None, None, false),
            (
                KvConnector::MoriIO,
                Some("WRITE"),
                4,
                Some(4),
                Some(1),
                true,
            ),
            (
                KvConnector::MoriIO,
                Some("WRITE"),
                4,
                Some(4),
                Some(0),
                true,
            ),
        ] {
            let mut p = pd_mock(WorkerType::Prefill);
            let mut d = pd_mock(WorkerType::Decode);
            p.set_dp_metrics(prefill_size).await;
            d.set_dp_metrics(2).await;
            let producer_params = json!({
                "do_remote_prefill": true,
                "do_remote_decode": false,
                "remote_engine_id": "producer",
                "remote_block_ids": [1, 2],
                "remote_dp_size": 8,
                "remote_dp_size_local": 4,
                "remote_dp_rank": 4,
                "remote_dp_rank_override": true,
            });
            if mode == Some("READ") {
                p.set_chat_response(json!({"kv_transfer_params": producer_params}))
                    .await;
            }
            let p_url = p.start().await.unwrap();
            let d_url = d.start().await.unwrap();
            clear_captured_requests(worker_port(&p_url));
            clear_captured_requests(worker_port(&d_url));
            let reservation = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            let discovery_address = reservation.local_addr().unwrap().to_string();
            let mut config = make_pd_config(vec![], vec![], 4);
            if role_overrides {
                config.prefill_data_parallel_size = Some(prefill_size);
                config.decode_data_parallel_size = Some(2);
            }
            config.kv_connector = connector;
            if let RoutingMode::VllmPrefillDecode {
                discovery_address: address,
                ..
            } = &mut config.mode
            {
                *address = Some(discovery_address.clone());
            }
            let ctx = common::create_test_context(config.clone());
            drop(reservation);
            let router: Arc<dyn vllm_router_rs::routers::RouterTrait> =
                Arc::from(RouterFactory::create_router(&ctx).await.unwrap());
            let zmq_context = zmq::Context::new();
            let sender = zmq_context.socket(zmq::DEALER).unwrap();
            sender.set_linger(0).unwrap();
            sender
                .connect(&format!("tcp://{discovery_address}"))
                .unwrap();
            let registered_p_url = if pinned_decode.is_some() {
                format!("{p_url}@3")
            } else {
                p_url.clone()
            };
            let registered_d_url = pinned_decode
                .map(|rank| format!("{d_url}@{rank}"))
                .unwrap_or_else(|| d_url.clone());
            for (url, role, dp_size, tp_size) in [
                (&registered_p_url, "P", reported_size, 2),
                (&registered_d_url, "D", Some(2), 1),
            ] {
                let mut registration = json!({
                    "type": role,
                    "http_address": url.trim_start_matches("http://"),
                    "zmq_address": "host:127.0.0.1,handshake:6301,notify:61005",
                });
                if let Some(mode) = mode {
                    registration["transfer_mode"] = json!(mode);
                    if let Some(size) = dp_size {
                        registration["dp_size"] = json!(size);
                    }
                    registration["tp_size"] = json!(tp_size);
                }
                sender
                    .send(rmp_serde::to_vec_named(&registration).unwrap(), 0)
                    .unwrap();
            }
            tokio::time::timeout(std::time::Duration::from_secs(5), async {
                loop {
                    let response = router.health(Request::new(Body::empty())).await;
                    if response.status().is_success() {
                        break;
                    }
                    tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                }
            })
            .await
            .unwrap();
            assert!(
                ctx.worker_registry.get_all().is_empty(),
                "ZMQ discovery must not expand Decode workers"
            );
            let app = common::test_app::create_test_app(router, Client::new(), &config);
            let should_reject = mode == Some("WRITE")
                && (reported_size.is_some_and(|size| size != prefill_size)
                    || (reported_size.is_none() && role_overrides));
            if should_reject {
                let response = pd_chat_response(app).await;
                assert_eq!(response.status().as_u16(), 500);
                let body = axum::body::to_bytes(response.into_body(), usize::MAX)
                    .await
                    .unwrap();
                let error = String::from_utf8_lossy(&body);
                assert!(
                    error.contains(if reported_size.is_some() {
                        "global DP 16"
                    } else {
                        "must report its global DP size"
                    }),
                    "{error}"
                );
                assert!(get_captured_requests(worker_port(&p_url)).is_empty());
                assert!(get_captured_requests(worker_port(&d_url)).is_empty());
                p.stop().await;
                d.stop().await;
                continue;
            }
            let requests = prefill_size.max(4);
            for _ in 0..requests {
                send_pd_chat(app.clone()).await;
            }
            {
                let expected = if pinned_decode.is_some() {
                    [3].into_iter().collect()
                } else if prefill_size > 1 {
                    (0..prefill_size).collect()
                } else {
                    Default::default()
                };
                assert_eq!(captured_ranks(worker_port(&p_url)), expected);
            }
            assert_eq!(
                captured_ranks(worker_port(&d_url)),
                pinned_decode.into_iter().collect()
            );
            let p_requests = get_captured_requests(worker_port(&p_url));
            let d_requests = get_captured_requests(worker_port(&d_url));
            assert_eq!(p_requests.len(), requests);
            assert_eq!(d_requests.len(), requests);
            if connector == KvConnector::MoriIO {
                for (p_request, d_request) in p_requests.iter().zip(d_requests.iter()) {
                    let p_params = &p_request.body.as_ref().unwrap()["kv_transfer_params"];
                    let d_params = &d_request.body.as_ref().unwrap()["kv_transfer_params"];
                    assert_eq!(p_params["remote_dp_size"], 2);
                    assert_eq!(
                        p_params.get("remote_dp_rank"),
                        pinned_decode.as_ref().map(|rank| json!(rank)).as_ref()
                    );
                    if mode == Some("WRITE") {
                        assert_eq!(d_params["remote_dp_size"], prefill_size);
                        assert_eq!(d_params["remote_tp_size"], 2);
                        assert!(d_params.get("is_request_leader").is_none());
                        if prefill_size > 1 {
                            let p_rank: usize = p_request.headers["x-data-parallel-rank"][0]
                                .parse()
                                .unwrap();
                            assert_eq!(d_params["remote_dp_rank"], p_rank);
                        } else {
                            assert_eq!(d_params["remote_dp_rank"], 0);
                        }
                        assert!(d_params.get("remote_dp_rank_override").is_none());
                        assert_eq!(p_params["transfer_id"], d_params["transfer_id"]);
                    } else {
                        assert_eq!(d_params, &producer_params);
                    }
                }
            }
            p.stop().await;
            d.stop().await;
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_discovery_rotates_prefill_ranks_per_endpoint() {
        for prefill_override in [Some(4), None] {
            let local_sizes = [4, if prefill_override.is_some() { 4 } else { 2 }];
            let mut p1 = pd_mock(WorkerType::Prefill);
            let mut p2 = pd_mock(WorkerType::Prefill);
            let mut d = pd_mock(WorkerType::Decode);
            p1.set_dp_metrics(local_sizes[0]).await;
            p2.set_dp_metrics(local_sizes[1]).await;
            let p1_url = p1.start().await.unwrap();
            let p2_url = p2.start().await.unwrap();
            let d_url = d.start().await.unwrap();
            for url in [&p1_url, &p2_url, &d_url] {
                clear_captured_requests(worker_port(url));
            }
            let reservation = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            let discovery_address = reservation.local_addr().unwrap().to_string();
            let mut config = make_pd_config(vec![], vec![], 1);
            config.prefill_data_parallel_size = prefill_override;
            config.decode_data_parallel_size = Some(2);
            if let RoutingMode::VllmPrefillDecode {
                discovery_address: address,
                ..
            } = &mut config.mode
            {
                *address = Some(discovery_address.clone());
            }
            let ctx = common::create_test_context(config.clone());
            drop(reservation);
            let router: Arc<dyn vllm_router_rs::routers::RouterTrait> =
                Arc::from(RouterFactory::create_router(&ctx).await.unwrap());
            let zmq_context = zmq::Context::new();
            let sender = zmq_context.socket(zmq::DEALER).unwrap();
            sender.set_linger(0).unwrap();
            sender
                .connect(&format!("tcp://{discovery_address}"))
                .unwrap();
            // D registers last, so readiness implies both P registrations arrived.
            for (url, role) in [(&p1_url, "P"), (&p2_url, "P"), (&d_url, "D")] {
                let registration = json!({
                    "type": role,
                    "http_address": url.trim_start_matches("http://"),
                    "zmq_address": "host:127.0.0.1,handshake:6301,notify:61005",
                });
                sender
                    .send(rmp_serde::to_vec_named(&registration).unwrap(), 0)
                    .unwrap();
            }
            tokio::time::timeout(std::time::Duration::from_secs(5), async {
                loop {
                    if router
                        .health(Request::new(Body::empty()))
                        .await
                        .status()
                        .is_success()
                    {
                        break;
                    }
                    tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                }
            })
            .await
            .unwrap();
            let app = common::test_app::create_test_app(router, Client::new(), &config);
            for _ in 0..16 {
                send_pd_chat(app.clone()).await;
            }
            for (url, local_size) in [(&p1_url, local_sizes[0]), (&p2_url, local_sizes[1])] {
                assert_eq!(get_captured_requests(worker_port(url)).len(), 8);
                assert_eq!(captured_ranks(worker_port(url)), (0..local_size).collect());
            }
            assert!(captured_ranks(worker_port(&d_url)).is_empty());
            for worker in [&mut p1, &mut p2, &mut d] {
                worker.stop().await;
            }
        }
    }

    /// Helper to create a RouterConfig with DP settings for Regular mode
    fn make_regular_config(worker_urls: Vec<String>, dp_size: usize) -> RouterConfig {
        RouterConfig {
            mode: RoutingMode::Regular { worker_urls },
            policy: PolicyConfig::RoundRobin,
            host: "127.0.0.1".to_string(),
            port: 0,
            max_payload_size: 256 * 1024 * 1024,
            request_timeout_secs: 10,
            worker_startup_timeout_secs: 5,
            worker_startup_check_interval_secs: 1,
            intra_node_data_parallel_size: dp_size,
            prefill_data_parallel_size: None,
            decode_data_parallel_size: None,
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
            health_check: vllm_router_rs::config::HealthCheckConfig::default(),
            enable_igw: false,
            connection_mode: ConnectionMode::Http,
            history_backend: vllm_router_rs::config::HistoryBackend::Memory,
            enable_profiling: false,
            profile_timeout_secs: 30,
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
            policy: PolicyConfig::RoundRobin,
            host: "127.0.0.1".to_string(),
            port: 0,
            max_payload_size: 256 * 1024 * 1024,
            request_timeout_secs: 10,
            worker_startup_timeout_secs: 5,
            worker_startup_check_interval_secs: 1,
            intra_node_data_parallel_size: dp_size,
            prefill_data_parallel_size: None,
            decode_data_parallel_size: None,
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
            health_check: vllm_router_rs::config::HealthCheckConfig::default(),
            enable_igw: false,
            connection_mode: ConnectionMode::Http,
            history_backend: vllm_router_rs::config::HistoryBackend::Memory,
            enable_profiling: false,
            profile_timeout_secs: 30,
            kv_connector: vllm_router_rs::config::KvConnector::Nixl,
            program_scheduling: None,
        }
    }

    // -----------------------------------------------------------------
    // Regular Router + DP > 1: transparent proxy
    // -----------------------------------------------------------------

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
