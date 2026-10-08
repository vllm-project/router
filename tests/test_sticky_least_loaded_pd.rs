//! End-to-end tests for the `sticky_least_loaded` policy in vLLM P/D
//! disaggregation mode.
//!
//! A real `VllmPrefillDecode` router fronts two prefill and two decode mock
//! workers, with `sticky_least_loaded` as both the prefill and the decode
//! policy. The tests check that the `X-Session-ID` header reaches both
//! policies on the request path: every turn of a session goes to the same
//! prefill and decode worker, new sessions are balanced across workers, and
//! `/finish_session` releases the session on both roles.

mod common;

#[cfg(test)]
mod sticky_least_loaded_pd_tests {
    use super::common;
    use axum::body::Body;
    use axum::extract::Request;
    use axum::Router;
    use common::mock_worker::{
        clear_captured_requests, get_captured_requests, HealthStatus, MockWorker, MockWorkerConfig,
        WorkerType,
    };
    use serde_json::json;
    use std::collections::{HashMap, HashSet};
    use std::sync::Arc;
    use tower::ServiceExt;
    use vllm_router_rs::config::{
        CircuitBreakerConfig, ConnectionMode, PolicyConfig, RetryConfig, RouterConfig, RoutingMode,
    };
    use vllm_router_rs::policies::{LoadBalancingPolicy, StickyLeastLoadedPolicy};
    use vllm_router_rs::routers::RouterFactory;
    use vllm_router_rs::server::AppContext;

    fn make_sticky_pd_config(
        prefill_urls: Vec<(String, Option<u16>)>,
        decode_urls: Vec<String>,
    ) -> RouterConfig {
        RouterConfig {
            mode: RoutingMode::VllmPrefillDecode {
                prefill_urls,
                decode_urls,
                prefill_policy: Some(PolicyConfig::StickyLeastLoaded),
                decode_policy: Some(PolicyConfig::StickyLeastLoaded),
                discovery_address: None,
            },
            policy: PolicyConfig::StickyLeastLoaded,
            host: "127.0.0.1".to_string(),
            port: 0,
            max_payload_size: 256 * 1024 * 1024,
            request_timeout_secs: 10,
            worker_startup_timeout_secs: 5,
            worker_startup_check_interval_secs: 1,
            intra_node_data_parallel_size: 1,
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

    /// Router over 2 prefill + 2 decode mock workers.
    struct PdFixture {
        app: Router,
        context: Arc<AppContext>,
        prefill_ports: Vec<u16>,
        decode_ports: Vec<u16>,
        workers: Vec<MockWorker>,
    }

    impl PdFixture {
        async fn start() -> Self {
            let mut workers = Vec::new();
            let mut prefill_urls = Vec::new();
            let mut decode_urls = Vec::new();
            for worker_type in [
                WorkerType::Prefill,
                WorkerType::Prefill,
                WorkerType::Decode,
                WorkerType::Decode,
            ] {
                let is_prefill = matches!(worker_type, WorkerType::Prefill);
                let mut worker = MockWorker::new(MockWorkerConfig {
                    port: 0,
                    worker_type,
                    health_status: HealthStatus::Healthy,
                    response_delay_ms: 0,
                    fail_rate: 0.0,
                });
                let url = worker.start().await.unwrap();
                clear_captured_requests(port_of(&url));
                if is_prefill {
                    prefill_urls.push(url);
                } else {
                    decode_urls.push(url);
                }
                workers.push(worker);
            }

            let config = make_sticky_pd_config(
                prefill_urls.iter().map(|u| (u.clone(), None)).collect(),
                decode_urls.clone(),
            );
            let context = common::create_test_context(config.clone());
            let router = RouterFactory::create_router(&context).await.unwrap();
            // Share the router's context so /finish_session reaches its PolicyRegistry.
            let app = common::test_app::create_test_app_with_context(
                Arc::from(router),
                Arc::clone(&context),
                &config,
            );

            Self {
                app,
                context,
                prefill_ports: prefill_urls.iter().map(|u| port_of(u)).collect(),
                decode_ports: decode_urls.iter().map(|u| port_of(u)).collect(),
                workers,
            }
        }

        fn policy(&self, prefill: bool) -> Arc<dyn LoadBalancingPolicy> {
            if prefill {
                self.context.policy_registry.get_prefill_policy()
            } else {
                self.context.policy_registry.get_decode_policy()
            }
        }

        fn active_sessions(&self, prefill: bool) -> usize {
            let policy = self.policy(prefill);
            assert_eq!(policy.name(), "sticky_least_loaded");
            policy
                .as_any()
                .downcast_ref::<StickyLeastLoadedPolicy>()
                .expect("sticky_least_loaded policy")
                .active_session_count()
        }

        async fn send(&self, request: Request<Body>) -> u16 {
            self.app
                .clone()
                .oneshot(request)
                .await
                .unwrap()
                .status()
                .as_u16()
        }

        /// Send one /v1/completions turn and return the (prefill, decode)
        /// worker ports that served it.
        async fn completion(&self, session_id: Option<&str>, prompt: &str) -> (u16, u16) {
            let before = self.served_counts();
            let mut request = Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json");
            if let Some(session_id) = session_id {
                request = request.header("x-session-id", session_id);
            }
            let body = json!({"model": "mock-model", "prompt": prompt, "max_tokens": 4});
            let request = request
                .body(Body::from(serde_json::to_string(&body).unwrap()))
                .unwrap();
            assert_eq!(self.send(request).await, 200);

            let after = self.served_counts();
            let served = |ports: &[u16]| -> u16 {
                let hit: Vec<u16> = ports
                    .iter()
                    .copied()
                    .filter(|p| after[p] > before[p])
                    .collect();
                assert_eq!(hit.len(), 1, "expected exactly one worker, got {:?}", hit);
                hit[0]
            };
            (served(&self.prefill_ports), served(&self.decode_ports))
        }

        async fn finish_session(&self, session_id: &str) {
            let request = Request::builder()
                .method("POST")
                .uri(format!("/finish_session?session_id={}", session_id))
                .body(Body::empty())
                .unwrap();
            assert_eq!(self.send(request).await, 200);
        }

        fn served_counts(&self) -> HashMap<u16, usize> {
            self.prefill_ports
                .iter()
                .chain(&self.decode_ports)
                .map(|&port| {
                    let n = get_captured_requests(port)
                        .iter()
                        .filter(|r| r.path == "/v1/completions")
                        .count();
                    (port, n)
                })
                .collect()
        }

        async fn stop(mut self) {
            for worker in &mut self.workers {
                worker.stop().await;
            }
        }
    }

    fn port_of(url: &str) -> u16 {
        url.rsplit(':').next().unwrap().parse().unwrap()
    }

    #[tokio::test]
    async fn test_pd_sessions_stick_and_balance_on_both_roles() {
        let fx = PdFixture::start().await;
        let sessions = ["session-a", "session-b", "session-c", "session-d"];

        let mut assigned: HashMap<&str, (u16, u16)> = HashMap::new();
        for turn in 0..3 {
            for session in sessions {
                // The prompt changes every turn: only the header identifies the session.
                let pair = fx
                    .completion(Some(session), &format!("{} turn {}", session, turn))
                    .await;
                let first = *assigned.entry(session).or_insert(pair);
                assert_eq!(pair, first, "{} moved on turn {}", session, turn);
            }
        }

        // New sessions go to the least-loaded worker, so 4 sessions split 2/2
        // across the two prefill workers and across the two decode workers.
        for (prefill, ports) in [(true, &fx.prefill_ports), (false, &fx.decode_ports)] {
            for port in ports {
                let n = assigned
                    .values()
                    .filter(|(p, d)| if prefill { p == port } else { d == port })
                    .count();
                assert_eq!(n, 2, "worker port {} has {} sessions", port, n);
            }
            assert_eq!(fx.active_sessions(prefill), sessions.len());
        }

        // /finish_session releases the session on both the prefill and decode policy.
        for session in sessions {
            fx.finish_session(session).await;
        }
        assert_eq!(fx.active_sessions(true), 0);
        assert_eq!(fx.active_sessions(false), 0);

        fx.stop().await;
    }

    #[tokio::test]
    async fn test_pd_requests_without_session_header_are_not_tracked() {
        let fx = PdFixture::start().await;

        let mut prefill_ports = HashSet::new();
        for i in 0..4 {
            let (prefill, _) = fx.completion(None, &format!("stateless {}", i)).await;
            prefill_ports.insert(prefill);
        }

        // Requests are still served, but no session state is recorded.
        assert!(!prefill_ports.is_empty());
        assert_eq!(fx.active_sessions(true), 0);
        assert_eq!(fx.active_sessions(false), 0);

        fx.stop().await;
    }
}
