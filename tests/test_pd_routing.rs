#[cfg(test)]
mod test_pd_routing {
    use vllm_router_rs::config::{
        CircuitBreakerConfig, ConnectionMode, PolicyConfig, RetryConfig, RouterConfig, RoutingMode,
    };
    use vllm_router_rs::routers::RouterFactory;

    #[tokio::test]
    async fn test_epd_handoff_preserves_metadata_and_isolates_ec_handles() {
        use axum::{
            extract::State,
            http::StatusCode,
            response::IntoResponse,
            routing::{get, post},
            Json, Router,
        };
        use serde_json::{json, Value};
        use std::sync::{Arc, Mutex};
        use vllm_router_rs::config::{ConfigValidator, EpdConfig};

        type Seen = Arc<Mutex<Vec<(String, Value)>>>;
        async fn encode(
            State(seen): State<Seen>,
            Json(body): Json<Value>,
        ) -> axum::response::Response {
            seen.lock().unwrap().push(("E".into(), body.clone()));
            if body["model"] == "encoder-fail" {
                return (StatusCode::SERVICE_UNAVAILABLE, "injected encoder error").into_response();
            }
            let hash = body["messages"][0]["content"][0]["uuid"].as_str().unwrap();
            Json(json!({
                "ec_transfer_params": {
                    (hash): {"metadata": {"image_grid_thw": [[1, 2, 3]]}},
                },
            }))
            .into_response()
        }
        async fn prefill(
            State(seen): State<Seen>,
            Json(body): Json<Value>,
        ) -> axum::response::Response {
            seen.lock().unwrap().push(("P".into(), body.clone()));
            if body["model"] == "fail" {
                return (StatusCode::INTERNAL_SERVER_ERROR, "injected prefill error")
                    .into_response();
            }
            if body["model"] == "missing-kv" {
                return Json(json!({"choices":[]})).into_response();
            }
            Json(json!({
                "kv_transfer_params": {
                    "remote_engine_id": "P",
                    "do_remote_prefill": true,
                },
            }))
            .into_response()
        }
        async fn decode(
            State(seen): State<Seen>,
            Json(body): Json<Value>,
        ) -> axum::response::Response {
            seen.lock().unwrap().push(("D".into(), body.clone()));
            if body["stream"] == true {
                return ([("content-type", "text/event-stream")], "data: [DONE]\n\n")
                    .into_response();
            }
            Json(json!({"choices":[{"message":{"content":"A"}}]})).into_response()
        }
        let seen: Seen = Arc::new(Mutex::new(Vec::new()));
        let app = Router::new()
            .route("/e/health", get(|| async { "ok" }))
            .route("/p/health", get(|| async { "ok" }))
            .route("/d/health", get(|| async { "ok" }))
            .route("/e/v1/chat/completions", post(encode))
            .route("/p/v1/chat/completions", post(prefill))
            .route("/d/v1/chat/completions", post(decode))
            .with_state(seen.clone());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        let mut config = RouterConfig {
            mode: RoutingMode::VllmPrefillDecode {
                prefill_urls: vec![(format!("{base}/p"), None)],
                decode_urls: vec![format!("{base}/d")],
                prefill_policy: None,
                decode_policy: None,
                discovery_address: None,
            },
            policy: PolicyConfig::Random,
            epd: Some(EpdConfig {
                encoder_urls: vec![format!("{base}/e")],
                consumer_zmq_addrs: Default::default(),
            }),
            worker_startup_timeout_secs: 5,
            ..RouterConfig::default()
        };
        // Mooncake EC reservations belong to P, not D.
        config
            .epd
            .as_mut()
            .unwrap()
            .consumer_zmq_addrs
            .insert(format!("{base}/d"), "tcp://localhost:1234".into());
        assert!(ConfigValidator::validate(&config).is_err());
        config.epd.as_mut().unwrap().consumer_zmq_addrs.clear();
        ConfigValidator::validate(&config).unwrap();
        let context = Arc::new(
            vllm_router_rs::server::AppContext::new(
                config,
                reqwest::Client::new(),
                64,
                None,
                vec![],
            )
            .unwrap(),
        );
        let router = RouterFactory::create_router(&context).await.unwrap();
        for (model, stream, status) in [
            ("test", false, StatusCode::OK),
            ("test", true, StatusCode::OK),
            ("fail", false, StatusCode::INTERNAL_SERVER_ERROR),
            ("missing-kv", false, StatusCode::INTERNAL_SERVER_ERROR),
            ("encoder-fail", false, StatusCode::SERVICE_UNAVAILABLE),
            ("bad-input", false, StatusCode::BAD_REQUEST),
        ] {
            seen.lock().unwrap().clear();
            let mut body = json!({
                "model": model,
                "stream": stream,
                "max_tokens": 16,
                "structured_outputs": {"choice": ["A", "B"]},
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
                        {"type": "text", "text": "choose"},
                    ],
                }],
            });
            if model == "bad-input" {
                body["messages"] = Value::Null;
            }
            let response = router
                .route_transparent(
                    None,
                    "/v1/chat/completions",
                    &axum::http::Method::POST,
                    body,
                )
                .await;
            assert_eq!(response.status(), status, "model={model}, stream={stream}");
            let bytes = axum::body::to_bytes(response.into_body(), 4096)
                .await
                .unwrap();
            if stream {
                assert!(String::from_utf8_lossy(&bytes).contains("[DONE]"));
            }
            let seen = seen.lock().unwrap();
            let phases: Vec<_> = seen.iter().map(|(phase, _)| phase.as_str()).collect();
            assert_eq!(
                phases,
                match model {
                    "bad-input" => vec![],
                    "encoder-fail" => vec!["E"],
                    "test" => vec!["E", "P", "D"],
                    _ => vec!["E", "P"],
                }
            );
            if seen.len() < 2 {
                continue;
            }
            let p = &seen[1].1;
            assert_eq!(p["max_tokens"], 1);
            assert_eq!(p["stream"], false);
            assert!(p["ec_transfer_params"]["ec_items"].is_array());
            assert_eq!(
                p["messages"][0]["content"][0]["image_embeds"]["image_grid_thw"],
                json!([1, 2, 3])
            );
            if status.is_success() {
                let d = &seen[2].1;
                assert_eq!(d["max_tokens"], 16);
                assert_eq!(d["messages"], p["messages"]);
                assert_eq!(d["structured_outputs"], p["structured_outputs"]);
                assert!(d.get("ec_transfer_params").is_none());
                assert_eq!(d["kv_transfer_params"]["remote_engine_id"], "P");
            }
        }
        server.abort();
    }

    // ========================================================================
    // Phase 1: Basic PD Components and Router Creation
    // ========================================================================

    #[test]
    fn test_worker_types() {
        use vllm_router_rs::core::{WorkerFactory, WorkerType};

        // Test worker creation for prefill servers
        let prefill_worker =
            WorkerFactory::create_prefill("http://prefill:8080".to_string(), Some(9000));
        assert_eq!(prefill_worker.url(), "http://prefill:8080");
        match prefill_worker.worker_type() {
            WorkerType::Prefill { bootstrap_port } => {
                assert_eq!(bootstrap_port, Some(9000));
            }
            _ => panic!("Expected Prefill worker type"),
        }

        // Test worker creation for decode servers
        let decode_worker = WorkerFactory::create_decode("http://decode:8080".to_string());
        assert_eq!(decode_worker.url(), "http://decode:8080");
        match decode_worker.worker_type() {
            WorkerType::Decode => (),
            _ => panic!("Expected Decode worker type"),
        }

        // Test regular worker creation
        let regular_worker = WorkerFactory::create_regular("http://regular:8080".to_string());
        assert_eq!(regular_worker.url(), "http://regular:8080");
        match regular_worker.worker_type() {
            WorkerType::Regular => (),
            _ => panic!("Expected Regular worker type"),
        }
    }

    #[tokio::test]
    async fn test_pd_router_configuration() {
        // Test PD router configuration with various policies
        // In the new structure, RoutingMode and PolicyConfig are separate
        let test_cases = vec![
            (
                RoutingMode::VllmPrefillDecode {
                    prefill_urls: vec![
                        ("http://prefill1:8080".to_string(), Some(9000)),
                        ("http://prefill2:8080".to_string(), None),
                    ],
                    decode_urls: vec![
                        "http://decode1:8080".to_string(),
                        "http://decode2:8080".to_string(),
                    ],
                    prefill_policy: None,
                    decode_policy: None,
                    discovery_address: None,
                },
                PolicyConfig::Random,
            ),
            (
                RoutingMode::VllmPrefillDecode {
                    prefill_urls: vec![("http://prefill:8080".to_string(), Some(9000))],
                    decode_urls: vec!["http://decode:8080".to_string()],
                    prefill_policy: None,
                    decode_policy: None,
                    discovery_address: None,
                },
                PolicyConfig::PowerOfTwo {
                    load_check_interval_secs: 5,
                },
            ),
            (
                RoutingMode::VllmPrefillDecode {
                    prefill_urls: vec![
                        ("http://p1:8080".to_string(), Some(9000)),
                        ("http://p2:8080".to_string(), Some(9001)),
                        ("http://p3:8080".to_string(), Some(9002)),
                    ],
                    decode_urls: vec!["http://d1:8080".to_string(), "http://d2:8080".to_string()],
                    prefill_policy: None,
                    decode_policy: None,
                    discovery_address: None,
                },
                PolicyConfig::CacheAware {
                    cache_threshold: 0.7,
                    balance_abs_threshold: 20,
                    balance_rel_threshold: 1.2,
                    eviction_interval_secs: 60,
                    max_tree_size: 1000000,
                },
            ),
        ];

        for (mode, policy) in test_cases {
            let config = RouterConfig {
                mode,
                policy,
                host: "127.0.0.1".to_string(),
                port: 3001,
                max_payload_size: 1024 * 1024,
                request_timeout_secs: 60,
                worker_startup_timeout_secs: 10,
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
                cors_allowed_origins: vec![],
                retry: RetryConfig::default(),
                circuit_breaker: CircuitBreakerConfig::default(),
                disable_retries: false,
                disable_circuit_breaker: false,
                health_check: vllm_router_rs::config::HealthCheckConfig::default(),
                enable_igw: false,
                rate_limit_tokens_per_second: None,
                connection_mode: ConnectionMode::Http,
                history_backend: vllm_router_rs::config::HistoryBackend::Memory,
                enable_profiling: false,
                profile_timeout_secs: 30,
                epd: None,
                kv_connector: vllm_router_rs::config::KvConnector::Nixl,
            };

            // Router creation will fail due to health checks, but config should be valid
            let app_context = vllm_router_rs::server::AppContext::new(
                config.clone(),
                reqwest::Client::new(),
                64,
                None,
                config.api_key_validation_urls.clone(),
            )
            .expect("Failed to create AppContext");
            let app_context = std::sync::Arc::new(app_context);
            let result = RouterFactory::create_router(&app_context).await;
            assert!(result.is_err());
            let error_msg = result.unwrap_err();
            // Error should be about health/timeout, not configuration
            assert!(
                error_msg.contains("healthy") || error_msg.contains("timeout"),
                "Unexpected error: {}",
                error_msg
            );
        }
    }
}
