#[cfg(test)]
mod test_pd_routing {
    use vllm_router_rs::config::{
        CircuitBreakerConfig, ConnectionMode, PolicyConfig, RetryConfig, RouterConfig, RoutingMode,
    };
    use vllm_router_rs::routers::RouterFactory;

    async fn mock_pd_router(
        decode: axum::routing::MethodRouter,
    ) -> (
        Box<dyn vllm_router_rs::routers::RouterTrait>,
        tokio::task::JoinHandle<()>,
    ) {
        mock_pd_router_with_mode(decode, false).await
    }

    async fn mock_pd_router_with_mode(
        decode: axum::routing::MethodRouter,
        push: bool,
    ) -> (
        Box<dyn vllm_router_rs::routers::RouterTrait>,
        tokio::task::JoinHandle<()>,
    ) {
        use axum::{extract::State, http::Uri, routing::get, routing::post, Json, Router};
        use serde_json::json;
        use std::sync::Arc;

        async fn prefill(State(push): State<bool>, uri: Uri) -> Json<serde_json::Value> {
            let usage = if uri.path().ends_with("/responses") {
                json!({"input_tokens": 70, "output_tokens": 1, "total_tokens": 71, "input_tokens_details": {"cached_tokens": 4}})
            } else {
                json!({"prompt_tokens": 70, "completion_tokens": 1, "total_tokens": 71, "prompt_tokens_details": {"cached_tokens": 4}})
            };
            let mut response = json!({
                "kv_transfer_params": {
                    "remote_engine_id": "P",
                    "do_remote_prefill": true,
                },
                "usage": usage,
            });
            if push {
                response["kv_transfer_params"]["remote_host"] = json!("127.0.0.1");
                response["kv_transfer_params"]["remote_port"] = json!(9100);
                response["kv_transfer_params"]["transfer_mode"] = json!("push");
            }
            Json(response)
        }
        let app = Router::new()
            .route("/p/health", get(|| async { "ok" }))
            .route("/d/health", get(|| async { "ok" }))
            .route("/p/v1/chat/completions", post(prefill))
            .route("/p/v1/completions", post(prefill))
            .route("/p/v1/responses", post(prefill))
            .with_state(push)
            .route("/d/v1/chat/completions", decode.clone())
            .route("/d/v1/responses", decode.clone())
            .route("/d/v1/completions", decode);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        let config = RouterConfig {
            mode: RoutingMode::VllmPrefillDecode {
                prefill_urls: vec![(format!("{base}/p"), None)],
                decode_urls: vec![format!("{base}/d")],
                prefill_policy: None,
                decode_policy: None,
                discovery_address: None,
            },
            policy: PolicyConfig::Random,
            kv_connector: vllm_router_rs::config::KvConnector::Nixl,
            disable_retries: true,
            disable_circuit_breaker: true,
            worker_startup_timeout_secs: 5,
            ..RouterConfig::default()
        };
        let client = reqwest::Client::builder()
            .no_proxy()
            .timeout(std::time::Duration::from_secs(5))
            .build()
            .unwrap();
        let context = Arc::new(
            vllm_router_rs::server::AppContext::new(config, client, 64, None, vec![]).unwrap(),
        );
        let router = RouterFactory::create_router(&context).await.unwrap();
        (router, server)
    }

    #[tokio::test]
    async fn test_pd_usage_through_sequential_and_nixl_push_router() {
        use axum::{
            http::{Method, StatusCode, Uri},
            response::IntoResponse,
            routing::post,
            Json,
        };
        use serde_json::{json, Value};

        async fn decode(uri: Uri, Json(body): Json<Value>) -> axum::response::Response {
            let responses = uri.path().ends_with("/responses");
            let usage = if responses {
                json!({"input_tokens": 999, "output_tokens": 8, "total_tokens": 1007,
                       "output_tokens_details": {"reasoning_tokens": 2}})
            } else {
                json!({"prompt_tokens": 999, "completion_tokens": 8, "total_tokens": 1007,
                       "prompt_tokens_details": {"cached_tokens": 99, "audio_tokens": 2}})
            };
            if body["stream"] == true {
                let delta = if responses {
                    json!({"type": "response.output_text.delta", "delta": "answer"})
                } else {
                    json!({"choices": [{"index": 0, "text": "answer"}], "usage": null})
                };
                let final_usage = if responses {
                    json!({"type": "response.completed", "response": {"usage": usage}})
                } else {
                    json!({"choices": [], "usage": usage})
                };
                return (
                    [("content-type", "text/event-stream")],
                    format!("data: {delta}\n\ndata: {final_usage}\n\ndata: [DONE]\n\n"),
                )
                    .into_response();
            }
            Json(json!({"choices": [{"text": "answer"}], "usage": usage})).into_response()
        }

        for push in [false, true] {
            let (router, server) = mock_pd_router_with_mode(post(decode), push).await;
            // The first NIXL push request learns the identity. The second and
            // all subsequent requests exercise concurrent P/D dispatch.
            for _ in 0..2 {
                for path in ["/v1/chat/completions", "/v1/completions", "/v1/responses"] {
                    for streaming in [false, true] {
                        let mut request =
                            json!({"model": "test", "stream": streaming, "max_tokens": 8});
                        match path {
                            "/v1/chat/completions" => {
                                request["messages"] = json!([{"role": "user", "content": "hello"}])
                            }
                            "/v1/responses" => {
                                request["input"] = json!("hello");
                                request["max_output_tokens"] = json!(8);
                            }
                            _ => request["prompt"] = json!("hello"),
                        }
                        if streaming && path != "/v1/responses" {
                            request["stream_options"] = json!({"include_usage": true});
                        }
                        let response = router
                            .route_transparent(None, path, &Method::POST, request)
                            .await;
                        assert_eq!(response.status(), StatusCode::OK);
                        let bytes = axum::body::to_bytes(response.into_body(), 4096)
                            .await
                            .unwrap();
                        let result: Value = if streaming {
                            let text = std::str::from_utf8(&bytes).unwrap();
                            assert!(text.ends_with("data: [DONE]\n\n"));
                            let frames: Vec<Value> = text
                                .split("\n\n")
                                .filter_map(|frame| frame.strip_prefix("data: "))
                                .filter(|data| *data != "[DONE]")
                                .map(|data| serde_json::from_str(data).unwrap())
                                .collect();
                            assert_eq!(frames.len(), 2);
                            assert!(frames[0].get("usage").is_none_or(Value::is_null));
                            frames.into_iter().last().unwrap()
                        } else {
                            serde_json::from_slice(&bytes).unwrap()
                        };
                        let usage = if streaming && path == "/v1/responses" {
                            assert!(result.get("usage").is_none());
                            &result["response"]["usage"]
                        } else {
                            &result["usage"]
                        };
                        let expected = if path == "/v1/responses" {
                            json!({"input_tokens": 70, "output_tokens": 8, "total_tokens": 78,
                                   "output_tokens_details": {"reasoning_tokens": 2},
                                   "input_tokens_details": {"cached_tokens": 4}})
                        } else {
                            json!({"prompt_tokens": 70, "completion_tokens": 8, "total_tokens": 78,
                                   "prompt_tokens_details": {"cached_tokens": 4, "audio_tokens": 2}})
                        };
                        assert_eq!(
                            usage, &expected,
                            "path={path}, streaming={streaming}, push={push}"
                        );
                    }
                }
            }
            server.abort();
        }
    }

    #[tokio::test]
    async fn test_pd_decode_429_passthrough_through_router() {
        use axum::{http::Method, http::StatusCode, routing::post};
        use serde_json::json;

        let (router, server) = mock_pd_router(post(|| async {
            (
                StatusCode::TOO_MANY_REQUESTS,
                [("retry-after", "7"), ("content-type", "text/plain")],
                "decode is overloaded\n",
            )
        }))
        .await;
        for path in ["/v1/chat/completions", "/v1/completions", "/v1/responses"] {
            for streaming in [false, true] {
                // A text error must bypass successful-response JSON/logprobs handling.
                let mut request = json!({
                    "model": "test",
                    "stream": streaming,
                    "max_tokens": 1,
                    "logprobs": if path == "/v1/chat/completions" { json!(true) } else { json!(1) },
                });
                if path == "/v1/chat/completions" {
                    request["messages"] = json!([{"role": "user", "content": "hello"}]);
                } else if path == "/v1/responses" {
                    request["input"] = json!("hello");
                } else {
                    request["prompt"] = json!("hello");
                }
                let response = router
                    .route_transparent(None, path, &Method::POST, request)
                    .await;
                assert_eq!(
                    response.status(),
                    StatusCode::TOO_MANY_REQUESTS,
                    "path={path}, streaming={streaming}",
                );
                assert_eq!(response.headers()["retry-after"], "7");
                assert_eq!(response.headers()["content-type"], "text/plain");
                let bytes = axum::body::to_bytes(response.into_body(), 4096)
                    .await
                    .unwrap();
                assert_eq!(bytes.as_ref(), b"decode is overloaded\n");
            }
        }
        server.abort();
    }

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
                    (format!("engine-{hash}")): {
                        "metadata": {"image_grid_thw": [[1, 2, 3]]},
                        "peer_host": "encoder",
                        "peer_port": 4321,
                        "size_bytes": 128,
                    },
                },
            }))
            .into_response()
        }
        async fn prefill(
            State(seen): State<Seen>,
            Json(body): Json<Value>,
        ) -> axum::response::Response {
            seen.lock().unwrap().push(("P".into(), body.clone()));
            if matches!(body["model"].as_str(), Some("prefill-400" | "prefill-422")) {
                let status = if body["model"] == "prefill-400" {
                    StatusCode::BAD_REQUEST
                } else {
                    StatusCode::UNPROCESSABLE_ENTITY
                };
                return (
                    status,
                    Json(json!({"error": "invalid sampling parameters"})),
                )
                    .into_response();
            }
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
            ("prefill-400", false, StatusCode::BAD_REQUEST),
            ("prefill-422", false, StatusCode::UNPROCESSABLE_ENTITY),
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
            if model.starts_with("prefill-") {
                assert_eq!(response.headers()["content-type"], "application/json");
            }
            let bytes = axum::body::to_bytes(response.into_body(), 4096)
                .await
                .unwrap();
            if stream {
                assert!(String::from_utf8_lossy(&bytes).contains("[DONE]"));
            }
            if model.starts_with("prefill-") {
                assert_eq!(
                    serde_json::from_slice::<Value>(&bytes).unwrap(),
                    json!({"error": "invalid sampling parameters"})
                );
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
            let uuid = p["messages"][0]["content"][0]["uuid"].as_str().unwrap();
            let hash = format!("engine-{uuid}");
            assert_eq!(p["ec_transfer_params"]["ec_items"][0]["mm_hash"], hash);
            assert_eq!(p["ec_transfer_params"][&hash]["peer_port"], 4321);
            assert!(p["ec_transfer_params"].get(uuid).is_none());
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

    /// With the NIXL connector, the first request caches the prefill push
    /// identity and subsequent requests take the concurrent fast path. The
    /// encoder stage must still run for those requests; otherwise raw media
    /// reaches P/D without a published embedding.
    #[tokio::test]
    async fn test_epd_nixl_push_still_encodes_after_identity_cached() {
        use axum::{
            extract::State,
            response::IntoResponse,
            routing::{get, post},
            Json, Router,
        };
        use serde_json::{json, Value};
        use std::sync::{Arc, Mutex};
        use vllm_router_rs::config::{EpdConfig, KvConnector};

        type Seen = Arc<Mutex<Vec<(String, Value)>>>;
        async fn encode(
            State(seen): State<Seen>,
            Json(body): Json<Value>,
        ) -> axum::response::Response {
            seen.lock().unwrap().push(("E".into(), body.clone()));
            let uuid = body["messages"][0]["content"][0]["uuid"].as_str().unwrap();
            Json(json!({
                "ec_transfer_params": {
                    (format!("engine-{uuid}")): {
                        "metadata": {"image_grid_thw": [[1, 2, 3]]},
                        "peer_host": "encoder",
                        "peer_port": 4321,
                        "size_bytes": 128,
                    },
                },
            }))
            .into_response()
        }
        async fn prefill(
            State(seen): State<Seen>,
            Json(body): Json<Value>,
        ) -> axum::response::Response {
            seen.lock().unwrap().push(("P".into(), body.clone()));
            if body["model"] == "prefill-422" {
                return (
                    axum::http::StatusCode::UNPROCESSABLE_ENTITY,
                    Json(json!({"error": "invalid sampling parameters"})),
                )
                    .into_response();
            }
            Json(json!({
                "kv_transfer_params": {
                    "remote_engine_id": "P",
                    "remote_host": "127.0.0.1",
                    "remote_port": 9100,
                    "transfer_mode": "push",
                },
            }))
            .into_response()
        }
        async fn decode(
            State(seen): State<Seen>,
            Json(body): Json<Value>,
        ) -> axum::response::Response {
            seen.lock().unwrap().push(("D".into(), body.clone()));
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
            .route("/p/generate", post(prefill))
            .route("/d/generate", post(decode))
            .with_state(seen.clone());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        let config = RouterConfig {
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
            kv_connector: KvConnector::Nixl,
            worker_startup_timeout_secs: 5,
            ..RouterConfig::default()
        };
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

        let chat_body = || {
            json!({
                "model": "test",
                "max_tokens": 16,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
                        {"type": "text", "text": "choose"},
                    ],
                }],
            })
        };

        // Invalid transparent bodies must be rejected before either PD path mutates them.
        for body in [Value::Null, json!([]), json!("text"), json!(7)] {
            let response = router
                .route_transparent(None, "/generate", &axum::http::Method::POST, body)
                .await;
            assert_eq!(response.status(), axum::http::StatusCode::BAD_REQUEST);
        }
        assert!(seen.lock().unwrap().is_empty());

        // First request: no cached push identity, sequential path.
        let response = router
            .route_transparent(
                None,
                "/v1/chat/completions",
                &axum::http::Method::POST,
                chat_body(),
            )
            .await;
        assert_eq!(response.status(), axum::http::StatusCode::OK);
        let _ = axum::body::to_bytes(response.into_body(), 4096).await;
        let phases: Vec<_> = seen
            .lock()
            .unwrap()
            .iter()
            .map(|(phase, _)| phase.clone())
            .collect();
        assert_eq!(phases, ["E", "P", "D"]);

        // Second request: the push identity is now cached, so the concurrent
        // fast path runs. The encoder must still be called and P/D must
        // receive the rewritten embeds metadata, not raw media.
        seen.lock().unwrap().clear();
        let response = router
            .route_transparent(
                None,
                "/v1/chat/completions",
                &axum::http::Method::POST,
                chat_body(),
            )
            .await;
        assert_eq!(response.status(), axum::http::StatusCode::OK);
        let _ = axum::body::to_bytes(response.into_body(), 4096).await;
        let seen_requests = seen.lock().unwrap().clone();
        let phases: Vec<_> = seen_requests
            .iter()
            .map(|(phase, _)| phase.as_str())
            .collect();
        // P and D are dispatched concurrently, so only their set is stable.
        assert_eq!(phases[0], "E");
        let mut pd: Vec<_> = phases[1..].to_vec();
        pd.sort_unstable();
        assert_eq!(pd, ["D", "P"]);

        let p = &seen_requests[1..]
            .iter()
            .find(|(phase, _)| phase == "P")
            .expect("prefill request")
            .1;
        let d = &seen_requests[1..]
            .iter()
            .find(|(phase, _)| phase == "D")
            .expect("decode request")
            .1;
        assert_eq!(
            p["messages"][0]["content"][0]["image_embeds"]["image_grid_thw"],
            json!([1, 2, 3])
        );
        assert!(d.get("ec_transfer_params").is_none());
        assert_eq!(d["kv_transfer_params"]["do_remote_prefill"], true);
        assert_eq!(d["kv_transfer_params"]["remote_engine_id"], "P");

        seen.lock().unwrap().clear();
        for body in [Value::Null, json!([]), json!("text"), json!(7)] {
            let response = router
                .route_transparent(None, "/generate", &axum::http::Method::POST, body)
                .await;
            assert_eq!(response.status(), axum::http::StatusCode::BAD_REQUEST);
        }
        assert!(seen.lock().unwrap().is_empty());
        let mut invalid = chat_body();
        invalid["model"] = json!("prefill-422");
        let response = router
            .route_transparent(
                None,
                "/v1/chat/completions",
                &axum::http::Method::POST,
                invalid,
            )
            .await;
        assert_eq!(
            response.status(),
            axum::http::StatusCode::UNPROCESSABLE_ENTITY
        );
        assert_eq!(response.headers()["content-type"], "application/json");
        let bytes = axum::body::to_bytes(response.into_body(), 4096)
            .await
            .unwrap();
        assert_eq!(
            serde_json::from_slice::<Value>(&bytes).unwrap(),
            json!({"error": "invalid sampling parameters"})
        );
        server.abort();
    }

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
                program_scheduling: None,
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
