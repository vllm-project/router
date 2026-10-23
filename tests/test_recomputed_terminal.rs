use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
    time::Duration,
};

use axum::{
    body::Body,
    extract::State,
    http::{Method, Request, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde_json::{json, Value};
use tokio::sync::Notify;
use vllm_router_rs::{
    config::{KvConnector, PolicyConfig, RouterConfig, RoutingMode},
    routers::RouterFactory,
};

#[derive(Clone, Default)]
struct Backend {
    push: bool,
    concurrent: Arc<AtomicBool>,
    decode_arrived: Arc<Notify>,
    seen: Arc<Mutex<Vec<(&'static str, Value)>>>,
}

fn prompt_logprobs() -> Value {
    json!([null, {"7":{"logprob":-0.5,"rank":1,"decoded_token":"hi"}}])
}

async fn prefill(State(state): State<Backend>, Json(body): Json<Value>) -> Response {
    let backend_failure = body["model"] == "backend-error";
    state.seen.lock().unwrap().push(("P", body));
    // This handshake proves D was dispatched before P returned. Merely observing
    // both requests, or a push identity, does not prove the concurrent path ran.
    if state.concurrent.load(Ordering::SeqCst)
        && tokio::time::timeout(Duration::from_secs(3), state.decode_arrived.notified())
            .await
            .is_err()
    {
        return (
            StatusCode::GATEWAY_TIMEOUT,
            "D did not arrive while P was pending",
        )
            .into_response();
    }
    let transfer = if state.push {
        json!({"remote_engine_id":"P", "remote_host":"127.0.0.1",
            "remote_port":9100, "transfer_mode":"push"})
    } else {
        json!({"do_remote_prefill":true, "do_remote_decode":false,
            "remote_engine_id":"P", "remote_block_ids":[1]})
    };
    // Keep the error fixture independent of the pre-existing merge-on-error
    // behavior: non-2xx bodies must bypass the recomputation guard.
    if backend_failure {
        return Json(json!({"choices":[], "kv_transfer_params":transfer})).into_response();
    }
    Json(json!({
        "prompt_logprobs":prompt_logprobs(),
        "choices":[{"index":0,"prompt_logprobs":prompt_logprobs(),"logprobs":{
            "tokens":["Q", "hi", "P_ONLY"], "token_logprobs":[null,-0.5,-9.0],
            "text_offset":[0,1,3], "top_logprobs":[null,{"hi":-0.5},{"P_ONLY":-9.0}]
        }}],
        "kv_transfer_params":transfer
    }))
    .into_response()
}

fn backend_error() -> Value {
    // A marker in an error response must not be rewritten as a successful-body interruption.
    json!({"error":{"message":"decode rejected request","type":"invalid_request_error"},
        "choices":[{"stop_reason":"recomputed"}]})
}

async fn decode(State(state): State<Backend>, Json(body): Json<Value>) -> Response {
    state.seen.lock().unwrap().push(("D", body.clone()));
    if state.concurrent.load(Ordering::SeqCst) {
        state.decode_arrived.notify_one();
    }
    if body["model"] == "backend-error" {
        return (StatusCode::UNPROCESSABLE_ENTITY, Json(backend_error())).into_response();
    }
    let mut choice = json!({"index":0,"finish_reason":"stop","message":{"content":"A"},
        "logprobs":{"tokens":["A"],"token_logprobs":[-0.25],
        "text_offset":[0],"top_logprobs":[{"A":-0.25}]}});
    if body["model"] == "interrupted" {
        choice["stop_reason"] = json!("recomputed");
    }
    if body["stream"] == true {
        choice.as_object_mut().unwrap().remove("message");
        choice["delta"] = json!({});
        let events = format!(
            "data: {}\n\ndata: {}\n\ndata: [DONE]\n\n",
            json!({"choices":[{"index":0,"delta":{"content":"A"},"finish_reason":null}]}),
            json!({"choices":[choice]})
        );
        ([("content-type", "text/event-stream")], events).into_response()
    } else {
        Json(json!({"choices":[choice]})).into_response()
    }
}

#[tokio::test]
async fn static_sequential_terminal_validation() {
    exercise(false, false).await;
}

#[tokio::test]
async fn static_nixl_push_concurrent_terminal_validation() {
    exercise(true, false).await;
}

#[tokio::test]
async fn discovery_sequential_terminal_validation() {
    exercise(false, true).await;
}

#[tokio::test]
async fn discovery_nixl_push_concurrent_terminal_validation() {
    exercise(true, true).await;
}

async fn exercise(push: bool, discovery: bool) {
    let state = Backend {
        push,
        ..Backend::default()
    };
    let app = Router::new()
        .route("/p/health", get(|| async { "ok" }))
        .route("/d/health", get(|| async { "ok" }))
        .route("/p/v1/chat/completions", post(prefill))
        .route("/p/v1/completions", post(prefill))
        .route("/d/v1/completions", post(decode))
        .route("/d/v1/chat/completions", post(decode))
        .with_state(state.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let base = format!("http://{address}");
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    // Discovery requires real ZMQ registration; no production registry hooks are added.
    let discovery_address = if discovery {
        let reservation = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        Some(reservation.local_addr().unwrap().to_string())
    } else {
        None
    };
    let config = RouterConfig {
        mode: RoutingMode::VllmPrefillDecode {
            prefill_urls: if discovery {
                vec![]
            } else {
                vec![(format!("{base}/p"), None)]
            },
            decode_urls: if discovery {
                vec![]
            } else {
                vec![format!("{base}/d")]
            },
            prefill_policy: None,
            decode_policy: None,
            discovery_address: discovery_address.clone(),
        },
        policy: PolicyConfig::Random,
        kv_connector: KvConnector::Nixl,
        worker_startup_timeout_secs: 5,
        ..RouterConfig::default()
    };
    let context = Arc::new(
        vllm_router_rs::server::AppContext::new(config, reqwest::Client::new(), 64, None, vec![])
            .unwrap(),
    );
    let router = RouterFactory::create_router(&context).await.unwrap();
    let zmq_context = zmq::Context::new();
    let registration = discovery_address.map(|addr| {
        let socket = zmq_context.socket(zmq::DEALER).unwrap();
        socket.set_linger(0).unwrap();
        socket.set_sndtimeo(1000).unwrap();
        socket.connect(&format!("tcp://{addr}")).unwrap();
        socket
    });
    let register = || {
        if let Some(socket) = &registration {
            for (kind, role, port) in [("P", "p", 9100), ("D", "d", 9200)] {
                // ROUTER receives [dealer identity, MessagePack registration].
                // JSON text is rejected by ServiceRegistry::parse_registration;
                // DEALER supplies the identity frame without a REQ empty delimiter.
                let message = rmp_serde::to_vec_named(&json!({
                    "type":kind,"http_address":format!("{address}/{role}"),
                    "zmq_address":format!("127.0.0.1:{port}")
                }))
                .unwrap();
                socket.send(message.as_slice(), 0).unwrap();
            }
        }
    };
    if discovery {
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                register();
                if router.health(Request::new(Body::empty())).await.status() == StatusCode::OK {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        })
        .await
        .expect("discovery must register both workers");
    }
    if push {
        let response = router
            .route_transparent(
                None,
                "/v1/chat/completions",
                &Method::POST,
                json!({"model":"normal","messages":[{"role":"user","content":"test"}]}),
            )
            .await;
        assert_eq!(response.status(), StatusCode::OK);
        axum::body::to_bytes(response.into_body(), 16384)
            .await
            .unwrap();
        let seen = state.seen.lock().unwrap();
        assert_eq!(
            seen.iter().map(|(phase, _)| *phase).collect::<Vec<_>>(),
            ["P", "D"]
        );
        drop(seen);
        state.concurrent.store(true, Ordering::SeqCst);
    }
    for path in ["/v1/chat/completions", "/v1/completions"] {
        for model in ["normal", "interrupted", "backend-error"] {
            for stream in [false, true] {
                let logprobs_cases: &[Option<bool>] = if stream {
                    &[Some(true)]
                } else {
                    &[None, Some(false), Some(true)]
                };
                for &logprobs in logprobs_cases {
                    register(); // Refresh the discovery lease before every request.
                    state.seen.lock().unwrap().clear();
                    let mut request = json!({"model":model,"messages":[{"role":"user","content":"test"}],
                        "prompt":"test","max_tokens":8,"stream":stream});
                    if let Some(value) = logprobs {
                        request["logprobs"] = json!(value);
                    }
                    let response = tokio::time::timeout(
                        Duration::from_secs(5),
                        router.route_transparent(None, path, &Method::POST, request),
                    )
                    .await
                    .expect("PD request must complete");
                    let status = response.status();
                    let body = String::from_utf8(
                        axum::body::to_bytes(response.into_body(), 16384)
                            .await
                            .unwrap()
                            .to_vec(),
                    )
                    .unwrap();
                    let seen = state.seen.lock().unwrap();
                    let mut phases: Vec<_> = seen.iter().map(|(phase, _)| *phase).collect();
                    if push {
                        phases.sort_unstable();
                        assert_eq!(phases, ["D", "P"]);
                    } else {
                        assert_eq!(phases, ["P", "D"]);
                    }
                    let d = &seen.iter().find(|(phase, _)| *phase == "D").unwrap().1;
                    assert_eq!(d["kv_transfer_params"]["remote_engine_id"], "P");
                    if push {
                        assert!(d["kv_transfer_params"]["remote_request_id"].is_string());
                    } else {
                        assert_eq!(d["kv_transfer_params"]["remote_block_ids"], json!([1]));
                    }
                    drop(seen);
                    if model == "backend-error" {
                        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{body}");
                        assert_eq!(
                            serde_json::from_str::<Value>(&body).unwrap(),
                            backend_error()
                        );
                    } else if model == "interrupted" {
                        assert_eq!(
                            status,
                            if stream {
                                StatusCode::OK
                            } else {
                                StatusCode::BAD_GATEWAY
                            },
                            "{body}"
                        );
                        assert_eq!(body.matches("upstream_recomputed").count(), 1, "{body}");
                        assert!(!body.contains("[DONE]"), "{body}");
                        assert!(!body.contains("\"finish_reason\":\"stop\""), "{body}");
                        if stream {
                            assert!(body.contains("\"content\":\"A\""), "{body}");
                        }
                    } else {
                        assert_eq!(status, StatusCode::OK, "{body}");
                        assert!(!body.contains("upstream_recomputed"), "{body}");
                        if stream {
                            let events: Vec<Value> = body
                                .lines()
                                .filter_map(|line| line.strip_prefix("data: "))
                                .filter(|data| *data != "[DONE]")
                                .map(|data| serde_json::from_str(data).unwrap())
                                .collect();
                            assert_eq!(events.len(), 2, "{body}");
                            assert_eq!(events[0]["choices"][0]["delta"]["content"], "A");
                            assert_eq!(events[1]["choices"][0]["finish_reason"], "stop");
                            assert!(body.trim_end().ends_with("data: [DONE]"), "{body}");
                            assert_eq!(body.matches("[DONE]").count(), 1);
                        } else {
                            let value: Value = serde_json::from_str(&body).unwrap();
                            assert_eq!(value["choices"][0]["message"]["content"], "A");
                            assert_eq!(value["choices"][0]["finish_reason"], "stop");
                            if logprobs.is_some() {
                                assert_eq!(value["prompt_logprobs"], prompt_logprobs());
                                let choice = &value["choices"][0];
                                assert_eq!(choice["prompt_logprobs"], prompt_logprobs());
                                assert_eq!(choice["logprobs"]["tokens"], json!(["Q", "hi", "A"]));
                                assert_eq!(
                                    choice["logprobs"]["token_logprobs"],
                                    json!([null, -0.5, -0.25])
                                );
                                assert_eq!(choice["logprobs"]["text_offset"], json!([0, 1, 3]));
                                assert_eq!(
                                    choice["logprobs"]["top_logprobs"],
                                    json!([null,{"hi":-0.5},{"A":-0.25}])
                                );
                            }
                        }
                    }
                }
            }
        }
        for stream in [false, true] {
            register();
            let response = router
                .route_transparent(
                    None,
                    path,
                    &Method::POST,
                    json!({"model":"interrupted","stop":["recomputed"],"prompt":"test",
                    "messages":[{"role":"user","content":"test"}],"stream":stream}),
                )
                .await;
            assert_eq!(response.status(), StatusCode::OK);
            let body = String::from_utf8(
                axum::body::to_bytes(response.into_body(), 16384)
                    .await
                    .unwrap()
                    .to_vec(),
            )
            .unwrap();
            assert!(body.contains("stop_reason"));
            assert!(!body.contains("upstream_recomputed"));
        }
    }
    server.abort();
}
