//! Exercises the public environment/configuration, asset loader and HTTP handler.
//! The child process isolates environment configuration from other Rust tests.

#![cfg(all(feature = "cpu", feature = "ready"))]

mod common;

use std::sync::{Arc, Mutex};

use axum::{
    body::Body,
    http::Request,
    routing::{get, post},
    Json,
};
use common::{create_test_context, test_app::create_test_app};
use serde_json::json;
use tower::ServiceExt;
use vllm_router_rs::{
    config::{PolicyConfig, RouterConfig, RoutingMode},
    policies::{KvAwarePolicy, LoadSample},
    routers::{
        http::router::Router,
        router_manager::{RouterId, RouterManager},
    },
};

#[test]
fn completion_public_startup_loads_assets_and_routes_owned_request() {
    if std::env::var_os("CMB_COMPLETION_STARTUP_CHILD").is_none() {
        let assets = tempfile::tempdir().unwrap();
        std::fs::write(
            assets.path().join("tokenizer.json"),
            include_str!("fixtures/tokenizer/completion_word_level.json"),
        )
        .unwrap();
        std::fs::write(
            assets.path().join("config.json"),
            r#"{"model_type":"llama","vocab_size":16}"#,
        )
        .unwrap();
        std::fs::write(
            assets.path().join("tokenizer_config.json"),
            json!({
                "bos_token":"<s>", "eos_token":"<s>",
                "chat_template":"{% for message in messages %}{{ message['content'] }}{% endfor %}"
            })
            .to_string(),
        )
        .unwrap();
        let result = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "completion_public_startup_loads_assets_and_routes_owned_request",
                "--nocapture",
            ])
            .env("CMB_COMPLETION_STARTUP_CHILD", "1")
            .env("VLLM_ROUTER_MODEL", assets.path())
            .env("VLLM_ROUTER_COMPLETION_MODEL_ALIAS", "public")
            .env("VLLM_ROUTER_COMPLETION_CACHE_MODEL", "engine")
            // HTTP Completion must not parse or construct the gRPC caches.
            .env("VLLM_ROUTER_L0_CACHE", "1")
            .env("VLLM_ROUTER_L1_CACHE", "1")
            .env("VLLM_ROUTER_L0_MAX_BYTES", "invalid")
            .env("VLLM_ROUTER_L1_MAX_BYTES", "invalid")
            .env_remove("VLLM_ROUTER_TOKENIZER")
            .env_remove("VLLM_ROUTER_KV_EVENT_SOURCES")
            .status()
            .unwrap();
        assert!(result.success(), "normal Completion startup child failed");
        return;
    }
    tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap().block_on(async {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let config = RouterConfig {
            mode: RoutingMode::Regular { worker_urls: vec![url.clone()] },
            policy: PolicyConfig::KvAware { router_temperature: 0.0, kv_penalty_host: 0.8, kv_penalty_external: 0.6, kv_penalty_disk: 0.3, static_kv_event_sources: Default::default() },
            worker_startup_timeout_secs: 5,
            worker_startup_check_interval_secs: 1,
            disable_retries: true,
            ..Default::default()
        };
        let ctx = create_test_context(config.clone());
        let captured = Arc::new(Mutex::new(Vec::new()));
        let app = axum::Router::new()
            .route("/health", get(|| async { "ok" }))
            .route("/get_server_info", get(|| async { Json(json!({"model_id":"engine","kv_block_size":4})) }))
            .route("/v1/completions", post({
                let (ctx, url, captured) = (ctx.clone(), url.clone(), captured.clone());
                move |body: bytes::Bytes| {
                    let (ctx, url, captured) = (ctx.clone(), url.clone(), captured.clone());
                    async move {
                        let worker = ctx.worker_registry.get_by_url(&url).unwrap();
                        let policy = ctx.policy_registry.get_default_policy();
                        let kv = policy.as_any().downcast_ref::<KvAwarePolicy>().unwrap();
                        assert_eq!(kv.slot_manager().unwrap().sample(&kv.target_for_worker(worker.as_ref())), LoadSample { active_prefill_tokens: 3, active_requests: 1 }, "real loaded tokenizer must prepare the known three-token prompt before dispatch");
                        captured.lock().unwrap().push(body);
                        Json(json!({"choices":[{"text":"ok","finish_reason":"stop"}]}))
                    }
                }
            }));
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let router = Arc::new(Router::new(vec![url], &ctx).await.expect("public Completion configuration must load actual assets"));
        let manager = RouterManager::new(config.clone(), ctx.client.clone(), ctx.worker_registry.clone(), ctx.policy_registry.clone());
        manager.register_router(RouterId::new("http-regular".into()), router);
        let app = create_test_app(Arc::new(manager), reqwest::Client::new(), &config);
        // A static Regular pool is unlabeled; omitting model exercises its
        // normal manager selection without injecting a synthetic model index.
        let raw = " { \"prompt\":\"hello world\", \"priority\":0 }\n";
        let response = app.clone().oneshot(Request::builder().method("POST").uri("/v1/completions").header("content-type", "application/json").body(Body::from(raw)).unwrap()).await.unwrap();
        assert_eq!(response.status(), axum::http::StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
        assert_eq!(serde_json::from_slice::<serde_json::Value>(&body).unwrap()["choices"][0]["text"], "ok");
        assert_eq!(captured.lock().unwrap().as_slice(), &[bytes::Bytes::from_static(raw.as_bytes())]);
        let response = app.oneshot(Request::builder().method("POST").uri("/v1/completions").header("content-type", "application/json").body(Body::from(r#"{"prompt":[-1]}"#)).unwrap()).await.unwrap();
        assert_eq!(response.status(), axum::http::StatusCode::BAD_REQUEST);
        assert_eq!(captured.lock().unwrap().len(), 1, "invalid prepared input cannot dispatch");
        for name in ["VLLM_ROUTER_L0_CACHE", "VLLM_ROUTER_L1_CACHE", "VLLM_ROUTER_L0_MAX_BYTES", "VLLM_ROUTER_L1_MAX_BYTES"] {
            std::env::remove_var(name);
        }
        let error = Router::new(vec!["grpc://127.0.0.1:1".into()], &ctx).await.unwrap_err();
        assert!(error.contains("requires static HTTP Regular kv_aware"), "{error}");
        server.abort();
    });
}
