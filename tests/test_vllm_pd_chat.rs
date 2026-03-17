/// Tests for VllmPD Router's route_chat after JSON pass-through migration.
/// The migration changed route_chat from serializing ChatCompletionRequest to
/// directly using serde_json::Value (body.clone()), removing the serialize+error path.
mod common;

use axum::http::StatusCode;
use serde_json::json;
use vllm_router_rs::config::{
    ConnectionMode, PolicyConfig, RetryConfig, RouterConfig, RoutingMode,
};
use vllm_router_rs::routers::http::vllm_pd_router::VllmPDRouter;
use vllm_router_rs::routers::RouterTrait;

fn vllm_pd_config() -> RouterConfig {
    RouterConfig {
        mode: RoutingMode::VllmPrefillDecode {
            prefill_urls: vec![],
            decode_urls: vec![],
            prefill_policy: None,
            decode_policy: None,
            discovery_address: None,
        },
        policy: PolicyConfig::RoundRobin,
        host: "127.0.0.1".to_string(),
        port: 3010,
        max_payload_size: 256 * 1024 * 1024,
        request_timeout_secs: 30,
        worker_startup_timeout_secs: 10,
        worker_startup_check_interval_secs: 1,
        discovery: None,
        intra_node_data_parallel_size: 1,
        api_key: None,
        api_key_validation_urls: vec![],
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
        circuit_breaker: vllm_router_rs::config::CircuitBreakerConfig::default(),
        disable_retries: false,
        disable_circuit_breaker: false,
        health_check: vllm_router_rs::config::HealthCheckConfig::default(),
        enable_igw: false,
        connection_mode: ConnectionMode::Http,
        model_path: None,
        tokenizer_path: None,
        history_backend: vllm_router_rs::config::HistoryBackend::Memory,
        enable_profiling: false,
        profile_timeout_secs: 30,
    }
}

// =============================================================================
// Test: VllmPD route_chat returns 503 when no workers are available
// =============================================================================

#[tokio::test]
async fn test_vllm_pd_route_chat_returns_503_without_workers() {
    let config = vllm_pd_config();
    let app_context = common::create_test_context(config);

    // Create VllmPDRouter in direct-URL mode with no workers
    let router = VllmPDRouter::new(vec![], vec![], None, &app_context)
        .await
        .unwrap();

    let body = json!({
        "messages": [{"role": "user", "content": "Hello!"}],
        "stream": false,
        "reasoning": {"effort": "high"},
        "custom_field": 42
    });

    let resp = router.route_chat(None, &body, None).await;

    assert_eq!(
        resp.status(),
        StatusCode::SERVICE_UNAVAILABLE,
        "VllmPD route_chat with no workers should return 503"
    );
}

// =============================================================================
// Test: VllmPD route_chat accepts JSON with extra fields without error
// =============================================================================

#[tokio::test]
async fn test_vllm_pd_route_chat_accepts_arbitrary_json() {
    let config = vllm_pd_config();
    let app_context = common::create_test_context(config);

    // Create VllmPDRouter in direct-URL mode with no workers
    let router = VllmPDRouter::new(vec![], vec![], None, &app_context)
        .await
        .unwrap();

    // Send JSON with fields that would have been rejected by ChatCompletionRequest deserialization.
    // After the migration, route_chat accepts any valid JSON — it should reach the
    // "no workers" check rather than failing on deserialization.
    let body = json!({
        "messages": [{"role": "user", "content": "Hello!"}],
        "stream": false,
        "unknown_future_field": "should not cause deserialization error",
        "reasoning": {"effort": "high"},
        "custom_vllm_param": 42,
        "nested": {"deeply": {"nested": true}}
    });

    let resp = router.route_chat(None, &body, None).await;

    // We expect 503 (no workers), NOT a deserialization error (400/500).
    // This proves that arbitrary JSON passes through without being rejected.
    assert_eq!(
        resp.status(),
        StatusCode::SERVICE_UNAVAILABLE,
        "VllmPD route_chat should accept arbitrary JSON fields (got {} instead of 503)",
        resp.status()
    );
}
