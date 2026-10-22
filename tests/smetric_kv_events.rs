mod common;
#[path = "common/kv_event_worker.rs"]
mod kv_event_worker;

use axum::{
    body::{to_bytes, Body},
    http::{Request, StatusCode},
};
use common::{create_test_context, test_app::create_test_app};
use kv_event_worker::EventWorker;
use serde_json::{json, Value};
use std::{sync::Arc, time::Duration};
use tower::ServiceExt;
use vllm_router_rs::{
    config::{PolicyConfig, RouterConfig, RoutingMode},
    routers::http::router::Router,
};

async fn selected_worker(app: &axum::Router) -> String {
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({"model":"test-model", "messages":[
            {"role":"user","content":"same history"},
            {"role":"assistant","content":"previous output"},
            {"role":"user","content":"next turn"}
        ], "max_tokens":1})
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = to_bytes(response.into_body(), 1 << 20).await.unwrap();
    let body: Value = serde_json::from_slice(&bytes).unwrap();
    body["choices"][0]["message"]["content"]
        .as_str()
        .unwrap()
        .to_string()
}

async fn wait_for_target(app: &axum::Router, expected: &str) {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(3);
    loop {
        let selected = selected_worker(app).await;
        if selected == expected {
            return;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "expected {expected}, still selected {selected}"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

#[tokio::test]
async fn smetric_grpc_placement_tracks_events_gaps_and_worker_removal() {
    let cold = EventWorker::spawn("cold").await;
    let hot = EventWorker::spawn("hot").await;
    let urls = vec![cold.url.clone(), hot.url.clone()];
    let config = RouterConfig {
        mode: RoutingMode::Regular { worker_urls: urls.clone() },
        policy: PolicyConfig::SMetric { config: serde_yaml::from_str(
            "C_LIN: 1\nC_ATT: 0\nPREFILL_RATE: 1000000\nSLACK: 1\nHIT_RATIO: 0.1\nTTFT_SLO_BASE: 10\nTTFT_SLO_PER_CHAR: 0\nEVICTION_INTERVAL_SECS: 0\n"
        ).unwrap() },
        disable_retries: true, disable_circuit_breaker: true,
        worker_startup_timeout_secs: 5, worker_startup_check_interval_secs: 1,
        ..RouterConfig::default()
    };
    let ctx = create_test_context(config.clone());
    let router = Router::new(urls, &ctx).await.unwrap();
    router.pin_test_token_ids((1..=9).collect());
    cold.wait_for_subscription().await;
    hot.wait_for_subscription().await;
    let hot_worker = ctx.worker_registry.get_by_url(&hot.url).unwrap();
    hot_worker.increment_load();
    let router = Arc::new(router);
    let app = create_test_app(router.clone(), reqwest::Client::new(), &config);
    let stored = json!({"type":"BlockStored", "block_hashes":[101,102],
        "parent_block_hash":null, "token_ids":[1,2,3,4,5,6,7,8], "block_size":4, "medium":"GPU"});
    hot.publish(1, stored.clone());
    wait_for_target(&app, "hot").await;
    hot.publish(
        2,
        json!({"type":"BlockRemoved", "block_hashes":[101,102], "medium":"GPU"}),
    );
    wait_for_target(&app, "cold").await;
    hot.publish(3, stored.clone());
    wait_for_target(&app, "hot").await;
    hot.publish(4, json!({"type":"AllBlocksCleared"}));
    wait_for_target(&app, "cold").await;
    hot.publish(5, stored.clone());
    wait_for_target(&app, "hot").await;
    hot.publish(
        7,
        json!({"type":"BlockStored", "block_hashes":[103],
        "parent_block_hash":null, "token_ids":[20,21,22,23], "block_size":4, "medium":"GPU"}),
    );
    wait_for_target(&app, "cold").await;
    hot.publish(8, stored);
    wait_for_target(&app, "hot").await;
    router.remove_worker(&hot.url);
    assert!(ctx.worker_registry.get_by_url(&hot.url).is_none());
    assert_eq!(selected_worker(&app).await, "cold");
    assert_eq!(hot_worker.load(), 1);
    for worker in [&cold, &hot] {
        for request in worker.state.captured.lock().unwrap().iter() {
            assert_eq!(request.token_ids, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
            assert!(!request.had_text_prompt);
        }
    }
    hot_worker.decrement_load();
}
