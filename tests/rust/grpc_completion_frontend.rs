#![cfg(all(feature = "cpu", feature = "ready"))]

mod common;
#[path = "common/grpc_frontend_fixture.rs"]
mod fixture;
#[path = "common/grpc_kv_fixture.rs"]
mod kv_fixture;

use serde_json::json;
use vllm_router_rs::backend::{prepare_grpc_completion, EngineFrontend};
use vllm_router_rs::protocols::spec::CompletionRequest;

#[tokio::test]
async fn grpc_prefix_selection_uses_the_ids_actually_dispatched() {
    use axum::body::to_bytes;
    use kv_fixture::{KvPublisher, TOPIC};
    use std::collections::HashMap;
    use vllm_router_rs::config::{PolicyConfig, RouterConfig, StaticKvEventSources};
    use vllm_router_rs::protocols::spec::ChatCompletionRequest;
    use vllm_router_rs::routers::http::router::Router;
    let model = fixture::model_fixture();
    let warm = common::mock_vllm_rs::MockVllmRsServer::spawn().await;
    let cold = common::mock_vllm_rs::MockVllmRsServer::spawn().await;
    let frontend = EngineFrontend::new();
    let bodies = [
        json!({"messages":[{"role":"user","content":"hello prefix"}]}),
        json!({"messages":[{"role":"user","content":[{"type":"text","text":"hello prefix"}]}]}),
        json!({"messages":[{"role":"user","content":"call f"},{"role":"assistant","content":"calling","tool_calls":[{"id":"c","type":"function","function":{"name":"f","arguments":"{}"}}]},{"role":"tool","tool_call_id":"c","content":"done"}]}),
        json!({"messages":[{"role":"user","content":"a longer chat to truncate"}],"truncate_prompt_tokens":6}),
    ];
    for mut body in bodies.into_iter().chain(std::iter::once(
        json!({"prompt":[1,2,3,4,5,6,7,8],"truncate_prompt_tokens":6}),
    )) {
        body["model"] = json!(model.path().to_str().unwrap());
        let completion = body.get("prompt").is_some();
        let expected = if completion {
            let request: CompletionRequest = serde_json::from_value(body.clone()).unwrap();
            let lowered = prepare_grpc_completion(&request).unwrap();
            frontend
                .prepare_token_ids(lowered.chat, lowered.token_ids, false)
                .await
                .unwrap()
                .tokenized
                .token_ids
        } else {
            frontend
                .prepare(serde_json::from_value(body.clone()).unwrap())
                .await
                .unwrap()
                .tokenized
                .token_ids
        };
        if body.get("truncate_prompt_tokens").is_some() {
            assert_eq!(expected.len(), 6);
        }
        let publisher = KvPublisher::bind();
        let config = RouterConfig {
            policy: PolicyConfig::KvAware {
                router_temperature: 0.0,
                kv_penalty_host: 0.8,
                kv_penalty_external: 0.6,
                kv_penalty_disk: 0.3,
                static_kv_event_sources: HashMap::from([(
                    warm.grpc_url.clone(),
                    StaticKvEventSources {
                        endpoints: vec![publisher.endpoint.clone()],
                        replay_endpoints: Some(vec![publisher.replay_endpoint.clone()]),
                        topic: TOPIC.into(),
                        block_size: 2,
                        model: "mock-model".into(),
                        hybrid: Some(publisher.layout("mock-model", 2)),
                    },
                )])
                .into(),
                peer_fetch: None,
            },
            disable_circuit_breaker: true,
            ..Default::default()
        };
        let context = common::create_test_context(config);
        let router = Router::new(vec![warm.grpc_url.clone(), cold.grpc_url.clone()], &context)
            .await
            .unwrap();
        let warm_worker = context.worker_registry.get_by_url(&warm.grpc_url).unwrap();
        warm_worker.increment_load();
        publisher
            .restore(&expected[..(expected.len() - 1) / 2 * 2], 2)
            .await;
        let before = warm.captured().len();
        let response = if completion {
            let request: CompletionRequest = serde_json::from_value(body).unwrap();
            router
                .route_typed_request(None, &request, "/v1/completions", None)
                .await
        } else {
            let request: ChatCompletionRequest = serde_json::from_value(body).unwrap();
            router
                .route_typed_request(None, &request, "/v1/chat/completions", None)
                .await
        };
        assert!(response.status().is_success());
        to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
        assert_eq!(warm.captured().len(), before + 1);
        assert!(cold.captured().is_empty());
        assert_eq!(warm.captured().last().unwrap().token_ids, expected.as_ref());
        assert_eq!(warm_worker.load(), 1);
    }
}

#[tokio::test]
async fn grpc_discovers_native_full_attention_without_layout_json() {
    use axum::body::to_bytes;
    use common::mock_vllm_rs::MockVllmRsServer;
    use kv_fixture::{KvPublisher, TOPIC};
    use vllm_router_rs::backend::{control::KvEventSourcesResponse, pb::KvEventSource};
    use vllm_router_rs::config::{PolicyConfig, RouterConfig};
    use vllm_router_rs::routers::http::router::Router;

    let publisher = KvPublisher::bind_with_seed(42);
    let legacy_publisher = KvPublisher::bind();
    let sources = |publisher: &KvPublisher| {
        vec![KvEventSource {
            transport: "zmq".into(),
            endpoint: publisher.endpoint.clone(),
            replay_endpoint: publisher.replay_endpoint.clone(),
            topic: TOPIC.into(),
            data_parallel_rank: Some(0),
            encoding: "msgpack".into(),
            schema_version: 1,
            ..Default::default()
        }]
    };
    let mut capability = serde_json::to_value(publisher.layout("mock-model", 2)).unwrap();
    capability["adapters"] = json!([]);
    let rank = capability["ranks"][0].as_object_mut().unwrap();
    rank.insert("transfer".into(), serde_json::Value::Null);
    let warm = MockVllmRsServer::spawn_with_kv_event_sources(
        2,
        KvEventSourcesResponse {
            sources: sources(&publisher),
            kv_routing_capabilities: Some(serde_json::to_vec(&capability).unwrap()),
        },
    )
    .await;
    let cold = MockVllmRsServer::spawn_with_kv_event_sources(
        2,
        KvEventSourcesResponse {
            sources: sources(&legacy_publisher),
            kv_routing_capabilities: None,
        },
    )
    .await;
    let context = common::create_test_context(RouterConfig {
        policy: PolicyConfig::KvAware {
            router_temperature: 0.0,
            kv_penalty_host: 0.8,
            kv_penalty_external: 0.6,
            kv_penalty_disk: 0.3,
            static_kv_event_sources: Default::default(),
            peer_fetch: None,
        },
        disable_circuit_breaker: true,
        ..Default::default()
    });
    let router = Router::new(vec![warm.grpc_url.clone(), cold.grpc_url.clone()], &context)
        .await
        .unwrap();
    let warm_worker = context.worker_registry.get_by_url(&warm.grpc_url).unwrap();
    warm_worker.increment_load();
    let model = fixture::model_fixture();
    let cached = [1, 2, 3, 4, 5, 6, 7];
    let uncached = [11, 12, 13, 14, 15, 16, 17];
    publisher.restore(&cached[..6], 2).await;
    for ids in [cached, uncached] {
        let request: CompletionRequest = serde_json::from_value(json!({
            "model": model.path().to_str().unwrap(),
            "prompt": ids,
            "max_tokens": 1,
        }))
        .unwrap();
        let response = router
            .route_typed_request(None, &request, "/v1/completions", None)
            .await;
        assert!(response.status().is_success());
        to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
    }
    // Only a verified native cache hit can outweigh the warm worker's load.
    assert_eq!(warm.captured().len(), 1);
    assert_eq!(warm.captured()[0].token_ids, cached);
    assert_eq!(cold.captured().len(), 1);
    assert_eq!(cold.captured()[0].token_ids, uncached);
}
