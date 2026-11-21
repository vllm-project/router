#![cfg(all(feature = "cpu", feature = "ready"))]

mod common;
#[path = "common/grpc_frontend_fixture.rs"]
mod fixture;

use serde_json::json;
use vllm_router_rs::backend::EngineFrontend;
use vllm_router_rs::protocols::spec::CompletionRequest;

#[tokio::test]
async fn grpc_prefix_selection_uses_the_ids_actually_dispatched() {
    use axum::body::to_bytes;
    use std::sync::Arc;
    use vllm_router_rs::config::{PolicyConfig, RouterConfig};
    use vllm_router_rs::kv_index::{
        CacheKey, HashMode, KvBlockIndexer, ResidencyOwner, RoutableTarget, StorageTier,
    };
    use vllm_router_rs::policies::KvAwarePolicy;
    use vllm_router_rs::protocols::spec::ChatCompletionRequest;
    use vllm_router_rs::routers::http::router::Router;
    let model = fixture::model_fixture();
    let warm = common::mock_vllm_rs::MockVllmRsServer::spawn().await;
    let cold = common::mock_vllm_rs::MockVllmRsServer::spawn().await;
    let config = RouterConfig {
        policy: PolicyConfig::KvAware {
            router_temperature: 0.0,
            kv_penalty_host: 0.8,
            kv_penalty_external: 0.6,
            kv_penalty_disk: 0.3,
            static_kv_event_sources: Default::default(),
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
    let frontend = EngineFrontend::new();
    let policy = context.policy_registry.get_default_policy();
    let policy = policy.as_any().downcast_ref::<KvAwarePolicy>().unwrap();
    let bodies = [
        json!({"messages":[{"role":"user","content":"hello prefix"}]}),
        json!({"messages":[{"role":"user","content":[{"type":"text","text":"hello prefix"}]}]}),
        json!({"messages":[{"role":"user","content":"call f"},{"role":"assistant","content":"calling","tool_calls":[{"id":"c","type":"function","function":{"name":"f","arguments":"{}"}}]},{"role":"tool","tool_call_id":"c","content":"done"}]}),
        json!({"messages":[{"role":"user","content":"a longer chat to truncate"}],"truncate_prompt_tokens":6}),
    ];
    for mut body in bodies.into_iter().chain(
        [
            json!({"prompt":[1,2,3,4,5,6,7,8],"truncate_prompt_tokens":6}),
            json!({"prompt": "hello prefix"}),
            json!({"prompt": "a longer text to truncate", "truncate_prompt_tokens": 6}),
        ]
        .into_iter(),
    ) {
        body["model"] = json!(model.path().to_str().unwrap());
        let completion = body.get("prompt").is_some();
        let expected = if completion {
            let request: CompletionRequest = serde_json::from_value(body.clone()).unwrap();
            let mut prepared = frontend.prepare_completions(&request).await.unwrap();
            assert_eq!(prepared.len(), 1);
            prepared.pop().unwrap().tokenized.token_ids
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
        let index = Arc::new(KvBlockIndexer::new());
        for (url, name) in [(&warm.grpc_url, "warm"), (&cold.grpc_url, "cold")] {
            policy.register_target_with_cache_identity(
                url,
                RoutableTarget {
                    instance_id: Arc::from(name),
                    dp_rank: 0,
                },
                CacheKey {
                    model: Arc::from("model"),
                    hash_mode: HashMode::Sha256,
                    block_size: 2,
                },
                index.clone(),
            );
        }
        let hashes = KvAwarePolicy::local_hashes(&expected[..(expected.len() - 1) / 2 * 2], 2);
        let blocks: Vec<_> = hashes
            .into_iter()
            .enumerate()
            .map(|(i, hash)| (Arc::from(format!("block-{i}")), hash))
            .collect();
        index.store(
            0,
            ResidencyOwner::Worker {
                source: Arc::from("warm"),
                dp_rank: 0,
                incarnation: 0,
            },
            StorageTier::Device,
            None,
            &blocks,
        );
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
async fn grpc_completion_text_prompt_is_encoded_with_special_tokens_by_default() {
    let model = fixture::model_fixture();
    let frontend = EngineFrontend::new();
    let mut request: CompletionRequest = serde_json::from_value(json!({
        "model": model.path().to_str().unwrap(),
        "prompt": "hello prefix",
    }))
    .unwrap();
    let prepared = frontend.prepare_completions(&request).await.unwrap();
    assert_eq!(prepared.len(), 1);
    let ids = &prepared[0].tokenized.token_ids;
    assert!(!ids.is_empty());
    // The fixture adds a BOS post-processor, so the default encode starts
    // with <s> (290).
    assert_eq!(ids[0], 290);

    request
        .other
        .insert("add_special_tokens".to_string(), json!(false));
    let prepared = frontend.prepare_completions(&request).await.unwrap();
    let ids = &prepared[0].tokenized.token_ids;
    assert!(!ids.is_empty());
    assert_ne!(ids[0], 290);
}

#[tokio::test]
async fn grpc_completion_batch_prepare_returns_one_request_per_prompt() {
    // Lowering is the fan-out unit Stage 3 will dispatch: a 2-prompt batch
    // prepares to 2 requests, in client order, each encoded separately.
    let model = fixture::model_fixture();
    let frontend = EngineFrontend::new();
    let request: CompletionRequest = serde_json::from_value(json!({
        "model": model.path().to_str().unwrap(),
        "prompt": ["first", "second"],
    }))
    .unwrap();
    let prepared = frontend.prepare_completions(&request).await.unwrap();
    assert_eq!(prepared.len(), 2);
    assert!(!prepared[0].tokenized.token_ids.is_empty());
    assert!(!prepared[1].tokenized.token_ids.is_empty());
    assert_ne!(
        prepared[0].tokenized.token_ids,
        prepared[1].tokenized.token_ids
    );

    let request: CompletionRequest = serde_json::from_value(json!({
        "model": model.path().to_str().unwrap(),
        "prompt": [[1], [2, 3]],
    }))
    .unwrap();
    let prepared = frontend.prepare_completions(&request).await.unwrap();
    assert_eq!(prepared.len(), 2);
    assert_eq!(prepared[0].tokenized.token_ids.as_ref(), &[1]);
    assert_eq!(prepared[1].tokenized.token_ids.as_ref(), &[2, 3]);
}

#[tokio::test]
async fn grpc_completion_batched_prompts_are_rejected_before_tokenize() {
    use axum::body::to_bytes;
    use axum::http::StatusCode;
    use vllm_router_rs::config::{PolicyConfig, RouterConfig};
    use vllm_router_rs::routers::http::router::Router;

    let model = fixture::model_fixture();
    let worker = common::mock_vllm_rs::MockVllmRsServer::spawn().await;
    let config = RouterConfig {
        policy: PolicyConfig::KvAware {
            router_temperature: 0.0,
            kv_penalty_host: 0.8,
            kv_penalty_external: 0.6,
            kv_penalty_disk: 0.3,
            static_kv_event_sources: Default::default(),
        },
        disable_circuit_breaker: true,
        ..Default::default()
    };
    let context = common::create_test_context(config);
    let router = Router::new(vec![worker.grpc_url.clone()], &context)
        .await
        .unwrap();

    // Batched with a model: 400, and nothing is dispatched.
    let request: CompletionRequest = serde_json::from_value(json!({
        "model": model.path().to_str().unwrap(),
        "prompt": ["first", "second"],
    }))
    .unwrap();
    let response = router
        .route_typed_request(None, &request, "/v1/completions", None)
        .await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
    assert!(
        String::from_utf8_lossy(&body).contains("batched prompts"),
        "{}",
        String::from_utf8_lossy(&body)
    );
    assert!(worker.captured().is_empty());

    // Batched without a model: still 400, not 503. The batch check runs
    // before tokenizer resolution, so a batch is never tokenized first —
    // the old order reached resolution here and answered 503.
    let request: CompletionRequest =
        serde_json::from_value(json!({"prompt": ["first", "second"]})).unwrap();
    let response = router
        .route_typed_request(None, &request, "/v1/completions", None)
        .await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
    assert!(
        String::from_utf8_lossy(&body).contains("batched prompts"),
        "{}",
        String::from_utf8_lossy(&body)
    );
    assert!(worker.captured().is_empty());
}

#[tokio::test]
async fn grpc_completion_text_prompt_bos_default_end_to_end() {
    use axum::body::to_bytes;
    use vllm_router_rs::config::{PolicyConfig, RouterConfig};
    use vllm_router_rs::routers::http::router::Router;

    let model = fixture::model_fixture();
    let worker = common::mock_vllm_rs::MockVllmRsServer::spawn().await;
    let config = RouterConfig {
        policy: PolicyConfig::KvAware {
            router_temperature: 0.0,
            kv_penalty_host: 0.8,
            kv_penalty_external: 0.6,
            kv_penalty_disk: 0.3,
            static_kv_event_sources: Default::default(),
        },
        disable_circuit_breaker: true,
        ..Default::default()
    };
    let context = common::create_test_context(config);
    let router = Router::new(vec![worker.grpc_url.clone()], &context)
        .await
        .unwrap();

    // Default (add_special_tokens omitted → true): the ids the worker
    // receives start with the fixture BOS token (290).
    let request: CompletionRequest = serde_json::from_value(json!({
        "model": model.path().to_str().unwrap(),
        "prompt": "hello prefix",
    }))
    .unwrap();
    let response = router
        .route_typed_request(None, &request, "/v1/completions", None)
        .await;
    assert!(response.status().is_success());
    to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
    assert_eq!(worker.captured().last().unwrap().token_ids[0], 290);

    // Explicit false: no BOS on the dispatched ids.
    let request: CompletionRequest = serde_json::from_value(json!({
        "model": model.path().to_str().unwrap(),
        "prompt": "hello prefix",
        "add_special_tokens": false,
    }))
    .unwrap();
    let response = router
        .route_typed_request(None, &request, "/v1/completions", None)
        .await;
    assert!(response.status().is_success());
    to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
    assert_ne!(worker.captured().last().unwrap().token_ids[0], 290);
}
