mod common;
#[path = "common/grpc_frontend_fixture.rs"]
mod fixture;

use serde_json::json;
use std::time::Duration;
use vllm_router_rs::backend::{prepare_grpc_completion, EngineFrontend};
use vllm_router_rs::protocols::spec::CompletionRequest;

#[tokio::test]
async fn completion_token_ids_bypass_chat_caches() {
    let model = fixture::model_fixture();
    let frontend =
        EngineFrontend::with_tokenizer_cache(Duration::from_secs(30), Default::default()).unwrap();
    let chat = frontend
        .prepare(fixture::chat_request(&model, "Hello 世界", false))
        .await
        .unwrap();
    let repeated = frontend
        .prepare(fixture::chat_request(&model, "Hello 世界", false))
        .await
        .unwrap();
    assert_eq!(chat.tokenized.cached_tokens, 0);
    assert_eq!(
        repeated.tokenized.cached_tokens,
        repeated.tokenized.token_ids.len()
    );
    let before = frontend.tokenizer_cache_stats().unwrap();
    let req: CompletionRequest = serde_json::from_value(json!({
        "model": model.path().to_str().unwrap(), "prompt": [4, 5, 6]
    }))
    .unwrap();
    let lowered = prepare_grpc_completion(&req).unwrap();
    let prepared = frontend
        .prepare_token_ids(lowered.chat, lowered.token_ids, lowered.emit_token_ids)
        .await
        .unwrap();
    assert_eq!(prepared.tokenized.token_ids.as_ref(), &[4, 5, 6]);
    assert_eq!(prepared.tokenized.cached_tokens, 0);
    assert_eq!(prepared.tokenized.encode_ms, 0.0);
    assert_eq!(frontend.tokenizer_cache_stats().unwrap(), before);
}

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
