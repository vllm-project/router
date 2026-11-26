#![cfg(all(feature = "cpu", feature = "ready"))]

mod common;
#[path = "common/grpc_frontend_fixture.rs"]
mod fixture;
#[path = "common/grpc_kv_fixture.rs"]
mod kv_fixture;

use serde_json::json;
use vllm_router_rs::backend::EngineFrontend;
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
async fn grpc_completion_streaming_prompt_logprobs_without_echo_is_rejected() {
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

    // Non-echo streaming with explicit `prompt_logprobs` is rejected:
    // an explicit count requires `echo` for the prompt text/offset
    // context, so the request never reaches the engine and no prompt
    // logprobs can leak into a chunk.
    let request: CompletionRequest = serde_json::from_value(json!({
        "model": model.path().to_str().unwrap(),
        "prompt": "hello prefix",
        "logprobs": 2,
        "prompt_logprobs": 2,
        "stream": true,
    }))
    .unwrap();
    let response = router
        .route_typed_request(None, &request, "/v1/completions", None)
        .await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
    assert!(
        String::from_utf8_lossy(&body).contains("prompt_logprobs requires echo"),
        "{}",
        String::from_utf8_lossy(&body)
    );
    assert!(worker.captured().is_empty());
}

#[tokio::test]
async fn grpc_completion_explicit_prompt_logprobs_validation_rejects_over_http() {
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

    // Streaming explicit `prompt_logprobs` still requires `echo` +
    // `logprobs`, and only 0 is allowed there; every rejected
    // combination answers 400 naming the failing condition and never
    // reaches the engine. Non-streaming standalone requests are
    // accepted instead (see
    // grpc_completion_standalone_prompt_logprobs_without_echo).
    for (mut body, message) in [
        (
            json!({"prompt_logprobs": 2, "echo": true, "stream": true}),
            "prompt_logprobs requires logprobs",
        ),
        (
            json!({"prompt_logprobs": 2, "logprobs": 2, "stream": true}),
            "prompt_logprobs requires echo",
        ),
        (
            json!({"prompt_logprobs": 2, "logprobs": 2, "echo": true, "stream": true}),
            "streaming prompt_logprobs must be 0",
        ),
        (
            json!({"prompt_logprobs": -1, "logprobs": 2, "echo": true, "stream": true}),
            "streaming prompt_logprobs must be 0",
        ),
    ] {
        body["model"] = json!(model.path().to_str().unwrap());
        body["prompt"] = json!("hello prefix");
        let request: CompletionRequest = serde_json::from_value(body).unwrap();
        let response = router
            .route_typed_request(None, &request, "/v1/completions", None)
            .await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
        assert!(
            String::from_utf8_lossy(&body).contains(message),
            "{}",
            String::from_utf8_lossy(&body)
        );
    }
    assert!(worker.captured().is_empty());
}

#[tokio::test]
async fn grpc_completion_explicit_prompt_logprobs_zero_streams_and_merges() {
    use axum::body::to_bytes;
    use common::mock_vllm_rs::{FrameLogprobs, MockVllmRs};
    use vllm_router_rs::backend::pb::candidate_tokens;
    use vllm_router_rs::config::{PolicyConfig, RouterConfig};
    use vllm_router_rs::routers::http::router::Router;

    let state = MockVllmRs {
        frames: Some(vec![
            ("".to_string(), vec![7, 8]),
            ("".to_string(), vec![9]),
        ]),
        frame_logprobs: Some(vec![
            FrameLogprobs {
                logprobs: vec![-0.5, -0.6],
                ranks: vec![1, 1],
                candidates: vec![],
            },
            FrameLogprobs {
                logprobs: vec![-0.7],
                ranks: vec![1],
                candidates: vec![],
            },
        ]),
        emit_prompt_logprobs: true,
        ..Default::default()
    };
    let model = fixture::model_fixture();
    let worker = common::mock_vllm_rs::MockVllmRsServer::spawn_with_state(state).await;
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

    // Streaming with explicit `prompt_logprobs: 0` is the one streaming
    // value allowed: the sampled token per prompt position, no extra
    // candidates. The prompt entries merge into the first chunk like
    // any other echo + logprobs request.
    let request: CompletionRequest = serde_json::from_value(json!({
        "model": model.path().to_str().unwrap(),
        "prompt": "hello prefix",
        "logprobs": 2,
        "echo": true,
        "prompt_logprobs": 0,
        "stream": true,
    }))
    .unwrap();
    let response = router
        .route_typed_request(None, &request, "/v1/completions", None)
        .await;
    assert!(response.status().is_success());
    let body = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
    let text = String::from_utf8(body.to_vec()).unwrap();
    assert!(text.trim_end().ends_with("data: [DONE]"), "{text}");
    let chunks: Vec<serde_json::Value> = text
        .split("\n\n")
        .filter_map(|frame| frame.strip_prefix("data: "))
        .filter(|data| *data != "[DONE]")
        .map(|data| serde_json::from_str(data).unwrap())
        .collect();
    assert!(!chunks.is_empty(), "{text}");
    let first = &chunks[0]["choices"][0];
    assert!(
        first["text"].as_str().unwrap().starts_with("hello prefix"),
        "{first}"
    );
    let first_logprobs = first["logprobs"]["token_logprobs"].as_array().unwrap();
    assert!(first_logprobs[0].is_null());
    assert_eq!(first_logprobs[first_logprobs.len() - 2], json!(-0.5));
    assert_eq!(first_logprobs[first_logprobs.len() - 1], json!(-0.6));
    // The wire request asked for prompt logprobs with no candidates:
    // 0 means the sampled token only.
    let captured = worker.captured();
    let options = captured[0].response_options.as_ref().unwrap();
    assert!(options.prompt_logprobs);
    assert!(options.prompt_candidates.is_none());
    assert!(matches!(
        options
            .output_candidates
            .as_ref()
            .and_then(|c| c.select.clone()),
        Some(candidate_tokens::Select::TopN(2))
    ));

    // Non-streaming with an explicit positive count merges the same
    // way: prompt entries first, output entries last.
    let request: CompletionRequest = serde_json::from_value(json!({
        "model": model.path().to_str().unwrap(),
        "prompt": "hello prefix",
        "logprobs": 2,
        "echo": true,
        "prompt_logprobs": 2,
    }))
    .unwrap();
    let response = router
        .route_typed_request(None, &request, "/v1/completions", None)
        .await;
    assert!(response.status().is_success());
    let body = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    let choice = &json["choices"][0];
    let token_logprobs = choice["logprobs"]["token_logprobs"].as_array().unwrap();
    assert!(token_logprobs[0].is_null());
    assert_eq!(
        &token_logprobs[token_logprobs.len() - 3..],
        &[json!(-0.5), json!(-0.6), json!(-0.7)]
    );
    assert!(token_logprobs[1..token_logprobs.len() - 3]
        .iter()
        .all(|value| *value == json!(-0.1)));
    let captured = worker.captured();
    let options = captured[1].response_options.as_ref().unwrap();
    assert!(matches!(
        options
            .prompt_candidates
            .as_ref()
            .and_then(|c| c.select.clone()),
        Some(candidate_tokens::Select::TopN(2))
    ));
}

#[tokio::test]
async fn grpc_completion_logprobs_echo_round_trip() {
    use axum::body::to_bytes;
    use common::mock_vllm_rs::{FrameLogprobs, MockVllmRs};
    use vllm_router_rs::backend::pb::candidate_token_info::TokenInfo;
    use vllm_router_rs::backend::pb::{candidate_tokens, CandidateTokenInfo};
    use vllm_router_rs::config::{PolicyConfig, RouterConfig};
    use vllm_router_rs::routers::http::router::Router;

    // Two streamed frames: ids [7, 8] then [9], each with logprobs, plus
    // one candidate token (id 10) at the first output position.
    let state = MockVllmRs {
        frames: Some(vec![
            ("".to_string(), vec![7, 8]),
            ("".to_string(), vec![9]),
        ]),
        frame_logprobs: Some(vec![
            FrameLogprobs {
                logprobs: vec![-0.5, -0.6],
                ranks: vec![1, 1],
                candidates: vec![
                    CandidateTokenInfo {
                        tokens: vec![TokenInfo {
                            id: 10,
                            logprob: -1.0,
                            rank: 2,
                        }],
                    },
                    CandidateTokenInfo { tokens: vec![] },
                ],
            },
            FrameLogprobs {
                logprobs: vec![-0.7],
                ranks: vec![1],
                candidates: vec![CandidateTokenInfo { tokens: vec![] }],
            },
        ]),
        emit_prompt_logprobs: true,
        ..Default::default()
    };
    let model = fixture::model_fixture();
    let worker = common::mock_vllm_rs::MockVllmRsServer::spawn_with_state(state).await;
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

    let request: CompletionRequest = serde_json::from_value(json!({
        "model": model.path().to_str().unwrap(),
        "prompt": "hello prefix",
        "logprobs": 2,
        "echo": true,
    }))
    .unwrap();
    let response = router
        .route_typed_request(None, &request, "/v1/completions", None)
        .await;
    assert!(response.status().is_success());
    let body = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    let choice = &json["choices"][0];
    // Echo: the prompt text leads the generated text.
    assert!(
        choice["text"].as_str().unwrap().starts_with("hello prefix"),
        "{choice}"
    );
    assert_eq!(choice["finish_reason"], "stop");
    let logprobs = &choice["logprobs"];
    let tokens = logprobs["tokens"].as_array().unwrap();
    let token_logprobs = logprobs["token_logprobs"].as_array().unwrap();
    let text_offset = logprobs["text_offset"].as_array().unwrap();
    let top_logprobs = logprobs["top_logprobs"].as_array().unwrap();
    // Prompt entries (first null, rest -0.1 from the mock) then 3 output
    // entries [-0.5, -0.6, -0.7]; the prompt length is derived, not assumed.
    assert_eq!(tokens.len(), token_logprobs.len());
    assert_eq!(tokens.len(), text_offset.len());
    assert_eq!(tokens.len(), top_logprobs.len());
    assert!(token_logprobs[0].is_null());
    assert_eq!(
        &token_logprobs[token_logprobs.len() - 3..],
        &[json!(-0.5), json!(-0.6), json!(-0.7)]
    );
    assert!(token_logprobs[1..token_logprobs.len() - 3]
        .iter()
        .all(|value| *value == json!(-0.1)));
    // Offsets start at 0 and the first output token starts where the prompt
    // text ends (12 chars for "hello prefix").
    assert_eq!(text_offset[0], 0);
    assert_eq!(text_offset[token_logprobs.len() - 3], 12);
    for window in text_offset.windows(2) {
        assert!(window[0].as_u64().unwrap() <= window[1].as_u64().unwrap());
    }
    // The sampled token joins the candidate map at the first output position.
    let top0 = top_logprobs[token_logprobs.len() - 3].as_object().unwrap();
    assert_eq!(top0.len(), 2);
    let mut values: Vec<f64> = top0.values().map(|v| v.as_f64().unwrap()).collect();
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(values, vec![-1.0, -0.5]);
    assert_eq!(json["usage"]["completion_tokens"], 3);

    // The wire request asked for exactly what the client asked for.
    let captured = worker.captured();
    assert_eq!(captured.len(), 1);
    let options = captured[0].response_options.as_ref().unwrap();
    assert!(options.output_logprobs);
    assert!(options.prompt_logprobs);
    assert!(matches!(
        options
            .output_candidates
            .as_ref()
            .and_then(|c| c.select.clone()),
        Some(candidate_tokens::Select::TopN(2))
    ));
    assert!(matches!(
        options
            .prompt_candidates
            .as_ref()
            .and_then(|c| c.select.clone()),
        Some(candidate_tokens::Select::TopN(2))
    ));
}

#[tokio::test]
async fn grpc_completion_split_emoji_before_finish_only_message_stays_aligned() {
    use axum::body::to_bytes;
    use common::mock_vllm_rs::{FrameLogprobs, MockVllmRs};
    use vllm_router_rs::backend::pb::CandidateTokenInfo;
    use vllm_router_rs::config::{PolicyConfig, RouterConfig};
    use vllm_router_rs::routers::http::router::Router;

    // The engine emits the first two bytes of an emoji, then a
    // finish-only message with no token ids and empty text. The text
    // decoder's terminal flush is discarded for that message (the text
    // falls back to the engine's empty text), so the logprob flush must
    // not run either — otherwise the tokens would gain a replacement
    // character the text does not have.
    let state = MockVllmRs {
        frames: Some(vec![
            ("".to_string(), vec![240, 159]),
            ("".to_string(), vec![]),
        ]),
        frame_logprobs: Some(vec![
            FrameLogprobs {
                logprobs: vec![-0.5, -0.6],
                ranks: vec![1, 1],
                candidates: vec![
                    CandidateTokenInfo { tokens: vec![] },
                    CandidateTokenInfo { tokens: vec![] },
                ],
            },
            FrameLogprobs {
                logprobs: vec![],
                ranks: vec![],
                candidates: vec![],
            },
        ]),
        ..Default::default()
    };
    let model = fixture::model_fixture();
    let worker = common::mock_vllm_rs::MockVllmRsServer::spawn_with_state(state).await;
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

    let request: CompletionRequest = serde_json::from_value(json!({
        "model": model.path().to_str().unwrap(),
        "prompt": "hello prefix",
        "logprobs": 2,
    }))
    .unwrap();
    let response = router
        .route_typed_request(None, &request, "/v1/completions", None)
        .await;
    assert!(response.status().is_success());
    let body = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    let choice = &json["choices"][0];
    assert_eq!(choice["finish_reason"], "stop");
    // The incomplete emoji contributes no text, and the logprob tokens
    // concatenate to exactly that text.
    assert_eq!(choice["text"].as_str().unwrap(), "");
    let logprobs = &choice["logprobs"];
    let tokens: Vec<&str> = logprobs["tokens"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t.as_str().unwrap())
        .collect();
    assert_eq!(tokens, vec!["", ""]);
    assert_eq!(tokens.concat(), choice["text"].as_str().unwrap());
    assert_eq!(logprobs["token_logprobs"], json!([-0.5, -0.6]));
    assert_eq!(json["usage"]["completion_tokens"], 2);
}

#[tokio::test]
async fn grpc_completion_prompt_top_logprobs_ignore_output_token_ids() {
    use axum::body::to_bytes;
    use common::mock_vllm_rs::MockVllmRs;
    use vllm_router_rs::backend::pb::candidate_tokens;
    use vllm_router_rs::config::{PolicyConfig, RouterConfig};
    use vllm_router_rs::routers::http::router::Router;

    // The output-only ID selection ([99], not a prompt id) must not
    // filter the sampled prompt token out of the prompt top_logprobs:
    // prompt candidates are TopN on the wire, never TokenIds.
    let state = MockVllmRs {
        emit_prompt_logprobs: true,
        ..Default::default()
    };
    let model = fixture::model_fixture();
    let worker = common::mock_vllm_rs::MockVllmRsServer::spawn_with_state(state).await;
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

    let request: CompletionRequest = serde_json::from_value(json!({
        "model": model.path().to_str().unwrap(),
        "prompt": [1, 2, 3],
        "logprobs": 2,
        "echo": true,
        "logprob_token_ids": [99],
        "return_token_ids": true,
        "return_tokens_as_token_ids": true,
    }))
    .unwrap();
    let response = router
        .route_typed_request(None, &request, "/v1/completions", None)
        .await;
    assert!(response.status().is_success());
    let body = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    let logprobs = &json["choices"][0]["logprobs"];
    let tokens = logprobs["tokens"].as_array().unwrap();
    assert_eq!(
        &tokens[..3],
        &[
            json!("token_id:1"),
            json!("token_id:2"),
            json!("token_id:3")
        ]
    );
    let top_logprobs = logprobs["top_logprobs"].as_array().unwrap();
    // The first prompt token is unscorable, so its top map stays null;
    // positions 1 and 2 carry the sampled token's logprob.
    assert!(top_logprobs[0].is_null());
    for (position, id) in [(1, 2), (2, 3)] {
        let top = top_logprobs[position]
            .as_object()
            .unwrap_or_else(|| panic!("prompt top_logprobs[{position}] must not be null"));
        assert_eq!(
            top.get(&format!("token_id:{id}")),
            Some(&json!(-0.1)),
            "sampled prompt token {id} must survive the output-only ID filter"
        );
    }

    // `token_ids` holds generated ids only under echo; the prompt ids
    // live in the separate `prompt_token_ids` field.
    assert_eq!(json["choices"][0]["token_ids"], json!([7, 8, 9]));
    assert_eq!(json["choices"][0]["prompt_token_ids"], json!([1, 2, 3]));

    // The wire request proves the asymmetry: TokenIds only for output
    // candidates, TopN for prompt candidates.
    let options = worker.captured()[0]
        .response_options
        .as_ref()
        .unwrap()
        .clone();
    assert!(matches!(
        options.output_candidates.and_then(|c| c.select),
        Some(candidate_tokens::Select::TokenIds(ids)) if ids.ids == vec![99]
    ));
    assert!(matches!(
        options.prompt_candidates.and_then(|c| c.select),
        Some(candidate_tokens::Select::TopN(2))
    ));
}

#[tokio::test]
async fn grpc_completion_echo_with_max_tokens_zero_returns_the_prompt_only() {
    use axum::body::to_bytes;
    use common::mock_vllm_rs::MockVllmRs;
    use vllm_router_rs::config::{PolicyConfig, RouterConfig};
    use vllm_router_rs::routers::http::router::Router;

    let state = MockVllmRs {
        emit_prompt_logprobs: true,
        zero_max_tokens_returns_empty: true,
        ..Default::default()
    };
    let model = fixture::model_fixture();
    let worker = common::mock_vllm_rs::MockVllmRsServer::spawn_with_state(state).await;
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

    // Echo + max_tokens: 0: the answer is the prompt, the one token the
    // engine had to generate is dropped.
    let request: CompletionRequest = serde_json::from_value(json!({
        "model": model.path().to_str().unwrap(),
        "prompt": "hello prefix",
        "logprobs": 2,
        "echo": true,
        "max_tokens": 0,
    }))
    .unwrap();
    let response = router
        .route_typed_request(None, &request, "/v1/completions", None)
        .await;
    assert!(response.status().is_success());
    let body = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    let choice = &json["choices"][0];
    assert_eq!(choice["text"], "hello prefix");
    assert_eq!(choice["finish_reason"], "stop");
    let token_logprobs = choice["logprobs"]["token_logprobs"].as_array().unwrap();
    assert!(token_logprobs[0].is_null());
    assert!(token_logprobs[1..]
        .iter()
        .all(|value| *value == json!(-0.1)));
    // The engine was asked for exactly one token.
    assert_eq!(worker.captured().last().unwrap().max_new_tokens, 1);
    // The dropped engine token is not a generated token: the client
    // usage reports zero completion tokens.
    assert_eq!(json["usage"]["completion_tokens"], 0);
    assert_eq!(
        json["usage"]["total_tokens"],
        json["usage"]["prompt_tokens"]
    );

    // Without echo the same request answers with empty text.
    let request: CompletionRequest = serde_json::from_value(json!({
        "model": model.path().to_str().unwrap(),
        "prompt": "hello prefix",
        "max_tokens": 0,
    }))
    .unwrap();
    let response = router
        .route_typed_request(None, &request, "/v1/completions", None)
        .await;
    assert!(response.status().is_success());
    let body = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["choices"][0]["text"], "");
    assert!(json["choices"][0]["logprobs"].is_null());
    // Without `echo` the engine is asked for zero tokens, and the reply
    // reports zero completion tokens, matching vLLM HTTP.
    assert_eq!(worker.captured().last().unwrap().max_new_tokens, 0);
    assert_eq!(json["usage"]["completion_tokens"], 0);
}

#[tokio::test]
async fn grpc_completion_logprobs_echo_streams_per_chunk() {
    use axum::body::to_bytes;
    use common::mock_vllm_rs::{FrameLogprobs, MockVllmRs};
    use vllm_router_rs::config::{PolicyConfig, RouterConfig};
    use vllm_router_rs::routers::http::router::Router;

    let state = MockVllmRs {
        frames: Some(vec![
            ("".to_string(), vec![7, 8]),
            ("".to_string(), vec![9]),
        ]),
        frame_logprobs: Some(vec![
            FrameLogprobs {
                logprobs: vec![-0.5, -0.6],
                ranks: vec![1, 1],
                candidates: vec![],
            },
            FrameLogprobs {
                logprobs: vec![-0.7],
                ranks: vec![1],
                candidates: vec![],
            },
        ]),
        emit_prompt_logprobs: true,
        ..Default::default()
    };
    let model = fixture::model_fixture();
    let worker = common::mock_vllm_rs::MockVllmRsServer::spawn_with_state(state).await;
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

    let request: CompletionRequest = serde_json::from_value(json!({
        "model": model.path().to_str().unwrap(),
        "prompt": "hello prefix",
        "logprobs": 2,
        "echo": true,
        "stream": true,
    }))
    .unwrap();
    let response = router
        .route_typed_request(None, &request, "/v1/completions", None)
        .await;
    assert!(response.status().is_success());
    let body = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
    let text = String::from_utf8(body.to_vec()).unwrap();
    assert!(text.trim_end().ends_with("data: [DONE]"), "{text}");
    let chunks: Vec<serde_json::Value> = text
        .split("\n\n")
        .filter_map(|frame| frame.strip_prefix("data: "))
        .filter(|data| *data != "[DONE]")
        .map(|data| serde_json::from_str(data).unwrap())
        .collect();
    assert!(!chunks.is_empty(), "{text}");
    // The first chunk leads with the echoed prompt and prompt logprobs
    // (first position null).
    let first = &chunks[0]["choices"][0];
    assert!(
        first["text"].as_str().unwrap().starts_with("hello prefix"),
        "{first}"
    );
    assert!(first["logprobs"]["token_logprobs"][0].is_null());
    // Prompt entries precede output entries in the first chunk: the
    // frame's output logprobs (-0.5, -0.6) come last.
    let first_logprobs = first["logprobs"]["token_logprobs"].as_array().unwrap();
    assert_eq!(first_logprobs[first_logprobs.len() - 2], json!(-0.5));
    assert_eq!(first_logprobs[first_logprobs.len() - 1], json!(-0.6));
    // text_offset is monotonic across chunks.
    let offsets: Vec<u64> = chunks
        .iter()
        .flat_map(|chunk| {
            chunk["choices"][0]["logprobs"]["text_offset"]
                .as_array()
                .cloned()
                .unwrap_or_default()
        })
        .map(|value| value.as_u64().unwrap())
        .collect();
    assert!(
        offsets.windows(2).all(|pair| pair[1] >= pair[0]),
        "{offsets:?}"
    );
    // A later chunk carries the first output token's logprob.
    assert!(chunks.iter().any(|chunk| {
        chunk["choices"][0]["logprobs"]["token_logprobs"]
            .as_array()
            .is_some_and(|values| values.iter().any(|v| *v == json!(-0.5)))
    }));
}

#[tokio::test]
async fn grpc_completion_echo_stream_token_ids_are_generated_only() {
    use axum::body::to_bytes;
    use common::mock_vllm_rs::MockVllmRs;
    use vllm_router_rs::config::{PolicyConfig, RouterConfig};
    use vllm_router_rs::routers::http::router::Router;

    let state = MockVllmRs {
        frames: Some(vec![
            ("".to_string(), vec![7, 8]),
            ("".to_string(), vec![9]),
        ]),
        ..Default::default()
    };
    let model = fixture::model_fixture();
    let worker = common::mock_vllm_rs::MockVllmRsServer::spawn_with_state(state).await;
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

    // Streaming echo: the first chunk carries the prompt ids in the
    // separate `prompt_token_ids` field; `token_ids` holds generated
    // ids only, on every chunk — as in vLLM HTTP.
    let request: CompletionRequest = serde_json::from_value(json!({
        "model": model.path().to_str().unwrap(),
        "prompt": [1, 2, 3],
        "echo": true,
        "stream": true,
        "return_token_ids": true,
    }))
    .unwrap();
    let response = router
        .route_typed_request(None, &request, "/v1/completions", None)
        .await;
    assert!(response.status().is_success());
    let body = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
    let text = String::from_utf8(body.to_vec()).unwrap();
    assert!(text.trim_end().ends_with("data: [DONE]"), "{text}");
    let chunks: Vec<serde_json::Value> = text
        .split("\n\n")
        .filter_map(|frame| frame.strip_prefix("data: "))
        .filter(|data| *data != "[DONE]")
        .map(|data| serde_json::from_str(data).unwrap())
        .collect();
    assert_eq!(chunks.len(), 2, "{text}");
    let first = &chunks[0]["choices"][0];
    assert_eq!(first["token_ids"], json!([7, 8]));
    assert_eq!(first["prompt_token_ids"], json!([1, 2, 3]));
    let second = &chunks[1]["choices"][0];
    assert_eq!(second["token_ids"], json!([9]));
    assert!(second["prompt_token_ids"].is_null());

    // Echo-only streaming: the first chunk still carries the prompt
    // ids separately, with empty `token_ids` (no generated ids
    // survive — the one token the engine had to generate is dropped).
    let request: CompletionRequest = serde_json::from_value(json!({
        "model": model.path().to_str().unwrap(),
        "prompt": [1, 2, 3],
        "echo": true,
        "stream": true,
        "max_tokens": 0,
        "return_token_ids": true,
    }))
    .unwrap();
    let response = router
        .route_typed_request(None, &request, "/v1/completions", None)
        .await;
    assert!(response.status().is_success());
    let body = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
    let text = String::from_utf8(body.to_vec()).unwrap();
    assert!(text.trim_end().ends_with("data: [DONE]"), "{text}");
    let chunks: Vec<serde_json::Value> = text
        .split("\n\n")
        .filter_map(|frame| frame.strip_prefix("data: "))
        .filter(|data| *data != "[DONE]")
        .map(|data| serde_json::from_str(data).unwrap())
        .collect();
    assert!(!chunks.is_empty(), "{text}");
    let first = &chunks[0]["choices"][0];
    assert_eq!(first["token_ids"], json!([]));
    assert_eq!(first["prompt_token_ids"], json!([1, 2, 3]));
}

#[tokio::test]
async fn grpc_completion_logprobs_echo_streams_monotonic_offsets_as_token_ids() {
    use axum::body::to_bytes;
    use common::mock_vllm_rs::{FrameLogprobs, MockVllmRs};
    use vllm_router_rs::config::{PolicyConfig, RouterConfig};
    use vllm_router_rs::routers::http::router::Router;

    // Token-ID display mode: prompt entries count `token_id:<id>` label
    // lengths, so the first generated offset must continue from the end
    // of the labels, not restart at the decoded prompt length.
    let state = MockVllmRs {
        frames: Some(vec![
            ("".to_string(), vec![7, 8]),
            ("".to_string(), vec![9]),
        ]),
        frame_logprobs: Some(vec![
            FrameLogprobs {
                logprobs: vec![-0.5, -0.6],
                ranks: vec![1, 1],
                candidates: vec![],
            },
            FrameLogprobs {
                logprobs: vec![-0.7],
                ranks: vec![1],
                candidates: vec![],
            },
        ]),
        emit_prompt_logprobs: true,
        ..Default::default()
    };
    let model = fixture::model_fixture();
    let worker = common::mock_vllm_rs::MockVllmRsServer::spawn_with_state(state).await;
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

    let request: CompletionRequest = serde_json::from_value(json!({
        "model": model.path().to_str().unwrap(),
        "prompt": "hello prefix",
        "logprobs": 2,
        "echo": true,
        "stream": true,
        "return_tokens_as_token_ids": true,
    }))
    .unwrap();
    let response = router
        .route_typed_request(None, &request, "/v1/completions", None)
        .await;
    assert!(response.status().is_success());
    let body = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
    let text = String::from_utf8(body.to_vec()).unwrap();
    assert!(text.trim_end().ends_with("data: [DONE]"), "{text}");
    let chunks: Vec<serde_json::Value> = text
        .split("\n\n")
        .filter_map(|frame| frame.strip_prefix("data: "))
        .filter(|data| *data != "[DONE]")
        .map(|data| serde_json::from_str(data).unwrap())
        .collect();
    assert!(!chunks.is_empty(), "{text}");
    // text_offset is monotonic across chunks, including across the
    // echo boundary where prompt entries meet generated entries.
    let offsets: Vec<u64> = chunks
        .iter()
        .flat_map(|chunk| {
            chunk["choices"][0]["logprobs"]["text_offset"]
                .as_array()
                .cloned()
                .unwrap_or_default()
        })
        .map(|value| value.as_u64().unwrap())
        .collect();
    assert!(
        offsets.windows(2).all(|pair| pair[1] >= pair[0]),
        "{offsets:?}"
    );
    // The prompt entries use label lengths, so the first generated
    // offset sits past the decoded prompt length (12).
    assert!(offsets.iter().any(|offset| *offset > 12), "{offsets:?}");
}

#[tokio::test]
async fn grpc_completion_late_prompt_info_is_dropped_not_merged() {
    use axum::body::to_bytes;
    use common::mock_vllm_rs::{FrameLogprobs, MockVllmRs};
    use vllm_router_rs::config::{PolicyConfig, RouterConfig};
    use vllm_router_rs::routers::http::router::Router;

    // PromptInfo arrives after the first output frame, as a
    // non-conforming engine might send it. The prompt logprobs are
    // dropped instead of merged into a later chunk: merging them
    // there would put prompt entries after output entries with
    // non-monotonic offsets.
    let state = MockVllmRs {
        frames: Some(vec![
            ("".to_string(), vec![7, 8]),
            ("".to_string(), vec![9]),
        ]),
        frame_logprobs: Some(vec![
            FrameLogprobs {
                logprobs: vec![-0.5, -0.6],
                ranks: vec![1, 1],
                candidates: vec![],
            },
            FrameLogprobs {
                logprobs: vec![-0.7],
                ranks: vec![1],
                candidates: vec![],
            },
        ]),
        emit_prompt_logprobs: true,
        prompt_info_after_first_frame: true,
        ..Default::default()
    };
    let model = fixture::model_fixture();
    let worker = common::mock_vllm_rs::MockVllmRsServer::spawn_with_state(state).await;
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

    let request: CompletionRequest = serde_json::from_value(json!({
        "model": model.path().to_str().unwrap(),
        "prompt": "hello prefix",
        "logprobs": 2,
        "echo": true,
        "stream": true,
    }))
    .unwrap();
    let response = router
        .route_typed_request(None, &request, "/v1/completions", None)
        .await;
    assert!(response.status().is_success());
    let body = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
    let text = String::from_utf8(body.to_vec()).unwrap();
    assert!(text.trim_end().ends_with("data: [DONE]"), "{text}");
    let chunks: Vec<serde_json::Value> = text
        .split("\n\n")
        .filter_map(|frame| frame.strip_prefix("data: "))
        .filter(|data| *data != "[DONE]")
        .map(|data| serde_json::from_str(data).unwrap())
        .collect();
    assert!(!chunks.is_empty(), "{text}");
    // The first chunk carries only the first frame's output logprobs:
    // PromptInfo had not arrived yet.
    let first = &chunks[0]["choices"][0];
    assert_eq!(
        first["logprobs"]["token_logprobs"],
        json!([-0.5, -0.6]),
        "{first}"
    );
    // No chunk carries a prompt entry (first prompt position null):
    // the late prompt logprobs were dropped, not merged into a later
    // chunk. The second chunk carries only the second frame's output
    // logprob.
    assert!(
        !chunks.iter().any(|chunk| {
            chunk["choices"][0]["logprobs"]["token_logprobs"]
                .as_array()
                .is_some_and(|values| values.iter().any(|value| value.is_null()))
        }),
        "{text}"
    );
    let second = chunks
        .iter()
        .find(|chunk| {
            chunk["choices"][0]["logprobs"]["token_logprobs"]
                .as_array()
                .is_some_and(|values| values.iter().any(|value| *value == json!(-0.7)))
        })
        .expect("second frame's output logprob present");
    assert_eq!(
        second["choices"][0]["logprobs"]["token_logprobs"],
        json!([-0.7]),
        "{second}"
    );
    // text_offset stays monotonic across chunks.
    let offsets: Vec<u64> = chunks
        .iter()
        .flat_map(|chunk| {
            chunk["choices"][0]["logprobs"]["text_offset"]
                .as_array()
                .cloned()
                .unwrap_or_default()
        })
        .map(|value| value.as_u64().unwrap())
        .collect();
    assert!(
        offsets.windows(2).all(|pair| pair[1] >= pair[0]),
        "{offsets:?}"
    );
}

#[tokio::test]
async fn grpc_completion_echo_with_truncated_text_prompt_stays_aligned() {
    use axum::body::to_bytes;
    use common::mock_vllm_rs::{FrameLogprobs, MockVllmRs};
    use vllm_router_rs::config::{PolicyConfig, RouterConfig};
    use vllm_router_rs::routers::http::router::Router;

    let state = MockVllmRs {
        frame_logprobs: Some(vec![FrameLogprobs {
            logprobs: vec![-0.5, -0.6, -0.7],
            ranks: vec![1, 1, 1],
            candidates: vec![],
        }]),
        emit_prompt_logprobs: true,
        ..Default::default()
    };
    let model = fixture::model_fixture();
    let worker = common::mock_vllm_rs::MockVllmRsServer::spawn_with_state(state).await;
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

    let request: CompletionRequest = serde_json::from_value(json!({
        "model": model.path().to_str().unwrap(),
        "prompt": "a longer text to truncate",
        "truncate_prompt_tokens": 6,
        "logprobs": 2,
        "echo": true,
    }))
    .unwrap();
    // The frontend's echo text is decoded from the truncated ids, not the
    // client's full string.
    let frontend = EngineFrontend::new();
    let prepared = frontend.prepare_completions(&request).await.unwrap();
    assert_eq!(prepared.len(), 1);
    assert_eq!(prepared[0].tokenized.token_ids.len(), 6);
    let echo_text = prepared[0]
        .completion
        .as_ref()
        .and_then(|options| options.echo_text.clone())
        .unwrap();
    assert_ne!(echo_text, "a longer text to truncate");

    let response = router
        .route_typed_request(None, &request, "/v1/completions", None)
        .await;
    assert!(response.status().is_success());
    let body = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    let choice = &json["choices"][0];
    // The reply echoes the truncated text, then the generated text (the
    // router decodes the mock's [7, 8, 9] itself; the mock's reply_text is
    // not used on this path).
    let text = choice["text"].as_str().unwrap();
    assert!(text.starts_with(&echo_text), "{text} vs {echo_text}");
    assert!(text.len() > echo_text.len());
    // Prompt logprobs cover exactly the 6 dispatched ids, then the 3
    // output entries; the first output token starts where the echoed text
    // ends.
    let logprobs = &choice["logprobs"];
    let tokens = logprobs["tokens"].as_array().unwrap();
    let token_logprobs = logprobs["token_logprobs"].as_array().unwrap();
    let text_offset = logprobs["text_offset"].as_array().unwrap();
    assert_eq!(tokens.len(), 9);
    assert_eq!(token_logprobs.len(), 9);
    assert_eq!(text_offset.len(), 9);
    assert!(token_logprobs[0].is_null());
    assert!(token_logprobs[1..6]
        .iter()
        .all(|value| *value == json!(-0.1)));
    assert_eq!(
        &token_logprobs[6..],
        &[json!(-0.5), json!(-0.6), json!(-0.7)]
    );
    assert_eq!(text_offset[6], echo_text.chars().count() as u64);
    // The worker received exactly the truncated ids.
    assert_eq!(worker.captured().last().unwrap().token_ids.len(), 6);
}

#[tokio::test]
async fn grpc_completion_standalone_prompt_logprobs_without_echo() {
    use axum::body::to_bytes;
    use common::mock_vllm_rs::{FrameLogprobs, MockVllmRs};
    use vllm_router_rs::config::{PolicyConfig, RouterConfig};
    use vllm_router_rs::routers::http::router::Router;

    let state = MockVllmRs {
        frame_logprobs: Some(vec![FrameLogprobs {
            logprobs: vec![-0.5, -0.6, -0.7],
            ranks: vec![1, 1, 1],
            candidates: vec![],
        }]),
        emit_prompt_logprobs: true,
        ..Default::default()
    };
    let model = fixture::model_fixture();
    let worker = common::mock_vllm_rs::MockVllmRsServer::spawn_with_state(state).await;
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

    // Standalone explicit `prompt_logprobs`: no `echo`, no `logprobs`.
    // vLLM takes the explicit value first, so this is a valid request.
    let request: CompletionRequest = serde_json::from_value(json!({
        "model": model.path().to_str().unwrap(),
        "prompt": "a longer text to truncate",
        "truncate_prompt_tokens": 6,
        "prompt_logprobs": 1,
    }))
    .unwrap();
    let frontend = EngineFrontend::new();
    let prepared = frontend.prepare_completions(&request).await.unwrap();
    assert_eq!(prepared.len(), 1);
    assert_eq!(prepared[0].tokenized.token_ids.len(), 6);
    // The prompt text is decoded for the merged offsets even without
    // `echo`.
    let echo_text = prepared[0]
        .completion
        .as_ref()
        .and_then(|options| options.echo_text.clone())
        .unwrap();

    let response = router
        .route_typed_request(None, &request, "/v1/completions", None)
        .await;
    assert!(response.status().is_success());
    let body = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    let choice = &json["choices"][0];
    // No echo: the reply text is the generated text only.
    let text = choice["text"].as_str().unwrap();
    assert!(!text.is_empty());
    assert!(!text.starts_with(&echo_text), "{text} vs {echo_text}");
    // No echo: the reply text is the generated text only, and the
    // prompt entries live in the separate `prompt_logprobs` field in
    // vLLM's array shape — `logprobs` stays null because output
    // logprobs were not requested.
    let logprobs = &choice["logprobs"];
    assert!(logprobs.is_null());
    let prompt_logprobs = &choice["prompt_logprobs"];
    let entries = prompt_logprobs.as_array().unwrap();
    assert_eq!(entries.len(), 6);
    // First position is null: nothing predicts the first prompt token.
    assert!(entries[0].is_null());
    // Every other position maps its token id to the sampled logprob
    // (the mock sends no candidates, rank 1 throughout).
    let prompt_ids = &prepared[0].tokenized.token_ids;
    let mut decoded_tail = String::new();
    for (index, entry) in entries.iter().enumerate().skip(1) {
        let map = entry.as_object().unwrap();
        assert_eq!(map.len(), 1);
        let id = prompt_ids[index].to_string();
        let candidate = &map[&id];
        assert_eq!(candidate["logprob"], json!(-0.1));
        assert_eq!(candidate["rank"], json!(1));
        let decoded = candidate["decoded_token"].as_str().unwrap();
        assert!(!decoded.is_empty());
        decoded_tail.push_str(decoded);
    }
    // The decoded tokens are the prompt text minus its first token.
    assert!(
        echo_text.ends_with(&decoded_tail),
        "{echo_text} vs {decoded_tail}"
    );
    assert!(!decoded_tail.is_empty());
    // The worker received exactly the truncated ids.
    assert_eq!(worker.captured().last().unwrap().token_ids.len(), 6);
}

#[tokio::test]
async fn grpc_completion_prompt_logprobs_plus_logprobs_without_echo() {
    use axum::body::to_bytes;
    use common::mock_vllm_rs::{FrameLogprobs, MockVllmRs};
    use vllm_router_rs::config::{PolicyConfig, RouterConfig};
    use vllm_router_rs::routers::http::router::Router;

    let state = MockVllmRs {
        frame_logprobs: Some(vec![FrameLogprobs {
            logprobs: vec![-0.5, -0.6, -0.7],
            ranks: vec![1, 1, 1],
            candidates: vec![],
        }]),
        emit_prompt_logprobs: true,
        ..Default::default()
    };
    let model = fixture::model_fixture();
    let worker = common::mock_vllm_rs::MockVllmRsServer::spawn_with_state(state).await;
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

    // Prompt logprobs plus output logprobs, no `echo`: the two live in
    // separate fields, and the output offsets start at 0 — relative to
    // the generated text, not the prompt.
    let request: CompletionRequest = serde_json::from_value(json!({
        "model": model.path().to_str().unwrap(),
        "prompt": "a longer text to truncate",
        "truncate_prompt_tokens": 6,
        "prompt_logprobs": 1,
        "logprobs": 2,
    }))
    .unwrap();
    let frontend = EngineFrontend::new();
    let prepared = frontend.prepare_completions(&request).await.unwrap();
    assert_eq!(prepared.len(), 1);
    assert_eq!(prepared[0].tokenized.token_ids.len(), 6);
    let echo_text = prepared[0]
        .completion
        .as_ref()
        .and_then(|options| options.echo_text.clone())
        .unwrap();

    let response = router
        .route_typed_request(None, &request, "/v1/completions", None)
        .await;
    assert!(response.status().is_success());
    let body = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    let choice = &json["choices"][0];
    // No echo: the reply text is the generated text only. The router
    // decodes the generated ids itself (completions do not prefer the
    // worker text), so the mock's ids [7, 8, 9] decode to control
    // characters, not the mock's frame text.
    let text = choice["text"].as_str().unwrap();
    assert!(!text.is_empty());
    assert!(!text.starts_with(&echo_text), "{text} vs {echo_text}");
    // `logprobs` holds output entries only, with offsets relative to
    // the generated text.
    let logprobs = &choice["logprobs"];
    assert!(!logprobs.is_null());
    let out_token_logprobs = logprobs["token_logprobs"].as_array().unwrap();
    assert_eq!(out_token_logprobs, &[json!(-0.5), json!(-0.6), json!(-0.7)]);
    let out_tokens = logprobs["tokens"].as_array().unwrap();
    let out_offset = logprobs["text_offset"].as_array().unwrap();
    assert_eq!(out_tokens.len(), 3);
    assert_eq!(out_offset.len(), 3);
    let mut expected_offset = 0u64;
    for (index, token) in out_tokens.iter().enumerate() {
        assert_eq!(out_offset[index], json!(expected_offset));
        expected_offset += token.as_str().unwrap().chars().count() as u64;
    }
    assert_eq!(expected_offset, text.chars().count() as u64);
    assert_eq!(
        out_tokens
            .iter()
            .map(|token| token.as_str().unwrap())
            .collect::<String>(),
        text
    );
    // The prompt entries live in the separate field, in vLLM's array
    // shape: first position null, then one token-id map per position.
    let prompt_logprobs = &choice["prompt_logprobs"];
    let entries = prompt_logprobs.as_array().unwrap();
    assert_eq!(entries.len(), 6);
    assert!(entries[0].is_null());
    let prompt_ids = &prepared[0].tokenized.token_ids;
    let mut decoded_tail = String::new();
    for (index, entry) in entries.iter().enumerate().skip(1) {
        let map = entry.as_object().unwrap();
        assert_eq!(map.len(), 1);
        let id = prompt_ids[index].to_string();
        let candidate = &map[&id];
        assert_eq!(candidate["logprob"], json!(-0.1));
        assert_eq!(candidate["rank"], json!(1));
        let decoded = candidate["decoded_token"].as_str().unwrap();
        assert!(!decoded.is_empty());
        decoded_tail.push_str(decoded);
    }
    assert!(
        echo_text.ends_with(&decoded_tail),
        "{echo_text} vs {decoded_tail}"
    );
    assert!(!decoded_tail.is_empty());
    // The worker received exactly the truncated ids.
    assert_eq!(worker.captured().last().unwrap().token_ids.len(), 6);
}

#[tokio::test]
async fn grpc_completion_echo_standalone_prompt_logprobs_without_logprobs() {
    use axum::body::to_bytes;
    use common::mock_vllm_rs::{FrameLogprobs, MockVllmRs};
    use vllm_router_rs::config::{PolicyConfig, RouterConfig};
    use vllm_router_rs::routers::http::router::Router;

    let state = MockVllmRs {
        frame_logprobs: Some(vec![FrameLogprobs {
            logprobs: vec![-0.5, -0.6, -0.7],
            ranks: vec![1, 1, 1],
            candidates: vec![],
        }]),
        emit_prompt_logprobs: true,
        ..Default::default()
    };
    let model = fixture::model_fixture();
    let worker = common::mock_vllm_rs::MockVllmRsServer::spawn_with_state(state).await;
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

    // Echo + standalone `prompt_logprobs`, no `logprobs`: vLLM builds
    // `logprobs` only when `request.logprobs` is not None, so the prompt
    // entries must not leak into `choices[].logprobs` — they live only
    // in the separate `prompt_logprobs` field.
    let request: CompletionRequest = serde_json::from_value(json!({
        "model": model.path().to_str().unwrap(),
        "prompt": [1, 2, 3],
        "echo": true,
        "prompt_logprobs": 1,
        "return_token_ids": true,
        "return_tokens_as_token_ids": true,
    }))
    .unwrap();
    let frontend = EngineFrontend::new();
    let prepared = frontend.prepare_completions(&request).await.unwrap();
    assert_eq!(prepared.len(), 1);
    let echo_text = prepared[0]
        .completion
        .as_ref()
        .and_then(|options| options.echo_text.clone())
        .unwrap();

    let response = router
        .route_typed_request(None, &request, "/v1/completions", None)
        .await;
    assert!(response.status().is_success());
    let body = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    let choice = &json["choices"][0];
    assert!(choice["text"].as_str().unwrap().starts_with(&echo_text));
    assert!(choice["logprobs"].is_null());
    let entries = choice["prompt_logprobs"].as_array().unwrap();
    assert_eq!(entries.len(), 3);
    assert!(entries[0].is_null());
    for (index, id) in [(1, 2), (2, 3)] {
        let map = entries[index].as_object().unwrap();
        assert_eq!(map.len(), 1);
        let candidate = &map[&id.to_string()];
        assert_eq!(candidate["logprob"], json!(-0.1));
        assert_eq!(candidate["rank"], json!(1));
    }
    assert_eq!(choice["prompt_token_ids"], json!([1, 2, 3]));
    assert_eq!(choice["token_ids"], json!([7, 8, 9]));

    // Same, echo-only (`max_tokens: 0`): the prompt is the whole
    // answer, `logprobs` still null, prompt entries still separate.
    let request: CompletionRequest = serde_json::from_value(json!({
        "model": model.path().to_str().unwrap(),
        "prompt": [1, 2, 3],
        "echo": true,
        "prompt_logprobs": 1,
        "max_tokens": 0,
        "return_token_ids": true,
        "return_tokens_as_token_ids": true,
    }))
    .unwrap();
    let response = router
        .route_typed_request(None, &request, "/v1/completions", None)
        .await;
    assert!(response.status().is_success());
    let body = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    let choice = &json["choices"][0];
    assert_eq!(choice["text"], echo_text);
    assert!(choice["logprobs"].is_null());
    let entries = choice["prompt_logprobs"].as_array().unwrap();
    assert_eq!(entries.len(), 3);
    assert!(entries[0].is_null());
    for (index, id) in [(1, 2), (2, 3)] {
        let map = entries[index].as_object().unwrap();
        assert_eq!(map.len(), 1);
        let candidate = &map[&id.to_string()];
        assert_eq!(candidate["logprob"], json!(-0.1));
        assert_eq!(candidate["rank"], json!(1));
    }
    assert_eq!(choice["prompt_token_ids"], json!([1, 2, 3]));
    // Echo-only carries no generated ids (the one token the engine had
    // to generate is dropped), so `token_ids` is empty.
    assert_eq!(choice["token_ids"], json!([]));
    assert_eq!(json["usage"]["completion_tokens"], 0);
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
