//! Model-specific roles must be rendered before gRPC dispatch.
mod common;
#[path = "common/grpc_frontend_fixture.rs"]
mod fixture;

use axum::body::to_bytes;
use common::mock_vllm_rs::MockVllmRsServer;
use serde_json::json;
use std::sync::Arc;
use std::time::Duration;
use tower::ServiceExt;
use vllm_router_rs::backend::EngineFrontend;
use vllm_router_rs::config::{PolicyConfig, RouterConfig, RoutingMode};
use vllm_router_rs::protocols::spec::ChatCompletionRequest;
use vllm_router_rs::routers::http::router::Router;

fn expected_ids(model: &tempfile::TempDir, prompt: &str) -> Vec<u32> {
    tokenizers::Tokenizer::from_file(model.path().join("tokenizer.json"))
        .unwrap()
        .encode(prompt, false)
        .unwrap()
        .get_ids()
        .to_vec()
}

#[tokio::test]
async fn active_developer_tools_return_400_without_worker_dispatch() {
    let model = fixture::model_fixture();
    let worker = MockVllmRsServer::spawn().await;
    let config = RouterConfig {
        mode: RoutingMode::Regular {
            worker_urls: vec![worker.grpc_url.clone()],
        },
        policy: PolicyConfig::RoundRobin,
        disable_circuit_breaker: true,
        ..Default::default()
    };
    let router = Router::new(
        vec![worker.grpc_url.clone()],
        &common::create_test_context(config.clone()),
    )
    .await
    .unwrap();
    let app = common::test_app::create_test_app(Arc::new(router), reqwest::Client::new(), &config);
    for choice in [
        json!(null),
        json!("auto"),
        json!("required"),
        json!({"type":"function", "function":{"name":"weather"}}),
    ] {
        let body = json!({
            "model": model.path().to_str().unwrap(),
            "messages": [{"role":"developer", "content":"policy", "tools":[{
                "type":"function", "function":{"name":"weather", "parameters":{"type":"object"}}
            }]}],
            "tool_choice": choice,
            "max_tokens":1
        });
        let response = app
            .clone()
            .oneshot(
                axum::http::Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(axum::body::Body::from(serde_json::to_vec(&body).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), 400);
        let payload = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        assert!(String::from_utf8_lossy(&payload).contains("active tool calling"));
    }
    assert!(worker.captured().is_empty());
}

#[tokio::test]
async fn developer_tools_with_none_keep_template_ids_and_cache_keys() {
    let model = fixture::model_fixture();
    let path = model.path().join("tokenizer_config.json");
    let mut config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    config["chat_template"] = json!("{% for m in messages %}{{ m.role }}|{{ m.content }}|{{ m.tools[0].function.name }}|{% if m.tools[0].function.strict %}strict{% else %}loose{% endif %}|{% if m.tools[0].function.defer_loading %}lazy{% else %}eager{% endif %}|{{ m.tools[0].function.parameters.properties.city.type }};{% endfor %}");
    std::fs::write(&path, config.to_string()).unwrap();
    let worker = MockVllmRsServer::spawn().await;
    let frontend =
        EngineFrontend::with_tokenizer_cache(Duration::from_secs(30), Default::default()).unwrap();
    for name in ["blue", "blue", "red"] {
        let request: ChatCompletionRequest = serde_json::from_value(json!({
            "model":model.path().to_str().unwrap(), "tool_choice":"none", "max_tokens":1,
            "messages":[{"role":"developer", "content":"policy", "tools":[{
                "type":"function", "defer_loading":true,
                "function":{"name":name, "strict":true, "defer_loading":false,
                    "parameters":{"type":"object", "properties":{"city":{"type":"string"}}}}
            }]}]
        }))
        .unwrap();
        let prepared = frontend.prepare(request).await.unwrap();
        let expected = expected_ids(
            &model,
            &format!("developer|policy|{name}|strict|eager|string;"),
        );
        assert_eq!(prepared.tokenized.token_ids.as_ref(), expected.as_slice());
        let response = frontend.dispatch(&worker.grpc_url, prepared).await;
        assert_eq!(response.status(), 200);
        to_bytes(response.into_body(), usize::MAX).await.unwrap();
        assert_eq!(worker.captured().last().unwrap().token_ids, expected);
    }
    let stats = frontend.tokenizer_cache_stats().unwrap();
    assert_eq!((stats.misses, stats.hits), (2, 1));
}

#[tokio::test]
async fn developer_and_request_tools_share_name_validation() {
    let model = fixture::model_fixture();
    let frontend = EngineFrontend::new();
    for (choice, initial, expected_error) in [
        (
            json!("none"),
            json!([{"type":"function", "function":{"name":"weather", "parameters":{}}}]),
            "duplicate",
        ),
        (
            json!({"type":"function", "function":{"name":"absent"}}),
            json!([]),
            "absent",
        ),
    ] {
        let request: ChatCompletionRequest = serde_json::from_value(json!({
            "model":model.path().to_str().unwrap(), "tool_choice":choice, "tools":initial,
            "messages":[{"role":"developer", "content":"policy", "tools":[{
                "type":"function", "function":{"name":"weather", "parameters":{}}
            }]}]
        }))
        .unwrap();
        let error = frontend
            .prepare(request)
            .await
            .err()
            .expect("invalid tool context must fail");
        assert!(error.to_string().to_lowercase().contains(expected_error));
    }
}

#[tokio::test]
async fn grpc_custom_role_reaches_worker_with_template_token_ids() {
    let model = fixture::model_fixture();
    let worker = MockVllmRsServer::spawn().await;
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": model.path().to_str().unwrap(),
        "messages": [
            {"role": "user", "content": "Summarize the context."},
            {"role": "latest_reminder", "content": "Keep the answer concise."}
        ],
        "max_tokens": 1
    }))
    .unwrap();
    let frontend = EngineFrontend::new();
    let prepared = frontend.prepare(request).await.unwrap();
    let tokenizer = tokenizers::Tokenizer::from_file(model.path().join("tokenizer.json")).unwrap();
    let expected = tokenizer
        .encode(
            "user: Summarize the context.\nlatest_reminder: Keep the answer concise.\nassistant: ",
            false,
        )
        .unwrap()
        .get_ids()
        .to_vec();
    assert!(!expected.is_empty());
    assert_eq!(prepared.tokenized.token_ids.as_ref(), expected.as_slice());
    let config = RouterConfig {
        mode: RoutingMode::Regular {
            worker_urls: vec![worker.grpc_url.clone()],
        },
        policy: PolicyConfig::RoundRobin,
        disable_circuit_breaker: true,
        ..Default::default()
    };
    let router = Router::new(
        vec![worker.grpc_url.clone()],
        &common::create_test_context(config.clone()),
    )
    .await
    .unwrap();
    let app = common::test_app::create_test_app(Arc::new(router), reqwest::Client::new(), &config);
    let body = json!({
        "model": model.path().to_str().unwrap(),
        "messages": [
            {"role": "user", "content": "Summarize the context."},
            {"role": "latest_reminder", "content": "Keep the answer concise."}
        ],
        "max_tokens": 1
    });
    let response = app
        .oneshot(
            axum::http::Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(axum::body::Body::from(serde_json::to_vec(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let captured = worker.captured();
    assert_eq!(captured.len(), 1);
    assert_eq!(captured[0].token_ids, expected);
}

#[tokio::test]
async fn custom_extension_fields_affect_template_ids_and_cache_keys() {
    let model = fixture::model_fixture();
    let mut config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(model.path().join("tokenizer_config.json")).unwrap())
            .unwrap();
    config["chat_template"] = json!("{% for m in messages %}{{ m.role }}|{{ m.content }}|{{ m.trace_key }}|{{ m.metadata.step }}|{{ m.labels[1] }};{% endfor %}");
    std::fs::write(
        model.path().join("tokenizer_config.json"),
        config.to_string(),
    )
    .unwrap();
    let worker = MockVllmRsServer::spawn().await;
    let frontend =
        EngineFrontend::with_tokenizer_cache(Duration::from_secs(30), Default::default()).unwrap();
    for key in ["blue", "blue", "red"] {
        let request: ChatCompletionRequest = serde_json::from_value(json!({
            "model": model.path().to_str().unwrap(),
            "messages": [{
                "role": "checkpoint", "content": "payload", "trace_key": key,
                "metadata": {"step": 3}, "labels": ["first", "second"]
            }],
            "max_tokens": 1
        }))
        .unwrap();
        let prepared = frontend.prepare(request).await.unwrap();
        let expected = expected_ids(&model, &format!("checkpoint|payload|{key}|3|second;"));
        assert_eq!(prepared.tokenized.token_ids.as_ref(), expected.as_slice());
        let response = frontend.dispatch(&worker.grpc_url, prepared).await;
        assert_eq!(response.status(), 200);
        to_bytes(response.into_body(), usize::MAX).await.unwrap();
        assert_eq!(worker.captured().last().unwrap().token_ids, expected);
    }
    let stats = frontend.tokenizer_cache_stats().unwrap();
    assert_eq!((stats.misses, stats.hits), (2, 1));
}

#[tokio::test]
async fn custom_text_parts_and_developer_messages_keep_role_and_unicode() {
    let model = fixture::model_fixture();
    let frontend = EngineFrontend::new();
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": model.path().to_str().unwrap(),
        "messages": [
            {"role": "developer", "content": "policy"},
            {"role": "checkpoint", "content": [{"type":"text", "text":"你"}, {"type":"text", "text":"好🌍"}]}
        ]
    }))
    .unwrap();
    let prepared = frontend.prepare(request).await.unwrap();
    let expected = expected_ids(&model, "developer: policy\ncheckpoint: 你好🌍\nassistant: ");
    assert_eq!(prepared.tokenized.token_ids.as_ref(), expected.as_slice());
}

#[tokio::test]
async fn custom_invalid_or_multimodal_content_is_rejected_before_grpc_dispatch() {
    let model = fixture::model_fixture();
    let frontend = EngineFrontend::new();
    for (content, expected_error) in [
        (json!(null), "invalid content"),
        (json!(42), "invalid content"),
        (
            json!({"text":"not a content block list"}),
            "invalid content",
        ),
        (
            json!([{"type":"image_url", "image_url":{"url":"https://example.invalid/image.png"}}]),
            "multimodal chat is not supported",
        ),
    ] {
        let request: ChatCompletionRequest = serde_json::from_value(json!({
            "model":model.path().to_str().unwrap(),
            "messages":[{"role":"checkpoint", "content":content}]
        }))
        .unwrap();
        let error = frontend
            .prepare(request)
            .await
            .err()
            .expect("invalid custom content must be rejected");
        assert!(error.to_string().contains(expected_error));
    }
}

#[tokio::test]
async fn deepseek_reminder_ids_reach_grpc_worker_for_both_dialects() {
    for model_type in ["deepseek_v4", "deepseek_v41"] {
        let model = fixture::model_fixture();
        std::fs::write(
            model.path().join("config.json"),
            json!({"model_type": model_type, "vocab_size": 293}).to_string(),
        )
        .unwrap();
        let worker = MockVllmRsServer::spawn().await;
        let request: ChatCompletionRequest = serde_json::from_value(json!({
            "model": model.path().to_str().unwrap(),
            "messages": [
                {"role": "user", "content": "Summarize the context."},
                {"role": "latest_reminder", "content": "Keep the answer concise."}
            ],
            "chat_template_kwargs": {"thinking": false},
            "max_tokens": 1
        }))
        .unwrap();
        let frontend = EngineFrontend::new();
        let prepared = frontend.prepare(request).await.unwrap();
        // Complete native prompt, including the assistant transition before
        // the reminder; encoding an arbitrary nonempty string is insufficient.
        let expected = expected_ids(&model, "<｜begin▁of▁sentence｜><｜User｜>Summarize the context.<｜Assistant｜></think><｜latest_reminder｜>Keep the answer concise.");
        assert_eq!(prepared.tokenized.token_ids.as_ref(), expected.as_slice());
        let response = frontend.dispatch(&worker.grpc_url, prepared).await;
        assert_eq!(response.status(), 200);
        to_bytes(response.into_body(), usize::MAX).await.unwrap();
        assert_eq!(worker.captured().len(), 1);
        assert_eq!(worker.captured()[0].token_ids, expected);
    }
}

#[tokio::test]
async fn native_renderer_rejects_unsupported_custom_role_before_dispatch() {
    let model = fixture::model_fixture();
    std::fs::write(
        model.path().join("config.json"),
        json!({"model_type": "deepseek_v4", "vocab_size": 293}).to_string(),
    )
    .unwrap();
    let frontend = EngineFrontend::new();
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": model.path().to_str().unwrap(),
        "messages": [{"role": "checkpoint", "content": "payload"}]
    }))
    .unwrap();
    let error = frontend
        .prepare(request)
        .await
        .err()
        .expect("native renderer must reject unsupported custom roles");
    assert!(error
        .to_string()
        .contains("chat role `checkpoint` is not supported"));
}

#[tokio::test]
async fn deepseek_system_reminder_ids_match_python_before_worker_dispatch() {
    let model = fixture::model_fixture();
    std::fs::write(
        model.path().join("config.json"),
        json!({"model_type": "deepseek_v4", "vocab_size": 293}).to_string(),
    )
    .unwrap();
    let worker = MockVllmRsServer::spawn().await;
    let frontend = EngineFrontend::new();
    for trailing_assistant in [false, true] {
        let mut messages = vec![
            json!({"role": "system", "content": "rules"}),
            json!({"role": "latest_reminder", "content": "Be brief"}),
        ];
        let prompt = if trailing_assistant {
            messages.push(json!({"role": "assistant", "content": "Hi there"}));
            "<｜begin▁of▁sentence｜>rules<｜latest_reminder｜>Be briefHi there<｜end▁of▁sentence｜>"
        } else {
            "<｜begin▁of▁sentence｜>rules<｜latest_reminder｜>Be brief"
        };
        let request: ChatCompletionRequest = serde_json::from_value(json!({
            "model": model.path().to_str().unwrap(),
            "messages": messages,
            "chat_template_kwargs": {"thinking": false},
            "max_tokens": 1
        }))
        .unwrap();
        let prepared = frontend.prepare(request).await.unwrap();
        let expected = expected_ids(&model, prompt);
        assert_eq!(prepared.tokenized.token_ids.as_ref(), expected.as_slice());
        let response = frontend.dispatch(&worker.grpc_url, prepared).await;
        assert_eq!(response.status(), 200);
        to_bytes(response.into_body(), usize::MAX).await.unwrap();
        assert_eq!(worker.captured().last().unwrap().token_ids, expected);
    }
    assert_eq!(worker.captured().len(), 2);
}
