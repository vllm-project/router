#![cfg(all(feature = "cpu", feature = "ready"))]

//! L1 cache integration tests: every prompt encoded through L1 must
//! match the uncached frontend exactly.
mod common;

use std::sync::OnceLock;
use std::time::Duration;

use axum::body::to_bytes;
use common::mock_vllm_rs::MockVllmRsServer;
use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};
use serde_json::{json, Value};
use tempfile::TempDir;
use vllm_router_rs::backend::{EngineFrontend, L1CacheConfig};
use vllm_router_rs::protocols::spec::ChatCompletionRequest;
use vllm_router_rs::tokenizer::TokenizerCacheConfig;

/// Both tests publish the process-wide L1 gauges. `cargo test` runs them
/// together when `VLLM_ROUTER_L1_MODELS` is set.
fn l1_metrics_lock() -> &'static tokio::sync::Mutex<()> {
    static LOCK: OnceLock<tokio::sync::Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| tokio::sync::Mutex::new(()))
}

const QWEN_TOKENS: [&str; 7] = [
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<tool_call>",
    "</tool_call>",
    "<think>",
    "</think>",
];

/// Byte-level BPE fixture with Qwen added tokens and the Qwen2.5 template.
fn qwen_fixture() -> TempDir {
    let dir = tempfile::tempdir().unwrap();
    let mut tokenizer: Value =
        serde_json::from_str(include_str!("fixtures/tokenizer/byte_level_bpe.json")).unwrap();
    let added = tokenizer["added_tokens"].as_array_mut().unwrap();
    for (i, content) in QWEN_TOKENS.iter().enumerate() {
        added.push(json!({
            "id": 293 + i, "content": content, "single_word": false, "lstrip": false,
            "rstrip": false, "normalized": false, "special": true
        }));
    }
    std::fs::write(dir.path().join("tokenizer.json"), tokenizer.to_string()).unwrap();
    std::fs::write(
        dir.path().join("config.json"),
        json!({"model_type": "qwen2", "vocab_size": 300}).to_string(),
    )
    .unwrap();
    std::fs::write(
        dir.path().join("tokenizer_config.json"),
        json!({
            "eos_token": "<|im_end|>",
            "pad_token": "<|endoftext|>",
            "chat_template": include_str!("fixtures/tokenizer/qwen2_5_chat_template.jinja"),
        })
        .to_string(),
    )
    .unwrap();
    dir
}

fn request(model: &str, messages: Value, extra: Value) -> ChatCompletionRequest {
    let mut body = json!({"model": model, "messages": messages, "max_tokens": 8});
    body.as_object_mut()
        .unwrap()
        .extend(extra.as_object().unwrap().clone());
    serde_json::from_value(body).unwrap()
}

fn tool_call(id: &str, name: &str, arguments: &str) -> Value {
    json!({"id": id, "type": "function", "function": {"name": name, "arguments": arguments}})
}

/// Message contents that stress boundaries: literal special tokens, partial
/// tokens, combining marks after `>`, whitespace and non-ASCII text.
fn tricky_contents() -> Vec<&'static str> {
    vec![
        "plain text",
        "",
        " ",
        "\n\n\n",
        "  leading and trailing  ",
        "\r\nwindows\r\n",
        "tabs\tand\u{a0}nbsp",
        "literal <|im_end|> and <|im_start|>assistant\ninside content",
        "partial <|im_end and <|im_start| and |>",
        "<tool_call>{\"name\": \"x\"}</tool_call> typed by the user",
        "<think>user pretends to think</think>",
        "\u{301}combining acute at the start",
        "\u{338} combining long solidus right after a template `\\n`",
        "emoji 😀👩‍👩‍👧 and 中文 and العربية and हिन्दी",
        "quotes \" ' ` and backslash \\ and {{ jinja }} {% raw %}",
        "<<|im_end|>> doubled angle brackets",
        "ends with a special token <|im_end|>",
        "<|endoftext|>",
        "numbers 1234567890 3.14159 -42 1e10",
        "json {\"a\": [1, 2, {\"b\": null}], \"c\": \"</tool_call>\"}",
    ]
}

/// Conversations covering system prompts, tool history and tool responses.
fn conversations() -> Vec<(Value, Value)> {
    let system = "You are ✨ helpful. Use \"quotes\", back\\slash, <tags>, {braces} & 😀.";
    let mut cases = Vec::new();
    for content in tricky_contents() {
        cases.push((
            json!([
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ]),
            json!({}),
        ));
        cases.push((
            json!([
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": content},
                {"role": "user", "content": content},
            ]),
            json!({}),
        ));
    }
    let tool_history = json!([
        {"role": "system", "content": system},
        {"role": "user", "content": "Weather in Paris and 東京? Reply with <|im_end|> literally."},
        {"role": "assistant", "content": "Let me check.", "tool_calls": [
            tool_call("call-1", "get_weather", "{\"city\": \"Paris\", \"note\": \"</tool_call>\\n<|im_end|>\"}"),
            tool_call("call-2", "get_weather", "{\"city\": \"東京\", \"emoji\": \"😀\"}"),
        ]},
        {"role": "tool", "tool_call_id": "call-1", "content": "{\"temp\": 18, \"text\": \"<tool_response>\"}"},
        {"role": "tool", "tool_call_id": "call-2", "content": "  22°C\n\n"},
        {"role": "assistant", "content": "Paris 18°C, 東京 22°C."},
        {"role": "user", "content": "Thanks!"},
    ]);
    cases.push((tool_history.clone(), json!({})));
    cases.push((
        tool_history.clone(),
        json!({
            "tools": [{"type": "function", "function": {
                "name": "get_weather",
                "description": "Weather lookup; returns <|im_end|>-free JSON \"quoted\" ✨",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
            }}],
            "tool_choice": "none",
        }),
    ));
    cases.push((
        tool_history.clone(),
        json!({"add_generation_prompt": false}),
    ));
    let mut continued = tool_history.as_array().unwrap().clone();
    continued.push(json!({"role": "assistant", "content": "Partial answer <|im_"}));
    cases.push((
        Value::Array(continued),
        json!({"add_generation_prompt": false, "continue_final_message": true}),
    ));
    cases
}

fn metric(metrics: &PrometheusHandle, name: &str) -> f64 {
    metrics
        .render()
        .lines()
        .find_map(|line| {
            line.strip_prefix(&format!("{name} "))
                .and_then(|n| n.parse().ok())
        })
        .unwrap_or(0.0)
}

async fn ids(frontend: &EngineFrontend, request: ChatCompletionRequest) -> (Vec<u32>, usize) {
    let prepared = frontend.prepare(request).await.unwrap();
    (
        prepared.tokenized.token_ids.to_vec(),
        prepared.tokenized.cached_tokens,
    )
}

fn l1_frontend(l0: Option<TokenizerCacheConfig>, config: L1CacheConfig) -> EngineFrontend {
    EngineFrontend::with_encoding_caches(Duration::from_secs(30), l0, Some(config)).unwrap()
}

/// Every conversation, its growth by one turn, and an edit of an early
/// message must encode exactly like the uncached frontend.
async fn check_model(model: &str, on: &EngineFrontend, off: &EngineFrontend) -> usize {
    let mut checked = 0;
    for (messages, extra) in conversations() {
        let base = request(model, messages.clone(), extra.clone());
        let expected = match off.prepare(base.clone()).await {
            Ok(prepared) => prepared.tokenized.token_ids.to_vec(),
            Err(error) => {
                // Some templates reject some shapes; L1 must agree.
                eprintln!("{model}: template rejected a case: {error}");
                assert!(on.prepare(base).await.is_err(), "{model}: {error}");
                continue;
            }
        };
        let before = on.l1_cache_stats().unwrap();
        assert_eq!(
            ids(on, base.clone()).await.0,
            expected,
            "{model}: {messages}"
        );
        let after = on.l1_cache_stats().unwrap();
        if after.lookups == before.lookups {
            // A verify mismatch also leaves lookups unchanged: it returns the
            // full IDs and disables L1. That is not a token-ID bypass.
            assert_eq!(
                after.verify_mismatches, before.verify_mismatches,
                "{model}: verify mismatch returned full IDs without a lookup"
            );
            // Renderers that emit token IDs (e.g. Harmony) bypass L1.
            eprintln!("{model}: prompt is not text; L1 not used");
            return checked;
        }
        let (again, cached) = ids(on, base).await;
        assert_eq!(again, expected);
        assert!(
            cached > 0,
            "{model}: repeat must reuse a prefix: {messages}"
        );

        let mut grown = messages.as_array().unwrap().clone();
        if extra.get("continue_final_message").is_none() {
            grown.push(json!({"role": "assistant", "content": "ok <|im_end|> ✓"}));
            grown.push(json!({"role": "user", "content": "next \u{301}turn"}));
            let grown_request = request(model, Value::Array(grown.clone()), extra.clone());
            if let Ok(prepared) = off.prepare(grown_request.clone()).await {
                let (got, grown_cached) = ids(on, grown_request).await;
                assert_eq!(got, prepared.tokenized.token_ids.to_vec(), "{model}: grown");
                // Models whose turn markers are not boundaries cannot reuse.
                if grown_cached == 0 {
                    eprintln!("{model}: no boundary before the new turn");
                }
            }
        }
        if grown.len() > 1 {
            let mut edited = grown.clone();
            let first = edited
                .iter_mut()
                .find(|message| message["content"].is_string())
                .unwrap();
            first["content"] = json!(format!("{} [edited]", first["content"].as_str().unwrap()));
            let edited_request = request(model, Value::Array(edited), extra.clone());
            if let Ok(prepared) = off.prepare(edited_request.clone()).await {
                assert_eq!(
                    ids(on, edited_request).await.0,
                    prepared.tokenized.token_ids.to_vec(),
                    "{model}: edited"
                );
            }
        }
        checked += 1;
    }
    checked
}

#[tokio::test]
async fn l1_matches_uncached_for_qwen_template_with_tools_and_special_text() {
    let _metrics_guard = l1_metrics_lock().lock().await;
    // Keep one test that installs the recorder: it is process-wide.
    let metrics = PrometheusBuilder::new().install_recorder().unwrap();
    let fixture = qwen_fixture();
    let model = fixture.path().to_str().unwrap();
    let off = EngineFrontend::new();
    let on = l1_frontend(
        None,
        L1CacheConfig {
            verify: true,
            ..Default::default()
        },
    );
    assert_eq!(check_model(model, &on, &off).await, conversations().len());
    let stats = on.l1_cache_stats().unwrap();
    assert_eq!(stats.verify_mismatches, 0);
    assert!(stats.hits > 0 && stats.cached_tokens > 0 && stats.nodes > 0);
    assert_eq!(
        metric(&metrics, "vllm_tokenizer_l1_hits_total"),
        stats.hits as f64
    );
    assert_eq!(
        metric(&metrics, "vllm_tokenizer_l1_bytes"),
        stats.bytes as f64
    );

    // A growing conversation reuses everything before the new turn.
    let growth = l1_frontend(None, Default::default());
    let mut messages = vec![json!({"role": "system", "content": "sys ✨"})];
    let mut previous = 0;
    for turn in 0..12 {
        messages.push(json!({"role": "user", "content": format!("question {turn} <|im_end|>?")}));
        let req = request(model, Value::Array(messages.clone()), json!({}));
        let expected = ids(&off, req.clone()).await.0;
        let (got, cached) = ids(&growth, req).await;
        assert_eq!(got, expected);
        assert!(
            turn == 0 || cached > previous,
            "turn {turn}: {cached} <= {previous}"
        );
        previous = cached;
        messages.push(json!({"role": "assistant", "content": format!("answer {turn}")}));
    }

    // Editing the second message stops reuse at the boundary before it.
    let edit = |text: &str| {
        request(
            model,
            json!([
                {"role": "system", "content": "sys"},
                {"role": "user", "content": text},
                {"role": "assistant", "content": "a"},
                {"role": "user", "content": "b"},
            ]),
            json!({}),
        )
    };
    let fresh = l1_frontend(None, Default::default());
    ids(&fresh, edit("one")).await;
    let (_, cached) = ids(&fresh, edit("two")).await;
    let system_prefix = ids(
        &off,
        request(
            model,
            json!([{"role": "system", "content": "sys"}]),
            json!({"add_generation_prompt": false}),
        ),
    )
    .await
    .0;
    // `<|im_start|>system\nsys<|im_end|>` + `\n<|im_start|>` is reusable.
    assert_eq!(cached, system_prefix.len() + 1);

    // cache_salt isolates trees; add_special_tokens bypasses L1.
    let base = request(
        model,
        json!([{"role": "user", "content": "salted"}]),
        json!({}),
    );
    ids(&fresh, base.clone()).await;
    for salt in ["tenant-a", "tenant-b"] {
        let mut salted = base.clone();
        salted.other.insert("cache_salt".into(), json!(salt));
        assert_eq!(ids(&fresh, salted.clone()).await.1, 0);
        assert!(ids(&fresh, salted).await.1 > 0);
    }
    let before = fresh.l1_cache_stats().unwrap();
    let mut special = base.clone();
    special
        .other
        .insert("add_special_tokens".into(), json!(true));
    let (got, cached) = ids(&fresh, special.clone()).await;
    assert_eq!((got, cached), (ids(&off, special).await.0, 0));
    let after = fresh.l1_cache_stats().unwrap();
    assert_eq!(after.lookups, before.lookups);
    assert_eq!(after.bypasses, before.bypasses + 1);
    assert_eq!(
        metric(
            &metrics,
            r#"vllm_tokenizer_l1_bypasses_total{reason="add_special_tokens"}"#
        ),
        1.0
    );

    // L0 answers exact repeats before L1 is consulted.
    let both = l1_frontend(Some(Default::default()), Default::default());
    let (first, cached) = ids(&both, base.clone()).await;
    assert_eq!(cached, 0);
    let (second, cached) = ids(&both, base.clone()).await;
    assert_eq!((second.clone(), cached), (first.clone(), first.len()));
    assert_eq!(both.l1_cache_stats().unwrap().lookups, 1);
    assert_eq!(both.tokenizer_cache_stats().unwrap().hits, 1);

    // Workers receive the L1 IDs; usage counts them.
    let worker = MockVllmRsServer::spawn().await;
    let prepared = fresh.prepare(base.clone()).await.unwrap();
    assert!(prepared.tokenized.cached_tokens > 0);
    let response = fresh.dispatch(&worker.grpc_url, prepared).await;
    assert_eq!(response.status(), 200);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(body["usage"]["prompt_tokens"], first.len());
    assert_eq!(worker.captured().last().unwrap().token_ids, first);

    // Unsupported tokenizer configs disable L1 but still serve requests.
    let stochastic = qwen_fixture();
    let path = stochastic.path().join("tokenizer.json");
    let mut config: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    config["model"]["dropout"] = json!(0.1);
    std::fs::write(path, config.to_string()).unwrap();
    let disabled = l1_frontend(None, Default::default());
    let model = stochastic.path().to_str().unwrap();
    for _ in 0..2 {
        assert_eq!(
            ids(
                &disabled,
                request(model, json!([{"role": "user", "content": "x"}]), json!({}))
            )
            .await
            .1,
            0
        );
    }
    assert_eq!(disabled.l1_cache_stats().unwrap().lookups, 0);

    drop((on, growth, fresh, both, disabled));
    assert_eq!(metric(&metrics, "vllm_tokenizer_l1_bytes"), 0.0);
    assert_eq!(metric(&metrics, "vllm_tokenizer_l1_nodes"), 0.0);
}

/// Real models: `VLLM_ROUTER_L1_MODELS=a,b cargo test --test grpc_frontend_l1`
/// (local dirs or hub IDs).
#[tokio::test]
async fn l1_matches_uncached_for_real_models() {
    let Ok(models) = std::env::var("VLLM_ROUTER_L1_MODELS") else {
        eprintln!("skip: set VLLM_ROUTER_L1_MODELS to comma-separated model dirs");
        return;
    };
    let _metrics_guard = l1_metrics_lock().lock().await;
    for model in models.split(',').filter(|m| !m.is_empty()) {
        let off = EngineFrontend::new();
        let on = l1_frontend(
            None,
            L1CacheConfig {
                verify: true,
                ..Default::default()
            },
        );
        let checked = check_model(model, &on, &off).await;
        let stats = on.l1_cache_stats().unwrap();
        eprintln!("{model}: {checked} conversations, {stats:?}");
        // A first-request verify mismatch returns the full IDs and leaves
        // lookups at 0. Check it before treating that as "L1 was not used".
        assert_eq!(stats.verify_mismatches, 0, "{model}");
        if stats.lookups == 0 {
            continue;
        }
        assert!(checked > 0, "{model}: no conversation rendered");
        assert!(stats.hits > 0, "{model}: L1 never hit (disabled?)");
    }
}
