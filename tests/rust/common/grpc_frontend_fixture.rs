//! Local model fixture for gRPC tests and benchmarks.
#![allow(dead_code)]
use serde_json::json;
use tempfile::TempDir;
use vllm_router_rs::protocols::spec::ChatCompletionRequest;

pub fn model_fixture() -> TempDir {
    let dir = tempfile::tempdir().unwrap();
    let mut tokenizer: serde_json::Value =
        serde_json::from_str(include_str!("../fixtures/tokenizer/byte_level_bpe.json")).unwrap();
    // Add BOS so add_special_tokens changes the IDs.
    tokenizer["post_processor"] = json!({
        "type": "TemplateProcessing",
        "single": [{"SpecialToken": {"id": "<s>", "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}}],
        "pair": [{"Sequence": {"id": "A", "type_id": 0}}, {"Sequence": {"id": "B", "type_id": 1}}],
        "special_tokens": {"<s>": {"id": "<s>", "ids": [290], "tokens": ["<s>"]}}
    });
    std::fs::write(dir.path().join("tokenizer.json"), tokenizer.to_string()).unwrap();
    std::fs::write(
        dir.path().join("config.json"),
        json!({"model_type": "llama", "vocab_size": 293}).to_string(),
    )
    .unwrap();
    std::fs::write(dir.path().join("tokenizer_config.json"), json!({
        "bos_token": "<s>", "eos_token": "</s>",
        "chat_template": "{% for message in messages %}{{ message['role'] }}: {{ message['content'] }}\n{% endfor %}{% if add_generation_prompt %}assistant: {% endif %}{{ suffix | default('') }}"
    }).to_string()).unwrap();
    dir
}

/// Same tiny BPE, with Qwen added tokens and the Qwen2.5 chat template.
/// L1 only stores a prefix that ends on an added-token boundary, and this
/// template places `<|im_start|>` / `<|im_end|>` between turns.
pub fn qwen_model_fixture() -> TempDir {
    let dir = tempfile::tempdir().unwrap();
    let mut tokenizer: serde_json::Value =
        serde_json::from_str(include_str!("../fixtures/tokenizer/byte_level_bpe.json")).unwrap();
    let added = tokenizer["added_tokens"].as_array_mut().unwrap();
    for (i, content) in [
        "<|endoftext|>",
        "<|im_start|>",
        "<|im_end|>",
        "<tool_call>",
        "</tool_call>",
        "<think>",
        "</think>",
    ]
    .into_iter()
    .enumerate()
    {
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
            "chat_template": include_str!("../fixtures/tokenizer/qwen2_5_chat_template.jinja"),
        })
        .to_string(),
    )
    .unwrap();
    dir
}

pub fn chat_request(model: &TempDir, text: &str, stream: bool) -> ChatCompletionRequest {
    serde_json::from_value(json!({
        "model": model.path().to_str().unwrap(),
        "messages": [{"role": "user", "content": text}],
        "max_tokens": 8,
        "stream": stream,
    }))
    .unwrap()
}
