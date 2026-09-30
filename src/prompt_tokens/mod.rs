//! Exact prompt extraction for a local Qwen3 Dense text tokenizer profile.
//!
//! These tokens are routing hints only. Callers must forward the original request
//! and use a non-affinity fallback on every error, never an approximate prompt.
//!
//! The restricted renderer is derived from the Qwen team's Apache-2.0 template
//! used by the Qwen3 Dense family. See tests/fixtures/kv_qwen3/README.md for
//! provenance and the separately pinned Qwen3-0.6B verification corpus.

use std::path::Path;

use serde_json::Value;
use sha2::{Digest, Sha256};
use tokenizers::Tokenizer;

use crate::protocols::spec::{CompletionRequest, PromptInput};

// This fingerprint identifies renderer semantics, not a model/revision or
// tokenizer vocabulary. Unknown templates disable Chat affinity only.
const QWEN3_CHAT_TEMPLATE_SHA256: &str =
    "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8";

/// A local tokenizer with validated Qwen3 Dense metadata. Never downloads assets.
#[derive(Debug)]
pub struct PromptTokenizer {
    tokenizer: Tokenizer,
    vocab_size: u64,
    chat_template_compatible: bool,
}

impl PromptTokenizer {
    /// Test-only direct-token input support. Production still requires the
    /// metadata-validated tokenizer through `load`; this empty vocabulary cannot
    /// stand in for model tokenization or the pinned Python oracle.
    #[cfg(test)]
    pub(crate) fn synthetic_for_test() -> Self {
        Self {
            tokenizer: Tokenizer::new(tokenizers::models::wordlevel::WordLevel::default()),
            vocab_size: u64::from(u32::MAX) + 1,
            chat_template_compatible: false,
        }
    }

    /// `local_path` is tokenizer.json or its containing directory. Neighboring
    /// config.json and tokenizer_config.json must describe a compatible Qwen3
    /// Dense profile. The configured workers must use these same assets; local
    /// metadata validation cannot establish the running workers' configuration.
    pub fn load(local_path: impl AsRef<Path>) -> Result<Self, String> {
        let path = local_path.as_ref();
        let path = if path.is_dir() {
            path.join("tokenizer.json")
        } else {
            path.to_path_buf()
        };
        let bytes = std::fs::read(&path).map_err(|error| format!("read tokenizer: {error}"))?;
        let directory = path
            .parent()
            .ok_or("tokenizer path has no parent directory")?;
        let read_json = |name| -> Result<Value, String> {
            let bytes = std::fs::read(directory.join(name))
                .map_err(|error| format!("read local {name}: {error}"))?;
            serde_json::from_slice(&bytes).map_err(|error| format!("parse local {name}: {error}"))
        };
        let model = read_json("config.json")?;
        let config = read_json("tokenizer_config.json")?;
        let tokenizer_json: Value = serde_json::from_slice(&bytes)
            .map_err(|error| format!("parse tokenizer.json: {error}"))?;
        let vocab_size = validate_profile_metadata(&model, &config, &tokenizer_json)?;
        let tokenizer = Tokenizer::from_bytes(&bytes)
            .map_err(|error| format!("load local tokenizer: {error}"))?;
        validate_token_semantics(&model, &tokenizer, vocab_size)?;

        // Transformers gives a standalone template file precedence over the
        // tokenizer configuration. Multiple named templates are outside this
        // restricted renderer; do not accidentally validate only one of them.
        let template_path = directory.join("chat_template.jinja");
        let standalone_template = if template_path.exists() {
            Some(
                std::fs::read_to_string(&template_path)
                    .map_err(|error| format!("read chat_template.jinja: {error}"))?,
            )
        } else {
            None
        };
        let template = standalone_template
            .as_deref()
            .or_else(|| config.get("chat_template").and_then(Value::as_str));
        let chat_template_compatible =
            !directory.join("chat_templates").exists() && compatible_chat_template(template);
        tracing::info!(
            tokenizer_sha256 = %format!("{:x}", Sha256::digest(&bytes)),
            chat_template_compatible,
            "loaded local Qwen3 Dense tokenizer profile"
        );
        if !chat_template_compatible {
            tracing::warn!("unrecognized Qwen3 Chat template; Chat KV affinity disabled");
        }
        Ok(Self {
            tokenizer,
            vocab_size,
            chat_template_compatible,
        })
    }

    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>, String> {
        self.tokenizer
            .encode(text, add_special_tokens)
            .map(|encoding| encoding.get_ids().to_vec())
            .map_err(|error| format!("tokenize prompt: {error}"))
    }

    /// Only a single string or a single token-ID sequence has one exact prefix.
    pub fn completion(&self, request: &CompletionRequest) -> Result<Vec<u32>, String> {
        if request.lora_path.is_some()
            || request.session_params.is_some()
            || request.suffix.is_some()
        {
            return Err("completion adapter, session, or suffix is outside the profile".into());
        }
        if let Some(key) = request
            .other
            .keys()
            .find(|key| key.as_str() != "add_special_tokens")
        {
            return Err(format!("unsupported completion field: {key}"));
        }
        let add_special_tokens = match request.other.get("add_special_tokens") {
            None => true,
            Some(Value::Bool(value)) => *value,
            Some(_) => return Err("add_special_tokens must be boolean".into()),
        };
        match &request.prompt {
            PromptInput::String(text) => self.encode(text, add_special_tokens),
            PromptInput::IntArray(ids) if !ids.is_empty() => ids
                .iter()
                .map(|&id| {
                    let id = u32::try_from(id).map_err(|_| "negative prompt token ID")?;
                    if u64::from(id) >= self.vocab_size {
                        return Err("prompt token ID exceeds configured model vocabulary".into());
                    }
                    Ok(id)
                })
                .collect(),
            PromptInput::IntArray(_) => Err("empty prompt token sequence".into()),
            PromptInput::StringArray(_) | PromptInput::IntBatch(_) => {
                Err("batched completion has no single exact prefix".into())
            }
        }
    }

    /// Use the lossless JSON body, before protocol deserialization can discard
    /// unknown message fields (especially reasoning, tools and multimodal data).
    pub fn chat_raw(&self, request: &Value) -> Result<Vec<u32>, String> {
        if !self.chat_template_compatible {
            return Err("local Chat template is outside the verified Qwen3 text profile".into());
        }
        self.encode(&render_qwen3_chat(request)?, false)
    }
}

fn compatible_chat_template(template: Option<&str>) -> bool {
    template.is_some_and(|template| {
        format!("{:x}", Sha256::digest(template.as_bytes())) == QWEN3_CHAT_TEMPLATE_SHA256
    })
}

fn validate_profile_metadata(
    model: &Value,
    config: &Value,
    tokenizer: &Value,
) -> Result<u64, String> {
    if model.get("model_type").and_then(Value::as_str) != Some("qwen3")
        || model.get("architectures") != Some(&serde_json::json!(["Qwen3ForCausalLM"]))
    {
        return Err("KV tokenizer profile requires Qwen3 Dense (qwen3 / Qwen3ForCausalLM)".into());
    }
    for key in [
        "num_experts",
        "num_experts_per_tok",
        "moe_intermediate_size",
        "vision_config",
        "text_config",
        "mamba_d_state",
        "hybrid_layer_pattern",
        "attention_chunk_size",
        "sliding_window",
    ] {
        if model.get(key).is_some_and(|value| !value.is_null()) {
            return Err(format!("unsupported Qwen3 Dense model field: {key}"));
        }
    }
    if model
        .get("use_sliding_window")
        .is_some_and(|value| !value.is_null() && value.as_bool() != Some(false))
        || model.get("layer_types").is_some_and(|value| {
            !value.is_null()
                && value.as_array().is_none_or(|types| {
                    types
                        .iter()
                        .any(|kind| kind.as_str() != Some("full_attention"))
                })
        })
    {
        return Err("KV tokenizer profile requires full-attention Qwen3 Dense".into());
    }
    let vocab_size = model
        .get("vocab_size")
        .and_then(Value::as_u64)
        .filter(|size| *size > 0 && *size <= u64::from(u32::MAX) + 1)
        .ok_or("missing or invalid model vocabulary size")?;
    if !matches!(
        config.get("tokenizer_class").and_then(Value::as_str),
        Some("Qwen2Tokenizer" | "Qwen2TokenizerFast")
    ) || config
        .get("bos_token")
        .is_some_and(|value| !value.is_null())
        || config
            .get("unk_token")
            .is_some_and(|value| !value.is_null())
        || config.get("eos_token").and_then(Value::as_str) != Some("<|im_end|>")
        || config.get("pad_token").and_then(Value::as_str) != Some("<|endoftext|>")
        || config.get("auto_map").is_some_and(|value| !value.is_null())
    {
        return Err("unsupported Qwen3 tokenizer class or special-token metadata".into());
    }
    // A Transformers loader override must not select a different tokenizer
    // file/backend from the tokenizer.json used here.
    for key in ["fast_tokenizer_files", "tokenizer_file", "backend"] {
        if config.get(key).is_some_and(|value| !value.is_null()) {
            return Err(format!("unsupported tokenizer loader override: {key}"));
        }
    }
    for key in ["cls_token", "sep_token", "mask_token"] {
        if config.get(key).is_some_and(|value| !value.is_null()) {
            return Err(format!(
                "unsupported tokenizer special-token override: {key}"
            ));
        }
    }
    if config.get("extra_special_tokens").is_some_and(|value| {
        !value.is_null()
            && !value.as_array().is_some_and(Vec::is_empty)
            && !value.as_object().is_some_and(serde_json::Map::is_empty)
    }) {
        return Err("unsupported extra_special_tokens override".into());
    }
    for flag in [
        "add_bos_token",
        "add_eos_token",
        "add_prefix_space",
        "split_special_tokens",
        "from_slow",
    ] {
        if config
            .get(flag)
            .is_some_and(|value| !value.is_null() && value.as_bool() != Some(false))
        {
            return Err(format!("unsupported tokenizer option: {flag}"));
        }
    }
    if tokenizer.pointer("/model/type").and_then(Value::as_str) != Some("BPE")
        || ["truncation", "padding"]
            .iter()
            .any(|key| tokenizer.get(key).is_some_and(|value| !value.is_null()))
    {
        return Err("Qwen3 requires an untruncated, unpadded BPE tokenizer".into());
    }
    if tokenizer
        .pointer("/model/dropout")
        .is_some_and(|value| !value.is_null() && value.as_f64() != Some(0.0))
    {
        return Err("KV prompt tokenization requires deterministic BPE (dropout disabled)".into());
    }
    let added = tokenizer
        .get("added_tokens")
        .and_then(Value::as_array)
        .ok_or("missing tokenizer added_tokens metadata")?;
    let configured = config
        .get("added_tokens_decoder")
        .and_then(Value::as_object)
        .ok_or("missing tokenizer_config added_tokens_decoder metadata")?;
    for (id, definition) in configured {
        let id: u64 = id.parse().map_err(|_| "invalid added-token ID")?;
        let actual = added
            .iter()
            .find(|token| token.get("id").and_then(Value::as_u64) == Some(id))
            .ok_or("tokenizer_config adds tokens absent from tokenizer.json")?;
        for field in [
            "content",
            "single_word",
            "lstrip",
            "rstrip",
            "normalized",
            "special",
        ] {
            if definition.get(field) != actual.get(field) {
                return Err(format!(
                    "tokenizer added-token definition mismatch: {id}/{field}"
                ));
            }
        }
    }
    if let Some(tokens) = config.get("additional_special_tokens") {
        let tokens = tokens
            .as_array()
            .ok_or("additional_special_tokens must be an array")?;
        for content in tokens {
            if !added.iter().any(|token| {
                token.get("content") == Some(content)
                    && token.get("special") == Some(&Value::Bool(true))
            }) {
                return Err("tokenizer_config changes special-token semantics".into());
            }
        }
    }
    Ok(vocab_size)
}

fn validate_token_semantics(
    model: &Value,
    tokenizer: &Tokenizer,
    vocab_size: u64,
) -> Result<(), String> {
    if tokenizer
        .get_vocab(true)
        .values()
        .any(|id| u64::from(*id) >= vocab_size)
    {
        return Err("tokenizer ID exceeds the configured model vocabulary".into());
    }
    for symbol in [
        "<|endoftext|>",
        "<|im_start|>",
        "<|im_end|>",
        "<think>",
        "</think>",
    ] {
        let id = tokenizer
            .token_to_id(symbol)
            .ok_or_else(|| format!("missing Qwen3 token: {symbol}"))?;
        let encoded = tokenizer
            .encode(symbol, false)
            .map_err(|error| error.to_string())?;
        if encoded.get_ids() != [id] {
            return Err(format!("Qwen3 token is not encoded atomically: {symbol}"));
        }
    }
    for (key, symbol) in [
        ("bos_token_id", "<|endoftext|>"),
        ("eos_token_id", "<|im_end|>"),
    ] {
        if model.get(key).and_then(Value::as_u64) != tokenizer.token_to_id(symbol).map(u64::from) {
            return Err(format!("model/tokenizer {key} mismatch"));
        }
    }
    // Qwen3's profile has no automatic BOS/EOS postprocessing. Validate the
    // actual backend, including a custom postprocessor embedded in its JSON.
    for text in ["", "profile check", "<|im_start|>user\nhello<|im_end|>\n"] {
        let plain = tokenizer
            .encode(text, false)
            .map_err(|error| error.to_string())?;
        let special = tokenizer
            .encode(text, true)
            .map_err(|error| error.to_string())?;
        if plain.get_ids() != special.get_ids() {
            return Err("Qwen3 tokenizer unexpectedly inserts special tokens".into());
        }
    }
    Ok(())
}

// These fields affect sampling, output, or routing metadata, not the rendered
// input in this verified text-only profile. Unknown fields fail closed because
// vLLM extensions can change prompt tokens or cache namespaces.
const CHAT_FIELDS: &[&str] = &[
    "model",
    "messages",
    "temperature",
    "top_p",
    "n",
    "stream",
    "stream_options",
    "stop",
    "max_tokens",
    "max_completion_tokens",
    "presence_penalty",
    "frequency_penalty",
    "logit_bias",
    "user",
    "seed",
    "logprobs",
    "top_logprobs",
    "response_format",
    "top_k",
    "min_p",
    "min_tokens",
    "repetition_penalty",
    "stop_token_ids",
    "ignore_eos",
    "skip_special_tokens",
    "spaces_between_special_tokens",
    "include_stop_str_in_output",
    "length_penalty",
    "priority",
    "request_id",
    "return_tokens_as_token_ids",
    "add_generation_prompt",
    "continue_final_message",
    "add_special_tokens",
    "chat_template_kwargs",
];

/// The restricted text branch of the recognized Qwen3 Dense Apache-2.0 template.
/// Supports an optional initial system message followed by alternating user and
/// assistant text, ending in a user message. Tools/reasoning/continuations fall
/// back instead of approximating the upstream template's other branches.
pub fn render_qwen3_chat(request: &Value) -> Result<String, String> {
    let object = request.as_object().ok_or("chat body must be an object")?;
    for key in object.keys() {
        if !CHAT_FIELDS.contains(&key.as_str()) {
            return Err(format!("unsupported chat field: {key}"));
        }
    }
    for (key, expected) in [
        ("add_generation_prompt", true),
        ("continue_final_message", false),
        ("add_special_tokens", false),
    ] {
        if let Some(value) = object.get(key) {
            if value.as_bool() != Some(expected) {
                return Err(format!("unsupported {key}"));
            }
        }
    }
    let mut enable_thinking = true;
    if let Some(kwargs) = object.get("chat_template_kwargs") {
        let kwargs = kwargs
            .as_object()
            .ok_or("chat_template_kwargs must be an object")?;
        for (key, value) in kwargs {
            if key != "enable_thinking" {
                return Err(format!("unsupported chat template argument: {key}"));
            }
            enable_thinking = value.as_bool().ok_or("enable_thinking must be boolean")?;
        }
    }
    let messages = object
        .get("messages")
        .and_then(Value::as_array)
        .filter(|messages| !messages.is_empty())
        .ok_or("messages must be a nonempty array")?;
    let mut rendered = String::new();
    let mut expected_role = "user";
    for (index, message) in messages.iter().enumerate() {
        let message = message.as_object().ok_or("message must be an object")?;
        if message.keys().any(|key| key != "role" && key != "content") {
            return Err("only role and string content are supported in each message".into());
        }
        let role = message
            .get("role")
            .and_then(Value::as_str)
            .ok_or("missing role")?;
        let content = message
            .get("content")
            .and_then(Value::as_str)
            .ok_or("message content must be text")?;
        if role == "system" && index == 0 {
            // An initial system message does not consume a user turn.
        } else if role == expected_role {
            expected_role = if role == "user" { "assistant" } else { "user" };
        } else {
            return Err("expected optional system, then alternating user/assistant text".into());
        }
        if content.contains("<think>")
            || content.contains("</think>")
            || content.contains("<tool_response>")
            || content.contains("</tool_response>")
        {
            return Err("reasoning or tool markers are outside the text profile".into());
        }
        rendered.push_str("<|im_start|>");
        rendered.push_str(role);
        rendered.push('\n');
        rendered.push_str(content);
        rendered.push_str("<|im_end|>\n");
    }
    if expected_role != "assistant" {
        return Err("the final message must be a user message".into());
    }
    rendered.push_str("<|im_start|>assistant\n");
    if !enable_thinking {
        rendered.push_str("<think>\n\n</think>\n\n");
    }
    Ok(rendered)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // Verification corpus pins only: runtime accepts compatible local Dense
    // profiles independently of model size, repository name and revision.
    const QWEN3_REVISION: &str = "c1899de289a04d12100db370d81485cdf75e47ca";
    const QWEN3_TOKENIZER_SHA256: &str =
        "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4";

    fn profile_metadata() -> (Value, Value, Value) {
        let model = json!({"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"],
            "vocab_size": 151936, "bos_token_id": 151643, "eos_token_id": 151645,
            "use_sliding_window": false, "sliding_window": null});
        let token = json!({"id":151644, "content":"<|im_start|>", "single_word":false,
            "lstrip":false, "rstrip":false, "normalized":false, "special":true});
        let mut definition = token.clone();
        definition.as_object_mut().unwrap().remove("id");
        let config = json!({"tokenizer_class":"Qwen2Tokenizer", "eos_token":"<|im_end|>",
            "bos_token":null, "pad_token":"<|endoftext|>", "add_bos_token":false,
            "add_prefix_space":false, "added_tokens_decoder":{"151644":definition},
            "additional_special_tokens":["<|im_start|>"]});
        let tokenizer = json!({"model":{"type":"BPE"}, "truncation":null,
            "padding":null, "added_tokens":[token]});
        (model, config, tokenizer)
    }

    #[test]
    fn dense_profile_does_not_pin_name_revision_or_model_size() {
        let (mut model, config, tokenizer) = profile_metadata();
        for (name, hidden_size, layers) in [
            ("custom-small", 1024, 28),
            ("local-medium", 2048, 28),
            ("custom-large", 2560, 36),
        ] {
            model["_name_or_path"] = json!(name);
            model["revision"] = json!("different-local-revision");
            model["hidden_size"] = json!(hidden_size);
            model["num_hidden_layers"] = json!(layers);
            assert_eq!(
                validate_profile_metadata(&model, &config, &tokenizer).unwrap(),
                151936
            );
        }
    }

    #[test]
    fn profile_rejects_moe_hybrid_and_tokenizer_overrides() {
        let (model, config, tokenizer) = profile_metadata();
        for (field, value) in [
            ("model_type", json!("qwen3_moe")),
            ("architectures", json!(["Qwen3MoeForCausalLM"])),
            ("num_experts", json!(128)),
            ("vision_config", json!({})),
            ("sliding_window", json!(4096)),
            ("use_sliding_window", json!(true)),
            ("layer_types", json!(["full_attention", "linear_attention"])),
            ("vocab_size", json!(0)),
        ] {
            let mut changed = model.clone();
            changed[field] = value;
            assert!(
                validate_profile_metadata(&changed, &config, &tokenizer).is_err(),
                "{field}"
            );
        }
        for (field, value) in [
            ("tokenizer_class", json!("CustomTokenizer")),
            ("add_bos_token", json!(true)),
            ("add_prefix_space", json!(true)),
            ("split_special_tokens", json!(true)),
            ("eos_token", json!("different")),
            ("auto_map", json!({"AutoTokenizer":"custom"})),
            ("fast_tokenizer_files", json!(["tokenizer.1.json"])),
            ("fast_tokenizer_files", json!([])),
            ("tokenizer_file", json!("other.json")),
            ("backend", json!("custom")),
            ("from_slow", json!(true)),
            ("from_slow", json!("false")),
            ("from_slow", json!(0)),
            (
                "extra_special_tokens",
                json!({"image_token":"<|custom_image|>"}),
            ),
            ("extra_special_tokens", json!(["<|custom_token|>"])),
            ("cls_token", json!("<|custom_cls|>")),
            ("sep_token", json!("<|custom_sep|>")),
            ("mask_token", json!("<|custom_mask|>")),
        ] {
            let mut changed = config.clone();
            changed[field] = value;
            assert!(
                validate_profile_metadata(&model, &changed, &tokenizer).is_err(),
                "{field}"
            );
        }
        let mut no_override = config.clone();
        no_override["fast_tokenizer_files"] = Value::Null;
        no_override["tokenizer_file"] = Value::Null;
        no_override["backend"] = Value::Null;
        no_override["from_slow"] = json!(false);
        no_override["extra_special_tokens"] = json!({});
        no_override["cls_token"] = Value::Null;
        no_override["sep_token"] = Value::Null;
        no_override["mask_token"] = Value::Null;
        assert!(validate_profile_metadata(&model, &no_override, &tokenizer).is_ok());
        for dropout in [json!(0.1), json!(1.0), json!("0"), json!(false)] {
            let mut changed = tokenizer.clone();
            changed["model"]["dropout"] = dropout;
            assert!(validate_profile_metadata(&model, &config, &changed).is_err());
        }
        for dropout in [Value::Null, json!(0.0)] {
            let mut changed = tokenizer.clone();
            changed["model"]["dropout"] = dropout;
            assert!(validate_profile_metadata(&model, &config, &changed).is_ok());
        }
        let mut changed = config.clone();
        changed["added_tokens_decoder"]["151644"]["lstrip"] = json!(true);
        assert!(validate_profile_metadata(&model, &changed, &tokenizer).is_err());
        let mut changed = tokenizer.clone();
        changed["truncation"] = json!({"max_length":16});
        assert!(validate_profile_metadata(&model, &config, &changed).is_err());
    }

    #[test]
    fn unknown_chat_template_keeps_exact_completion_available() {
        assert!(!compatible_chat_template(None));
        assert!(!compatible_chat_template(Some("custom {{ messages }}")));
        let tokenizer = PromptTokenizer::synthetic_for_test();
        let request = serde_json::from_value(json!({"prompt":[1,2,3]})).unwrap();
        assert_eq!(tokenizer.completion(&request).unwrap(), [1, 2, 3]);
        assert!(tokenizer
            .chat_raw(&json!({"messages":[{"role":"user","content":"hello"}]}))
            .is_err());
    }

    #[test]
    fn direct_token_ids_respect_model_vocabulary() {
        let mut tokenizer = PromptTokenizer::synthetic_for_test();
        tokenizer.vocab_size = 100;
        let valid = serde_json::from_value(json!({"prompt":[99]})).unwrap();
        let invalid = serde_json::from_value(json!({"prompt":[100]})).unwrap();
        assert_eq!(tokenizer.completion(&valid).unwrap(), [99]);
        assert!(tokenizer.completion(&invalid).is_err());
    }

    #[test]
    #[ignore = "requires a complete local Qwen3 Dense model/tokenizer profile"]
    fn local_dense_profile_loads_metadata_and_chat_template() {
        let directory = std::env::var("QWEN3_PROFILE_DIRECTORY").expect(
            "QWEN3_PROFILE_DIRECTORY with config.json, tokenizer_config.json, tokenizer.json",
        );
        // Exercise the actual production loader, including the model metadata,
        // vocabulary and template checks that the fixed tokenizer oracle does
        // not cover. This test intentionally does not pin model size/revision.
        let tokenizer = PromptTokenizer::load(directory).unwrap();
        assert!(
            tokenizer.chat_template_compatible,
            "this test expects the standard Qwen3 Chat template"
        );
        let fixtures: Value =
            serde_json::from_str(include_str!("../../tests/fixtures/kv_qwen3/prompts.json"))
                .unwrap();
        for case in fixtures["chat"].as_array().unwrap() {
            assert!(
                !tokenizer.chat_raw(&case["request"]).unwrap().is_empty(),
                "{}",
                case["name"]
            );
        }
        // A direct ID comes from this profile's vocabulary, not fixed 0.6B IDs.
        let id = tokenizer.tokenizer.token_to_id("<|im_start|>").unwrap();
        let request = serde_json::from_value(json!({"prompt":[id]})).unwrap();
        assert_eq!(tokenizer.completion(&request).unwrap(), [id]);
    }

    fn tokenizer() -> PromptTokenizer {
        // Small synthetic tokenizer tests extraction and rejection independent
        // of network/model assets. The pinned oracle tests token IDs separately.
        PromptTokenizer::synthetic_for_test()
    }

    #[test]
    fn completion_token_ids_are_not_encoded_as_text() {
        let body = serde_json::from_value(json!({"prompt": [151644, 8948, 198, 42]})).unwrap();
        assert_eq!(
            tokenizer().completion(&body).unwrap(),
            [151644, 8948, 198, 42]
        );
    }

    #[test]
    fn completion_ambiguous_or_transformed_inputs_fall_back() {
        for body in [
            json!({"prompt": [[1, 2]]}),
            json!({"prompt": ["hello"]}),
            json!({"prompt": [-1]}),
            json!({"prompt": [1], "cache_salt": "salt"}),
            json!({"prompt": [1], "truncate_prompt_tokens": 1}),
            json!({"prompt": [1], "prompt_embeds": []}),
            json!({"prompt": [1], "add_special_tokens": "false"}),
            json!({"prompt": [1], "suffix": "tail"}),
        ] {
            let body = serde_json::from_value(body).unwrap();
            assert!(tokenizer().completion(&body).is_err());
        }
    }

    #[test]
    fn public_rendered_fixtures_match() {
        let fixtures: Value =
            serde_json::from_str(include_str!("../../tests/fixtures/kv_qwen3/prompts.json"))
                .unwrap();
        for case in fixtures["chat"].as_array().unwrap() {
            assert_eq!(
                render_qwen3_chat(&case["request"]).unwrap(),
                case["rendered"].as_str().unwrap(),
                "{}",
                case["name"]
            );
        }
    }

    #[test]
    fn lossy_or_unverified_chat_forms_fall_back() {
        let base = json!({"messages": [{"role": "user", "content": "Hello"}]});
        for (key, value) in [
            ("tools", json!([])),
            ("cache_salt", json!("salt")),
            ("truncate_prompt_tokens", json!(8)),
            ("chat_template", json!("custom")),
            ("lora_path", json!("adapter")),
            ("continue_final_message", json!(true)),
            ("add_generation_prompt", json!(false)),
            ("add_special_tokens", json!(true)),
            ("chat_template_kwargs", json!({"documents": []})),
            ("chat_template_kwargs", json!({"enable_thinking": "false"})),
        ] {
            let mut request = base.clone();
            request[key] = value;
            assert!(render_qwen3_chat(&request).is_err(), "{key}");
        }
        for extra in [
            "name",
            "reasoning",
            "reasoning_content",
            "tool_calls",
            "unknown",
        ] {
            let mut request = base.clone();
            request["messages"][0][extra] = Value::Null;
            assert!(render_qwen3_chat(&request).is_err(), "{extra}");
        }
        for messages in [
            json!([]),
            json!([{"role": "user", "content": [{"type":"text", "text":"Hello"}]}]),
            json!([{"role":"assistant", "content":"Hello"}]),
            json!([{"role":"user", "content":"<think>reason</think>"}]),
            json!([{"role":"system", "content":"alone"}]),
        ] {
            assert!(render_qwen3_chat(&json!({"messages": messages})).is_err());
        }
    }

    #[test]
    #[ignore = "requires the hash-pinned local tokenizer"]
    fn pinned_python_oracle_token_ids() {
        let path = std::env::var("QWEN3_TOKENIZER_PATH").expect("QWEN3_TOKENIZER_PATH");
        let path = Path::new(&path);
        let path = if path.is_dir() {
            path.join("tokenizer.json")
        } else {
            path.to_path_buf()
        };
        let bytes = std::fs::read(path).unwrap();
        assert_eq!(
            format!("{:x}", Sha256::digest(&bytes)),
            QWEN3_TOKENIZER_SHA256
        );
        // The fixed golden isolates tokenizer/renderer parity. Runtime metadata
        // validation is tested separately and no longer pins these bytes.
        let tokenizer = PromptTokenizer {
            tokenizer: Tokenizer::from_bytes(bytes).unwrap(),
            vocab_size: 151936,
            chat_template_compatible: true,
        };
        let oracle: Value = match std::env::var("QWEN3_ORACLE_PATH") {
            Ok(path) => serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap(),
            Err(_) => serde_json::from_str(include_str!(
                "../../tests/fixtures/kv_qwen3/python_oracle.json"
            ))
            .unwrap(),
        };
        assert_eq!(oracle["mode"], "canonical-tokenizer");
        assert_eq!(oracle["tokenizer_sha256"], QWEN3_TOKENIZER_SHA256);
        assert_eq!(oracle["revision"], QWEN3_REVISION);
        let fixtures: Value =
            serde_json::from_str(include_str!("../../tests/fixtures/kv_qwen3/prompts.json"))
                .unwrap();
        let expected_chat = fixtures["chat"].as_array().unwrap();
        let expected_completion = fixtures["completion"].as_array().unwrap();
        assert_eq!(
            oracle["chat"].as_array().unwrap().len(),
            expected_chat.len()
        );
        assert_eq!(
            oracle["completion"].as_array().unwrap().len(),
            expected_completion.len()
        );
        for (case, fixture) in oracle["chat"].as_array().unwrap().iter().zip(expected_chat) {
            assert_eq!(case["request"], fixture["request"]);
            assert_eq!(case["rendered"], fixture["rendered"]);
            let actual = tokenizer.chat_raw(&case["request"]).unwrap();
            let expected: Vec<u32> = serde_json::from_value(case["token_ids"].clone()).unwrap();
            assert_eq!(actual, expected, "{}", case["name"]);
        }
        for (case, fixture) in oracle["completion"]
            .as_array()
            .unwrap()
            .iter()
            .zip(expected_completion)
        {
            assert_eq!(case["request"], fixture["request"]);
            let request = serde_json::from_value(case["request"].clone()).unwrap();
            let expected: Vec<u32> = serde_json::from_value(case["token_ids"].clone()).unwrap();
            assert_eq!(
                tokenizer.completion(&request).unwrap(),
                expected,
                "{}",
                case["name"]
            );
        }
    }
}
