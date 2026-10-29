//! OpenAI chat fields → `vllm-proto` `GenerateRequest`.
//!
//! This version always sets `prompt = TokenIds`. Proto also has text /
//! `media` / KV-transfer fields; they are unused here. Sampling-default
//! merge still lives in worker `vllm-text`; this only fills proto.

use crate::backend::pb;
use crate::protocols::spec::{
    ChatCompletionRequest, CompletionRequest, GenerationRequest, PromptInput, StringOrArray,
};

/// Passthrough keys the gRPC pipeline forwards to the worker. Anything else is
/// rejected: a field the adapter cannot express must not be dropped silently.
const SUPPORTED_OTHER: [&str; 7] = [
    "add_special_tokens",
    "cache_salt",
    "documents",
    "lora_name",
    "priority",
    "truncate_prompt_tokens",
    "watermarking",
];

/// Completions-only response option; never part of the chat shape.
const RETURN_TOKEN_IDS: &str = "return_token_ids";

// Consumed by routing, never sent as generation options.
const ROUTING_ONLY: [&str; 4] = ["vllm_xargs", "agent_hint", "session_id", "user_id"];

fn ensure_supported_other<'a>(
    keys: impl Iterator<Item = &'a str>,
    endpoint: &str,
) -> Result<(), String> {
    let unsupported: Vec<&str> = keys
        .filter(|key| !SUPPORTED_OTHER.contains(key) && !ROUTING_ONLY.contains(key))
        .collect();
    if unsupported.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "unsupported gRPC {endpoint} fields: {}",
            unsupported.join(", ")
        ))
    }
}

pub fn requires_worker_output_text(request: &ChatCompletionRequest) -> bool {
    !request.no_stop_trim
        && (request.stop.as_ref().is_some_and(|stop| match stop {
            StringOrArray::String(value) => !value.is_empty(),
            StringOrArray::Array(values) => !values.is_empty(),
        }) || request
            .stop_token_ids
            .as_ref()
            .is_some_and(|ids| !ids.is_empty()))
}

/// Wire adapter: OpenAI chat fields → crates.io `vllm-proto` 0.2.
pub fn chat_to_generate_request(
    request: &ChatCompletionRequest,
    token_ids: Vec<u32>,
    request_id: String,
) -> Result<pb::GenerateRequest, String> {
    chat_to_generate_request_with_progress(request, token_ids, request_id, false)
}

pub fn chat_to_generate_request_with_progress(
    request: &ChatCompletionRequest,
    token_ids: Vec<u32>,
    request_id: String,
    include_prefill_progress: bool,
) -> Result<pb::GenerateRequest, String> {
    if request.n.unwrap_or(1) > 1 {
        return Err("n > 1 is not supported by vLLM 0.29 GenerateStream; use n=1".to_string());
    }
    if request.logprobs || request.top_logprobs.is_some() {
        return Err(
            "chat logprobs are not yet supported by the router gRPC response adapter".to_string(),
        );
    }
    if request.functions.is_some() || request.function_call.is_some() {
        return Err(
            "deprecated functions/function_call are not supported on gRPC; use tools/tool_choice"
                .to_string(),
        );
    }
    if request.lora_path.is_some() {
        return Err(
            "lora_path cannot be mapped to the gRPC loaded LoRA name; use lora_name".to_string(),
        );
    }
    if request.return_hidden_states || request.echo == Some(true) {
        return Err(
            "return_hidden_states and echo are not supported by the gRPC response adapter"
                .to_string(),
        );
    }
    ensure_supported_other(request.other.keys().map(String::as_str), "chat")?;
    other_bool(request, "add_special_tokens")?;
    // Validate types before using the same scope as routing.
    other_string(request, "lora_name")?;
    other_string(request, "cache_salt")?;
    let (lora_name, cache_salt) = request.cache_identity();
    let max_new_tokens = request
        .max_completion_tokens
        .or(request.max_tokens)
        .unwrap_or(0);

    let (stop_strings, stop_token_ids) = split_stops(request)?;
    let logit_bias = request
        .logit_bias
        .as_ref()
        .map(|bias| {
            bias.iter()
                .map(|(token, value)| {
                    token
                        .parse::<u32>()
                        .map(|id| (id, *value))
                        .map_err(|_| format!("logit_bias key is not a token id: {token}"))
                })
                .collect::<Result<_, _>>()
        })
        .transpose()?
        .unwrap_or_default();

    let mut decoding = pb::DecodingParameters {
        presence_penalty: request.presence_penalty.unwrap_or(0.0),
        frequency_penalty: request.frequency_penalty.unwrap_or(0.0),
        repetition_penalty: request.repetition_penalty.unwrap_or(0.0),
        logit_bias,
        ..Default::default()
    };
    decoding.structured_output = structured_output(request)?;

    if let Some(top_k) = request.top_k {
        if top_k == -1 {
            return Err(
                "top_k=-1 is not supported by the vLLM gRPC protocol; omit top_k or use a positive value"
                    .to_string(),
            );
        }
        if top_k < 0 {
            return Err("top_k must be positive for the vLLM gRPC protocol".to_string());
        }
    }

    Ok(pb::GenerateRequest {
        request_id,
        model: request.model.clone().unwrap_or_default(),
        prompt: Some(pb::generate_request::Prompt::TokenIds(pb::TokenIds {
            ids: token_ids,
        })),
        // EngineFrontend resolves model generation-config inheritance before
        // this adapter. The 1.0 fallback covers direct/test callers and matches
        // OpenAI when the model has no configured default; protobuf omission
        // cannot be used because vLLM gRPC rewrites it to greedy 0.0.
        temperature: Some(request.temperature.unwrap_or(1.0)),
        sampling: Some(pb::RandomSampling {
            num_sequences: request.n.unwrap_or(0),
            top_k: request.top_k.map(|top_k| top_k as u32),
            top_p: request.top_p,
            min_p: request.min_p,
            seed: request.seed,
        }),
        decoding: Some(decoding),
        stopping: Some(pb::StoppingCriteria {
            max_new_tokens,
            min_new_tokens: request.min_tokens.unwrap_or(0),
            stop_token_ids,
            stop_strings,
            include_stop_strings: request.no_stop_trim,
            ignore_eos: request.ignore_eos,
            thinking_token_budget: None,
        }),
        response: Some(pb::ResponseOptions {
            // Worker text carries the exact character-level trim boundary for
            // hidden stop strings/tokens; token IDs alone cannot represent it.
            output_text: Some(requires_worker_output_text(request)),
            output_token_ids: true,
            skip_special_tokens: Some(request.skip_special_tokens),
            ..Default::default()
        }),
        kv: Some(pb::KvCacheParameters {
            cache_salt: cache_salt.unwrap_or_default().to_string(),
            ..Default::default()
        }),
        truncate_prompt_tokens: other_u32(request, "truncate_prompt_tokens")?.unwrap_or(0),
        priority: other_i32(request, "priority")?.unwrap_or(0),
        session_id: request
            .session_params
            .as_ref()
            .and_then(|params| params.get("session_id"))
            .and_then(serde_json::Value::as_str)
            .filter(|id| !id.is_empty())
            .map(str::to_string),
        lora_name: lora_name.unwrap_or_default().to_string(),
        watermarking: other_bool(request, "watermarking")?,
        include_prefill_progress,
        ..Default::default()
    })
}

fn structured_output(
    request: &ChatCompletionRequest,
) -> Result<Option<pb::decoding_parameters::StructuredOutput>, String> {
    use pb::decoding_parameters::StructuredOutput;

    let mut selected = Vec::new();
    if let Some(regex) = &request.regex {
        selected.push(StructuredOutput::Regex(regex.clone()));
    }
    if let Some(ebnf) = &request.ebnf {
        selected.push(StructuredOutput::Grammar(ebnf.clone()));
    }
    if let Some(format) = &request.response_format {
        match format {
            crate::protocols::spec::ResponseFormat::Text => {}
            crate::protocols::spec::ResponseFormat::JsonObject => {
                selected.push(StructuredOutput::JsonObject(true));
            }
            crate::protocols::spec::ResponseFormat::JsonSchema { json_schema } => {
                selected.push(StructuredOutput::Json(json_schema.schema.to_string()));
            }
        }
    }
    if let Some(params) = &request.structured_outputs {
        if let Some(json) = &params.json {
            selected.push(StructuredOutput::Json(json.to_string()));
        }
        if let Some(regex) = &params.regex {
            selected.push(StructuredOutput::Regex(regex.clone()));
        }
        if let Some(choices) = &params.choice {
            selected.push(StructuredOutput::Choice(
                pb::decoding_parameters::StringChoices {
                    choices: choices.clone(),
                },
            ));
        }
        if let Some(grammar) = &params.grammar {
            selected.push(StructuredOutput::Grammar(grammar.clone()));
        }
        if params.json_object == Some(true) {
            selected.push(StructuredOutput::JsonObject(true));
        }
        if let Some(tag) = &params.structural_tag {
            selected.push(StructuredOutput::StructuralTag(tag.clone()));
        }
    }
    match selected.len() {
        0 => Ok(None),
        1 => Ok(selected.pop()),
        _ => Err("only one structured output constraint may be specified".to_string()),
    }
}

fn split_stops(request: &ChatCompletionRequest) -> Result<(Vec<String>, Vec<u32>), String> {
    let stop_strings = match &request.stop {
        Some(StringOrArray::String(s)) if !s.is_empty() => vec![s.clone()],
        Some(StringOrArray::Array(items)) => items.clone(),
        _ => Vec::new(),
    };
    let stop_token_ids = request
        .stop_token_ids
        .as_deref()
        .unwrap_or_default()
        .iter()
        .map(|id| u32::try_from(*id).map_err(|_| format!("stop_token_ids cannot contain {id}")))
        .collect::<Result<_, _>>()?;
    Ok((stop_strings, stop_token_ids))
}

fn other_string<'a>(
    request: &'a ChatCompletionRequest,
    key: &str,
) -> Result<Option<&'a str>, String> {
    request
        .other
        .get(key)
        .map(|value| {
            value
                .as_str()
                .ok_or_else(|| format!("{key} must be a string"))
        })
        .transpose()
}

fn other_bool(request: &ChatCompletionRequest, key: &str) -> Result<Option<bool>, String> {
    request
        .other
        .get(key)
        .map(|value| {
            value
                .as_bool()
                .ok_or_else(|| format!("{key} must be a boolean"))
        })
        .transpose()
}

fn other_u32(request: &ChatCompletionRequest, key: &str) -> Result<Option<u32>, String> {
    request
        .other
        .get(key)
        .map(|value| {
            value
                .as_u64()
                .and_then(|value| u32::try_from(value).ok())
                .ok_or_else(|| format!("{key} must be a non-negative 32-bit integer"))
        })
        .transpose()
}

fn other_i32(request: &ChatCompletionRequest, key: &str) -> Result<Option<i32>, String> {
    request
        .other
        .get(key)
        .map(|value| {
            value
                .as_i64()
                .and_then(|value| i32::try_from(value).ok())
                .ok_or_else(|| format!("{key} must be a 32-bit integer"))
        })
        .transpose()
}

/// A `/v1/completions` request lowered onto the gRPC pipeline.
#[derive(Debug)]
pub struct PreparedCompletion {
    pub chat: ChatCompletionRequest,
    pub token_ids: Vec<u32>,
    pub emit_token_ids: bool,
}

/// Completions → chat shape, prompt ids, and response options for a gRPC pool.
///
/// The gRPC wire protocol is token_ids-only, so the client's ids go straight
/// through: no chat template and no router-side tokenize. Fields without a
/// chat-shape equivalent are rejected — dropping them would answer a different
/// request than the client sent.
pub fn prepare_grpc_completion(request: &CompletionRequest) -> Result<PreparedCompletion, String> {
    let PromptInput::IntArray(ids) = &request.prompt else {
        return Err(
            "gRPC /v1/completions requires a single token-id prompt (list[int])".to_string(),
        );
    };
    if ids.is_empty() {
        return Err("completion prompt token_ids must not be empty".to_string());
    }
    let token_ids = ids
        .iter()
        .map(|id| {
            u32::try_from(*id)
                .map_err(|_| format!("prompt token ids must be non-negative 32-bit integers: {id}"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let rejected: [(&str, bool, Option<&str>); 7] = [
        ("suffix", request.suffix.is_some(), None),
        ("best_of", request.best_of.is_some(), None),
        (
            "json_schema",
            request.json_schema.is_some(),
            Some("use regex or ebnf"),
        ),
        ("logprobs", request.logprobs.is_some(), None),
        ("echo", request.echo, None),
        ("return_hidden_states", request.return_hidden_states, None),
        ("n > 1", request.n.unwrap_or(1) > 1, Some("use n=1")),
    ];
    if let Some((field, _, hint)) = rejected.iter().find(|(_, present, _)| *present) {
        let hint = hint.map(|hint| format!("; {hint}")).unwrap_or_default();
        return Err(format!(
            "{field} is not supported by the gRPC /v1/completions path{hint}"
        ));
    }
    if request.other.contains_key("documents") {
        return Err("documents is not supported by the gRPC /v1/completions path".to_string());
    }
    // Router-side only: the adapter asks the worker for ids unconditionally, so
    // this gates what the reply writes back rather than the wire request.
    let emit_token_ids = match request.other.get(RETURN_TOKEN_IDS) {
        None => false,
        Some(value) => value
            .as_bool()
            .ok_or_else(|| "return_token_ids must be a boolean".to_string())?,
    };
    ensure_supported_other(
        request
            .other
            .keys()
            .map(String::as_str)
            .filter(|key| *key != RETURN_TOKEN_IDS),
        "/v1/completions",
    )?;
    let other = request
        .other
        .iter()
        .filter(|(key, _)| {
            key.as_str() != RETURN_TOKEN_IDS && !ROUTING_ONLY.contains(&key.as_str())
        })
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect();
    let chat = ChatCompletionRequest {
        model: request.model.clone(),
        messages: Vec::new(),
        temperature: request.temperature,
        top_p: request.top_p,
        n: request.n,
        stream: request.stream,
        stream_options: request.stream_options.clone(),
        stop: request.stop.clone(),
        max_tokens: request.max_tokens,
        max_completion_tokens: None,
        presence_penalty: request.presence_penalty,
        frequency_penalty: request.frequency_penalty,
        logit_bias: request.logit_bias.clone(),
        user: request.user.clone(),
        seed: request.seed,
        logprobs: false,
        top_logprobs: None,
        response_format: None,
        tools: None,
        tool_choice: None,
        parallel_tool_calls: None,
        functions: None,
        function_call: None,
        top_k: request.top_k,
        min_p: request.min_p,
        min_tokens: request.min_tokens,
        repetition_penalty: request.repetition_penalty,
        regex: request.regex.clone(),
        ebnf: request.ebnf.clone(),
        stop_token_ids: request.stop_token_ids.clone(),
        no_stop_trim: request.no_stop_trim,
        ignore_eos: request.ignore_eos,
        add_generation_prompt: None,
        continue_final_message: false,
        skip_special_tokens: request.skip_special_tokens,
        lora_path: request.lora_path.clone(),
        session_params: request.session_params.clone(),
        chat_template_kwargs: None,
        return_hidden_states: false,
        echo: None,
        reasoning_effort: None,
        structured_outputs: None,
        other,
        ..Default::default()
    };
    Ok(PreparedCompletion {
        chat,
        token_ids,
        emit_token_ids,
    })
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
mod tests {
    use super::*;

    fn chat(json: serde_json::Value) -> ChatCompletionRequest {
        serde_json::from_value(json).unwrap()
    }

    #[test]
    fn completion_routing_metadata_preserves_worker_cache_scope() {
        let request: CompletionRequest = serde_json::from_value(serde_json::json!({
            "prompt": [1, 2], "lora_name": "adapter", "cache_salt": "tenant",
            "session_id": "s", "user_id": "u", "agent_hint": {"agent_id": "a"},
            "vllm_xargs": {"agentic_context": {"session_id": "s"}},
            "session_params": {"session_id": "typed-session"}, "return_token_ids": true
        }))
        .unwrap();
        let prepared = prepare_grpc_completion(&request).unwrap();
        assert_eq!(prepared.token_ids, vec![1, 2]);
        assert!(prepared.emit_token_ids);
        assert_eq!(prepared.chat.cache_identity(), request.cache_identity());
        assert_eq!(prepared.chat.session_params, request.session_params);
        for key in ROUTING_ONLY {
            assert!(!prepared.chat.other.contains_key(key));
        }
        let proto =
            chat_to_generate_request(&prepared.chat, prepared.token_ids, "r".into()).unwrap();
        assert_eq!(proto.lora_name, "adapter");
        assert_eq!(proto.kv.unwrap().cache_salt, "tenant");
    }

    #[test]
    fn token_ids_only_no_text_prompt() {
        // This-version default: convert always emits TokenIds. Not a forever
        // ban on a text prompt field if that is added later.
        let req = chat(serde_json::json!({
            "model": "qwen",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 32,
            "temperature": 0.2
        }));
        let proto = chat_to_generate_request(&req, vec![11, 22, 33], "req-1".into()).unwrap();
        match proto.prompt.unwrap() {
            pb::generate_request::Prompt::TokenIds(ids) => {
                assert_eq!(ids.ids, vec![11, 22, 33]);
            }
            other => panic!("expected token_ids, got {other:?}"),
        }
        assert_eq!(proto.stopping.unwrap().max_new_tokens, 32);
        assert_eq!(proto.temperature, Some(0.2));
        assert_eq!(proto.model, "qwen");
        assert_eq!(proto.response.unwrap().output_text, Some(false));
    }

    #[test]
    fn prefers_max_completion_tokens() {
        let req = chat(serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 8,
            "max_completion_tokens": 64
        }));
        let proto = chat_to_generate_request(&req, vec![1], "r".into()).unwrap();
        assert_eq!(proto.stopping.unwrap().max_new_tokens, 64);
    }

    #[test]
    fn preserves_openai_defaults_when_omitted() {
        let req = chat(serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}]
        }));
        let proto = chat_to_generate_request(&req, vec![1], "r".into()).unwrap();
        assert_eq!(proto.temperature, Some(1.0));
        assert_eq!(proto.stopping.unwrap().max_new_tokens, 0);
    }

    #[test]
    fn missing_model_stays_empty_on_grpc_wire() {
        let req = chat(serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}]
        }));
        let proto = chat_to_generate_request(&req, vec![1], "r".into()).unwrap();
        assert_eq!(proto.model, "");
    }

    #[test]
    fn rejects_top_k_minus_one_for_grpc() {
        let req = chat(serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "top_k": -1
        }));
        let err = chat_to_generate_request(&req, vec![1], "r".into()).unwrap_err();
        assert!(err.contains("top_k=-1"));

        let req = chat(serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "top_k": -2
        }));
        let err = chat_to_generate_request(&req, vec![1], "r".into()).unwrap_err();
        assert!(err.contains("top_k must be positive"));
    }

    #[test]
    fn rejects_unadapted_multi_choice_and_logprobs() {
        let n = chat(serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "n": 2
        }));
        assert!(chat_to_generate_request(&n, vec![1], "r".into())
            .unwrap_err()
            .contains("n > 1"));

        let logprobs = chat(serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "logprobs": true
        }));
        assert!(chat_to_generate_request(&logprobs, vec![1], "r".into())
            .unwrap_err()
            .contains("logprobs"));
    }

    #[test]
    fn lowers_json_schema_and_logit_bias() {
        let req = chat(serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "logit_bias": {"42": -10.0},
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "answer",
                    "schema": {"type": "object"}
                }
            }
        }));
        let proto = chat_to_generate_request(&req, vec![1], "r".into()).unwrap();
        let decoding = proto.decoding.unwrap();
        assert_eq!(decoding.logit_bias.get(&42), Some(&-10.0));
        assert!(matches!(
            decoding.structured_output,
            Some(pb::decoding_parameters::StructuredOutput::Json(_))
        ));
    }

    #[test]
    fn requests_worker_text_for_hidden_stop_boundary() {
        let req = chat(serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "stop": ["END"]
        }));
        let proto = chat_to_generate_request(&req, vec![1], "r".into()).unwrap();
        assert_eq!(proto.response.unwrap().output_text, Some(true));

        let visible = chat(serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "stop": ["END"],
            "no_stop_trim": true
        }));
        let proto = chat_to_generate_request(&visible, vec![1], "r".into()).unwrap();
        assert_eq!(proto.response.unwrap().output_text, Some(false));
    }

    fn completion(json: serde_json::Value) -> CompletionRequest {
        serde_json::from_value(json).unwrap()
    }

    #[test]
    fn lowers_completion_onto_chat_shape() {
        let req = completion(serde_json::json!({
            "model": "qwen",
            "prompt": [1, 2, 3],
            "max_tokens": 16,
            "temperature": 0.3,
            "top_k": 10,
            "stop": ["END"],
            "return_token_ids": true
        }));
        let prepared = prepare_grpc_completion(&req).unwrap();
        assert_eq!(prepared.token_ids, vec![1, 2, 3]);
        assert!(prepared.emit_token_ids);
        assert_eq!(prepared.chat.max_tokens, Some(16));
        assert_eq!(prepared.chat.top_k, Some(10));
        assert!(prepared.chat.messages.is_empty());
        assert!(prepared.chat.stop.is_some());
    }

    #[test]
    fn token_ids_are_withheld_unless_requested() {
        let default =
            prepare_grpc_completion(&completion(serde_json::json!({"prompt": [1]}))).unwrap();
        assert!(!default.emit_token_ids);

        let explicit_false = prepare_grpc_completion(&completion(serde_json::json!({
            "prompt": [1],
            "return_token_ids": false
        })))
        .unwrap();
        assert!(!explicit_false.emit_token_ids);

        let wrong_type = prepare_grpc_completion(&completion(serde_json::json!({
            "prompt": [1],
            "return_token_ids": "yes"
        })))
        .unwrap_err();
        assert!(wrong_type.contains("boolean"), "{wrong_type}");
    }

    #[test]
    fn rejects_fields_the_grpc_adapter_cannot_express() {
        for (body, field) in [
            (
                serde_json::json!({"prompt": [1], "suffix": "tail"}),
                "suffix",
            ),
            (serde_json::json!({"prompt": [1], "best_of": 3}), "best_of"),
            (serde_json::json!({"prompt": [1], "n": 2}), "n > 1"),
            (
                serde_json::json!({"prompt": [1], "json_schema": "{\"type\":\"object\"}"}),
                "json_schema",
            ),
            (
                serde_json::json!({"prompt": [1], "logprobs": 3}),
                "logprobs",
            ),
            (serde_json::json!({"prompt": [1], "echo": true}), "echo"),
            (
                serde_json::json!({"prompt": [1], "return_hidden_states": true}),
                "return_hidden_states",
            ),
            (
                serde_json::json!({"prompt": [1], "documents": [{"content": "x"}]}),
                "documents",
            ),
        ] {
            let err = prepare_grpc_completion(&completion(body)).unwrap_err();
            assert!(err.contains(field), "{err}");
        }

        let off = completion(serde_json::json!({"prompt": [1], "echo": false}));
        assert!(!prepare_grpc_completion(&off)
            .unwrap()
            .chat
            .echo
            .unwrap_or(false));
    }

    #[test]
    fn rejects_unknown_passthrough_fields_instead_of_dropping_them() {
        let err = prepare_grpc_completion(&completion(serde_json::json!({
            "prompt": [1],
            "guided_decoding_backend": "xgrammar"
        })))
        .unwrap_err();
        assert!(err.contains("guided_decoding_backend"), "{err}");

        let chat_err = chat_to_generate_request(
            &chat(serde_json::json!({
                "messages": [{"role": "user", "content": "hi"}],
                "guided_decoding_backend": "xgrammar"
            })),
            vec![1],
            "r".into(),
        )
        .unwrap_err();
        assert!(chat_err.contains("guided_decoding_backend"), "{chat_err}");
    }

    #[test]
    fn rejects_prompts_without_usable_token_ids() {
        let text = prepare_grpc_completion(&completion(serde_json::json!({"prompt": "hello"})))
            .unwrap_err();
        assert!(text.contains("list[int]"), "{text}");

        let negative = prepare_grpc_completion(&completion(serde_json::json!({"prompt": [1, -2]})))
            .unwrap_err();
        assert!(negative.contains("non-negative"), "{negative}");
    }
}
