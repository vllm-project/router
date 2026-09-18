//! OpenAI chat fields → `vllm-proto` `GenerateRequest`.
//!
//! This version always sets `prompt = TokenIds`. Proto also has text /
//! `media` / KV-transfer fields; they are unused here. Sampling-default
//! merge still lives in worker `vllm-text`; this only fills proto.

use crate::backend::pb;
use crate::protocols::spec::{ChatCompletionRequest, StringOrArray};

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
    let supported_other = [
        "add_special_tokens",
        "cache_salt",
        "documents",
        "lora_name",
        "priority",
        "truncate_prompt_tokens",
        "watermarking",
    ];
    let unsupported: Vec<_> = request
        .other
        .keys()
        .filter(|key| !supported_other.contains(&key.as_str()))
        .cloned()
        .collect();
    if !unsupported.is_empty() {
        return Err(format!(
            "unsupported gRPC chat fields: {}",
            unsupported.join(", ")
        ));
    }
    other_bool(request, "add_special_tokens")?;
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

    Ok(pb::GenerateRequest {
        request_id,
        model: request
            .model
            .clone()
            .filter(|s| !s.is_empty())
            .or_else(|| {
                std::env::var("VLLM_ROUTER_MODEL")
                    .ok()
                    .filter(|s| !s.is_empty())
            })
            .unwrap_or_default(),
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
            top_k: request.top_k.unwrap_or(0).max(0) as u32,
            top_p: request.top_p.unwrap_or(0.0),
            min_p: request.min_p.unwrap_or(0.0),
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
            cache_salt: other_string(request, "cache_salt")?
                .unwrap_or_default()
                .to_string(),
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
        lora_name: other_string(request, "lora_name")?
            .unwrap_or_default()
            .to_string(),
        watermarking: other_bool(request, "watermarking")?,
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

#[cfg(test)]
mod tests {
    use super::*;

    fn chat(json: serde_json::Value) -> ChatCompletionRequest {
        serde_json::from_value(json).unwrap()
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
}
