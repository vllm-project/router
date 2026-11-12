//! Tokenizer/prompt-preparation equivalence fingerprints.
//!
//! These helpers live beside `EngineFrontend` preparation: they describe the
//! inputs that can change the rendered token IDs, so a prepared count and the
//! Engine's own count are only comparable when they match.

use crate::program_scheduling::TokenizerFingerprint;
use crate::protocols::spec::ChatCompletionRequest;

/// Identity of a tokenizer/prompt-preparation equivalence class.
///
/// The Engine must tokenize the same model with the same template options and
/// the same frontend assets for a prepared count to describe Engine input. The
/// fingerprint covers every input that changes that mapping, so a mismatch
/// revokes exactly one class instead of disabling a whole deployment.
/// Canonical JSON for a fingerprint: object keys sorted at every level.
///
/// `chat_template_kwargs` is a `HashMap`, so plain serialization would make the
/// fingerprint depend on iteration order and silently split one equivalence
/// class into many.
pub(crate) fn canonical_json(value: &serde_json::Value) -> String {
    fn write(value: &serde_json::Value, out: &mut String) {
        match value {
            serde_json::Value::Object(map) => {
                let mut keys = map.keys().collect::<Vec<_>>();
                keys.sort();
                out.push('{');
                for (index, key) in keys.into_iter().enumerate() {
                    if index > 0 {
                        out.push(',');
                    }
                    out.push_str(&serde_json::to_string(key).unwrap_or_default());
                    out.push(':');
                    write(&map[key], out);
                }
                out.push('}');
            }
            serde_json::Value::Array(items) => {
                out.push('[');
                for (index, item) in items.iter().enumerate() {
                    if index > 0 {
                        out.push(',');
                    }
                    write(item, out);
                }
                out.push(']');
            }
            other => out.push_str(&serde_json::to_string(other).unwrap_or_default()),
        }
    }
    let mut out = String::new();
    write(value, &mut out);
    out
}

pub(crate) fn prepared_chat_fingerprint(
    request: &ChatCompletionRequest,
    encode_backend: &str,
) -> TokenizerFingerprint {
    let env_model = std::env::var("VLLM_ROUTER_MODEL").unwrap_or_default();
    let env_tokenizer = std::env::var("VLLM_ROUTER_TOKENIZER").unwrap_or_default();
    // Only inputs that change the rendered token IDs belong in the fingerprint.
    // Per-request identity (`vllm_xargs`, `agent_hint`, session params),
    // sampling knobs and credentials must not split one equivalence class into
    // thousands of fingerprints.
    let options = canonical_json(&serde_json::json!({
        "chat_template_kwargs": request.chat_template_kwargs,
        "add_generation_prompt": request.add_generation_prompt,
        "continue_final_message": request.continue_final_message,
        "reasoning_effort": request.reasoning_effort,
        "response_format": request.response_format,
        "add_special_tokens": request.other.get("add_special_tokens"),
        "chat_template": request.other.get("chat_template"),
        "tools": request.tools,
        "functions": request.functions,
        "tool_choice": request.tool_choice,
        "parallel_tool_calls": request.parallel_tool_calls,
    }));
    TokenizerFingerprint::from_parts(&[
        "prepared_token_ids",
        request.model.as_deref().unwrap_or_default(),
        &env_model,
        &env_tokenizer,
        &options,
        encode_backend,
    ])
}

/// Fingerprint for a client-supplied token-ID prompt.
///
/// Those IDs are forwarded verbatim, so the only way the Engine count can differ
/// is per-request options (for example an implicit special token) or a model
/// mix-up; both are covered here.
pub(crate) fn client_token_ids_fingerprint(
    model_id: Option<&str>,
    model_pool: &str,
    endpoint: &str,
) -> TokenizerFingerprint {
    TokenizerFingerprint::from_parts(&[
        "client_token_ids",
        model_pool,
        endpoint,
        model_id.unwrap_or_default(),
    ])
}

/// Exact input-token count of a transparent JSON body, when its prompt is one
/// token-ID array.
pub(crate) fn exact_prompt_tokens_from_json(request: Option<&serde_json::Value>) -> Option<usize> {
    let request = request?;
    // The Engine must not prepend special tokens for `len(prompt)` to be the
    // exact Engine input count.
    if request
        .get("add_special_tokens")
        .and_then(serde_json::Value::as_bool)
        != Some(false)
    {
        return None;
    }
    let array = request.get("prompt")?.as_array()?;
    if array.is_empty() {
        return None;
    }
    if !array.iter().all(|value| value.as_u64().is_some()) {
        return None;
    }
    // The Engine keeps only the last N tokens; `0`/absent means no truncation.
    // An invalid limit falls back to the estimator instead of guessing.
    let truncate = match request.get("truncate_prompt_tokens") {
        None => None,
        Some(value) if value.is_null() => None,
        Some(value) => Some(usize::try_from(value.as_u64()?).ok()?),
    };
    let tokens = array.len();
    Some(match truncate {
        Some(limit) if limit > 0 && tokens > limit => limit,
        _ => tokens,
    })
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
mod tests {
    use super::*;

    #[test]
    fn response_format_must_have_its_own_verification_class() {
        let request = |format: serde_json::Value| -> ChatCompletionRequest {
            serde_json::from_value(serde_json::json!({
                "model": "model", "messages": [{"role": "user", "content": "hello"}],
                "response_format": format
            }))
            .unwrap()
        };
        assert_ne!(
            prepared_chat_fingerprint(&request(serde_json::Value::Null), "vllm-tokenizer"),
            prepared_chat_fingerprint(
                &request(serde_json::json!({"type":"json_object"})),
                "vllm-tokenizer"
            )
        );
    }

    #[test]
    fn reasoning_effort_must_have_its_own_verification_class() {
        let request = |effort: &str| -> ChatCompletionRequest {
            serde_json::from_value(serde_json::json!({
                "model":"model", "messages":[{"role":"user","content":"hello"}],
                "reasoning_effort":effort
            }))
            .unwrap()
        };
        assert_ne!(
            prepared_chat_fingerprint(&request("none"), "vllm-tokenizer"),
            prepared_chat_fingerprint(&request("high"), "vllm-tokenizer")
        );
    }
}
