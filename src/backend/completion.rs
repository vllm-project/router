//! Pure preparation for the restricted single-prompt Completion contract.
//!
//! This does not authenticate tokenizer assets or the remote worker's input/hash
//! contract. Runtime integration must validate the tokenizer definition before
//! loading it, supply the worker's model vocabulary bound, and verify alignment.
//! No HTTP body rewriting or dispatch is performed here.

use serde_json::Value;
use vllm_tokenizer::Tokenizer;

use crate::protocols::spec::{CompletionRequest, PromptInput};

/// Request-owned input tokens, borrowed by policy and reused across retries.
#[derive(Debug)]
pub struct PreparedCompletion {
    token_ids: Vec<u32>,
}

impl PreparedCompletion {
    pub fn token_ids(&self) -> &[u32] {
        &self.token_ids
    }
}

/// Reject tokenizer definitions that silently truncate or pad exact input.
///
/// Validate the same definition that will be loaded into the tokenizer. This
/// checks Hugging Face syntax, not asset provenance or worker conformance, and
/// cannot certify an already-created arbitrary `Tokenizer` trait object.
pub fn validate_completion_tokenizer_definition(definition: &Value) -> Result<(), String> {
    for field in ["truncation", "padding"] {
        if definition.get(field).is_some_and(|value| !value.is_null()) {
            return Err(format!("completion tokenizer {field} must be disabled"));
        }
    }
    serde_json::from_value::<tokenizers::Tokenizer>(definition.clone())
        .map(|_| ())
        .map_err(|error| format!("invalid completion tokenizer definition: {error}"))
}

/// Prepare one text prompt or one token-ID sequence without mutating the request.
///
/// Text defaults to `add_special_tokens = true`; explicit IDs are never encoded
/// or given additional special tokens. IDs are bounded by the configured model
/// vocabulary, which may exceed the tokenizer vocabulary. Unsupported input
/// modifiers fail closed; typed sampling/output fields are left untouched.
/// Empty encoded input is rejected rather than treated as a positive prefix.
pub fn prepare_completion(
    request: &CompletionRequest,
    tokenizer: &dyn Tokenizer,
    model_vocab_size: u32,
) -> Result<PreparedCompletion, String> {
    if model_vocab_size == 0 {
        return Err("completion model vocabulary must be nonzero".into());
    }
    if request.lora_path.is_some() {
        return Err("completion preparation does not support lora_path".into());
    }
    if request.session_params.is_some() {
        return Err("completion preparation does not support session_params".into());
    }
    if request.suffix.is_some() {
        return Err("completion preparation does not support suffix".into());
    }
    let mut add_special_tokens = true;
    for (field, value) in &request.other {
        if field != "add_special_tokens" {
            return Err(format!("completion preparation does not support {field}"));
        }
        add_special_tokens = value
            .as_bool()
            .ok_or_else(|| "completion add_special_tokens must be a boolean".to_string())?;
    }
    let token_ids = match &request.prompt {
        PromptInput::String(text) => tokenizer
            .encode(text, add_special_tokens)
            .map_err(|error| format!("completion tokenization failed: {error}"))?,
        PromptInput::IntArray(ids) => ids
            .iter()
            .map(|&id| {
                u32::try_from(id)
                    .map_err(|_| format!("completion prompt contains negative token ID {id}"))
            })
            .collect::<Result<Vec<_>, _>>()?,
        PromptInput::StringArray(_) | PromptInput::IntBatch(_) => {
            return Err("completion preparation does not support batched prompts".into());
        }
    };
    if token_ids.is_empty() {
        return Err("completion preparation requires nonempty token input".into());
    }
    if let Some(id) = token_ids.iter().find(|&&id| id >= model_vocab_size) {
        return Err(format!(
            "completion token ID {id} is outside model vocabulary size {model_vocab_size}"
        ));
    }
    Ok(PreparedCompletion { token_ids })
}
