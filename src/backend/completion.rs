//! Pure preparation for the restricted single-prompt Completion contract.
//!
//! This does not authenticate tokenizer assets or the remote worker's input/hash
//! contract. Runtime integration must validate the tokenizer definition before
//! loading it, supply the worker's model vocabulary bound, and verify alignment.
//! No HTTP body rewriting or dispatch is performed here.

use serde_json::Value;
use vllm_tokenizer::Tokenizer;

use crate::protocols::spec::{CompletionRequest, PromptInput};

/// Distinguishes an ineligible optimization from invalid input or service failure.
#[derive(Debug, PartialEq, Eq)]
pub enum CompletionPreparationError {
    Unsupported(String),
    InvalidRequest(String),
    ServiceFailure(String),
}

impl std::fmt::Display for CompletionPreparationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unsupported(message)
            | Self::InvalidRequest(message)
            | Self::ServiceFailure(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for CompletionPreparationError {}

/// Reject tokenizer definitions that silently truncate or pad exact input.
///
/// Validate the same definition that will be loaded into the tokenizer. This
/// checks this `tokenizer.json`, not asset provenance or worker conformance,
/// and cannot certify an already-created arbitrary `Tokenizer` trait object.
/// `HuggingFaceTokenizer::new_hf` also merges added tokens from the sibling
/// `tokenizer_config.json`; callers must keep both files immutable and validate
/// the effective loaded backend separately when establishing worker alignment.
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
/// This diagnostic adapter preserves error text but discards error categories;
/// routing fallback decisions must use [`prepare_completion_classified`].
pub fn prepare_completion(
    request: &CompletionRequest,
    tokenizer: &dyn Tokenizer,
    model_vocab_size: u32,
) -> Result<Vec<u32>, String> {
    prepare_completion_classified(request, tokenizer, model_vocab_size)
        .map_err(|error| error.to_string())
}

/// Classify preparation without certifying the remote worker's serving contract.
///
/// Success is exact preparation for this restricted input contract. Explicit IDs
/// and the special-token switch are checked before unsupported modifiers, so an
/// ineligible optimization cannot mask known invalid input. Only `Unsupported`
/// may use an independently valid ordinary execution path.
pub fn prepare_completion_classified(
    request: &CompletionRequest,
    tokenizer: &dyn Tokenizer,
    model_vocab_size: u32,
) -> Result<Vec<u32>, CompletionPreparationError> {
    use CompletionPreparationError::{InvalidRequest, ServiceFailure, Unsupported};

    if model_vocab_size == 0 {
        return Err(ServiceFailure(
            "completion model vocabulary must be nonzero".into(),
        ));
    }
    match &request.prompt {
        PromptInput::IntArray(ids) => validate_prompt_ids(ids, model_vocab_size)?,
        PromptInput::IntBatch(batch) => {
            for ids in batch {
                validate_prompt_ids(ids, model_vocab_size)?;
            }
        }
        _ => {}
    }
    let add_special_tokens = request
        .other
        .get("add_special_tokens")
        .map(|value| {
            value.as_bool().ok_or_else(|| {
                InvalidRequest("completion add_special_tokens must be a boolean".into())
            })
        })
        .transpose()?
        .unwrap_or(true);
    if request.lora_path.is_some() {
        return Err(Unsupported(
            "completion preparation does not support lora_path".into(),
        ));
    }
    if request.session_params.is_some() {
        return Err(Unsupported(
            "completion preparation does not support session_params".into(),
        ));
    }
    if request.suffix.is_some() {
        return Err(Unsupported(
            "completion preparation does not support suffix".into(),
        ));
    }
    for (field, value) in &request.other {
        // vLLM 0.29 Completion priority controls engine scheduling, not input
        // rendering or cache identity. Other extension fields remain a valid
        // ordinary-path fallback until their input and namespace contracts are
        // established; in particular LoRA and salt are not sampling options.
        if field != "add_special_tokens" && !(field == "priority" && value.as_i64().is_some()) {
            return Err(Unsupported(format!(
                "completion preparation does not support {field}"
            )));
        }
    }
    let token_ids = match &request.prompt {
        PromptInput::String(text) => {
            let ids = tokenizer
                .encode(text, add_special_tokens)
                .map_err(|error| {
                    ServiceFailure(format!("completion tokenization failed: {error}"))
                })?;
            if let Some(id) = ids.iter().find(|&&id| id >= model_vocab_size) {
                return Err(ServiceFailure(format!(
                    "completion token ID {id} is outside model vocabulary size {model_vocab_size}"
                )));
            }
            ids
        }
        PromptInput::IntArray(ids) => ids.iter().map(|&id| id as u32).collect(),
        PromptInput::StringArray(_) | PromptInput::IntBatch(_) => {
            return Err(Unsupported(
                "completion preparation does not support batched prompts".into(),
            ));
        }
    };
    if token_ids.is_empty() {
        return Err(InvalidRequest(
            "completion preparation requires nonempty token input".into(),
        ));
    }
    Ok(token_ids)
}

fn validate_prompt_ids(
    ids: &[i32],
    model_vocab_size: u32,
) -> Result<(), CompletionPreparationError> {
    use CompletionPreparationError::InvalidRequest;
    if ids.is_empty() {
        return Err(InvalidRequest(
            "completion preparation requires nonempty token input".into(),
        ));
    }
    for &id in ids {
        if id < 0 {
            return Err(InvalidRequest(format!(
                "completion prompt contains negative token ID {id}"
            )));
        }
        if id as u32 >= model_vocab_size {
            return Err(InvalidRequest(format!(
                "completion token ID {id} is outside model vocabulary size {model_vocab_size}"
            )));
        }
    }
    Ok(())
}
