//! Whether a tokenizer always maps the same text to the same IDs.
use anyhow::Result;
use vllm_text::backend::hf::{ResolvedModelFiles, TokenizerSource};

/// Check source files because the tokenizer trait cannot report determinism.
pub(crate) async fn deterministic_model(model: &str) -> Result<bool> {
    let files = ResolvedModelFiles::new(model).await?;
    match files.tokenizer {
        TokenizerSource::Tiktoken(_) | TokenizerSource::Tekken(_) => Ok(true),
        TokenizerSource::HuggingFace(path) => {
            let json: serde_json::Value = serde_json::from_slice(&std::fs::read(path)?)?;
            Ok(deterministic_hf_model(&json["model"]))
        }
    }
}

pub(crate) fn deterministic_hf_model(model: &serde_json::Value) -> bool {
    match model["type"].as_str() {
        Some("BPE") => model["dropout"].is_null() || model["dropout"].as_f64() == Some(0.0),
        // Normal encode does not sample.
        Some("WordPiece" | "WordLevel" | "Unigram") => true,
        _ => false,
    }
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn only_known_deterministic_models_are_cacheable() {
        for dropout in [json!(null), json!(0.0)] {
            assert!(deterministic_hf_model(
                &json!({"type": "BPE", "dropout": dropout})
            ));
        }
        for model in [
            json!({"type": "BPE", "dropout": 0.1}),
            json!({"type": "BPE", "dropout": "unknown"}),
            json!({"type": "future"}),
            json!(null),
        ] {
            assert!(!deterministic_hf_model(&model));
        }
    }
}
