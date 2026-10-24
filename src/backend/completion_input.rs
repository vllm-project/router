//! Startup snapshot for the opt-in, static HTTP Completion input contract.
//!
//! The operator binds an alias to immutable local frontend assets and an
//! identically configured Worker cohort. This is not remote attestation.

use std::future::Future;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde_json::Value;
use vllm_tokenizer::DynTokenizer;

use super::preprocess::FrontendHandle;
use super::{validate_completion_tokenizer_definition, EngineFrontend};

#[derive(Clone)]
pub(crate) struct CompletionInput {
    /// Exact served alias; distinct from local asset paths and event identity.
    pub model: String,
    pub cache_model: String,
    pub vocab_size: u32,
    pub tokenizer: DynTokenizer,
}

impl std::fmt::Debug for CompletionInput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompletionInput")
            .field("model", &self.model)
            .field("cache_model", &self.cache_model)
            .field("vocab_size", &self.vocab_size)
            .finish_non_exhaustive()
    }
}

impl CompletionInput {
    pub async fn from_env(frontend: &EngineFrontend) -> Result<Option<Self>> {
        let Some(model) = std::env::var("VLLM_ROUTER_COMPLETION_MODEL_ALIAS").ok() else {
            return Ok(None);
        };
        let path = PathBuf::from(std::env::var("VLLM_ROUTER_MODEL").context(
            "Completion requires VLLM_ROUTER_MODEL as an immutable local model directory",
        )?);
        let cache_model =
            std::env::var("VLLM_ROUTER_COMPLETION_CACHE_MODEL").unwrap_or_else(|_| model.clone());
        Self::load(
            &model,
            &cache_model,
            &path,
            frontend.completion_frontend(&model),
        )
        .await
        .map(Some)
    }

    pub(super) async fn load(
        model: &str,
        cache_model: &str,
        path: &Path,
        frontend: impl Future<Output = Result<FrontendHandle>>,
    ) -> Result<Self> {
        if model.trim().is_empty() || model != model.trim() {
            bail!("VLLM_ROUTER_COMPLETION_MODEL_ALIAS must be a nonempty exact served-model alias");
        }
        if cache_model.trim().is_empty() || cache_model != cache_model.trim() {
            bail!(
                "VLLM_ROUTER_COMPLETION_CACHE_MODEL must be a nonempty exact cache-model identity"
            );
        }
        let path = path.canonicalize().context("Completion model directory")?;
        if !path.is_dir() {
            bail!("Completion model assets must be a local directory");
        }
        let config: Value =
            serde_json::from_slice(&tokio::fs::read(path.join("config.json")).await?)
                .context("Completion model config")?;
        let vocab_size = config
            .get("vocab_size")
            .and_then(Value::as_u64)
            .and_then(|size| u32::try_from(size).ok())
            .filter(|size| *size != 0)
            .context("Completion config.json requires a nonzero u32 vocab_size")?;
        // Do not poll the generic resolver until the local-only Completion
        // contract is valid: its normal gRPC path may resolve remote assets.
        let handle = frontend.await?;
        // Resolve through the existing alias-keyed cache. A cache hit must
        // validate its loaded source, not a freshly recomputed environment path.
        let source = Path::new(
            handle
                .asset_source()
                .context("Completion requires a real tokenizer backend")?,
        )
        .canonicalize()
        .context("Completion frontend asset directory")?;
        if !source.is_dir() {
            bail!("Completion frontend assets must be a local directory");
        }
        let definition: Value =
            serde_json::from_slice(&tokio::fs::read(source.join("tokenizer.json")).await?)
                .context("Completion tokenizer definition")?;
        validate_completion_tokenizer_definition(&definition).map_err(anyhow::Error::msg)?;
        let tokenizer = handle
            .tokenizer()
            .context("Completion requires a real tokenizer backend")?;
        Ok(Self {
            model: model.to_owned(),
            cache_model: cache_model.to_owned(),
            vocab_size,
            tokenizer,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    #[tokio::test]
    async fn completion_invalid_local_contract_never_polls_frontend_resolver() {
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(directory.path().join("config.json"), r#"{"vocab_size":0}"#).unwrap();
        let missing = directory.path().join("not-a-local-model");
        let attempts = AtomicUsize::new(0);
        for (model, cache_model, path, expected) in [
            (" public", "engine", directory.path(), "MODEL_ALIAS"),
            ("public", " engine", directory.path(), "CACHE_MODEL"),
            ("public", "engine", missing.as_path(), "model directory"),
            ("public", "engine", directory.path(), "vocab_size"),
        ] {
            let error = CompletionInput::load(model, cache_model, path, async {
                attempts.fetch_add(1, Ordering::Relaxed);
                Err(anyhow::anyhow!("frontend resolver must not be polled"))
            })
            .await
            .unwrap_err();
            assert!(error.to_string().contains(expected), "{error:#}");
            assert_eq!(attempts.load(Ordering::Relaxed), 0);
        }
    }
}
