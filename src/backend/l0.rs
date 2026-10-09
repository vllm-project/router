//! Shared L0 encoding cache for gRPC models.
use std::mem::size_of;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Once};

use anyhow::{bail, Context, Result};
use vllm_text::backend::hf::{ResolvedModelFiles, TokenizerSource};
use vllm_text::Prompt;
use vllm_tokenizer::DynTokenizer;

use crate::tokenizer::cache::{entry_overhead_bytes, EncodeCache};
use crate::tokenizer::TokenizerCacheConfig;

#[derive(Clone, Hash, PartialEq, Eq)]
pub(crate) struct EncodingKey {
    instance: u64,
    add_special_tokens: bool,
    cache_salt: Option<String>,
    text: String,
}

pub(crate) type FrontendCache = EncodeCache<EncodingKey, Vec<u32>>;

#[derive(Clone)]
pub(crate) struct PromptEncoder {
    tokenizer: DynTokenizer,
    cache: Option<Arc<FrontendCache>>,
    instance: u64,
    model_id: Arc<str>,
    oversized_warning: Arc<Once>,
}

impl PromptEncoder {
    // Caching requires a deterministic tokenizer with fixed configuration.
    pub(crate) fn new(
        model_id: &str,
        tokenizer: DynTokenizer,
        cache: Option<Arc<FrontendCache>>,
    ) -> Self {
        static NEXT_INSTANCE: AtomicU64 = AtomicU64::new(0);
        Self {
            tokenizer,
            cache,
            // Clones share an ID; new instances get distinct IDs.
            instance: NEXT_INSTANCE.fetch_add(1, Ordering::Relaxed),
            model_id: Arc::from(model_id),
            oversized_warning: Arc::new(Once::new()),
        }
    }

    pub(crate) fn encode(
        &self,
        prompt: Prompt,
        add_special_tokens: bool,
        cache_salt: Option<&str>,
    ) -> vllm_tokenizer::Result<Vec<u32>> {
        let text = match prompt {
            Prompt::TokenIds(ids) => return Ok(ids),
            Prompt::Text(text) => text,
        };
        let Some(cache) = &self.cache else {
            return self.tokenizer.encode(&text, add_special_tokens);
        };
        let key = EncodingKey {
            instance: self.instance,
            add_special_tokens,
            cache_salt: cache_salt.map(str::to_owned),
            text,
        };
        cache.get_or_encode(
            &key,
            || self.tokenizer.encode(&key.text, add_special_tokens),
            |ids| {
                let bytes = entry_overhead_bytes::<EncodingKey, Vec<u32>>()
                    + key.text.len()
                    + key.cache_salt.as_ref().map_or(0, String::len)
                    + size_of::<Vec<u32>>()
                    + ids.len() * size_of::<u32>();
                if bytes > cache.config().max_entry_bytes {
                    self.oversized_warning.call_once(|| {
                        tracing::warn!(
                            model_id = %self.model_id,
                            estimated_bytes = bytes,
                            max_entry_bytes = cache.config().max_entry_bytes,
                            "L0 entry exceeds cache limit; encoding will not be cached"
                        );
                    });
                }
                bytes
            },
        )
    }
}

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

fn deterministic_hf_model(model: &serde_json::Value) -> bool {
    match model["type"].as_str() {
        Some("BPE") => model["dropout"].is_null() || model["dropout"].as_f64() == Some(0.0),
        // Normal encode does not sample.
        Some("WordPiece" | "WordLevel" | "Unigram") => true,
        _ => false,
    }
}

pub(crate) fn config_from_env() -> Result<Option<TokenizerCacheConfig>> {
    parse_config(|name| std::env::var(name).ok())
}

fn parse_config(read: impl Fn(&str) -> Option<String>) -> Result<Option<TokenizerCacheConfig>> {
    match read("VLLM_ROUTER_L0_CACHE").as_deref() {
        None | Some("0" | "false") => return Ok(None),
        Some("1" | "true") => {}
        _ => bail!("VLLM_ROUTER_L0_CACHE must be 0, 1, false or true"),
    }
    let mut config = TokenizerCacheConfig::default();
    for (name, value) in [
        ("VLLM_ROUTER_L0_MAX_ENTRIES", &mut config.max_entries),
        ("VLLM_ROUTER_L0_MAX_BYTES", &mut config.max_bytes),
        (
            "VLLM_ROUTER_L0_MAX_ENTRY_BYTES",
            &mut config.max_entry_bytes,
        ),
    ] {
        if let Some(raw) = read(name) {
            *value = raw
                .parse()
                .with_context(|| format!("invalid {name}: {raw}"))?;
        }
    }
    config.validate()?;
    Ok(Some(config))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::sync::atomic::AtomicUsize;
    use vllm_tokenizer::{HuggingFaceTokenizer, Tokenizer, TokenizerError};

    struct CountingTokenizer {
        inner: HuggingFaceTokenizer,
        calls: AtomicUsize,
        suffix: u32,
    }

    impl CountingTokenizer {
        fn new(suffix: u32) -> Arc<Self> {
            Arc::new(Self {
                inner: HuggingFaceTokenizer::new_hf(std::path::Path::new(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/tests/fixtures/tokenizer/byte_level_bpe.json"
                )))
                .unwrap(),
                calls: AtomicUsize::new(0),
                suffix,
            })
        }
    }

    impl Tokenizer for CountingTokenizer {
        fn encode(&self, text: &str, special: bool) -> vllm_tokenizer::Result<Vec<u32>> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            if text == "fail" {
                return Err(TokenizerError("test failure".into()));
            }
            let mut ids = self.inner.encode(text, special)?;
            ids.extend([self.suffix, u32::from(special)]);
            Ok(ids)
        }
        fn encode_ordinary(&self, text: &str) -> vllm_tokenizer::Result<Vec<u32>> {
            self.inner.encode_ordinary(text)
        }
        fn decode(&self, ids: &[u32], special: bool) -> vllm_tokenizer::Result<String> {
            self.inner.decode(ids, special)
        }
        fn token_to_id(&self, token: &str) -> Option<u32> {
            self.inner.token_to_id(token)
        }
        fn id_to_token(&self, id: u32) -> Option<String> {
            self.inner.id_to_token(id)
        }
    }

    fn encode(encoder: &PromptEncoder, text: &str, special: bool) -> Vec<u32> {
        encoder
            .encode(Prompt::Text(text.into()), special, None)
            .unwrap()
    }

    #[test]
    fn hits_skip_encoding_and_isolate_instances_options_and_prompts() {
        let cache = Arc::new(FrontendCache::new(Default::default()).unwrap());
        let inner = CountingTokenizer::new(10);
        let encoder = PromptEncoder::new("test-model", inner.clone(), Some(cache.clone()));
        let uncached = PromptEncoder::new("test-model", CountingTokenizer::new(10), None);
        for text in ["Hello", "hello", "你好 👋", ""] {
            for special in [false, true] {
                let expected = encode(&uncached, text, special);
                assert_eq!(encode(&encoder, text, special), expected);
                assert_eq!(encode(&encoder.clone(), text, special), expected);
            }
        }
        assert_eq!(inner.calls.load(Ordering::Relaxed), 8);
        assert_eq!(cache.stats().hits, 8);
        // A new encoder must miss, even with the same tokenizer Arc.
        let other = PromptEncoder::new("test-model", inner.clone(), Some(cache.clone()));
        encode(&other, "Hello", false);
        assert_eq!(inner.calls.load(Ordering::Relaxed), 9);
        let changed = PromptEncoder::new(
            "test-model",
            CountingTokenizer::new(20),
            Some(cache.clone()),
        );
        assert_ne!(
            encode(&changed, "Hello", false),
            encode(&encoder, "Hello", false)
        );
        assert_eq!(cache.stats().entries, 10);
    }

    #[test]
    fn cache_salt_isolates_entries_and_counts_retained_bytes() {
        let cache = Arc::new(FrontendCache::new(Default::default()).unwrap());
        let inner = CountingTokenizer::new(0);
        let encoder = PromptEncoder::new("test-model", inner.clone(), Some(cache.clone()));
        let expected = encode(
            &PromptEncoder::new("test-model", CountingTokenizer::new(0), None),
            "Hello",
            false,
        );
        let mut unsalted_bytes = 0;
        for salt in [None, Some(""), Some("tenant-a"), Some("tenant-b")] {
            let before = cache.stats().bytes;
            for _ in 0..2 {
                assert_eq!(
                    encoder
                        .encode(Prompt::Text("Hello".into()), false, salt)
                        .unwrap(),
                    expected
                );
            }
            let retained = cache.stats().bytes - before;
            if salt.is_none() {
                unsalted_bytes = retained;
            }
            assert_eq!(retained, unsalted_bytes + salt.map_or(0, str::len));
        }
        let stats = cache.stats();
        assert_eq!((stats.hits, stats.misses, stats.entries), (4, 4, 4));
        assert_eq!(inner.calls.load(Ordering::Relaxed), 4);

        let small = Arc::new(
            FrontendCache::new(TokenizerCacheConfig {
                max_entry_bytes: unsalted_bytes,
                ..Default::default()
            })
            .unwrap(),
        );
        let encoder = PromptEncoder::new("test-model", inner, Some(small.clone()));
        assert_eq!(encode(&encoder, "Hello", false), expected);
        assert_eq!(
            encoder
                .encode(Prompt::Text("Hello".into()), false, Some("tenant-a"))
                .unwrap(),
            expected
        );
        assert_eq!(small.stats().oversized, 1);
        assert_eq!(small.stats().bytes, unsalted_bytes);
    }

    #[test]
    fn errors_and_token_ids_are_never_cached() {
        let cache = Arc::new(FrontendCache::new(Default::default()).unwrap());
        let inner = CountingTokenizer::new(0);
        let encoder = PromptEncoder::new("test-model", inner.clone(), Some(cache.clone()));
        for _ in 0..2 {
            assert!(encoder
                .encode(Prompt::Text("fail".into()), false, None)
                .is_err());
            assert_eq!(
                encoder
                    .encode(Prompt::TokenIds(vec![5, 4, 3]), true, Some("tenant-a"))
                    .unwrap(),
                [5, 4, 3]
            );
        }
        assert_eq!(inner.calls.load(Ordering::Relaxed), 2);
        assert_eq!(cache.stats().misses, 2);
        assert_eq!(cache.stats().entries, 0);
        assert_eq!(cache.stats().bytes, 0);
    }

    #[test]
    fn shared_budget_evicts_across_models_and_skips_oversized_inputs() {
        let measure = Arc::new(FrontendCache::new(Default::default()).unwrap());
        encode(
            &PromptEncoder::new(
                "test-model",
                CountingTokenizer::new(0),
                Some(measure.clone()),
            ),
            "a",
            false,
        );
        let per_entry = measure.stats().bytes;
        for config in [
            TokenizerCacheConfig {
                max_entries: 1,
                max_bytes: per_entry * 4,
                max_entry_bytes: per_entry * 2,
            },
            TokenizerCacheConfig {
                max_entries: 100,
                max_bytes: per_entry,
                max_entry_bytes: per_entry,
            },
        ] {
            let cache = Arc::new(FrontendCache::new(config).unwrap());
            let a =
                PromptEncoder::new("test-model", CountingTokenizer::new(0), Some(cache.clone()));
            let b =
                PromptEncoder::new("test-model", CountingTokenizer::new(1), Some(cache.clone()));
            encode(&a, "a", false);
            encode(&b, "a", false);
            assert_eq!(cache.stats().entries, 1);
            assert_eq!(cache.stats().bytes, per_entry);
            assert_eq!(cache.stats().evictions, 1);
            encode(&a, "a", false);
            assert_eq!(cache.stats().evictions, 2);
            encode(&a, &"long ".repeat(100), false);
            assert_eq!(cache.stats().oversized, 1);
            assert_eq!(cache.stats().bytes, per_entry);
            cache.clear();
            assert_eq!(cache.stats().bytes, 0);
        }
    }

    #[test]
    fn concurrent_models_stay_within_one_budget() {
        let config = TokenizerCacheConfig {
            max_entries: 8,
            max_bytes: 4096,
            max_entry_bytes: 1024,
        };
        let cache = Arc::new(FrontendCache::new(config).unwrap());
        let a = PromptEncoder::new("test-model", CountingTokenizer::new(0), Some(cache.clone()));
        let b = PromptEncoder::new("test-model", CountingTokenizer::new(1), Some(cache.clone()));
        std::thread::scope(|scope| {
            for t in 0..8 {
                let encoder = if t % 2 == 0 { a.clone() } else { b.clone() };
                scope.spawn(move || {
                    for i in 0..100 {
                        let text = format!("Hello {}", i % 8);
                        let expected = encoder.tokenizer.encode(&text, false).unwrap();
                        assert_eq!(encode(&encoder, &text, false), expected);
                    }
                });
            }
        });
        let stats = cache.stats();
        assert_eq!(stats.hits + stats.misses, 800);
        assert!(stats.entries <= 8);
        assert!(stats.bytes <= 4096);
        assert!(stats.evictions > 0);
    }

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

    #[test]
    fn config_is_opt_in_and_validates_shared_limits() {
        let parse = |values: &[(&str, &str)]| {
            parse_config(|key| {
                values
                    .iter()
                    .find(|(k, _)| *k == key)
                    .map(|(_, v)| v.to_string())
            })
        };
        assert_eq!(parse(&[]).unwrap(), None);
        assert_eq!(parse(&[("VLLM_ROUTER_L0_MAX_BYTES", "bad")]).unwrap(), None);
        assert_eq!(
            parse(&[("VLLM_ROUTER_L0_CACHE", "true")]).unwrap(),
            Some(Default::default())
        );
        assert!(parse(&[("VLLM_ROUTER_L0_CACHE", "maybe")]).is_err());
        for (name, value) in [
            ("VLLM_ROUTER_L0_MAX_ENTRIES", "0"),
            ("VLLM_ROUTER_L0_MAX_BYTES", "1"),
            ("VLLM_ROUTER_L0_MAX_ENTRY_BYTES", "bad"),
        ] {
            assert!(parse(&[("VLLM_ROUTER_L0_CACHE", "1"), (name, value)]).is_err());
        }
    }
}
